from __future__ import annotations

import numpy as np
import pandas as pd

import analyzer


def _market_frame(include_vix: bool = True, periods: int = 300) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    base = np.linspace(100.0, 130.0, periods)
    frame = pd.DataFrame({"SPY Close": base}, index=index)
    if include_vix:
        frame["VIX"] = np.linspace(18.0, 24.0, periods) + np.sin(np.arange(periods) / 9.0)
    return frame


class _NoOpLog:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def test_regime_model_uses_spy_and_vix_when_available():
    model = analyzer.RegimeModel(_market_frame(include_vix=True), {"regime_min_rows": 252})

    obs, mode = model._build_hmm_observations(log_details=False)

    assert mode == "SPY+VIX"
    assert list(obs.columns) == ["spy_ret", "vix_z"]
    assert len(obs) >= 252


def test_regime_model_falls_back_to_spy_only_without_vix():
    model = analyzer.RegimeModel(_market_frame(include_vix=False), {"regime_min_rows": 252})

    obs, mode = model._build_hmm_observations(log_details=False)

    assert mode == "SPY"
    assert list(obs.columns) == ["spy_ret"]
    assert len(obs) >= 252


def test_regime_train_fits_hmm_only_through_training_cutoff(monkeypatch):
    index = pd.date_range("2023-01-01", periods=120, freq="D", tz="UTC")
    X = pd.DataFrame(
        {
            "feature_a": np.sin(np.arange(120) / 5.0),
            "feature_b": np.cos(np.arange(120) / 7.0),
        },
        index=index,
    )
    y = pd.Series(np.tile([0, 1, 2], 40), index=index)
    captured = {}

    def fake_fit_hmm(self, fit_end=None):
        captured["fit_end"] = fit_end
        return pd.Series(1, index=index, name="regime")

    monkeypatch.setattr(analyzer.RegimeModel, "_fit_hmm", fake_fit_hmm)
    monkeypatch.setattr(analyzer, "HAS_XGB", False)
    monkeypatch.setattr(analyzer, "log", _NoOpLog())
    model = analyzer.RegimeModel(_market_frame(periods=120), {"train_split": 0.60, "feat_var_thresh": 0.0})

    assert model.train(X, y) is True
    assert captured["fit_end"] == index[71]


def test_fit_hmm_does_not_fit_future_market_rows(monkeypatch):
    class TrackingHMM:
        fit_rows = None

        def __init__(self, n_components=2, covariance_type="full", n_iter=100, random_state=42):
            self.n_components = n_components
            self.means_ = np.array([[-0.01, 0.0], [0.01, 0.0]], dtype=float)
            self.covars_ = np.array([np.eye(2), np.eye(2)], dtype=float)

        def fit(self, obs):
            TrackingHMM.fit_rows = len(obs)
            return self

        def predict(self, obs):
            return (np.asarray(obs)[:, 0] >= 0.0).astype(int)

    monkeypatch.setattr(analyzer, "HAS_HMM", True)
    monkeypatch.setattr(analyzer._hmm, "GaussianHMM", TrackingHMM)
    monkeypatch.setattr(analyzer, "log", _NoOpLog())
    market = _market_frame(include_vix=True, periods=300)
    model = analyzer.RegimeModel(market, {"regime_min_rows": 50})

    regimes = model._fit_hmm(fit_end=market.index[199])

    assert TrackingHMM.fit_rows == 200
    assert regimes is not None
    assert len(regimes) == 300


def test_trade_signal_policy_blocks_uncertain_predictions():
    label, reason = analyzer._trade_signal_from_policy(
        np.array([0.15, 0.30, 0.55]),
        {"buy_threshold": 0.50, "sell_threshold": 0.50, "margin_threshold": 0.05, "cost_gate": 0.01},
        {"BUY": 0.05, "SELL": 0.05},
        tx_cost=0.01,
        uncertainty_ok=False,
    )

    assert label == 1
    assert reason == "conformal_gate"


def test_trade_signal_policy_respects_margin_gate_before_edge_checks():
    label, reason = analyzer._trade_signal_from_policy(
        np.array([0.31, 0.34, 0.35]),
        {"buy_threshold": 0.34, "sell_threshold": 0.34, "margin_threshold": 0.08, "cost_gate": 0.0},
        {"BUY": 0.20, "SELL": 0.20},
        tx_cost=0.0,
    )

    assert label == 1
    assert reason == "margin_gate"


def test_trade_signal_policy_prefers_buy_when_probability_and_edge_clear_threshold():
    label, reason = analyzer._trade_signal_from_policy(
        np.array([0.10, 0.15, 0.75]),
        {"buy_threshold": 0.60, "sell_threshold": 0.60, "margin_threshold": 0.10, "cost_gate": 0.02},
        {"BUY": 0.08, "SELL": 0.08},
        tx_cost=0.02,
    )

    assert label == 2
    assert reason is None


def test_execution_state_marks_model_guard_when_no_deployable_candidate():
    state = analyzer._build_execution_state(
        {
            "signal": "BUY",
            "selection_status": "reference_only_no_deployable_candidate",
            "deployment_eligible": False,
        },
        {
            "deployment_eligible": False,
            "eligibility_failures": ["conformal_unusable"],
        },
        {"router_status": "fallback", "fallback_reason": "no_eligible_family", "routing_actionable": False},
    )

    assert state["execution_status"] == "ABSTAIN_MODEL_UNRELIABLE"
    assert state["execution_gate"] == "model_guard"
    assert state["abstain_reason"] == "model_guard"
    assert state["eligibility_failures"] == ["conformal_unusable"]


def test_execution_state_marks_uncertain_when_conformal_is_not_usable():
    state = analyzer._build_execution_state(
        {
            "signal": "BUY",
            "probability_margin": 0.03,
            "prediction_set": ["BUY", "HOLD"],
            "set_size": 2,
        },
        {
            "deployment_eligible": True,
            "conformal": {
                "usable_for_execution": False,
                "usability_failures": ["singleton_rate_too_low"],
            },
            "decision_policy": {"margin_threshold": 0.05, "cost_gate": 0.01},
        },
        {"router_status": "active", "routing_actionable": True},
    )

    assert state["execution_status"] == "ABSTAIN_UNCERTAIN"
    assert state["execution_gate"] == "conformal_gate"
    assert state["execution_gate_details"]["conformal_failures"] == ["singleton_rate_too_low"]
    assert state["execution_gate_details"]["set_size"] == 2


def test_execution_state_marks_hold_signal_as_neutral():
    state = analyzer._build_execution_state(
        {"signal": "HOLD", "probability_margin": 0.01},
        {
            "deployment_eligible": True,
            "decision_policy": {"margin_threshold": 0.05, "cost_gate": 0.01},
        },
        {"router_status": "active", "routing_actionable": True},
    )

    assert state["execution_status"] == "HOLD_NEUTRAL"
    assert state["execution_gate"] == "neutral_hold"
    assert state["abstain_reason"] == "neutral_hold"
