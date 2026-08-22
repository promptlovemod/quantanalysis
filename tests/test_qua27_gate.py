from __future__ import annotations

from utils.portfolio_tools import DEFAULT_BENCHMARK_THRESHOLDS, build_quality_gate


def _selected_scope() -> dict:
    return {
        "selected_candidate_model": "PatchTST",
        "selected_candidate_family": "dl_family",
        "selected_candidate_is_tree": False,
        "static_backtest_scope": "selected_candidate",
        "walkforward_scope": "selected_candidate",
        "cpcv_scope": "selected_candidate",
    }


def _make_signal(evidence_scope: dict | None = None, wf_sharpe=0.8, cpcv_p5=0.2) -> dict:
    signal = {
        "pipeline_status": "OK",
        "walkforward_backtest": {"wf_sharpe": wf_sharpe},
        "cpcv": {"sharpe_p5": cpcv_p5},
        "signal": {"execution_status": "ACTIONABLE"},
        "selection": {},
    }
    if evidence_scope is not None:
        signal["evidence_scope"] = evidence_scope
    return signal


def _make_diag(include_reliability: bool = True) -> dict:
    diag = {
        "seed_stability": {"stable": True},
        "calibration": {"ece": 0.05, "classwise": {"BUY": {"recall": 0.6}}},
    }
    if include_reliability:
        diag["reliability_score"] = 4.0
    return diag


def _make_bundle(signal: dict, diag: dict) -> dict:
    return {
        "signal_data": signal,
        "fund_data": {},
        "mc_data": {},
        "diag_data": diag,
        "artifact_presence": {"signal_data": True, "fund_data": False, "mc_data": False, "diag_data": True},
    }


def test_tree_family_scoped_walkforward_excluded_from_gate_metrics():
    bundle = _make_bundle(
        _make_signal(evidence_scope={
            "selected_candidate_model": "GBM",
            "walkforward_scope": "tree_family_diagnostics",
            "cpcv_scope": "tree_family_diagnostics",
        }),
        _make_diag(),
    )

    gate = build_quality_gate({"TREE": bundle}, DEFAULT_BENCHMARK_THRESHOLDS)

    assert gate["metrics"]["median_wf_sharpe"] is None
    assert gate["metrics"]["positive_wf_share"] is None
    assert gate["metrics"]["median_cpcv_p5"] is None
    assert gate["coverage"]["walkforward"]["count"] == 1
    assert gate["coverage"]["walkforward_scope_eligible"]["count"] == 0
    assert gate["coverage"]["cpcv_scope_eligible"]["count"] == 0
    assert gate["data_availability"]["wf_sharpe"]["available"] is False
    assert gate["data_availability"]["cpcv_p5"]["available"] is False
    assert "legacy_evidence_scope_assumed" not in gate["per_ticker"]["TREE"]["warnings"]
    assert gate["per_ticker"]["TREE"]["metrics"]["wf_sharpe"] == 0.8


def test_missing_evidence_scope_treated_as_legacy_with_warning():
    bundle = _make_bundle(_make_signal(), _make_diag())

    gate = build_quality_gate({"LEGACY": bundle}, DEFAULT_BENCHMARK_THRESHOLDS)

    assert gate["metrics"]["median_wf_sharpe"] == 0.8
    assert gate["metrics"]["median_cpcv_p5"] == 0.2
    assert gate["coverage"]["walkforward_scope_eligible"]["count"] == 1
    assert gate["coverage"]["cpcv_scope_eligible"]["count"] == 1
    assert "legacy_evidence_scope_assumed" in gate["per_ticker"]["LEGACY"]["warnings"]


def test_missing_reliability_reported_as_insufficient_evidence():
    bundle = _make_bundle(
        _make_signal(evidence_scope=_selected_scope()),
        _make_diag(include_reliability=False),
    )

    gate = build_quality_gate({"NOREL": bundle}, DEFAULT_BENCHMARK_THRESHOLDS)

    assert gate["insufficient_evidence_checks"] == ["insufficient_evidence:min_reliability_score_mean"]
    check = next(c for c in gate["checks"] if c["name"] == "min_reliability_score_mean")
    assert check["passed"] is False
    assert gate["data_availability"]["reliability_score_mean"]["count"] == 0
    assert gate["data_availability"]["reliability_score_mean"]["available"] is False
    assert "missing_reliability_score" in gate["per_ticker"]["NOREL"]["warnings"]


def test_selected_candidate_scope_metrics_computed_as_before():
    bundle = _make_bundle(_make_signal(evidence_scope=_selected_scope()), _make_diag())

    gate = build_quality_gate({"GOOD": bundle}, DEFAULT_BENCHMARK_THRESHOLDS)

    assert gate["metrics"]["median_wf_sharpe"] == 0.8
    assert gate["metrics"]["positive_wf_share"] == 1.0
    assert gate["metrics"]["median_cpcv_p5"] == 0.2
    assert gate["metrics"]["seed_stable_rate"] == 1.0
    assert gate["metrics"]["reliability_score_mean"] == 4.0
    assert gate["metrics"]["median_ece"] == 0.05
    assert gate["metrics"]["median_buy_recall"] == 0.6
    assert gate["insufficient_evidence_checks"] == []
    assert all(entry["available"] for entry in gate["data_availability"].values())
    assert gate["status"] == "PASS"


def test_multiple_testing_ledger_counts_leaderboards():
    signal = _make_signal(evidence_scope=_selected_scope())
    signal["selection_leaderboard"] = [
        {"model": f"m{i}", "robust_score": 0.1 * i} for i in range(5)
    ]
    bundle_a = _make_bundle(signal, _make_diag())
    signal_b = _make_signal(evidence_scope=_selected_scope())
    signal_b["selection_leaderboard"] = [{"model": "x", "robust_score": 0.3}]
    bundle_b = _make_bundle(signal_b, _make_diag())

    gate = build_quality_gate({"A": bundle_a, "B": bundle_b}, DEFAULT_BENCHMARK_THRESHOLDS)

    mt = gate["multiple_testing"]
    assert mt["n_tickers"] == 2
    assert mt["leaderboard_size_median"] == 3.0
    assert mt["leaderboard_size_max"] == 5
    assert mt["n_trials_upper_bound"] == 10
    assert mt["expected_max_abs_sharpe_null"] is not None
    assert "caveat" in mt


def test_multiple_testing_ledger_absent_leaderboard_is_null_safe():
    bundle = _make_bundle(_make_signal(evidence_scope=_selected_scope()), _make_diag())

    gate = build_quality_gate({"SOLO": bundle}, DEFAULT_BENCHMARK_THRESHOLDS)

    mt = gate["multiple_testing"]
    assert mt["leaderboard_size_median"] is None
    assert mt["leaderboard_size_max"] is None
    assert mt["n_trials_upper_bound"] is None
    assert mt["expected_max_abs_sharpe_null"] is None
