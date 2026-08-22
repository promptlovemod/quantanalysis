import json
from pathlib import Path

import numpy as np

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:  # pragma: no cover - optional
    HAS_YF = False

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


DEFAULT_OPTIMIZER = "mean_variance"
DEFAULT_BENCHMARK_THRESHOLDS = {
    "min_success_rate": 0.80,
    "min_positive_wf_share": 0.40,
    "min_median_wf_sharpe": 0.00,
    "min_median_cpcv_p5": -0.25,
    "min_seed_stable_rate": 0.60,
    "min_reliability_score_mean": 3.0,
    "max_median_ece": 0.10,
    "min_median_buy_recall": 0.10,
}


def load_report_bundle(base_dir, ticker: str) -> dict:
    base = Path(base_dir) / ticker
    out = {}
    artifact_presence = {}
    for key, fn in [
        ("signal_data", f"{ticker}_signal.json"),
        ("fund_data", f"{ticker}_fundamentals.json"),
        ("mc_data", f"{ticker}_montecarlo.json"),
        ("diag_data", f"{ticker}_diagnostics.json"),
    ]:
        path = base / fn
        artifact_presence[key] = bool(path.exists())
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    out[key] = json.load(f)
            except Exception:
                out[key] = {}
        else:
            out[key] = {}
    out["artifact_presence"] = artifact_presence
    return out


def _safe_float(value, default=0.0) -> float:
    if value is None:
        return None if default is None else float(default)
    try:
        return float(value)
    except Exception:
        return None if default is None else float(default)


def build_expected_return_inputs(tickers: list, stock_data: dict) -> dict:
    inputs = {}
    for ticker in tickers:
        bundle = stock_data.get(ticker, {})
        signal_data = bundle.get("signal_data", {}) or {}
        mc_data = bundle.get("mc_data", {}) or {}
        fund_data = bundle.get("fund_data", {}) or {}

        sig = signal_data.get("signal", {}) or {}
        direction = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}.get(str(sig.get("signal", "HOLD")), 0.0)
        conf = _safe_float(sig.get("confidence"), 0.33)

        mc_risk = mc_data.get("risk_summary", {}) or {}
        current_price = _safe_float(mc_data.get("current_price"), 1.0)
        model_targets = []
        for model_name in ("GBM", "Merton", "Heston"):
            terminal = _safe_float((mc_risk.get(model_name, {}) or {}).get("median_1yr"), current_price)
            if current_price > 0:
                model_targets.append((terminal / current_price) - 1.0)
        mc_mu = float(np.mean(model_targets)) if model_targets else 0.0

        composite = _safe_float(fund_data.get("composite"), 50.0)
        fund_adjustment = (composite - 50.0) / 100.0

        mu = direction * conf * 0.6 + mc_mu * 0.3 + fund_adjustment * 0.1
        inputs[ticker] = {
            "direction": direction,
            "confidence": conf,
            "mc_mu": mc_mu,
            "fund_adjustment": fund_adjustment,
            "expected_return": float(mu),
        }
    return inputs


def fetch_return_matrix(tickers: list, lookback_days: int = 252):
    if not HAS_YF or not tickers:
        return np.empty((0, 0)), [], "unavailable"
    import pandas as _pd
    period = f"{max(int(lookback_days * 2), 365)}d"
    try:
        hist = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        # yfinance returns a MultiIndex (metric, ticker) when len(tickers) > 1
        # and a flat Index (metric,) for a single ticker.  Handle both cases.
        if isinstance(hist.columns, _pd.MultiIndex):
            if "Close" in hist.columns.get_level_values(0):
                close = hist["Close"]
            else:
                return np.empty((0, 0)), [], "unavailable"
        else:
            if "Close" in hist.columns:
                close = hist[["Close"]]
                if len(tickers) == 1:
                    close.columns = tickers
            else:
                return np.empty((0, 0)), [], "unavailable"
        if len(tickers) == 1 and isinstance(close, _pd.Series):
            close = close.to_frame(name=tickers[0])
        close = close.dropna(how="all").ffill().dropna(how="any")
        rets = close.pct_change().dropna().tail(int(lookback_days))
        if rets.empty:
            return np.empty((0, 0)), [], "empty"
        return rets.to_numpy(dtype=float), list(rets.columns), "yfinance"
    except Exception:
        return np.empty((0, 0)), [], "error"


def _heuristic_weights(mu_arr, min_weight, max_weight):
    positive = np.maximum(mu_arr, 0.0)
    if positive.sum() <= 0:
        positive = np.ones_like(mu_arr, dtype=float)
    raw = positive / positive.sum()
    raw = np.clip(raw, min_weight, max_weight)
    return raw / raw.sum()


def _mean_variance_weights(mu_arr, cov_arr, min_weight, max_weight):
    n = len(mu_arr)
    w0 = _heuristic_weights(mu_arr, min_weight, max_weight)

    def objective(w):
        port_ret = float(w @ mu_arr)
        port_var = float(w @ cov_arr @ w)
        return -(port_ret / (np.sqrt(max(port_var, 1e-10))))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9})
    if not res.success:
        raise RuntimeError(res.message)
    weights = np.clip(res.x, min_weight, max_weight)
    return weights / weights.sum()


def _risk_parity_weights(cov_arr, min_weight, max_weight):
    n = cov_arr.shape[0]
    w0 = np.full(n, 1.0 / n, dtype=float)

    def objective(w):
        port_var = cov_arr @ w
        total = max(float(w @ port_var), 1e-10)
        rc = w * port_var / np.sqrt(total)
        target = np.full_like(rc, rc.mean())
        return float(np.sum((rc - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9})
    if not res.success:
        raise RuntimeError(res.message)
    weights = np.clip(res.x, min_weight, max_weight)
    return weights / weights.sum()


def _black_litterman_weights(mu_arr, cov_arr, market_weights, min_weight, max_weight):
    tau = 0.05
    delta = 2.5
    pi = delta * cov_arr @ market_weights
    omega = np.diag(np.diag(tau * cov_arr))
    middle = np.linalg.inv(np.linalg.inv(tau * cov_arr) + np.linalg.inv(omega))
    posterior_mu = middle @ (np.linalg.inv(tau * cov_arr) @ pi + np.linalg.inv(omega) @ mu_arr)
    return _mean_variance_weights(posterior_mu, cov_arr, min_weight, max_weight), posterior_mu


def _cvar_weights(mu_arr, scenario_returns, min_weight, max_weight, alpha: float = 0.95):
    n = len(mu_arr)
    w0 = _heuristic_weights(mu_arr, min_weight, max_weight)

    def objective(w):
        port = scenario_returns @ w
        cutoff = np.quantile(port, 1.0 - alpha)
        tail = port[port <= cutoff]
        cvar = -float(tail.mean()) if len(tail) else -float(cutoff)
        return cvar - float(mu_arr @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9})
    if not res.success:
        raise RuntimeError(res.message)
    weights = np.clip(res.x, min_weight, max_weight)
    return weights / weights.sum()


def compute_portfolio_weights(tickers: list,
                              stock_data: dict,
                              optimizer: str = DEFAULT_OPTIMIZER,
                              lookback_days: int = 252,
                              min_weight: float = 0.02,
                              max_weight: float = 0.25) -> dict:
    expected_inputs = build_expected_return_inputs(tickers, stock_data)
    valid = [ticker for ticker in tickers if ticker in expected_inputs]
    if not valid:
        return {
            "requested_method": optimizer,
            "method_used": "heuristic",
            "fallback_reason": "no_valid_inputs",
            "weights": {},
            "expected_returns": {},
            "covariance_source": "unavailable",
        }

    mu_arr = np.array([expected_inputs[t]["expected_return"] for t in valid], dtype=float)
    ret_matrix, columns, cov_source = fetch_return_matrix(valid, lookback_days=lookback_days)
    cov_arr = None
    fallback_reason = ""
    if len(columns) >= 2 and ret_matrix.size:
        aligned_idx = [columns.index(t) for t in valid if t in columns]
        if len(aligned_idx) == len(valid):
            cov_arr = LedoitWolf().fit(ret_matrix[:, aligned_idx]).covariance_
    if cov_arr is None:
        cov_arr = np.diag(np.maximum(np.abs(mu_arr), 0.05) ** 2)
        fallback_reason = "covariance_fallback"
        cov_source = "diagonal_fallback"

    method_used = optimizer or DEFAULT_OPTIMIZER
    try:
        if method_used == "risk_parity":
            weights = _risk_parity_weights(cov_arr, min_weight, max_weight)
        elif method_used == "black_litterman":
            market_caps = []
            for ticker in valid:
                fund = stock_data.get(ticker, {}).get("fund_data", {}) or {}
                market_caps.append(_safe_float((fund.get("fundamentals", {}) or {}).get("marketCap"), 0.0))
            market_caps = np.array(market_caps, dtype=float)
            if market_caps.sum() <= 0:
                market_caps = np.full(len(valid), 1.0 / len(valid), dtype=float)
            else:
                market_caps = market_caps / market_caps.sum()
            weights, posterior_mu = _black_litterman_weights(mu_arr, cov_arr, market_caps, min_weight, max_weight)
            mu_arr = np.asarray(posterior_mu, dtype=float)
        elif method_used == "cvar":
            if ret_matrix.size == 0:
                raise RuntimeError("missing_return_scenarios")
            missing = [t for t in valid if t not in columns]
            if missing:
                raise RuntimeError(f"cvar_missing_tickers:{','.join(missing)}")
            aligned_idx = [columns.index(t) for t in valid]
            weights = _cvar_weights(mu_arr, ret_matrix[:, aligned_idx], min_weight, max_weight)
        elif method_used == "heuristic":
            weights = _heuristic_weights(mu_arr, min_weight, max_weight)
        else:
            weights = _mean_variance_weights(mu_arr, cov_arr, min_weight, max_weight)
            method_used = "mean_variance"
    except Exception as exc:
        fallback_reason = str(exc) or f"{method_used}_fallback"
        method_used = "heuristic"
        weights = _heuristic_weights(mu_arr, min_weight, max_weight)

    return {
        "requested_method": optimizer,
        "method_used": method_used,
        "fallback_reason": fallback_reason,
        "weights": _rounded_weight_map(valid, weights),
        "expected_returns": {
            ticker: {
                **expected_inputs[ticker],
                "expected_return": round(float(expected_inputs[ticker]["expected_return"]), 6),
            }
            for ticker in valid
        },
        "covariance_source": cov_source,
        "lookback_days": int(lookback_days),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
    }


def _median_or_none(values):
    clean = [float(v) for v in values if v is not None]
    return float(np.median(clean)) if clean else None


def _mean_or_none(values):
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def _meets_lower(metric, threshold):
    return metric is not None and float(metric) >= float(threshold)


def _meets_upper(metric, threshold):
    return metric is not None and float(metric) <= float(threshold)


def _rounded_weight_map(tickers, weights, decimals: int = 6):
    if not tickers:
        return {}
    rounded = [round(float(w), decimals) for w in weights]
    if len(rounded) == 1:
        rounded[0] = 1.0
    else:
        # Ensure the adjustment term stays non-negative — clipped optimisers
        # can produce weights that sum slightly above 1.0 after rounding.
        last = round(1.0 - sum(rounded[:-1]), decimals)
        rounded[-1] = max(0.0, last)
    return {ticker: float(weight) for ticker, weight in zip(tickers, rounded)}


def _coverage_stat(count: int, total: int) -> dict:
    total = int(total or 0)
    count = int(count or 0)
    return {
        "count": count,
        "total": total,
        "missing": max(0, total - count),
        "ratio": float(count / total) if total else 0.0,
    }


def build_quality_gate(stock_data: dict, thresholds: dict | None = None) -> dict:
    thresholds = dict(DEFAULT_BENCHMARK_THRESHOLDS | (thresholds or {}))
    tickers = list(stock_data.keys())
    pipeline_ok = []
    wf_sharpes = []
    cpcv_p5 = []
    seed_stable = []
    reliability_scores = []
    eces = []
    buy_recalls = []
    actionable = []
    per_ticker = {}

    signal_present_count = 0
    diag_present_count = 0
    wf_count = 0
    cpcv_count = 0
    seed_count = 0
    reliability_count = 0
    calibration_count = 0
    buy_recall_count = 0
    robust_leaderboard = []
    calibration_leaderboard = []
    conformal_leaderboard = []
    buy_recall_leaderboard = []
    sell_recall_leaderboard = []
    macro_pr_auc_leaderboard = []
    non_hold_recall_leaderboard = []
    eligible_deployment_leaderboard = []
    router_family_frequency = {}
    router_status_frequency = {}
    routing_eval_deltas = []
    routing_cpcv_deltas = []
    routing_robust_deltas = []
    routing_improved_flags = []
    rejection_counts = {}
    execution_status_counts = {}
    mc_status_frequency = {}
    mc_dispersion_frequency = {}
    mc_fallback_count = 0
    conformal_usable_count = 0
    conformal_unusable_count = 0
    conformal_blocked_candidate_total = 0
    conformal_blocked_ticker_count = 0
    conformal_blocked_selected_count = 0
    conformal_singleton_rate_by_class = {}
    conformal_coverage_by_class = {}
    conformal_set_size_histogram = {}
    reference_only_count = 0
    reference_matches_deployment_count = 0
    no_eligible_family_count = 0
    eligible_candidate_counts_by_family = {}
    family_eligible_win_counts = {}
    directional_rejection_counts = {}
    sector_rows = {}
    vol_bucket_rows = {}

    for ticker in tickers:
        bundle = stock_data[ticker] or {}
        signal = bundle.get("signal_data", {}) or {}
        diag = bundle.get("diag_data", {}) or {}
        fund = bundle.get("fund_data", {}) or {}
        mc = bundle.get("mc_data", {}) or {}
        artifacts = bundle.get("artifact_presence", {}) or {}
        signal_present = bool(artifacts.get("signal_data")) or bool(signal)
        diag_present = bool(artifacts.get("diag_data")) or bool(diag)
        signal_present_count += int(signal_present)
        diag_present_count += int(diag_present)

        pipeline_is_ok = signal.get("pipeline_status") == "OK"
        pipeline_ok.append(pipeline_is_ok)
        wf = (signal.get("walkforward_backtest", {}) or {}).get("wf_sharpe")
        cpcv = (signal.get("cpcv", {}) or {}).get("sharpe_p5")
        seed = (diag.get("seed_stability", {}) or {}).get("stable")
        if seed is None:
            seed = (signal.get("seed_stability", {}) or {}).get("stable")
        reliability = diag.get("reliability_score")
        cal = (diag.get("calibration", {}) or {})
        if not cal:
            cal = signal.get("calibration", {}) or {}
        if not cal:
            cal = signal.get("calibration_diagnostics", {}) or {}
        classwise = cal.get("classwise", {}) or {}
        ece = cal.get("ece")
        if ece is None:
            ece = cal.get("post_calibration_ece")
        buy_recall = ((classwise.get("BUY", {}) or {}).get("recall"))
        selection = signal.get("selection", {}) or {}
        router = signal.get("router", {}) or {}
        conformal = signal.get("conformal", {}) or signal.get("conformal_prediction", {}) or {}
        selected_signal = signal.get("signal", {}) or {}
        selected_eval = selection.get("evaluation", {}) or {}
        execution_status = str(selected_signal.get("execution_status", "") or "UNKNOWN")
        actionable_flag = execution_status == "ACTIONABLE"
        actionable.append(float(actionable_flag))
        execution_status_counts[execution_status] = execution_status_counts.get(execution_status, 0) + 1
        selection_status = str(selection.get("selection_status", "") or "")
        reference_model_used = selection.get("reference_model_used")
        deployment_model_used = selection.get("deployment_model_used")
        display_model = deployment_model_used or reference_model_used or selected_signal.get("model_used") or selected_signal.get("model")
        display_family = (
            selection.get("deployment_family_used")
            or selection.get("reference_family_used")
            or selection.get("family")
            or router.get("chosen_family")
        )
        if selection_status == "reference_only_no_deployable_candidate":
            reference_only_count += 1
        if bool(selection.get("reference_matches_deployment", False)):
            reference_matches_deployment_count += 1
        if str(router.get("fallback_reason", "") or "") == "no_eligible_family":
            no_eligible_family_count += 1
        candidate_counts = selection.get("candidate_counts", {}) or {}
        for fam, count in (candidate_counts.get("eligible_by_family", {}) or {}).items():
            eligible_candidate_counts_by_family[str(fam)] = (
                eligible_candidate_counts_by_family.get(str(fam), 0) + int(count or 0)
            )
        conformal_block_count = int(selection.get("conformal_blocked_otherwise_healthy_count", 0) or 0)
        conformal_blocked_candidate_total += conformal_block_count
        conformal_blocked_ticker_count += int(conformal_block_count > 0)
        for reason, count in (selection.get("directional_rejection_counts", {}) or {}).items():
            directional_rejection_counts[str(reason)] = (
                directional_rejection_counts.get(str(reason), 0) + int(count or 0)
            )
        eligibility_failures = (
            selected_signal.get("eligibility_failures")
            or selection.get("eligibility_failures")
            or []
        )
        for reason in eligibility_failures:
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1

        has_wf = wf is not None
        has_cpcv = cpcv is not None
        has_seed = seed is not None
        has_reliability = reliability is not None
        has_calibration = bool(cal) and ece is not None
        has_buy_recall = buy_recall is not None

        wf_count += int(has_wf)
        cpcv_count += int(has_cpcv)
        seed_count += int(has_seed)
        reliability_count += int(has_reliability)
        calibration_count += int(has_calibration)
        buy_recall_count += int(has_buy_recall)

        wf_sharpes.append(wf if has_wf else None)
        cpcv_p5.append(cpcv if has_cpcv else None)
        seed_stable.append(bool(seed) if has_seed else None)
        reliability_scores.append(reliability if has_reliability else None)
        eces.append(ece if has_calibration else None)
        buy_recalls.append(buy_recall if has_buy_recall else None)

        warnings = []
        if not signal_present:
            warnings.append("missing_signal_json")
        if not diag_present:
            warnings.append("missing_diagnostics_json")
        if signal_present and not pipeline_is_ok:
            warnings.append("pipeline_not_ok")
        if not has_calibration:
            warnings.append("missing_calibration")
        if not has_reliability:
            warnings.append("missing_reliability_score")

        robust_score = selection.get("robust_score")
        if robust_score is not None:
            robust_leaderboard.append({
                "ticker": ticker,
                "model": display_model,
                "family": display_family,
                "robust_score": float(robust_score),
                "wf_sharpe": wf,
                "cpcv_p5": cpcv,
                "selection_status": selection_status,
            })
        if has_calibration:
            calibration_leaderboard.append({
                "ticker": ticker,
                "model": display_model,
                "family": display_family,
                "post_calibration_nll": _safe_float(cal.get("post_calibration_nll"), None),
                "post_calibration_ece": _safe_float(cal.get("post_calibration_ece", ece), None),
                "multiclass_brier": _safe_float(cal.get("multiclass_brier", cal.get("brier_score")), None),
            })
        sell_recall = ((classwise.get("SELL", {}) or {}).get("recall"))
        macro_pr_auc = selected_eval.get("macro_pr_auc", cal.get("pr_auc_macro"))
        non_hold_recall_min = selected_eval.get("non_hold_recall_min")
        if buy_recall is not None:
            buy_recall_leaderboard.append({
                "ticker": ticker,
                "model": display_model,
                "buy_recall": _safe_float(buy_recall, 0.0),
            })
        if sell_recall is not None:
            sell_recall_leaderboard.append({
                "ticker": ticker,
                "model": display_model,
                "sell_recall": _safe_float(sell_recall, 0.0),
            })
        if macro_pr_auc is not None:
            macro_pr_auc_leaderboard.append({
                "ticker": ticker,
                "model": display_model,
                "macro_pr_auc": _safe_float(macro_pr_auc, 0.0),
            })
        if non_hold_recall_min is not None:
            non_hold_recall_leaderboard.append({
                "ticker": ticker,
                "model": display_model,
                "non_hold_recall_min": _safe_float(non_hold_recall_min, 0.0),
            })
        deployment_summary = selection.get("deployment_champion") or {}
        if deployment_summary:
            family = str(deployment_summary.get("family", "") or "UNKNOWN")
            family_eligible_win_counts[family] = family_eligible_win_counts.get(family, 0) + 1
            eligible_deployment_leaderboard.append({
                "ticker": ticker,
                "model": deployment_summary.get("model"),
                "family": family,
                "robust_score": _safe_float(deployment_summary.get("robust_score"), 0.0),
                "macro_pr_auc": _safe_float(deployment_summary.get("macro_pr_auc"), 0.0),
                "non_hold_recall_min": _safe_float(deployment_summary.get("non_hold_recall_min"), 0.0),
            })
        if conformal:
            conformal_leaderboard.append({
                "ticker": ticker,
                "model": display_model,
                "family": display_family,
                "sharpness": _safe_float(conformal.get("sharpness"), 0.0),
                "avg_set_size": _safe_float(conformal.get("avg_set_size"), 0.0),
                "singleton_rate": _safe_float(conformal.get("singleton_rate"), 0.0),
                "abstain_rate": _safe_float(conformal.get("abstain_rate"), 0.0),
                "usable_for_execution": bool(conformal.get("usable_for_execution", False)),
            })
            if bool(conformal.get("usable_for_execution", False)):
                conformal_usable_count += 1
            else:
                conformal_unusable_count += 1
            if bool(conformal.get("blocked_otherwise_healthy_model", False)):
                conformal_blocked_selected_count += 1
            for cls, value in (conformal.get("class_conditional_singleton_rate", {}) or {}).items():
                conformal_singleton_rate_by_class.setdefault(str(cls), []).append(float(value))
            for cls, value in (conformal.get("class_conditional_coverage", {}) or {}).items():
                conformal_coverage_by_class.setdefault(str(cls), []).append(float(value))
            for set_size, count in (conformal.get("set_size_distribution", {}) or {}).items():
                key = str(set_size)
                conformal_set_size_histogram[key] = conformal_set_size_histogram.get(key, 0) + int(count or 0)
        chosen_family = str(router.get("chosen_family", "") or "")
        if chosen_family:
            router_family_frequency[chosen_family] = router_family_frequency.get(chosen_family, 0) + 1
        router_status = str(router.get("router_status", "") or "unknown")
        router_status_frequency[router_status] = router_status_frequency.get(router_status, 0) + 1
        delta_eval = router.get("delta_eval_sharpe")
        delta_cpcv = router.get("delta_cpcv_p5_sharpe")
        delta_robust = router.get("delta_robust_score")
        if delta_eval is not None:
            routing_eval_deltas.append(float(delta_eval))
        if delta_cpcv is not None:
            routing_cpcv_deltas.append(float(delta_cpcv))
        if delta_robust is not None:
            routing_robust_deltas.append(float(delta_robust))
        if router.get("routing_improved_vs_global") is not None:
            routing_improved_flags.append(bool(router.get("routing_improved_vs_global")))

        mc_reliability = mc.get("mc_reliability", {}) or {}
        mc_status = str(mc_reliability.get("mc_reliability_status", "") or "unknown")
        mc_status_frequency[mc_status] = mc_status_frequency.get(mc_status, 0) + 1
        if mc_reliability.get("vol_model_fallback"):
            mc_fallback_count += 1
        for model_name, risk in (mc.get("risk_summary", {}) or {}).items():
            disp = str((risk.get("calibration_check", {}) or {}).get("dispersion_label", "") or "unknown")
            key = f"{model_name}:{disp}"
            mc_dispersion_frequency[key] = mc_dispersion_frequency.get(key, 0) + 1

        sector = (
            ((fund.get("fundamentals", {}) or {}).get("sector"))
            or ((signal.get("fundamentals", {}) or {}).get("fundamentals", {}) or {}).get("sector")
            or "UNKNOWN"
        )
        vol = _safe_float((signal.get("regime", {}) or {}).get("ann_vol"), 0.0)
        vol_bucket = "low_vol" if vol < 0.25 else "mid_vol" if vol < 0.45 else "high_vol"
        for bucket, store in ((sector, sector_rows), (vol_bucket, vol_bucket_rows)):
            stats = store.setdefault(bucket, {"n": 0, "positive_wf": 0, "wf_sharpes": [], "robust_scores": []})
            stats["n"] += 1
            if wf is not None and float(wf) > 0:
                stats["positive_wf"] += 1
            if wf is not None:
                stats["wf_sharpes"].append(float(wf))
            if robust_score is not None:
                stats["robust_scores"].append(float(robust_score))

        per_ticker[ticker] = {
            "artifacts": {
                "signal_json": signal_present,
                "diagnostics_json": diag_present,
            },
            "pipeline_status": signal.get("pipeline_status"),
            "pipeline_ok": bool(pipeline_is_ok),
            "contributes": {
                "walkforward": has_wf,
                "cpcv": has_cpcv,
                "seed_stability": has_seed,
                "reliability": has_reliability,
                "calibration": has_calibration,
                "buy_recall": has_buy_recall,
            },
            "metrics": {
                "wf_sharpe": wf,
                "cpcv_p5": cpcv,
                "seed_stable": bool(seed) if has_seed else None,
                "reliability_score": reliability,
                "ece": ece if has_calibration else None,
                "buy_recall": buy_recall,
                "router_delta_eval_sharpe": delta_eval,
                "router_delta_cpcv_p5_sharpe": delta_cpcv,
                "router_delta_robust_score": delta_robust,
                "execution_status": execution_status,
                "actionable": actionable_flag,
                "router_status": router_status,
                "macro_pr_auc": macro_pr_auc,
                "mc_reliability_status": mc_status,
                "selection_status": selection_status,
            },
            "warnings": warnings,
            "eligibility_failures": list(eligibility_failures),
            "conformal_usable": bool(conformal.get("usable_for_execution", False)) if conformal else False,
            "conformal_class_conditional_singleton_rate": dict(conformal.get("class_conditional_singleton_rate", {}) or {}) if conformal else {},
            "conformal_class_conditional_coverage": dict(conformal.get("class_conditional_coverage", {}) or {}) if conformal else {},
            "conformal_set_size_distribution": dict(conformal.get("set_size_distribution", {}) or {}) if conformal else {},
            "reference_model_used": reference_model_used,
            "deployment_model_used": deployment_model_used,
            "eligible_candidate_counts_by_family": dict(candidate_counts.get("eligible_by_family", {}) or {}),
            "conformal_blocked_otherwise_healthy_count": conformal_block_count,
            "directional_rejection_counts": dict(selection.get("directional_rejection_counts", {}) or {}),
        }

    metrics = {
        "n_tickers": len(tickers),
        "success_rate": float(np.mean(pipeline_ok)) if pipeline_ok else 0.0,
        "positive_wf_share": _mean_or_none([float(v > 0) for v in wf_sharpes if v is not None]),
        "median_wf_sharpe": _median_or_none(wf_sharpes),
        "median_cpcv_p5": _median_or_none(cpcv_p5),
        "seed_stable_rate": _mean_or_none([float(bool(v)) for v in seed_stable if v is not None]),
        "reliability_score_mean": _mean_or_none(reliability_scores),
        "median_ece": _median_or_none(eces),
        "median_buy_recall": _median_or_none(buy_recalls),
        "actionable_rate": _mean_or_none(actionable),
    }

    checks = [
        ("min_success_rate", metrics["success_rate"] >= thresholds["min_success_rate"]),
        ("min_positive_wf_share", _meets_lower(metrics["positive_wf_share"], thresholds["min_positive_wf_share"])),
        ("min_median_wf_sharpe", _meets_lower(metrics["median_wf_sharpe"], thresholds["min_median_wf_sharpe"])),
        ("min_median_cpcv_p5", _meets_lower(metrics["median_cpcv_p5"], thresholds["min_median_cpcv_p5"])),
        ("min_seed_stable_rate", _meets_lower(metrics["seed_stable_rate"], thresholds["min_seed_stable_rate"])),
        ("min_reliability_score_mean", _meets_lower(metrics["reliability_score_mean"], thresholds["min_reliability_score_mean"])),
        ("max_median_ece", _meets_upper(metrics["median_ece"], thresholds["max_median_ece"])),
        ("min_median_buy_recall", _meets_lower(metrics["median_buy_recall"], thresholds["min_median_buy_recall"])),
    ]

    coverage = {
        "signal_json": _coverage_stat(signal_present_count, len(tickers)),
        "diagnostics_json": _coverage_stat(diag_present_count, len(tickers)),
        "pipeline_ok": _coverage_stat(sum(1 for ok in pipeline_ok if ok), len(tickers)),
        "walkforward": _coverage_stat(wf_count, len(tickers)),
        "cpcv": _coverage_stat(cpcv_count, len(tickers)),
        "seed_stability": _coverage_stat(seed_count, len(tickers)),
        "reliability": _coverage_stat(reliability_count, len(tickers)),
        "calibration": _coverage_stat(calibration_count, len(tickers)),
        "buy_recall": _coverage_stat(buy_recall_count, len(tickers)),
    }

    coverage_warnings = []
    if signal_present_count < len(tickers):
        coverage_warnings.append("signal_coverage_incomplete")
    if diag_present_count < len(tickers):
        coverage_warnings.append("diagnostics_coverage_incomplete")
    if calibration_count < len(tickers):
        coverage_warnings.append("calibration_coverage_incomplete")

    failure_map = {
        "min_success_rate": "low_pipeline_success_rate",
        "min_positive_wf_share": "weak_walkforward_robustness",
        "min_median_wf_sharpe": "negative_median_walkforward_sharpe",
        "min_median_cpcv_p5": "weak_cpcv_tail",
        "min_seed_stable_rate": "low_seed_stability",
        "min_reliability_score_mean": "low_reliability_score",
        "max_median_ece": "poor_calibration",
        "min_median_buy_recall": "weak_buy_recall",
    }
    primary_failure_reasons = list(coverage_warnings)
    primary_failure_reasons.extend(
        failure_map[name] for name, passed in checks
        if not passed and failure_map.get(name) not in primary_failure_reasons
    )

    passed = sum(1 for _, ok in checks if ok)
    if passed >= 6:
        status = "PASS"
    elif passed >= 4:
        status = "MARGINAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "passed_checks": passed,
        "total_checks": len(checks),
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": [{"name": name, "passed": bool(ok)} for name, ok in checks],
        "coverage": coverage,
        "coverage_warnings": coverage_warnings,
        "primary_failure_reasons": primary_failure_reasons,
        "per_ticker": per_ticker,
        "robust_score_leaderboard": sorted(robust_leaderboard, key=lambda row: row.get("robust_score", float("-inf")), reverse=True),
        "calibration_leaderboard": sorted(
            calibration_leaderboard,
            key=lambda row: (
                float("inf") if row.get("post_calibration_nll") is None else row.get("post_calibration_nll")
            ),
        ),
        "conformal_sharpness_leaderboard": sorted(
            conformal_leaderboard,
            key=lambda row: row.get("sharpness", float("-inf")),
            reverse=True,
        ),
        "buy_recall_leaderboard": sorted(
            buy_recall_leaderboard,
            key=lambda row: row.get("buy_recall", float("-inf")),
            reverse=True,
        ),
        "sell_recall_leaderboard": sorted(
            sell_recall_leaderboard,
            key=lambda row: row.get("sell_recall", float("-inf")),
            reverse=True,
        ),
        "macro_pr_auc_leaderboard": sorted(
            macro_pr_auc_leaderboard,
            key=lambda row: row.get("macro_pr_auc", float("-inf")),
            reverse=True,
        ),
        "non_hold_recall_leaderboard": sorted(
            non_hold_recall_leaderboard,
            key=lambda row: row.get("non_hold_recall_min", float("-inf")),
            reverse=True,
        ),
        "eligible_deployment_leaderboard": sorted(
            eligible_deployment_leaderboard,
            key=lambda row: row.get("robust_score", float("-inf")),
            reverse=True,
        ),
        "router_family_selection_frequency": {
            fam: float(count / max(len(tickers), 1))
            for fam, count in sorted(router_family_frequency.items(), key=lambda item: item[0])
        },
        "router_status_frequency": {
            key: int(val) for key, val in sorted(router_status_frequency.items(), key=lambda item: item[0])
        },
        "routing_uplift": {
            "median_eval_sharpe_delta": _median_or_none(routing_eval_deltas),
            "median_cpcv_p5_delta": _median_or_none(routing_cpcv_deltas),
            "median_robust_score_delta": _median_or_none(routing_robust_deltas),
            "improved_share": _mean_or_none([float(v) for v in routing_improved_flags]),
        },
        "actionable_summary": {
            "actionable_rate": metrics.get("actionable_rate"),
            "execution_status_counts": execution_status_counts,
        },
        "model_rejection_counts": rejection_counts,
        "mc_reliability_summary": {
            "status_frequency": mc_status_frequency,
            "fallback_vol_count": int(mc_fallback_count),
            "dispersion_frequency": mc_dispersion_frequency,
        },
        "conformal_usability_summary": {
            "usable_count": int(conformal_usable_count),
            "unusable_count": int(conformal_unusable_count),
            "conformal_block_rate": float(conformal_blocked_ticker_count / max(len(tickers), 1)),
            "conformal_blocked_ticker_count": int(conformal_blocked_ticker_count),
            "conformal_blocked_candidate_count": int(conformal_blocked_candidate_total),
            "otherwise_healthy_blocked_selected_count": int(conformal_blocked_selected_count),
            "otherwise_healthy_blocked_candidate_count": int(conformal_blocked_candidate_total),
            "singleton_rate_by_class": {
                cls: _mean_or_none(values)
                for cls, values in sorted(conformal_singleton_rate_by_class.items(), key=lambda item: item[0])
            },
            "class_conditional_coverage": {
                cls: _mean_or_none(values)
                for cls, values in sorted(conformal_coverage_by_class.items(), key=lambda item: item[0])
            },
            "set_size_histogram": {
                key: int(val)
                for key, val in sorted(conformal_set_size_histogram.items(), key=lambda item: item[0])
            },
        },
        "selection_summary": {
            "reference_only_count": int(reference_only_count),
            "reference_only_rate": float(reference_only_count / max(len(tickers), 1)),
            "reference_matches_deployment_count": int(reference_matches_deployment_count),
            "no_eligible_family_count": int(no_eligible_family_count),
            "no_eligible_family_rate": float(no_eligible_family_count / max(len(tickers), 1)),
            "eligible_candidate_counts_by_family": {
                fam: int(count) for fam, count in sorted(eligible_candidate_counts_by_family.items(), key=lambda item: item[0])
            },
            "family_level_eligible_win_rate": {
                fam: float(count / max(len(tickers), 1))
                for fam, count in sorted(family_eligible_win_counts.items(), key=lambda item: item[0])
            },
            "directional_rejection_counts": {
                reason: int(count) for reason, count in sorted(directional_rejection_counts.items(), key=lambda item: item[0])
            },
        },
        "per_sector_win_table": [
            {
                "sector": key,
                "n": int(val["n"]),
                "positive_wf_share": float(val["positive_wf"] / max(val["n"], 1)),
                "median_wf_sharpe": float(np.median(val["wf_sharpes"])) if val["wf_sharpes"] else None,
                "median_robust_score": float(np.median(val["robust_scores"])) if val["robust_scores"] else None,
            }
            for key, val in sorted(sector_rows.items(), key=lambda item: item[0])
        ],
        "per_volatility_bucket_win_table": [
            {
                "bucket": key,
                "n": int(val["n"]),
                "positive_wf_share": float(val["positive_wf"] / max(val["n"], 1)),
                "median_wf_sharpe": float(np.median(val["wf_sharpes"])) if val["wf_sharpes"] else None,
                "median_robust_score": float(np.median(val["robust_scores"])) if val["robust_scores"] else None,
            }
            for key, val in sorted(vol_bucket_rows.items(), key=lambda item: item[0])
        ],
        "net_of_cost_walkforward_metrics": {
            "median_wf_sharpe": metrics.get("median_wf_sharpe"),
            "positive_wf_share": metrics.get("positive_wf_share"),
        },
    }
