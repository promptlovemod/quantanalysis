import datetime
import json
from collections import Counter
from pathlib import Path


DEFAULT_CONFORMAL_LIMITS = {
    "n_classes": 3,
    "min_singleton_rate": 0.05,
    "max_avg_set_size": 2.6,
    "max_full_set_rate": 0.70,
    "min_coverage_ratio": 0.80,
}

DEFAULT_FOCUS_TICKERS = ("AAPL", "CLPT", "RKLB")


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _report_paths(report_dir: Path) -> dict:
    ticker = report_dir.name
    return {
        "signal": report_dir / f"{ticker}_signal.json",
        "montecarlo": report_dir / f"{ticker}_montecarlo.json",
        "dashboard_consistency": report_dir / f"{ticker}_dashboard_consistency.json",
        "master_log": report_dir / f"{ticker}_master.log",
        "dashboard": report_dir / f"{ticker}_dashboard.html",
    }


def _add_check(checks: list, failures: list, warnings: list,
               name: str, passed: bool, severity: str = "error",
               details: str | dict | None = None):
    record = {
        "name": str(name),
        "passed": bool(passed),
        "severity": str(severity),
        "details": details,
    }
    checks.append(record)
    if record["passed"]:
        return
    if severity == "warning":
        warnings.append(record)
    else:
        failures.append(record)


def validate_signal_artifact(signal_json: dict | None,
                             mc_json: dict | None = None,
                             limits: dict | None = None) -> dict:
    cfg = dict(DEFAULT_CONFORMAL_LIMITS)
    cfg.update(dict(limits or {}))
    payload = dict(signal_json or {})
    signal = dict(payload.get("signal", {}) or {})
    selection = dict(payload.get("selection", {}) or {})
    router = dict(payload.get("router", {}) or {})
    conformal = dict(payload.get("conformal", {}) or payload.get("conformal_prediction", {}) or {})
    backtest = dict(payload.get("backtest", {}) or {})
    backtest_audit = dict(payload.get("backtest_audit", {}) or {})
    mc_reliability = dict((mc_json or {}).get("mc_reliability", {}) or {})

    execution_status = str(signal.get("execution_status", "") or "")
    execution_gate = str(signal.get("execution_gate", "") or "")
    abstain_reason = signal.get("abstain_reason")
    deployment_eligible = bool(signal.get("deployment_eligible", False))
    reference_model = selection.get("reference_model_used") or signal.get("reference_model_used")
    deployment_model = selection.get("deployment_model_used") or signal.get("deployment_model_used")
    model_used = signal.get("model_used")
    selection_status = str(
        selection.get("selection_status", signal.get("selection_status", "")) or ""
    )
    router_status = str(router.get("router_status", "") or "")
    routing_actionable = bool(router.get("routing_actionable", False))

    checks = []
    failures = []
    warnings = []

    expected_gate_map = {
        "ACTIONABLE": {"actionable"},
        "HOLD_NEUTRAL": {"neutral_hold", "neutral_expectation"},
        "ABSTAIN_NO_EDGE": {"edge_gate"},
        "ABSTAIN_UNCERTAIN": {"conformal_gate", "margin_gate"},
        "ABSTAIN_MODEL_UNRELIABLE": {"model_guard", "pipeline_error"},
    }
    if execution_status:
        _add_check(
            checks,
            failures,
            warnings,
            "execution_status_gate_alignment",
            execution_gate in expected_gate_map.get(execution_status, {execution_gate}),
            details={"execution_status": execution_status, "execution_gate": execution_gate},
        )

    if execution_status == "ACTIONABLE":
        _add_check(
            checks,
            failures,
            warnings,
            "actionable_requires_deployment_model",
            deployment_model not in (None, "") and deployment_eligible,
            details={
                "deployment_model_used": deployment_model,
                "deployment_eligible": deployment_eligible,
            },
        )
    elif execution_status.startswith("ABSTAIN"):
        _add_check(
            checks,
            failures,
            warnings,
            "abstain_reason_present",
            abstain_reason not in (None, ""),
            severity="warning",
            details={"execution_status": execution_status, "abstain_reason": abstain_reason},
        )

    if selection_status == "deployed":
        _add_check(
            checks,
            failures,
            warnings,
            "selection_status_model_alignment",
            deployment_model not in (None, "") and deployment_eligible,
            details={
                "selection_status": selection_status,
                "deployment_model_used": deployment_model,
                "deployment_eligible": deployment_eligible,
            },
        )
    elif selection_status == "reference_only_no_deployable_candidate":
        _add_check(
            checks,
            failures,
            warnings,
            "selection_status_model_alignment",
            reference_model not in (None, "")
            and deployment_model in (None, "")
            and not deployment_eligible
            and execution_status == "ABSTAIN_MODEL_UNRELIABLE",
            details={
                "selection_status": selection_status,
                "reference_model_used": reference_model,
                "deployment_model_used": deployment_model,
                "deployment_eligible": deployment_eligible,
                "execution_status": execution_status,
            },
        )

    if deployment_model in (None, ""):
        _add_check(
            checks,
            failures,
            warnings,
            "deployment_model_matches_signal_model",
            model_used in (None, ""),
            details={"model_used": model_used, "deployment_model_used": deployment_model},
        )
    else:
        _add_check(
            checks,
            failures,
            warnings,
            "deployment_model_matches_signal_model",
            str(model_used or "") == str(deployment_model),
            details={"model_used": model_used, "deployment_model_used": deployment_model},
        )

    if router_status == "fallback":
        _add_check(
            checks,
            failures,
            warnings,
            "router_fallback_alignment",
            not routing_actionable,
            details={
                "router_status": router_status,
                "routing_actionable": routing_actionable,
                "fallback_reason": router.get("fallback_reason"),
            },
        )
    elif router_status == "active":
        _add_check(
            checks,
            failures,
            warnings,
            "router_active_alignment",
            routing_actionable and router.get("chosen_family") not in (None, ""),
            details={
                "router_status": router_status,
                "routing_actionable": routing_actionable,
                "chosen_family": router.get("chosen_family"),
            },
        )

    prediction_set = list(signal.get("prediction_set", []) or [])
    set_size = int(signal.get("set_size", 0) or 0)
    is_singleton = bool(signal.get("is_conformal_singleton", False))
    _add_check(
        checks,
        failures,
        warnings,
        "conformal_signal_shape",
        (not prediction_set and set_size == 0) or len(prediction_set) == set_size,
        details={"prediction_set": prediction_set, "set_size": set_size},
    )
    _add_check(
        checks,
        failures,
        warnings,
        "conformal_singleton_flag_alignment",
        (set_size == 1) == is_singleton if set_size >= 0 else True,
        details={"set_size": set_size, "is_conformal_singleton": is_singleton},
    )

    if conformal:
        bypass_active = bool(conformal.get("conformal_bypass_active", False))
        coverage = _safe_float(conformal.get("coverage"), 0.0)
        target_coverage = _safe_float(conformal.get("target_coverage"), 0.90)
        singleton_rate = _safe_float(conformal.get("singleton_rate"), 0.0)
        avg_set_size = _safe_float(conformal.get("avg_set_size"), float(cfg["n_classes"]))
        full_set_rate = _safe_float(
            conformal.get("full_set_rate"),
            1.0 if avg_set_size >= float(cfg["n_classes"]) else 0.0,
        )
        coverage_floor = float(target_coverage) * float(cfg["min_coverage_ratio"])
        usable = bool(conformal.get("usable_for_execution", False))
        contradictions = []
        if usable and not bypass_active and coverage < coverage_floor:
            contradictions.append("coverage_below_floor")
        if usable and not bypass_active and singleton_rate < float(cfg["min_singleton_rate"]):
            contradictions.append("singleton_rate_below_floor")
        if usable and not bypass_active and avg_set_size > float(cfg["max_avg_set_size"]):
            contradictions.append("avg_set_size_above_ceiling")
        if usable and not bypass_active and full_set_rate > float(cfg["max_full_set_rate"]):
            contradictions.append("full_set_rate_above_ceiling")
        if usable and not bypass_active and bool(conformal.get("degenerate_execution_conformal", False)):
            contradictions.append("degenerate_execution_conformal")
        if bypass_active and not conformal.get("conformal_bypass_reason"):
            contradictions.append("missing_conformal_bypass_reason")
        _add_check(
            checks,
            failures,
            warnings,
            "conformal_usable_alignment",
            not contradictions,
            details={
                "usable_for_execution": usable,
                "conformal_bypass_active": bypass_active,
                "contradictions": contradictions,
                "coverage": coverage,
                "coverage_floor": coverage_floor,
                "singleton_rate": singleton_rate,
                "avg_set_size": avg_set_size,
                "full_set_rate": full_set_rate,
            },
        )

    if backtest:
        _add_check(
            checks,
            failures,
            warnings,
            "backtest_audit_present",
            bool(backtest_audit),
            severity="warning",
            details={"has_backtest": True, "has_backtest_audit": bool(backtest_audit)},
        )
    if backtest_audit:
        _add_check(
            checks,
            failures,
            warnings,
            "backtest_warning_has_flags",
            backtest_audit.get("sanity_status") != "warning"
            or bool(backtest_audit.get("sanity_flags")),
            severity="warning",
            details={
                "sanity_status": backtest_audit.get("sanity_status"),
                "sanity_flags": backtest_audit.get("sanity_flags"),
            },
        )
        n_bars = int(backtest_audit.get("n_bars", 0) or 0)
        n_trades = int(backtest_audit.get("n_trades", 0) or 0)
        if n_bars > 0:
            _add_check(
                checks,
                failures,
                warnings,
                "backtest_trade_count_feasible",
                n_trades <= n_bars,
                details={"n_bars": n_bars, "n_trades": n_trades},
            )

    if mc_reliability:
        primary_status = str(mc_reliability.get("mc_reliability_status", "") or "")
        baseline_status = str(
            mc_reliability.get("baseline_reliability_status", "") or primary_status
        )
        scenario_status = str(mc_reliability.get("scenario_reliability_status", "") or "")
        _add_check(
            checks,
            failures,
            warnings,
            "mc_primary_matches_baseline",
            not baseline_status or primary_status == baseline_status,
            severity="warning",
            details={
                "mc_reliability_status": primary_status,
                "baseline_reliability_status": baseline_status,
            },
        )
        _add_check(
            checks,
            failures,
            warnings,
            "mc_scenario_status_visible",
            bool(scenario_status),
            severity="warning",
            details={"scenario_reliability_status": scenario_status or None},
        )

    failure_names = [row["name"] for row in failures]
    warning_names = [row["name"] for row in warnings]
    status = "ERROR" if failures else "WARNING" if warnings else "OK"
    return {
        "generated_at": datetime.datetime.now().isoformat(),
        "status": status,
        "check_count": int(len(checks)),
        "failure_count": int(len(failures)),
        "warning_count": int(len(warnings)),
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "failure_names": failure_names,
        "warning_names": warning_names,
    }


def _latest_successful_ticker(report_dirs: list[Path]) -> str | None:
    latest = None
    for report_dir in report_dirs:
        signal_path = _report_paths(report_dir)["signal"]
        signal = _load_json(signal_path)
        if not signal or signal.get("pipeline_status") != "OK":
            continue
        stamp = signal.get("generated")
        sort_key = str(stamp or signal_path.stat().st_mtime)
        candidate = (sort_key, report_dir.name)
        if latest is None or candidate[0] > latest[0]:
            latest = candidate
    return latest[1] if latest else None


def _positive_directional_edge(signal: dict) -> bool:
    edge_minus_cost = dict(signal.get("edge_minus_cost", {}) or {})
    if any((_safe_float(val, 0.0) or 0.0) > 0.0 for val in edge_minus_cost.values()):
        return True
    probs = dict(signal.get("probabilities", {}) or {})
    hold = _safe_float(probs.get("HOLD"), 0.0) or 0.0
    directional = max(
        _safe_float(probs.get("BUY"), 0.0) or 0.0,
        _safe_float(probs.get("SELL"), 0.0) or 0.0,
    )
    return directional > hold


def _defect_entry(ticker: str,
                  subsystem: str,
                  symptom: str,
                  likely_root_cause: str,
                  confidence: float,
                  blast_radius: str,
                  classification: str,
                  issue_code: str,
                  evidence: dict | None = None,
                  recommended_fix: str | None = None) -> dict:
    return {
        "ticker": ticker,
        "issue_code": issue_code,
        "subsystem": subsystem,
        "symptom": symptom,
        "likely_root_cause": likely_root_cause,
        "confidence": float(confidence),
        "blast_radius": blast_radius,
        "classification": classification,
        "evidence": dict(evidence or {}),
        "recommended_fix": recommended_fix,
    }


def _ticker_summary(report_dir: Path) -> tuple[dict, list]:
    ticker = report_dir.name
    paths = _report_paths(report_dir)
    signal = _load_json(paths["signal"])
    mc = _load_json(paths["montecarlo"])
    dashboard_consistency = _load_json(paths["dashboard_consistency"])
    invariants = validate_signal_artifact(signal, mc if mc else None) if signal else {}

    sig = dict(signal.get("signal", {}) or {})
    selection = dict(signal.get("selection", {}) or {})
    conformal = dict(signal.get("conformal", {}) or signal.get("conformal_prediction", {}) or {})
    backtest_audit = dict(signal.get("backtest_audit", {}) or {})
    mc_rel = dict(mc.get("mc_reliability", {}) or {})

    summary = {
        "ticker": ticker,
        "paths": {key: str(path) for key, path in paths.items()},
        "signal_generated_at": signal.get("generated"),
        "pipeline_status": signal.get("pipeline_status"),
        "execution_status": sig.get("execution_status"),
        "execution_gate": sig.get("execution_gate"),
        "selection_status": selection.get("selection_status"),
        "reference_model_used": selection.get("reference_model_used") or sig.get("reference_model_used"),
        "deployment_model_used": selection.get("deployment_model_used") or sig.get("deployment_model_used"),
        "conformal_method": conformal.get("conformal_method") or sig.get("conformal_method"),
        "conformal_usable": bool(conformal.get("usable_for_execution", False)) if conformal else False,
        "conformal_singleton_rate": _safe_float(conformal.get("singleton_rate"), None),
        "conformal_avg_set_size": _safe_float(conformal.get("avg_set_size"), None),
        "conformal_degenerate": bool(conformal.get("degenerate_execution_conformal", False)),
        "positive_directional_edge": _positive_directional_edge(sig) if sig else False,
        "static_backtest_warning": backtest_audit.get("sanity_status") == "warning",
        "mc_primary_status": mc_rel.get("mc_reliability_status"),
        "mc_baseline_status": mc_rel.get("baseline_reliability_status"),
        "mc_scenario_status": mc_rel.get("scenario_reliability_status"),
        "dashboard_consistency_status": dashboard_consistency.get("status") if dashboard_consistency else None,
        "artifact_invariants": invariants,
    }

    defects = []
    if invariants and invariants.get("status") == "ERROR":
        defects.append(_defect_entry(
            ticker=ticker,
            subsystem="analyzer_pipeline",
            symptom="Final signal artifact violates output invariants.",
            likely_root_cause="State transition or serialization logic is emitting contradictory final fields.",
            confidence=0.99,
            blast_radius="global",
            classification="code_bug",
            issue_code="artifact_invariant_failure",
            evidence={"failure_names": invariants.get("failure_names", [])},
            recommended_fix="Harden final-signal serialization and keep invariant tests on every report write.",
        ))

    if dashboard_consistency and dashboard_consistency.get("status") == "WARNING":
        defects.append(_defect_entry(
            ticker=ticker,
            subsystem="dashboard_truth",
            symptom="Dashboard consistency validator found a display-vs-JSON mismatch.",
            likely_root_cause="Dashboard rendering is drifting from JSON truth or a stale artifact is being shown.",
            confidence=0.90,
            blast_radius="global",
            classification="stale_reporting",
            issue_code="dashboard_truth_mismatch",
            evidence={"mismatches": dashboard_consistency.get("mismatches", [])[:5]},
            recommended_fix="Map dashboard fields only from canonical JSON paths and keep the consistency report visible.",
        ))

    if conformal and not bool(conformal.get("usable_for_execution", False)):
        if bool(conformal.get("degenerate_execution_conformal", False)):
            defects.append(_defect_entry(
                ticker=ticker,
                subsystem="conformal_execution",
                symptom="Execution conformal collapsed into degenerate wide/full prediction sets.",
                likely_root_cause="Cross-conformal tuning is not producing sharp sets, either because calibrated probabilities are too flat or nonconformity scoring is too conservative.",
                confidence=0.95,
                blast_radius="global",
                classification="policy_bug",
                issue_code="conformal_degeneracy",
                evidence={
                    "singleton_rate": conformal.get("singleton_rate"),
                    "avg_set_size": conformal.get("avg_set_size"),
                    "degeneracy_reason": conformal.get("degeneracy_reason"),
                },
                recommended_fix="Inspect nonconformity score distributions and fail fast when all tuned candidates collapse to full sets.",
            ))
        if selection.get("selection_status") == "reference_only_no_deployable_candidate" and _positive_directional_edge(sig):
            defects.append(_defect_entry(
                ticker=ticker,
                subsystem="selection_execution",
                symptom="No deployable candidate exists even though the final classifier shows directional probability or positive edge.",
                likely_root_cause="Execution gating is dominated by conformal unusability or overly strict deployability filters.",
                confidence=0.92,
                blast_radius="global",
                classification="policy_bug",
                issue_code="conformal_blocking_deployment",
                evidence={
                    "selection_status": selection.get("selection_status"),
                    "prediction_set": sig.get("prediction_set"),
                    "edge_minus_cost": sig.get("edge_minus_cost"),
                    "conformal_failures": (sig.get("execution_gate_details", {}) or {}).get("conformal_failures"),
                },
                recommended_fix="Separate raw classifier quality from execution gating and add richer conformal diagnostics to explain the block.",
            ))

    if backtest_audit.get("sanity_status") == "warning":
        defects.append(_defect_entry(
            ticker=ticker,
            subsystem="static_backtest",
            symptom="Static in-sample backtest metrics are implausibly strong relative to walk-forward/CPCV evidence.",
            likely_root_cause="Static compounding, exposure accounting, or trade aggregation can still overstate audit-surface returns.",
            confidence=0.90,
            blast_radius="global",
            classification="stale_reporting",
            issue_code="static_backtest_implausible",
            evidence={
                "sanity_flags": backtest_audit.get("sanity_flags", []),
                "strategy_return_pct": backtest_audit.get("strategy_return_pct"),
                "strategy_return_vs_buyhold_ratio": backtest_audit.get("strategy_return_vs_buyhold_ratio"),
            },
            recommended_fix="Keep static backtest demoted and extend audit checks around compounding, re-entry, and cost application.",
        ))

    primary_status = str(mc_rel.get("mc_reliability_status", "") or "")
    baseline_status = str(mc_rel.get("baseline_reliability_status", "") or primary_status)
    if primary_status and baseline_status and primary_status != baseline_status:
        defects.append(_defect_entry(
            ticker=ticker,
            subsystem="monte_carlo_summary",
            symptom="Primary Monte Carlo reliability badge disagrees with the baseline-model status.",
            likely_root_cause="Scenario or stress-model miscalibration is still bleeding into the main reliability label.",
            confidence=0.88,
            blast_radius="global",
            classification="code_bug",
            issue_code="mc_primary_badge_mismatch",
            evidence={
                "mc_reliability_status": primary_status,
                "baseline_reliability_status": baseline_status,
                "scenario_reliability_status": mc_rel.get("scenario_reliability_status"),
            },
            recommended_fix="Keep baseline reliability as the primary badge and surface scenario reliability separately.",
        ))

    summary["defects"] = defects
    return summary, defects


def _iter_report_dirs(reports_dir: Path) -> list[Path]:
    dirs = []
    for child in sorted(reports_dir.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if list(child.glob("*_signal.json")):
            dirs.append(child)
    return dirs


def _recommended_fix_order(defects: list[dict]) -> list[dict]:
    codes = Counter(defect.get("issue_code") for defect in defects)
    plan = []
    if codes.get("artifact_invariant_failure"):
        plan.append({
            "priority": 1,
            "topic": "State consistency",
            "why": "Contradictory final-artifact fields still exist in live outputs.",
            "actions": [
                "Run invariant validation on every final signal write.",
                "Fail tests when selection, execution, router, or conformal fields disagree.",
            ],
            "evidence_count": int(codes["artifact_invariant_failure"]),
        })
    if codes.get("dashboard_truth_mismatch"):
        plan.append({
            "priority": len(plan) + 1,
            "topic": "Dashboard truthfulness",
            "why": "Rendered dashboard values can still drift from JSON truth.",
            "actions": [
                "Keep dashboard fields mapped only from canonical JSON paths.",
                "Surface consistency-report warnings directly in the dashboard output.",
            ],
            "evidence_count": int(codes["dashboard_truth_mismatch"]),
        })
    if codes.get("conformal_degeneracy") or codes.get("conformal_blocking_deployment"):
        plan.append({
            "priority": len(plan) + 1,
            "topic": "Conformal execution quality",
            "why": "Execution conformal remains the main blocker to deployability.",
            "actions": [
                "Inspect nonconformity score distributions and qhat behavior by class.",
                "Keep singleton-focused tuning but stop early when every candidate degenerates to full sets.",
                "Report otherwise-healthy models blocked only by conformal.",
            ],
            "evidence_count": int(codes["conformal_degeneracy"] + codes["conformal_blocking_deployment"]),
        })
    if codes.get("static_backtest_implausible"):
        plan.append({
            "priority": len(plan) + 1,
            "topic": "Static backtest sanity",
            "why": "In-sample backtest still produces implausible numbers compared with OOS evidence.",
            "actions": [
                "Audit compounding, re-entry logic, and transaction-cost application.",
                "Keep static backtest visually secondary whenever sanity flags fire.",
            ],
            "evidence_count": int(codes["static_backtest_implausible"]),
        })
    if codes.get("mc_primary_badge_mismatch"):
        plan.append({
            "priority": len(plan) + 1,
            "topic": "Monte Carlo reliability aggregation",
            "why": "Primary MC reliability can still be polluted by scenario-model behavior.",
            "actions": [
                "Keep baseline-model reliability as the primary badge.",
                "Expose scenario reliability and aggregation policy separately in JSON and dashboard output.",
            ],
            "evidence_count": int(codes["mc_primary_badge_mismatch"]),
        })
    return plan


def build_repo_debug_audit(reports_dir: str | Path = "reports",
                           focus_tickers: list[str] | None = None) -> dict:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_dirs = _iter_report_dirs(root)
    latest_success = _latest_successful_ticker(report_dirs)
    focus = []
    for ticker in list(focus_tickers or []) + list(DEFAULT_FOCUS_TICKERS):
        if ticker and ticker not in focus and (root / ticker).exists():
            focus.append(ticker)
    if latest_success and latest_success not in focus:
        focus.append(latest_success)

    per_ticker = {}
    defects = []
    invariant_status_frequency = Counter()
    dashboard_warning_count = 0
    conformal_block_count = 0
    backtest_warning_count = 0

    for report_dir in report_dirs:
        summary, ticker_defects = _ticker_summary(report_dir)
        per_ticker[report_dir.name] = summary
        defects.extend(ticker_defects)
        inv_status = str((summary.get("artifact_invariants", {}) or {}).get("status", "MISSING") or "MISSING")
        invariant_status_frequency[inv_status] += 1
        dashboard_warning_count += int(summary.get("dashboard_consistency_status") == "WARNING")
        conformal_block_count += int(
            not bool(summary.get("conformal_usable", True))
            and summary.get("selection_status") == "reference_only_no_deployable_candidate"
        )
        backtest_warning_count += int(bool(summary.get("static_backtest_warning", False)))

    classification_frequency = Counter(defect.get("classification") for defect in defects)
    issue_frequency = Counter(defect.get("issue_code") for defect in defects)
    return {
        "generated": datetime.datetime.now().isoformat(),
        "reports_dir": str(root.resolve()),
        "latest_successful_ticker": latest_success,
        "focus_tickers": focus,
        "summary": {
            "report_count": int(len(report_dirs)),
            "defect_count": int(len(defects)),
            "dashboard_warning_count": int(dashboard_warning_count),
            "conformal_block_count": int(conformal_block_count),
            "static_backtest_warning_count": int(backtest_warning_count),
        },
        "artifact_summary": {
            "invariant_status_frequency": dict(invariant_status_frequency),
            "classification_frequency": dict(classification_frequency),
            "issue_frequency": dict(issue_frequency),
        },
        "per_ticker": per_ticker,
        "defect_ledger": defects,
        "recommended_fix_order": _recommended_fix_order(defects),
    }


def render_repo_debug_audit_markdown(audit: dict) -> str:
    summary = dict(audit.get("summary", {}) or {})
    artifact_summary = dict(audit.get("artifact_summary", {}) or {})
    focus = list(audit.get("focus_tickers", []) or [])
    defects = list(audit.get("defect_ledger", []) or [])
    plan = list(audit.get("recommended_fix_order", []) or [])

    lines = [
        "# Repo Debug Audit",
        "",
        f"- Generated: {audit.get('generated', 'N/A')}",
        f"- Reports scanned: {summary.get('report_count', 0)}",
        f"- Defects logged: {summary.get('defect_count', 0)}",
        f"- Latest successful ticker: {audit.get('latest_successful_ticker') or 'N/A'}",
        f"- Focus tickers: {', '.join(focus) if focus else 'N/A'}",
        "",
        "## Artifact Summary",
        "",
        f"- Invariant status frequency: {artifact_summary.get('invariant_status_frequency', {})}",
        f"- Classification frequency: {artifact_summary.get('classification_frequency', {})}",
        f"- Issue frequency: {artifact_summary.get('issue_frequency', {})}",
        "",
        "## Defect Ledger",
        "",
    ]
    if not defects:
        lines.append("- No defects detected.")
    else:
        for defect in defects:
            lines.append(
                f"- [{defect.get('ticker')}] {defect.get('issue_code')}: "
                f"{defect.get('symptom')} "
                f"(subsystem={defect.get('subsystem')}, class={defect.get('classification')}, "
                f"blast={defect.get('blast_radius')})"
            )
            lines.append(f"  Root cause: {defect.get('likely_root_cause')}")
            if defect.get("recommended_fix"):
                lines.append(f"  Fix: {defect.get('recommended_fix')}")
    lines.extend(["", "## Recommended Fix Order", ""])
    if not plan:
        lines.append("- No follow-up steps required.")
    else:
        for step in plan:
            lines.append(
                f"- P{step.get('priority')}: {step.get('topic')} "
                f"(evidence={step.get('evidence_count', 0)})"
            )
            lines.append(f"  Why: {step.get('why')}")
            for action in step.get("actions", []):
                lines.append(f"  Action: {action}")
    lines.append("")
    return "\n".join(lines)
