import datetime
import json
from pathlib import Path


def _bool_map(values: dict | None) -> dict:
    return {str(key): bool(val) for key, val in (values or {}).items()}


def _artifact_presence(paths: dict[str, str | Path]) -> dict:
    return {
        name: {
            "path": str(Path(path)),
            "exists": Path(path).exists(),
        }
        for name, path in paths.items()
    }


def _missing_artifacts(artifacts: dict) -> list[str]:
    return [name for name, meta in artifacts.items() if not meta.get("exists")]


def summarize_single_run(
    ticker: str,
    out_dir: str | Path,
    module_success: dict | None,
    *,
    signal_json: dict | None = None,
    fund_json: dict | None = None,
    mc_json: dict | None = None,
    consistency_report: dict | None = None,
    dashboard_path: str | Path | None = None,
) -> dict:
    report_dir = Path(out_dir)
    module_status = _bool_map(module_success)
    artifacts = _artifact_presence(
        {
            "signal_json": report_dir / f"{ticker}_signal.json",
            "fundamentals_json": report_dir / f"{ticker}_fundamentals.json",
            "montecarlo_json": report_dir / f"{ticker}_montecarlo.json",
            "dashboard_html": dashboard_path or (report_dir / f"{ticker}_dashboard.html"),
            "analysis_png": report_dir / f"{ticker}_analysis.png",
            "selection_png": report_dir / f"{ticker}_selection_diagnostics.png",
            "dl_models_png": report_dir / f"{ticker}_dl_models.png",
            "montecarlo_png": report_dir / f"{ticker}_montecarlo.png",
        }
    )
    missing_artifacts = _missing_artifacts(artifacts)
    signal_payload = dict(signal_json or {})
    pipeline_errors = list(signal_payload.get("pipeline_errors", []) or [])
    fatal_pipeline_errors = [err for err in pipeline_errors if isinstance(err, dict) and err.get("fatal")]
    artifact_invariants = dict(signal_payload.get("artifact_invariants", {}) or {})
    consistency_status = (consistency_report or {}).get("status") or "N/A"
    invariant_status = artifact_invariants.get("status") or "N/A"
    pipeline_status = str(signal_payload.get("pipeline_status", "UNKNOWN") or "UNKNOWN").upper()

    overall_status = "OK"
    if not all(module_status.values()) or pipeline_status == "FAILED":
        overall_status = "FAILED"
    elif consistency_status != "OK" or invariant_status not in {"OK", "N/A"} or missing_artifacts:
        overall_status = "WARNING"

    return {
        "ticker": ticker,
        "generated_at": datetime.datetime.now().isoformat(),
        "overall_status": overall_status,
        "module_status": module_status,
        "module_success_count": int(sum(1 for ok in module_status.values() if ok)),
        "module_total": int(len(module_status)),
        "failed_modules": [name for name, ok in module_status.items() if not ok],
        "pipeline_status": pipeline_status,
        "pipeline_error_count": int(len(pipeline_errors)),
        "fatal_pipeline_error_count": int(len(fatal_pipeline_errors)),
        "dashboard_consistency_status": consistency_status,
        "artifact_invariant_status": invariant_status,
        "missing_artifact_count": int(len(missing_artifacts)),
        "missing_artifacts": missing_artifacts,
        "artifacts": artifacts,
        "data_timestamps": {
            "signal_generated_at": signal_payload.get("generated"),
            "fund_generated_at": (fund_json or {}).get("generated"),
            "mc_generated_at": (mc_json or {}).get("generated"),
        },
    }


def summarize_portfolio_run(
    tickers: list[str],
    failed: list[str],
    portfolio_summary: dict | None,
    *,
    stock_data: dict | None = None,
    dashboard_path: str | Path | None = None,
    extra_artifacts: dict | None = None,
) -> dict:
    summary = dict(portfolio_summary or {})
    optimizer = dict(summary.get("optimizer", {}) or {})
    quality_gate = dict(summary.get("quality_gate", {}) or {})
    stock_data = dict(stock_data or {})
    failed_set = {str(t) for t in (failed or [])}
    pipeline_failed = []
    for ticker in tickers:
        signal_data = ((stock_data.get(ticker, {}) or {}).get("signal_data", {}) or {})
        if str(signal_data.get("pipeline_status", "UNKNOWN") or "UNKNOWN").upper() == "FAILED":
            pipeline_failed.append(ticker)

    artifact_map = {
        "portfolio_summary_json": Path("reports") / "portfolio_summary.json",
        "portfolio_dashboard_html": dashboard_path or (Path("reports") / "portfolio_dashboard.html"),
        "portfolio_optimizer_png": Path("reports") / "portfolio_optimizer.png",
        "benchmark_quality_png": Path("reports") / "benchmark_quality.png",
        "panel_summary_png": Path("reports") / "panel_summary.png",
    }
    for name, path in (extra_artifacts or {}).items():
        artifact_map[str(name)] = Path(path)
    artifacts = _artifact_presence(artifact_map)
    missing_artifacts = _missing_artifacts(artifacts)

    overall_status = "OK"
    if failed_set or pipeline_failed:
        overall_status = "FAILED"
    elif quality_gate.get("status") in {"FAIL", "MARGINAL"} or missing_artifacts:
        overall_status = "WARNING"

    return {
        "generated_at": datetime.datetime.now().isoformat(),
        "overall_status": overall_status,
        "ticker_count": int(len(tickers)),
        "failed_tickers": sorted(failed_set),
        "pipeline_failed_tickers": sorted(pipeline_failed),
        "quality_gate_status": quality_gate.get("status") or "N/A",
        "allocation_status": optimizer.get("allocation_status") or summary.get("allocation_status") or "unknown",
        "actionable_universe_size": int(summary.get("actionable_universe_size", 0) or 0),
        "non_actionable_universe_size": int(summary.get("non_actionable_universe_size", 0) or 0),
        "cash_weight": float(summary.get("cash_weight", 0.0) or 0.0),
        "missing_artifact_count": int(len(missing_artifacts)),
        "missing_artifacts": missing_artifacts,
        "artifacts": artifacts,
    }


def write_orchestration_summary(path: str | Path, payload: dict) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path
