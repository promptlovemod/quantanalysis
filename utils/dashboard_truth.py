import json
import math
from pathlib import Path


CRITICAL_FIELD_MAP = {
    "final_signal": ("signal", ("signal", "signal")),
    "confidence": ("signal", ("signal", "confidence")),
    "selection_status": ("signal", ("signal", "selection_status")),
    "reference_model": ("signal", ("selection", "reference_model_used")),
    "deployment_model": ("signal", ("selection", "deployment_model_used")),
    "conformal_method": ("signal", ("conformal", "conformal_method")),
    "conformal_target_coverage": ("signal", ("conformal", "target_coverage")),
    "walkforward_sharpe": ("signal", ("walkforward_backtest", "wf_sharpe")),
    "cpcv_p5": ("signal", ("cpcv", "sharpe_p5")),
    "mc_reliability_status": ("montecarlo", ("mc_reliability", "mc_reliability_status")),
}


def _get_nested(data, path):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur.get(key)
    return cur


def extract_dashboard_truth(signal_json: dict | None, mc_json: dict | None) -> dict:
    sources = {
        "signal": dict(signal_json or {}),
        "montecarlo": dict(mc_json or {}),
    }
    return {
        field: _get_nested(sources[source_name], path)
        for field, (source_name, path) in CRITICAL_FIELD_MAP.items()
    }


def build_debug_footer_context(signal_json: dict | None,
                               dashboard_generated_at: str) -> dict:
    payload = dict(signal_json or {})
    selection = dict(payload.get("selection", {}) or {})
    conformal = dict(payload.get("conformal", {}) or {})
    return {
        "signal_generated_at": payload.get("generated"),
        "dashboard_generated_at": dashboard_generated_at,
        "conformal_method": conformal.get("conformal_method"),
        "reference_model": selection.get("reference_model_used"),
        "deployment_model": selection.get("deployment_model_used"),
        "schema_version": payload.get("schema_version"),
    }


def _normalize_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return str(value)


def _values_match(displayed, source) -> bool:
    lhs = _normalize_value(displayed)
    rhs = _normalize_value(source)
    if lhs is None and rhs is None:
        return True
    if isinstance(lhs, float) and isinstance(rhs, float):
        return math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)
    return lhs == rhs


def validate_dashboard_payload(display_payload: dict,
                               signal_json: dict | None,
                               mc_json: dict | None,
                               dashboard_generated_at: str) -> dict:
    source_truth = extract_dashboard_truth(signal_json, mc_json)
    checked_fields = []
    mismatches = []
    for field, (source_name, path) in CRITICAL_FIELD_MAP.items():
        displayed = (display_payload or {}).get(field)
        source_val = source_truth.get(field)
        matched = _values_match(displayed, source_val)
        record = {
            "field": field,
            "source": source_name,
            "path": ".".join(path),
            "displayed_value": displayed,
            "source_value": source_val,
            "match": bool(matched),
        }
        checked_fields.append(record)
        if not matched:
            mismatches.append(record)
    return {
        "status": "OK" if not mismatches else "WARNING",
        "checked_fields": checked_fields,
        "mismatches": mismatches,
        "source_timestamps": {
            "signal_generated_at": (signal_json or {}).get("generated"),
            "montecarlo_generated_at": (mc_json or {}).get("generated"),
        },
        "dashboard_generated_at": dashboard_generated_at,
    }


def write_dashboard_consistency_report(path: str | Path, report: dict) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return out_path
