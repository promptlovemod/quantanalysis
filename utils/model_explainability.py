import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:  # pragma: no cover - optional
    shap = None
    HAS_SHAP = False


def _extract_importance(model, feature_names):
    if not feature_names:
        return np.array([], dtype=float)
    imp = None
    if hasattr(model, "feature_importances_"):
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            imp = None
    elif hasattr(model, "get_feature_importance"):
        try:
            imp = np.asarray(model.get_feature_importance(), dtype=float)
        except Exception:
            imp = None
    elif hasattr(model, "coef_"):
        try:
            coef = np.asarray(model.coef_, dtype=float)
            imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        except Exception:
            imp = None
    if imp is None or len(imp) != len(feature_names):
        return np.array([], dtype=float)
    return imp


def _resolve_shap_values(values, class_index: int):
    if isinstance(values, list):
        idx = min(max(int(class_index), 0), len(values) - 1)
        return np.asarray(values[idx], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        idx = min(max(int(class_index), 0), arr.shape[-1] - 1)
        return arr[:, :, idx]
    return arr


def _top_rows(feature_names, values, top_n: int):
    if values is None or len(values) != len(feature_names):
        return []
    order = np.argsort(np.abs(values))[::-1][:top_n]
    rows = []
    for idx in order:
        rows.append({
            "feature": str(feature_names[idx]),
            "value": float(values[idx]),
            "abs_value": float(abs(values[idx])),
        })
    return rows


def build_explainability(model,
                         X_reference,
                         X_latest,
                         feature_names,
                         class_index: int = 0,
                         model_name: str = "",
                         top_n: int = 10) -> dict:
    feature_names = list(feature_names or [])
    if not feature_names:
        return {
            "available": False,
            "model_name": model_name,
            "method": "unavailable",
            "reason": "missing_feature_names",
            "top_global": [],
            "top_local": [],
        }

    X_reference = np.asarray(X_reference, dtype=float)
    X_latest = np.asarray(X_latest, dtype=float)
    if X_reference.ndim == 1:
        X_reference = X_reference.reshape(1, -1)
    if X_latest.ndim == 1:
        X_latest = X_latest.reshape(1, -1)
    if X_reference.shape[1] != len(feature_names) or X_latest.shape[1] != len(feature_names):
        return {
            "available": False,
            "model_name": model_name,
            "method": "unavailable",
            "reason": "shape_mismatch",
            "top_global": [],
            "top_local": [],
        }

    payload = {
        "available": False,
        "model_name": model_name,
        "method": "unavailable",
        "top_global": [],
        "top_local": [],
    }

    if HAS_SHAP:
        try:
            sample = X_reference[: min(len(X_reference), 256)]
            explainer = shap.TreeExplainer(model)
            shap_ref = _resolve_shap_values(explainer.shap_values(sample), class_index)
            shap_latest = _resolve_shap_values(explainer.shap_values(X_latest[:1]), class_index)
            global_vals = np.mean(np.abs(shap_ref), axis=0)
            local_vals = np.asarray(shap_latest[0], dtype=float)
            payload.update({
                "available": True,
                "method": "shap_tree",
                "top_global": _top_rows(feature_names, global_vals, top_n),
                "top_local": _top_rows(feature_names, local_vals, top_n),
            })
            return payload
        except Exception as exc:
            payload["shap_error"] = str(exc)

    imp = _extract_importance(model, feature_names)
    if imp.size:
        payload.update({
            "available": True,
            "method": "model_importance",
            "top_global": _top_rows(feature_names, imp, top_n),
            "top_local": _top_rows(feature_names, imp * X_latest[0], top_n),
        })
        return payload

    payload["reason"] = "unsupported_model"
    return payload
