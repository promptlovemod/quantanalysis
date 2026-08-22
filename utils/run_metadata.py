import datetime
import hashlib
import json
from pathlib import Path

try:
    import numpy as _np
except ImportError:  # pragma: no cover - optional
    _np = None


DEFAULT_CONFIG_VERSION = "v14.0"


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if _np is not None:
        if isinstance(value, _np.bool_):
            return bool(value)
        if isinstance(value, _np.integer):
            return int(value)
        if isinstance(value, _np.floating):
            return float(value)
        if isinstance(value, _np.ndarray):
            return value.tolist()
    return value


def stable_hash(payload, length: int = 12) -> str:
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def normalize_universe(universe) -> list:
    items = []
    for item in universe or []:
        token = str(item).strip().upper()
        if token and token not in items:
            items.append(token)
    return items


def build_run_metadata(mode: str,
                       seed=None,
                       config: dict | None = None,
                       config_version: str | None = None,
                       enabled_models=None,
                       optimizer: str | None = None,
                       watchlist_path=None,
                       universe=None,
                       extra: dict | None = None,
                       started_at: str | None = None) -> dict:
    started_at = started_at or datetime.datetime.now().isoformat()
    config_payload = _json_ready(config or {})
    config_hash = stable_hash(config_payload)
    norm_universe = normalize_universe(universe)
    universe_hash = stable_hash(norm_universe) if norm_universe else ""
    base = {
        "run_id": f"{datetime.datetime.now():%Y%m%dT%H%M%S}_{config_hash[:8]}",
        "mode": str(mode),
        "started_at": started_at,
        "seed": seed,
        "config_version": config_version or DEFAULT_CONFIG_VERSION,
        "config_hash": config_hash,
        "enabled_models": list(enabled_models or []),
        "optimizer": optimizer or "",
        "watchlist_path": str(watchlist_path) if watchlist_path else "",
        "universe_hash": universe_hash,
        "universe_size": len(norm_universe),
    }
    if extra:
        base.update(_json_ready(extra))
    return base


def complete_run_metadata(metadata: dict, status: str | None = None) -> dict:
    payload = dict(metadata or {})
    payload["completed_at"] = datetime.datetime.now().isoformat()
    if status:
        payload["status"] = status
    return payload


def append_experiment_record(base_dir, metadata: dict,
                             status: str = "OK",
                             summary: dict | None = None) -> Path:
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = complete_run_metadata(metadata, status=status)
    if summary:
        record["summary"] = _json_ready(summary)
    path = out_dir / "experiment_runs.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=True, default=str)
        f.write("\n")
    return path
