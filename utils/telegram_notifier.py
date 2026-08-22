import logging
import os
import time
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency at runtime
    requests = None


LOG = logging.getLogger("telegram_notifier")
TIMEOUT = 10
RETRIES = 2
MAX_ERROR_CHARS = 3500
DEFAULT_DELAY_MS = 5000
DEFAULT_RATE_LIMIT_SECONDS = 0.0
DEFAULT_ALERT_SUPPRESSION_SECONDS = 0.0


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default) or default).strip())
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default) or default).strip())
    except Exception:
        return float(default)


def progress_enabled() -> bool:
    return _env_truthy("TELEGRAM_ENABLED") and _env_truthy("TELEGRAM_PROGRESS_ENABLED")


def _final_notifications_enabled() -> bool:
    return _env_truthy("TELEGRAM_ENABLED") and not _env_truthy("TELEGRAM_SUPPRESS_CHILD")


def _is_enabled() -> bool:
    return _final_notifications_enabled() or progress_enabled()


def _credentials() -> tuple[str | None, str | None]:
    return (
        str(os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip() or None,
        str(os.environ.get("TELEGRAM_CHAT_ID", "") or "").strip() or None,
    )


def _api_url(method: str) -> str | None:
    token, _ = _credentials()
    if not token:
        return None
    return f"https://api.telegram.org/bot{token}/{method}"


def _state_path() -> Path:
    root = str(os.environ.get("TELEGRAM_STATE_DIR", "") or "").strip()
    if root:
        return Path(root) / "delivery_state.json"
    return Path("reports") / ".telegram" / "delivery_state.json"


def _load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {"last_attempt_ts": None, "notifications": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_attempt_ts": None, "notifications": {}}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _apply_rate_limit(*, sleeper=time.sleep) -> float:
    min_interval = max(0.0, _float_env("TELEGRAM_RATE_LIMIT_SECONDS", DEFAULT_RATE_LIMIT_SECONDS))
    if min_interval <= 0:
        return 0.0
    state = _load_state()
    now = time.time()
    last_attempt_raw = state.get("last_attempt_ts")
    if last_attempt_raw in (None, ""):
        wait_seconds = 0.0
    else:
        last_attempt = float(last_attempt_raw or 0.0)
        wait_seconds = max(0.0, min_interval - max(0.0, now - last_attempt))
    if wait_seconds > 0:
        sleeper(wait_seconds)
        now += wait_seconds
    state["last_attempt_ts"] = now
    _save_state(state)
    return wait_seconds


def _delivery_signature(kind: str, text: str = "", paths: Iterable[str] | None = None) -> str:
    normalized_paths = [str(Path(path)) for path in (paths or [])]
    payload = f"{kind}|{str(text or '').strip()}|{'|'.join(normalized_paths)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _delivery_suppressed(kind: str, signature: str) -> bool:
    suppression_window = max(
        0.0,
        _float_env("TELEGRAM_ALERT_SUPPRESSION_SECONDS", DEFAULT_ALERT_SUPPRESSION_SECONDS),
    )
    if suppression_window <= 0:
        return False
    state = _load_state()
    bucket = ((state.get("notifications", {}) or {}).get(kind, {}) or {})
    last_signature = str(bucket.get("signature", "") or "")
    last_sent_ts = float(bucket.get("sent_at_ts", 0.0) or 0.0)
    if not last_signature or last_signature != signature:
        return False
    return (time.time() - last_sent_ts) < suppression_window


def _record_delivery(kind: str, signature: str, delivered: bool) -> None:
    state = _load_state()
    notifications = state.setdefault("notifications", {})
    bucket = notifications.setdefault(kind, {})
    bucket["signature"] = signature
    bucket["sent_at_ts"] = time.time()
    bucket["delivered"] = bool(delivered)
    _save_state(state)


def _post(method: str, *, data=None, files=None):
    if not _is_enabled():
        return None
    if requests is None:
        LOG.warning("Telegram notifier unavailable: requests is not installed")
        return None
    url = _api_url(method)
    _, chat_id = _credentials()
    if not url or not chat_id:
        LOG.warning("Telegram notifier skipped: missing bot token or chat id")
        return None

    payload = dict(data or {})
    payload.setdefault("chat_id", chat_id)
    _apply_rate_limit()
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.post(url, data=payload, files=files, timeout=TIMEOUT)
            if resp.ok:
                try:
                    return resp.json()
                except Exception:
                    return {"ok": True}
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:  # pragma: no cover - network path
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < RETRIES:
            time.sleep(1.0 + attempt)
    LOG.warning("Telegram notifier failed (%s)", last_error or "unknown error")
    return None


def _ok_result(payload) -> bool:
    return bool(payload and payload.get("ok", True))


def send_message(text: str) -> dict | None:
    if not _is_enabled():
        return None
    payload = str(text or "").strip() or "Quant analyzer update."
    return _post("sendMessage", data={"text": payload})


def edit_message_text(message_id: int | str, text: str) -> bool:
    if not progress_enabled():
        return False
    try:
        msg_id = int(message_id)
    except Exception:
        return False
    payload = _post("editMessageText", data={"message_id": msg_id, "text": str(text or "").strip() or "Quant analyzer update."})
    return _ok_result(payload)


def send_chat_action(action: str = "typing") -> bool:
    if not _is_enabled():
        return False
    payload = _post("sendChatAction", data={"action": str(action or "typing")})
    return _ok_result(payload)


def send_photo(photo_path: str, caption: str | None = None) -> bool:
    if not _final_notifications_enabled():
        return False
    path = Path(photo_path)
    if not path.exists() or path.suffix.lower() != ".png":
        LOG.warning("Telegram photo skipped: invalid PNG path %s", photo_path)
        return False
    try:
        with open(path, "rb") as fh:
            return _post(
                "sendPhoto",
                data={"caption": caption or ""},
                files={"photo": fh},
            ) is not None
    except Exception as exc:
        LOG.warning("Telegram photo failed (%s)", exc)
        return False


def send_document(file_path: str, caption: str | None = None) -> bool:
    if not _final_notifications_enabled():
        return False
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        LOG.warning("Telegram document skipped: invalid path %s", file_path)
        return False
    try:
        with open(path, "rb") as fh:
            return _post(
                "sendDocument",
                data={"caption": caption or ""},
                files={"document": fh},
            ) is not None
    except Exception as exc:
        LOG.warning("Telegram document failed (%s)", exc)
        return False


def send_error(text: str) -> bool:
    if not _final_notifications_enabled():
        return False
    payload = str(text or "").strip()
    if not payload:
        payload = "Quant analyzer failed with no error text."
    payload = payload[-MAX_ERROR_CHARS:]
    return _post("sendMessage", data={"text": payload}) is not None


def result_delay_seconds() -> float:
    delay_ms = max(0, _int_env("TELEGRAM_DELAY_BEFORE_RESULT_MS", DEFAULT_DELAY_MS))
    return float(delay_ms) / 1000.0


@dataclass
class TelegramProgressSession:
    run_name: str
    enabled: bool = False
    message_id: int | None = None

    def start(self, text: str) -> bool:
        if not self.enabled:
            return False
        if self.message_id is not None:
            return self.update(text)
        payload = send_message(text)
        try:
            self.message_id = int((((payload or {}).get("result", {}) or {}).get("message_id")))
        except Exception:
            self.message_id = None
        return self.message_id is not None

    def attach(self, message_id: int | str | None) -> bool:
        if not self.enabled:
            return False
        try:
            self.message_id = int(message_id) if message_id is not None else None
        except Exception:
            self.message_id = None
        return self.message_id is not None

    def update(self, text: str) -> bool:
        if not self.enabled or self.message_id is None:
            return False
        return edit_message_text(self.message_id, text)

    def mark_finalizing(self, text: str = "Finalizing result...") -> bool:
        return self.update(text)

    def mark_failed(self, text: str = "Failed") -> bool:
        return self.update(text)


def create_progress_session(run_name: str, message_id: int | str | None = None) -> TelegramProgressSession:
    session = TelegramProgressSession(run_name=str(run_name or "Quant analyzer"), enabled=progress_enabled())
    if message_id is not None:
        session.attach(message_id)
    return session


def notify_success(photo_path: str | Iterable[str], caption: str | None = None) -> bool:
    paths = list(photo_path) if isinstance(photo_path, (list, tuple, set)) else [str(photo_path)]
    signature = _delivery_signature("success", text=caption or "", paths=paths)
    if _delivery_suppressed("success", signature):
        LOG.info("Telegram success notification suppressed as duplicate")
        return False
    if isinstance(photo_path, (list, tuple, set)):
        ok = False
        for idx, path in enumerate(photo_path):
            label = caption if idx == 0 else f"{caption or 'Result'} ({idx + 1})"
            ok = send_photo(str(path), caption=label) or ok
        if ok:
            _record_delivery("success", signature, True)
        return ok
    ok = send_photo(str(photo_path), caption=caption)
    if ok:
        _record_delivery("success", signature, True)
    return ok


def notify_failure(run_name: str, err_text: str | None = None, log_path: str | None = None) -> bool:
    caption = f"[{run_name}] failed"
    signature = _delivery_signature("failure", text=f"{caption}\n{err_text or ''}", paths=[log_path] if log_path else [])
    if _delivery_suppressed("failure", signature):
        LOG.info("Telegram failure notification suppressed as duplicate")
        return False
    if log_path:
        path = Path(log_path)
        if path.exists():
            ok = send_document(str(path), caption=caption)
            if ok:
                _record_delivery("failure", signature, True)
            return ok
    text = str(err_text or "").strip()
    text = text[-MAX_ERROR_CHARS:]
    message = f"{caption}\n\n{text}" if text else caption
    ok = send_error(message)
    if ok:
        _record_delivery("failure", signature, True)
    return ok
