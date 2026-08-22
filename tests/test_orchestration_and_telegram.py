from __future__ import annotations

from pathlib import Path

from utils.orchestration_summary import summarize_portfolio_run, summarize_single_run
from utils import telegram_notifier as notifier


def test_summarize_single_run_flags_failed_modules_and_missing_artifacts(tmp_path: Path):
    out_dir = tmp_path / "AAPL"
    out_dir.mkdir()
    (out_dir / "AAPL_signal.json").write_text("{}", encoding="utf-8")
    (out_dir / "AAPL_fundamentals.json").write_text("{}", encoding="utf-8")
    (out_dir / "AAPL_dashboard.html").write_text("<html></html>", encoding="utf-8")

    summary = summarize_single_run(
        "AAPL",
        out_dir,
        {"ML": True, "FUND": False, "MC": True},
        signal_json={"pipeline_status": "FAILED", "pipeline_errors": [{"fatal": True}]},
        fund_json={},
        mc_json={},
        consistency_report={"status": "OK"},
        dashboard_path=out_dir / "AAPL_dashboard.html",
    )

    assert summary["overall_status"] == "FAILED"
    assert summary["failed_modules"] == ["FUND"]
    assert summary["fatal_pipeline_error_count"] == 1
    assert "montecarlo_json" in summary["missing_artifacts"]


def test_summarize_portfolio_run_reports_warning_from_gate(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "portfolio_summary.json").write_text("{}", encoding="utf-8")
    (reports_dir / "portfolio_dashboard.html").write_text("<html></html>", encoding="utf-8")

    summary = summarize_portfolio_run(
        ["AAPL", "MSFT"],
        [],
        {
            "quality_gate": {"status": "FAIL"},
            "optimizer": {"allocation_status": "cash_no_actionable_names"},
            "actionable_universe_size": 0,
            "non_actionable_universe_size": 2,
            "cash_weight": 1.0,
        },
        stock_data={
            "AAPL": {"signal_data": {"pipeline_status": "OK"}},
            "MSFT": {"signal_data": {"pipeline_status": "OK"}},
        },
        dashboard_path=reports_dir / "portfolio_dashboard.html",
        extra_artifacts={
            "portfolio_summary_json": reports_dir / "portfolio_summary.json",
            "portfolio_optimizer_png": reports_dir / "portfolio_optimizer.png",
        },
    )

    assert summary["overall_status"] == "WARNING"
    assert summary["quality_gate_status"] == "FAIL"
    assert summary["allocation_status"] == "cash_no_actionable_names"


def test_telegram_rate_limit_sleeps_until_slot_available(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TELEGRAM_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_RATE_LIMIT_SECONDS", "0.5")
    timeline = iter([0.0, 0.1])
    sleep_calls: list[float] = []

    monkeypatch.setattr(notifier.time, "time", lambda: next(timeline))

    first_wait = notifier._apply_rate_limit(sleeper=lambda seconds: sleep_calls.append(seconds))
    second_wait = notifier._apply_rate_limit(sleeper=lambda seconds: sleep_calls.append(seconds))

    assert first_wait == 0.0
    assert round(second_wait, 3) == 0.4
    assert sleep_calls == [0.4]


def test_telegram_duplicate_failure_can_be_suppressed(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TELEGRAM_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALERT_SUPPRESSION_SECONDS", "60")
    calls: list[tuple[str, str | None, str | None]] = []

    monkeypatch.setattr(notifier, "send_document", lambda path, caption=None: calls.append((path, caption, None)) or True)
    monkeypatch.setattr(notifier, "send_error", lambda text: calls.append(("", None, text)) or True)

    log_path = tmp_path / "run.log"
    log_path.write_text("boom", encoding="utf-8")
    first = notifier.notify_failure("portfolio", "boom", log_path=str(log_path))
    second = notifier.notify_failure("portfolio", "boom", log_path=str(log_path))

    assert first is True
    assert second is False
    assert len(calls) == 1
