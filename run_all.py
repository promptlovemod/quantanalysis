# -*- coding: utf-8 -*-
"""
Stock ML Analyzer  V.0.7.0  —  Master Runner
==========================================
NEW 19  Portfolio Scanner Mode  (--portfolio watchlist.txt)
        Batch-runs all three analysis modules for every ticker in a
        watchlist file, then aggregates results into a single portfolio
        ranking dashboard with mean-variance allocation suggestions.

NEW 20  Mean-Variance Allocation
        scipy.optimize computes optimal long-only weights from ML signals,
        Monte Carlo expected returns, and historical volatility.
        Max 25% per position.

NEW 21  DuckDB speed-up  (in analyzer.py)
        Cached OHLCV means repeat tickers are not re-downloaded.

NEW 22  Tiingo data source  (optional, in analyzer.py)
        Set TIINGO_API_KEY at the top of analyzer.py for institutional-grade
        adjusted OHLCV. Falls back to yfinance automatically.

Usage:
  python run_all.py                         interactive (asks single or portfolio)
  python run_all.py AAPL                    single stock deep-dive
  python run_all.py --portfolio watch.txt   portfolio scan
  python run_all.py --portfolio watch.txt --parallel 2
"""
import subprocess, sys, json, webbrowser, time, base64, argparse, os, traceback, textwrap
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
from utils.run_metadata import (
    DEFAULT_CONFIG_VERSION,
    append_experiment_record,
    build_run_metadata,
    complete_run_metadata,
)
from utils.dashboard_truth import (
    build_debug_footer_context,
    extract_dashboard_truth,
    validate_dashboard_payload,
    write_dashboard_consistency_report,
)
from utils.debug_audit import build_repo_debug_audit, render_repo_debug_audit_markdown
from utils.telegram_notifier import (
    create_progress_session,
    notify_failure,
    notify_success,
    result_delay_seconds as telegram_result_delay_seconds,
    send_chat_action,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable or []
        def __iter__(self): return iter(self._it)
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def set_description(self, s): pass
        def close(self): pass
        @staticmethod
        def write(s): print(s)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TELEGRAM_PROGRESS_SESSION = None
TELEGRAM_RUN_NAME = "run_all"
TELEGRAM_FAILURE_LOG_PATH = None


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


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stock ML Analyzer  V.0.7.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py AAPL
  python run_all.py --portfolio watchlist.txt
  python run_all.py --portfolio watchlist.txt --parallel 2
  python run_all.py --debug benchmark
  python run_all.py --debug diagnostic RKLB
  python run_all.py --debug audit
        """
    )
    parser.add_argument('ticker', nargs='?', default=None,
                        help='Ticker symbol for single-stock mode')
    parser.add_argument('--portfolio', '-p', metavar='FILE',
                        help='Path to watchlist .txt file (one ticker per line)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Workers for parallel portfolio scan (default: 1)')
    parser.add_argument('--gpu-jobs', type=int, default=1,
                        help='Maximum concurrent GPU-bound analyzer jobs for benchmark mode (default: 1)')
    parser.add_argument('--panel', action='store_true',
                        help='Use shared cross-sectional panel ML for watchlist runs')
    parser.add_argument('--optimizer', default=DEFAULT_OPTIMIZER,
                        choices=['mean_variance', 'risk_parity', 'black_litterman', 'cvar', 'heuristic'],
                        help='Portfolio optimizer for watchlist runs')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run soft benchmark quality gate after the watchlist run')
    parser.add_argument('--benchmark-watchlist', metavar='FILE',
                        help='Watchlist file to use for benchmark evaluation')
    parser.add_argument('--debug', choices=['benchmark', 'diagnostic', 'audit'],
                        help='Developer workflow: benchmark, single-ticker diagnostic, or repo audit mode')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _validate_ticker(ticker: str):
    if HAS_YF:
        try:
            info  = yf.Ticker(ticker).fast_info
            price = getattr(info, 'last_price', None)
            if price is None:
                raise ValueError("No price returned")
            qtype = getattr(info, 'quote_type', '')
            print(f"\n  OK: {ticker}  |  last price = ${price:.2f}  |  type = {qtype}")
        except Exception as e:
            print(f"\n  WARNING: Could not validate {ticker}: {e}")
            ans = input("  Continue anyway? (y/n): ").strip().lower()
            if ans != 'y':
                print("  Exiting."); sys.exit(0)


def img_to_base64(path: Path) -> str:
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('ascii')
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


def _fmt_time(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"


def _progress_session(run_name: str):
    global TELEGRAM_PROGRESS_SESSION, TELEGRAM_RUN_NAME
    TELEGRAM_RUN_NAME = str(run_name or "run_all")
    if TELEGRAM_PROGRESS_SESSION is None:
        TELEGRAM_PROGRESS_SESSION = create_progress_session(TELEGRAM_RUN_NAME)
    return TELEGRAM_PROGRESS_SESSION


def _progress_text(run_name: str,
                   phase_index: int,
                   phase_total: int,
                   phase_label: str,
                   *,
                   completed: int | None = None,
                   total: int | None = None,
                   ticker: str | None = None,
                   analyzer_stage: str | None = None,
                   status: str | None = None) -> str:
    lines = [str(run_name or "Quant analyzer")]
    lines.append(f"Phase {int(phase_index)}/{int(phase_total)}: {phase_label}")
    if total is not None and total > 0:
        ticker_line = f"Progress: {int(completed or 0)}/{int(total)}"
        if ticker:
            ticker_line += f" | Current: {ticker}"
        lines.append(ticker_line)
    elif ticker:
        lines.append(f"Ticker: {ticker}")
    if analyzer_stage:
        lines.append(f"Analyzer stage: {analyzer_stage}")
    if status:
        lines.append(f"Status: {status}")
    return "\n".join(lines)


def _progress_start_or_update(run_name: str,
                              phase_index: int,
                              phase_total: int,
                              phase_label: str,
                              **kwargs):
    session = _progress_session(run_name)
    text = _progress_text(run_name, phase_index, phase_total, phase_label, **kwargs)
    if session is None or not getattr(session, "enabled", False):
        return session
    if getattr(session, "message_id", None) is None:
        session.start(text)
    else:
        session.update(text)
    return session


def _progress_child_env(run_name: str,
                        phase_index: int,
                        phase_total: int,
                        phase_label: str,
                        ticker: str,
                        ticker_index: int | None = None,
                        ticker_total: int | None = None) -> dict:
    session = _progress_session(run_name)
    env = {
        "TELEGRAM_PROGRESS_ENABLED": os.environ.get("TELEGRAM_PROGRESS_ENABLED", ""),
        "TELEGRAM_PROGRESS_RUN_NAME": str(run_name or "Quant analyzer"),
        "TELEGRAM_PROGRESS_PHASE_INDEX": str(int(phase_index)),
        "TELEGRAM_PROGRESS_PHASE_TOTAL": str(int(phase_total)),
        "TELEGRAM_PROGRESS_PHASE_LABEL": str(phase_label),
        "TELEGRAM_PROGRESS_TICKER": str(ticker or ""),
    }
    if ticker_index is not None:
        env["TELEGRAM_PROGRESS_TICKER_INDEX"] = str(int(ticker_index))
    if ticker_total is not None:
        env["TELEGRAM_PROGRESS_TICKER_TOTAL"] = str(int(ticker_total))
    if session is not None and getattr(session, "message_id", None) is not None:
        env["TELEGRAM_PROGRESS_MESSAGE_ID"] = str(session.message_id)
    return env


def _child_env(env: dict | None = None) -> dict:
    out = dict(os.environ)
    if env:
        out.update(env)
    out["TELEGRAM_SUPPRESS_CHILD"] = "1"
    return out


def _tail_text(path: Path, max_chars: int = 3200) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except Exception:
        return ""


def _write_error_log(path: Path, err_text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(err_text or ""), encoding="utf-8")
    return path


def _notify_run_success(mode: str,
                        ticker: str | None = None,
                        run_type: str | None = None,
                        progress_session=None) -> bool:
    out_dir = Path("reports")
    photo_paths = []
    caption = None
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if mode == "single" and ticker:
        base = out_dir / ticker
        for candidate in [
            base / f"{ticker}_analysis.png",
            base / f"{ticker}_selection_diagnostics.png",
            base / f"{ticker}_dl_models.png",
            base / f"{ticker}_montecarlo.png",
            base / f"{ticker}_montecarlo_volatility.png",
            base / f"{ticker}_montecarlo_diagnostics.png",
            base / f"{ticker}_dcf_surface_3d.png",
        ]:
            if candidate.exists():
                photo_paths.append(candidate)
        signal_path = base / f"{ticker}_signal.json"
        if signal_path.exists():
            try:
                signal_data = json.loads(signal_path.read_text(encoding="utf-8"))
                sig = (signal_data.get("signal", {}) or {})
                caption = (
                    f"{ticker} | {timestamp}\n"
                    f"{sig.get('model_used', 'model')} | {sig.get('signal', 'N/A')} | "
                    f"{float(sig.get('confidence', 0.0) or 0.0):.2f}"
                )
            except Exception:
                caption = f"{ticker} | {timestamp}"
        else:
            caption = f"{ticker} | {timestamp}"
    elif mode == "benchmark":
        photo_paths = [path for path in [
            out_dir / "benchmark_quality.png",
            out_dir / "portfolio_optimizer.png",
        ] if path.exists()]
        caption = f"Benchmark | {timestamp}"
    elif mode == "panel":
        photo_paths = [path for path in [
            out_dir / "panel_summary.png",
            out_dir / "portfolio_optimizer.png",
            out_dir / "benchmark_quality.png",
        ] if path.exists()]
        caption = f"Panel run | {timestamp}"
    else:
        photo_paths = [path for path in [
            out_dir / "portfolio_optimizer.png",
            out_dir / "benchmark_quality.png",
        ] if path.exists()]
        caption = f"Portfolio run | {timestamp}"
    if not photo_paths:
        return False
    if progress_session is not None and getattr(progress_session, "enabled", False):
        progress_session.mark_finalizing("Finalizing result...")
        send_chat_action("upload_photo")
        time.sleep(telegram_result_delay_seconds())
    return notify_success([str(path) for path in photo_paths], caption=caption)


def _notify_run_failure(run_name: str,
                        log_path: Path | None = None,
                        err_text: str | None = None,
                        progress_session=None) -> bool:
    tail = str(err_text or "").strip()
    if not tail and log_path is not None:
        tail = _tail_text(log_path)
    if not tail:
        tail = f"{run_name} failed without a captured traceback."
    if progress_session is not None and getattr(progress_session, "enabled", False):
        progress_session.mark_failed("Failed")
    return notify_failure(run_name, tail, log_path=str(log_path) if log_path else None)


def _has_cuda_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_benchmark_workers(n_workers: int, gpu_jobs: int = 1) -> tuple[int, str | None]:
    requested = max(1, int(n_workers or 1))
    gpu_limit = max(1, int(gpu_jobs or 1))
    if not _has_cuda_gpu():
        return requested, None
    effective = min(requested, gpu_limit)
    if effective < requested:
        reason = ("CUDA detected - benchmark analyzer jobs are serialized to "
                  f"{effective} GPU worker(s) for stability")
        return effective, reason
    return effective, None


def _analyzer_runtime_env(force_diagnostics: bool = False) -> dict | None:
    if not force_diagnostics:
        return None
    env = dict(os.environ)
    env["ANALYZER_FORCE_DIAGNOSTICS"] = "1"
    env["ANALYZER_RUN_CONTEXT"] = "benchmark"
    return env


def run_module(label: str, script: str, ticker: str,
               master_bar=None, log_path: Path = None,
               silent: bool = False,
               env: dict | None = None) -> tuple:
    """Run a single analysis module. Returns (elapsed_sec, success_bool)."""
    sep = "─" * 54
    msg = f"\n{sep}\n  {label}  [{ticker}]\n{sep}"
    if master_bar:   tqdm.write(msg)
    elif not silent: print(msg)

    t0 = time.time()
    child_env = _child_env(env)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n{'='*54}\n  {label}  [{ticker}]\n{'='*54}\n")
            log_f.flush()
            proc = subprocess.Popen(
                [sys.executable, script, ticker],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=child_env,
            )
            for line in proc.stdout:
                if not silent:
                    sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
                    sys.stdout.flush()
                log_f.write(line)
            proc.wait()
            result = proc
    else:
        result = subprocess.run(
            [sys.executable, script, ticker],
            capture_output=silent,
            env=child_env,
        )

    elapsed_sec = time.time() - t0
    success     = (result.returncode == 0)
    status = f"  Done in {_fmt_time(elapsed_sec)}" if success else \
             f"  WARNING: {script} exited with code {result.returncode}"
    if master_bar:   tqdm.write(status)
    elif not silent: print(status)
    return elapsed_sec, success


def _format_module_timings(result: dict | None) -> str:
    if not isinstance(result, dict):
        return ""
    parts = []
    for name in ("ML", "FUND", "MC"):
        info = result.get(name, {}) or {}
        if "elapsed" in info:
            parts.append(f"{name}={_fmt_time(float(info.get('elapsed', 0.0) or 0.0))}")
    return ", ".join(parts)


def _run_timed_ticker_task(ticker: str, worker, *args, **kwargs) -> dict:
    started_dt = datetime.datetime.now()
    t0 = time.time()
    try:
        result = worker(ticker, *args, **kwargs)
        return {
            "ticker": ticker,
            "started_at": started_dt.isoformat(),
            "started_label": started_dt.strftime("%H:%M:%S"),
            "elapsed": time.time() - t0,
            "result": result,
            "error": None,
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "started_at": started_dt.isoformat(),
            "started_label": started_dt.strftime("%H:%M:%S"),
            "elapsed": time.time() - t0,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _new_batch_progress(total: int, label: str) -> dict:
    return {
        "label": label,
        "total": max(0, int(total or 0)),
        "started_at": time.time(),
        "completed": 0,
        "tickers": {},
    }


def _batch_elapsed(progress: dict) -> float:
    return max(0.0, time.time() - float(progress.get("started_at", time.time())))


def _record_batch_completion(progress: dict,
                             task: dict,
                             success: bool,
                             result: dict | None,
                             bar=None):
    ticker = task.get("ticker", "?")
    progress["completed"] += 1
    ticker_elapsed = float(task.get("elapsed", 0.0) or 0.0)
    progress["tickers"][ticker] = {
        "elapsed_sec": ticker_elapsed,
        "started_at": task.get("started_at"),
        "started_label": task.get("started_label"),
        "success": bool(success),
        "module_timings": {
            name: float((info or {}).get("elapsed", 0.0) or 0.0)
            for name, info in (result or {}).items()
            if isinstance(info, dict) and "elapsed" in info
        },
        "error": task.get("error"),
    }
    total_elapsed = _batch_elapsed(progress)
    module_str = _format_module_timings(result)
    line = (
        f"  [{progress['completed']}/{progress['total']}] "
        f"{'✓' if success else '✗'} {ticker:<6} "
        f"start={task.get('started_label', '--:--:--')} "
        f"ticker={_fmt_time(ticker_elapsed)} total={_fmt_time(total_elapsed)}"
    )
    if module_str:
        line += f"  [{module_str}]"
    if task.get("error"):
        line += f"  error={task['error']}"
    if bar:
        bar.set_postfix(
            last=ticker,
            ok='✓' if success else '✗',
            total=_fmt_time(total_elapsed),
            last_t=_fmt_time(ticker_elapsed),
            refresh=True,
        )
        tqdm.write(line)
    else:
        print(line)
    run_name = {
        "portfolio": "Portfolio run",
        "panel-support": "Panel run",
        "benchmark": "Benchmark run",
    }.get(str(progress.get("label", "") or ""), "Portfolio run")
    _progress_start_or_update(
        run_name=run_name,
        phase_index=3,
        phase_total=3,
        phase_label="Monte Carlo",
        completed=progress["completed"],
        total=progress["total"],
        ticker=ticker,
        status="Completed" if success else "Failed",
    )


def _batch_elapsed_summary(progress: dict) -> dict:
    ticker_stats = progress.get("tickers", {}) or {}
    elapsed_values = [float(v.get("elapsed_sec", 0.0) or 0.0) for v in ticker_stats.values()]
    total_elapsed = _batch_elapsed(progress)
    fastest = slowest = None
    if ticker_stats:
        fastest_ticker, fastest_info = min(
            ticker_stats.items(), key=lambda item: float(item[1].get("elapsed_sec", 0.0) or 0.0))
        slowest_ticker, slowest_info = max(
            ticker_stats.items(), key=lambda item: float(item[1].get("elapsed_sec", 0.0) or 0.0))
        fastest = {"ticker": fastest_ticker, "elapsed_sec": round(float(fastest_info.get("elapsed_sec", 0.0) or 0.0), 3)}
        slowest = {"ticker": slowest_ticker, "elapsed_sec": round(float(slowest_info.get("elapsed_sec", 0.0) or 0.0), 3)}
    return {
        "label": progress.get("label", "batch"),
        "total_tickers": int(progress.get("total", 0)),
        "completed_tickers": int(progress.get("completed", 0)),
        "total_elapsed_sec": round(total_elapsed, 3),
        "total_elapsed_human": _fmt_time(total_elapsed),
        "average_ticker_sec": round(float(np.mean(elapsed_values)), 3) if elapsed_values else None,
        "average_ticker_human": _fmt_time(float(np.mean(elapsed_values))) if elapsed_values else None,
        "fastest_ticker": fastest,
        "slowest_ticker": slowest,
        "failed_tickers": int(sum(1 for info in ticker_stats.values() if not info.get("success"))),
    }


def _print_batch_summary(progress: dict):
    summary = _batch_elapsed_summary(progress)
    print()
    print(f"  Batch summary ({summary['label']}):")
    print(f"    Completed     : {summary['completed_tickers']}/{summary['total_tickers']}")
    print(f"    Total elapsed : {summary['total_elapsed_human']}")
    if summary.get("average_ticker_human"):
        print(f"    Avg / ticker  : {summary['average_ticker_human']}")
    if summary.get("fastest_ticker"):
        print(f"    Fastest       : {summary['fastest_ticker']['ticker']}  {_fmt_time(summary['fastest_ticker']['elapsed_sec'])}")
    if summary.get("slowest_ticker"):
        print(f"    Slowest       : {summary['slowest_ticker']['ticker']}  {_fmt_time(summary['slowest_ticker']['elapsed_sec'])}")
    print(f"    Failed count  : {summary['failed_tickers']}")


def _print_timing(module_times: dict, module_success: dict):
    print(); print("  " + "─"*44); print("  Module timing:")
    for name, sec in module_times.items():
        bar_w = int(sec / max(module_times.values()) * 20) if module_times else 0
        ok    = "✓" if module_success.get(name) else "✗"
        print(f"    {ok} {name:<8}  {_fmt_time(sec):<12}  {'#'*bar_w}")
    print(f"  {'─'*44}\n")


def _open_browser(path: Path):
    try:
        webbrowser.open(f"file://{path.resolve()}")
        print("  Opening dashboard in browser...")
    except Exception:
        print(f"  --> Open manually: {path.resolve()}")


def _portfolio_tools():
    from utils.portfolio_tools import (
        build_quality_gate,
        compute_portfolio_weights,
        load_report_bundle,
    )
    return build_quality_gate, compute_portfolio_weights, load_report_bundle


def _save_chart(fig, path: Path):
    if not HAS_MPL:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close(fig)
    return path


def _build_optimizer_chart(portfolio_summary: dict, stock_data: dict) -> Path | None:
    if not HAS_MPL:
        return None
    optimizer = portfolio_summary.get('optimizer', {}) or {}
    weights = optimizer.get('weights', {}) or {}
    expected = optimizer.get('expected_returns', {}) or {}
    tickers = list(weights.keys())
    if not tickers:
        return None

    colors = []
    for ticker in tickers:
        if ticker == 'CASH':
            colors.append('#8B949E')
            continue
        sig = ((stock_data.get(ticker, {}).get('signal_data', {}).get('signal', {}) or {}).get('signal', 'HOLD'))
        colors.append({'BUY': '#00C853', 'HOLD': '#FFD600', 'SELL': '#D50000'}.get(sig, '#58A6FF'))

    weight_vals = [weights.get(t, 0.0) * 100 for t in tickers]
    expected_vals = [
        float((expected.get(t, {}) or {}).get('allocatable_expected_return',
              (expected.get(t, {}) or {}).get('expected_return', 0.0))) * 100
        for t in tickers
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0D1117')
    for ax in axes:
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
        ax.grid(color='#30363D', alpha=0.5, linewidth=0.5)

    axes[0].barh(tickers, weight_vals, color=colors, alpha=0.85)
    axes[0].set_title('Portfolio Weights', color='#E6EDF3', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Weight (%)', color='#E6EDF3')

    axes[1].barh(tickers, expected_vals, color=colors, alpha=0.85)
    axes[1].axvline(0, color='#30363D', linewidth=0.8)
    axes[1].set_title('Expected Return Inputs', color='#E6EDF3', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Expected Return (%)', color='#E6EDF3')

    fig.suptitle(
        f"Portfolio Optimizer - {optimizer.get('method_used', DEFAULT_OPTIMIZER).replace('_', ' ').title()}",
        color='#E6EDF3',
        fontsize=13,
        fontweight='bold',
    )
    fig.text(
        0.5, 0.01,
        f"Requested={optimizer.get('requested_method', DEFAULT_OPTIMIZER)}  |  "
        f"Covariance={optimizer.get('covariance_source', 'N/A')}  |  "
        f"Fallback={optimizer.get('fallback_reason') or 'none'}  |  "
        f"Allocation={optimizer.get('allocation_status', 'N/A')}  |  "
        f"Cash={float(optimizer.get('cash_weight', 0.0) or 0.0):.0%}",
        ha='center', color='#8B949E', fontsize=8,
    )
    return _save_chart(fig, Path('reports') / 'portfolio_optimizer.png')


def _build_benchmark_chart(payload: dict) -> Path | None:
    if not HAS_MPL:
        return None
    gate = payload.get('quality_gate', {}) or {}
    checks = gate.get('checks', []) or []
    metrics = gate.get('metrics', {}) or {}
    coverage = gate.get('coverage', {}) or {}
    reasons = gate.get('primary_failure_reasons', []) or []
    robust_top = (gate.get('robust_score_leaderboard', []) or [])[:3]
    calibration_top = (gate.get('calibration_leaderboard', []) or [])[:2]
    conformal_top = (gate.get('conformal_sharpness_leaderboard', []) or [])[:2]
    router_freq = gate.get('router_family_selection_frequency', {}) or {}
    routing_uplift = gate.get('routing_uplift', {}) or {}
    actionable_summary = gate.get('actionable_summary', {}) or {}
    router_status_frequency = gate.get('router_status_frequency', {}) or {}
    mc_summary = gate.get('mc_reliability_summary', {}) or {}
    conformal_summary = gate.get('conformal_usability_summary', {}) or {}
    selection_summary = gate.get('selection_summary', {}) or {}
    if not checks:
        return None

    labels = [textwrap.fill(str(c.get('name', 'check') or 'check'), width=22) for c in checks]
    values = [1 if c.get('passed') else 0 for c in checks]
    colors = ['#00C853' if v else '#D50000' for v in values]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18.5, 8.2),
        constrained_layout=True,
        gridspec_kw={'width_ratios': [1.25, 1.0]},
    )
    fig.patch.set_facecolor('#0D1117')
    for ax in axes:
        ax.set_facecolor('#161B22')
        for spine in ax.spines.values():
            spine.set_color('#30363D')

    ypos = list(range(len(labels)))
    axes[0].barh(ypos, values, color=colors, alpha=0.9)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels, color='#E6EDF3')
    axes[0].set_xlim(0, 1)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Fail', 'Pass'], color='#8B949E')
    axes[0].tick_params(axis='y', colors='#E6EDF3', labelsize=8)
    axes[0].grid(axis='x', color='#30363D', alpha=0.55, linewidth=0.5)
    axes[0].set_title('Benchmark Quality Checks', color='#E6EDF3', fontsize=12, fontweight='bold')

    axes[1].axis('off')
    lines = [
        f"Status: {gate.get('status', 'N/A')}",
        f"Passed: {gate.get('passed_checks', 0)}/{gate.get('total_checks', 0)}",
        f"Success rate: {metrics.get('success_rate', 0):.0%}",
        f"Actionable rate: {float(actionable_summary.get('actionable_rate', 0.0) or 0.0):.0%}",
        f"Signal coverage: {(coverage.get('signal_json', {}) or {}).get('count', 0)}/{metrics.get('n_tickers', 0)}",
        f"Diag coverage: {(coverage.get('diagnostics_json', {}) or {}).get('count', 0)}/{metrics.get('n_tickers', 0)}",
        f"Median WF Sharpe: {metrics.get('median_wf_sharpe', 'N/A')}",
        f"Median CPCV p5: {metrics.get('median_cpcv_p5', 'N/A')}",
        f"Seed stable rate: {metrics.get('seed_stable_rate', 0):.0%}",
        f"Mean reliability: {metrics.get('reliability_score_mean', 'N/A')}",
        f"Median ECE: {metrics.get('median_ece', 'N/A')}",
        f"Median BUY recall: {metrics.get('median_buy_recall', 'N/A')}",
        f"Reference-only rate: {float(selection_summary.get('reference_only_rate', 0.0) or 0.0):.0%}",
    ]
    if reasons:
        lines.append(f"Main issues: {', '.join(reasons[:3])}")
    if robust_top:
        lines.append("Top robust: " + ", ".join(
            f"{row.get('ticker')}:{row.get('robust_score', 0):+.2f}" for row in robust_top
        ))
    if calibration_top:
        lines.append("Top calibrated: " + ", ".join(
            f"{row.get('ticker')}:{row.get('post_calibration_ece', 0):.3f}" for row in calibration_top
        ))
    if conformal_top:
        lines.append("Top conformal: " + ", ".join(
            f"{row.get('ticker')}:{row.get('sharpness', 0):.2f}" for row in conformal_top
        ))
    if selection_summary:
        eligible_wins = selection_summary.get('family_level_eligible_win_rate', {}) or {}
        if eligible_wins:
            lines.append("Eligible wins: " + ", ".join(
                f"{fam}={ratio:.0%}" for fam, ratio in eligible_wins.items()
            ))
        rejection_counts = selection_summary.get('directional_rejection_counts', {}) or {}
        if rejection_counts:
            top_rejections = list(rejection_counts.items())[:3]
            lines.append("Directional rejects: " + ", ".join(
                f"{name}={count}" for name, count in top_rejections
            ))
    if router_freq:
        lines.append("Router freq: " + ", ".join(
            f"{fam}={ratio:.0%}" for fam, ratio in router_freq.items()
        ))
    if router_status_frequency:
        lines.append("Router status: " + ", ".join(
            f"{name}={count}" for name, count in router_status_frequency.items()
        ))
    if routing_uplift:
        lines.append(
            "Router uplift: "
            f"ΔWF={routing_uplift.get('median_eval_sharpe_delta', 'N/A')}  "
            f"ΔCPCV={routing_uplift.get('median_cpcv_p5_delta', 'N/A')}"
        )
    if mc_summary:
        status_frequency = mc_summary.get('status_frequency', {}) or {}
        if status_frequency:
            lines.append("MC status: " + ", ".join(
                f"{name}={count}" for name, count in status_frequency.items()
            ))
        lines.append(f"MC fallback-vol: {mc_summary.get('fallback_vol_count', 0)}")
    if conformal_summary:
        lines.append(
            "Conformal blocked: "
            f"{int(conformal_summary.get('conformal_blocked_ticker_count', 0))}/"
            f"{metrics.get('n_tickers', 0)} "
            f"({float(conformal_summary.get('conformal_block_rate', 0.0) or 0.0):.0%})"
        )
    if conformal_summary:
        usable = int(conformal_summary.get('usable_count', 0) or 0)
        unusable = int(conformal_summary.get('unusable_count', 0) or 0)
        lines.append(f"Conformal usable: {usable}/{usable + unusable}")
    axes[1].text(
        0.01, 0.99, "\n".join(lines),
        va='top', ha='left', color='#E6EDF3', fontsize=10, family='monospace'
    )
    fig.suptitle('Benchmark Quality Gate', color='#E6EDF3', fontsize=13, fontweight='bold')
    return _save_chart(fig, Path('reports') / 'benchmark_quality.png')


def _build_panel_chart(summary: dict) -> Path | None:
    if not HAS_MPL:
        return None
    model_accuracy = summary.get('model_accuracy', {}) or {}
    tickers = summary.get('tickers', {}) or {}
    if not model_accuracy and not tickers:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True)
    fig.patch.set_facecolor('#0D1117')
    for ax in axes:
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
        ax.grid(color='#30363D', alpha=0.5, linewidth=0.5)

    if model_accuracy:
        names = list(model_accuracy.keys())
        wrapped_names = [textwrap.fill(str(name), width=18) for name in names]
        vals = [float(model_accuracy[k]) * 100 for k in names]
        axes[0].barh(wrapped_names, vals, color='#58A6FF', alpha=0.85)
        axes[0].set_title('Shared Model Accuracy', color='#E6EDF3', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Accuracy (%)', color='#E6EDF3')
        axes[0].tick_params(axis='y', labelsize=8)
    else:
        axes[0].axis('off')

    if tickers:
        names = list(tickers.keys())
        confs = [float((tickers[k] or {}).get('confidence', 0.0)) * 100 for k in names]
        colors = []
        for ticker in names:
            sig = (tickers.get(ticker) or {}).get('signal', 'HOLD')
            colors.append({'BUY': '#00C853', 'HOLD': '#FFD600', 'SELL': '#D50000'}.get(sig, '#58A6FF'))
        axes[1].bar(names, confs, color=colors, alpha=0.85)
        axes[1].set_title('Panel Latest Confidence by Ticker', color='#E6EDF3', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Confidence (%)', color='#E6EDF3')
        axes[1].tick_params(axis='x', rotation=30, labelsize=8)
    else:
        axes[1].axis('off')

    panel_context = summary.get('panel_context', {}) or {}
    fig.suptitle(
        f"Panel Summary - {panel_context.get('shared_model', 'Shared Model')}",
        color='#E6EDF3',
        fontsize=13,
        fontweight='bold',
    )
    return _save_chart(fig, Path('reports') / 'panel_summary.png')


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _default_watchlist() -> str:
    return "watchlist_example.txt"


def _default_benchmark_watchlist() -> str:
    preferred = Path("benchmark_universe.txt")
    return str(preferred) if preferred.exists() else _default_watchlist()


def _build_portfolio_summary(tickers: list,
                             stock_data: dict,
                             optimizer: str,
                             watchlist_path: str,
                             quality_gate: dict | None = None,
                             mode: str = "standard") -> dict:
    _, compute_portfolio_weights, _ = _portfolio_tools()
    optimizer_summary = compute_portfolio_weights(
        tickers,
        stock_data,
        optimizer=optimizer,
    )
    run_metadata = complete_run_metadata(build_run_metadata(
        mode=f"portfolio_{mode}",
        config={
            "optimizer": optimizer,
            "benchmark_thresholds": DEFAULT_BENCHMARK_THRESHOLDS,
        },
        config_version=DEFAULT_CONFIG_VERSION,
        optimizer=optimizer_summary.get("method_used", optimizer),
        watchlist_path=watchlist_path,
        universe=tickers,
        extra={"watchlist_path": watchlist_path},
    ), status="OK")
    summary = {
        "generated": run_metadata["completed_at"],
        "run_metadata": run_metadata,
        "watchlist_path": watchlist_path,
        "tickers": tickers,
        "optimizer": optimizer_summary,
        "quality_gate": quality_gate or {},
        "actionable_universe_size": int(len(optimizer_summary.get("allocatable_tickers", []) or [])),
        "non_actionable_universe_size": int(len(optimizer_summary.get("excluded_tickers", []) or [])),
        "cash_weight": float(optimizer_summary.get("cash_weight", 0.0) or 0.0),
        "allocation_status": optimizer_summary.get("allocation_status") or "unknown",
    }
    _write_json(Path("reports") / "portfolio_summary.json", summary)
    _build_optimizer_chart(summary, stock_data)
    append_experiment_record(Path("reports"), run_metadata, status="OK", summary={
        "watchlist_path": watchlist_path,
        "optimizer": optimizer_summary.get("method_used", optimizer),
        "quality_gate": (quality_gate or {}).get("status"),
    })
    return summary


def _evaluate_quality_gate(watchlist_path: str,
                           stock_data: dict | None = None,
                           elapsed_summary: dict | None = None) -> dict:
    build_quality_gate, _, _ = _portfolio_tools()
    tickers = load_watchlist(watchlist_path)
    stock_data = stock_data or {t: _load_stock_json(t) for t in tickers}
    quality_gate = build_quality_gate(stock_data, DEFAULT_BENCHMARK_THRESHOLDS)
    metadata = complete_run_metadata(build_run_metadata(
        mode="benchmark_quality",
        config={"benchmark_thresholds": DEFAULT_BENCHMARK_THRESHOLDS},
        config_version=DEFAULT_CONFIG_VERSION,
        watchlist_path=watchlist_path,
        universe=tickers,
        extra={"watchlist_path": watchlist_path},
    ), status=quality_gate.get("status", "UNKNOWN"))
    payload = {
        "generated": metadata["completed_at"],
        "run_metadata": metadata,
        "watchlist_path": watchlist_path,
        "tickers": tickers,
        "quality_gate": quality_gate,
        "coverage": quality_gate.get("coverage", {}),
        "primary_failure_reasons": quality_gate.get("primary_failure_reasons", []),
        "per_ticker": quality_gate.get("per_ticker", {}),
        "elapsed_summary": elapsed_summary or {},
    }
    _write_json(Path("reports") / "benchmark_quality.json", payload)
    _build_benchmark_chart(payload)
    append_experiment_record(
        Path("reports"),
        metadata,
        status=quality_gate.get("status", "UNKNOWN"),
        summary={
            **quality_gate,
            "elapsed_summary": elapsed_summary or {},
        },
    )
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-STOCK MODE
# ─────────────────────────────────────────────────────────────────────────────
def run_single(ticker: str):
    global TELEGRAM_FAILURE_LOG_PATH
    modules = [
        ("ML",   "[ML]    ML Analysis + BiLSTM + TFT + Transformer",  "analyzer.py"),
        ("FUND", "[FUND]  Fundamental Analysis",                        "fundamental.py"),
        ("MC",   "[MC]    Monte Carlo Simulation",                      "monte_carlo.py"),
    ]
    print(f"\n  Starting full analysis for: {ticker}")
    print(f"  All outputs --> reports/{ticker}/")
    print(f"  {len(modules)} modules queued\n")

    master_bar = tqdm(
        modules, desc="  Overall", unit="module", ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        file=sys.stdout, leave=True, position=0,
    )
    module_times = {}; module_success = {}
    log_path = Path("reports") / ticker / f"{ticker}_master.log"
    TELEGRAM_FAILURE_LOG_PATH = log_path
    progress_session = _progress_start_or_update(
        run_name=f"Single run: {ticker}",
        phase_index=1,
        phase_total=3,
        phase_label="ML analyzer",
        ticker=ticker,
        status="Starting",
    )

    for idx, (short, label, script) in enumerate(master_bar, 1):
        master_bar.set_postfix(current=short, refresh=True)
        phase_label = {"ML": "ML analyzer", "FUND": "Fundamentals", "MC": "Monte Carlo"}.get(short, short)
        child_env = None
        if short == "ML":
            child_env = _progress_child_env(f"Single run: {ticker}", idx, 3, phase_label, ticker)
        _progress_start_or_update(
            run_name=f"Single run: {ticker}",
            phase_index=idx,
            phase_total=3,
            phase_label=phase_label,
            ticker=ticker,
            analyzer_stage="Queued" if short == "ML" else None,
            status="Running",
        )
        elapsed_sec, success = run_module(label, script, ticker, master_bar, log_path, env=child_env)
        module_times[short]   = elapsed_sec
        module_success[short] = success

    master_bar.set_postfix(status="DONE", refresh=True)
    master_bar.close()
    _print_timing(module_times, module_success)

    path = build_single_dashboard(ticker, module_success)
    if all(module_success.values()):
        _notify_run_success("single", ticker=ticker, progress_session=progress_session)
    else:
        _notify_run_failure(f"single-{ticker}", log_path=log_path, progress_session=progress_session)
    _open_browser(path)
    print()
    print("=" * 54)
    print(f"  Done!  Reports in: reports/{ticker}/")
    print("=" * 54)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SCANNER (NEW 19)
# ─────────────────────────────────────────────────────────────────────────────
def load_watchlist(path: str) -> list:
    """Read tickers from a text file (one per line; # = comment)."""
    p = Path(path)
    if not p.exists():
        print(f"\n  ERROR: Watchlist file not found: {path}")
        print("  See watchlist_example.txt for format.")
        sys.exit(1)
    tickers = []
    for line in p.read_text(encoding='utf-8').splitlines():
        t = line.split('#')[0].strip().upper()
        if t:
            tickers.append(t)
    if not tickers:
        print(f"\n  ERROR: No tickers found in {path}"); sys.exit(1)
    return list(dict.fromkeys(tickers))   # deduplicate, preserve order


def _run_ticker_batch(ticker: str,
                      force_ml_diagnostics: bool = False,
                      run_name: str = "Portfolio run",
                      ticker_index: int | None = None,
                      ticker_total: int | None = None) -> dict:
    """Run all 3 modules silently for one ticker. Returns per-module timing/success."""
    modules  = [("ML","analyzer.py"), ("FUND","fundamental.py"), ("MC","monte_carlo.py")]
    log_path = Path("reports") / ticker / f"{ticker}_master.log"
    results  = {}
    for short, script in modules:
        env = _analyzer_runtime_env(force_ml_diagnostics) if short == "ML" else {}
        if short == "ML":
            env = {
                **(env or {}),
                **_progress_child_env(run_name, 1, 3, "ML analyzer", ticker, ticker_index, ticker_total),
            }
        elif short == "FUND":
            _progress_start_or_update(run_name, 2, 3, "Fundamentals", completed=(ticker_index or 0) - 1, total=ticker_total, ticker=ticker, status="Running")
        elif short == "MC":
            _progress_start_or_update(run_name, 3, 3, "Monte Carlo", completed=(ticker_index or 0) - 1, total=ticker_total, ticker=ticker, status="Running")
        elapsed, success = run_module(f"[{short}] {ticker}", script,
                                      ticker, log_path=log_path, silent=True,
                                      env=env)
        results[short] = {'elapsed': elapsed, 'success': success}
    return results


def _run_ticker_analyzer_only(ticker: str,
                              force_diagnostics: bool = False,
                              run_name: str = "Benchmark run",
                              ticker_index: int | None = None,
                              ticker_total: int | None = None) -> dict:
    log_path = Path("reports") / ticker / f"{ticker}_master.log"
    env = {
        **(_analyzer_runtime_env(force_diagnostics) or {}),
        **_progress_child_env(run_name, 1, 3, "ML analyzer", ticker, ticker_index, ticker_total),
    }
    elapsed, success = run_module(f"[ML] {ticker}", "analyzer.py",
                                  ticker, log_path=log_path, silent=True,
                                  env=env)
    return {"ML": {"elapsed": elapsed, "success": success}}


def _run_ticker_support_batch(ticker: str,
                              run_name: str = "Panel run",
                              ticker_index: int | None = None,
                              ticker_total: int | None = None) -> dict:
    modules = [("FUND", "fundamental.py"), ("MC", "monte_carlo.py")]
    log_path = Path("reports") / ticker / f"{ticker}_master.log"
    results = {"ML": {"elapsed": 0.0, "success": True}}
    for short, script in modules:
        phase_index = 2 if short == "FUND" else 3
        phase_label = "Fundamentals" if short == "FUND" else "Monte Carlo"
        _progress_start_or_update(run_name, phase_index, 3, phase_label, completed=(ticker_index or 0) - 1, total=ticker_total, ticker=ticker, status="Running")
        elapsed, success = run_module(f"[{short}] {ticker}", script,
                                      ticker, log_path=log_path, silent=True)
        results[short] = {"elapsed": elapsed, "success": success}
    return results


def run_portfolio(watchlist_path: str,
                  n_workers: int = 1,
                  run_type: str = "standard",
                  optimizer: str = DEFAULT_OPTIMIZER,
                  benchmark: bool = False,
                  benchmark_watchlist: str | None = None):
    """Batch-process watchlist in standard or panel mode."""
    global TELEGRAM_FAILURE_LOG_PATH
    tickers = load_watchlist(watchlist_path)
    run_name = "Panel run" if run_type == "panel" else "Portfolio run"
    run_start = time.time()
    print()
    print("=" * 60)
    print(f"  Portfolio Scanner  v13.0  —  {len(tickers)} tickers")
    print("=" * 60)
    print(f"  Watchlist : {watchlist_path}")
    print(f"  Run type  : {run_type}")
    print(f"  Optimizer : {optimizer}")
    print(f"  Workers   : {n_workers}")
    print(f"  Output    : reports/portfolio_dashboard.html")
    print()

    all_results = {}
    failed = []
    TELEGRAM_FAILURE_LOG_PATH = Path("reports") / "run_all_error.log"
    progress_session = _progress_start_or_update(
        run_name=run_name,
        phase_index=1,
        phase_total=3,
        phase_label="ML analyzer",
        completed=0,
        total=len(tickers),
        status="Starting",
    )
    batch_progress = _new_batch_progress(
        len(tickers) if run_type == "standard" else 0,
        "portfolio" if run_type == "standard" else "panel-support",
    )

    if run_type == "panel":
        from panel_runner import run_panel
        print("  Running shared panel ML model...\n")
        panel_result = run_panel(watchlist_path)
        try:
            with open(panel_result.get("summary_path"), encoding="utf-8") as f:
                _build_panel_chart(json.load(f))
        except Exception:
            pass
        panel_success = set(panel_result.get("tickers", {}).keys())
        for ticker in tickers:
            if ticker not in panel_success:
                all_results[ticker] = {"ML": {"elapsed": 0.0, "success": False}}
                failed.append(ticker)
        work_items = [t for t in tickers if t in panel_success]
        batch_progress = _new_batch_progress(len(work_items), "panel-support")
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_run_timed_ticker_task, t, _run_ticker_support_batch, run_name, i, len(work_items)): t
                    for i, t in enumerate(work_items, 1)
                }
                bar = tqdm(as_completed(futures), total=len(futures),
                           desc="  Support", unit="ticker", ncols=70, file=sys.stdout)
                for fut in bar:
                    ticker = futures[fut]
                    task = fut.result()
                    result = task.get("result") or {"ML": {"elapsed": 0.0, "success": False}}
                    all_results[ticker] = result
                    ok = task.get("error") is None and all(v["success"] for v in result.values())
                    _record_batch_completion(batch_progress, task, ok, result, bar=bar)
                    if not ok and ticker not in failed:
                        failed.append(ticker)
                    # (old try/except bar-update — replaced by _record_batch_completion)
                bar.close()
        else:
            for i, ticker in enumerate(work_items, 1):
                task = _run_timed_ticker_task(ticker, _run_ticker_support_batch, run_name, i, len(work_items))
                print(
                    f"\n  [{i}/{len(work_items)}] Finalizing {ticker} "
                    f"(start {task['started_label']} | batch {_fmt_time(_batch_elapsed(batch_progress))})..."
                )
                all_results[ticker] = task.get("result") or {"ML": {"elapsed": 0.0, "success": False}}
                ok = task.get("error") is None and all(v['success'] for v in all_results[ticker].values())
                _record_batch_completion(batch_progress, task, ok, all_results[ticker])
                # (removed dead code: print(f"  {'✓' if ok else '✗'} {ticker:<6}"))
                if not ok and ticker not in failed:
                    failed.append(ticker)
    else:
        if n_workers > 1:
            print(f"  Running {len(tickers)} tickers in parallel ({n_workers} workers)...\n")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_run_timed_ticker_task, t, _run_ticker_batch, benchmark, run_name, i, len(tickers)): t
                    for i, t in enumerate(tickers, 1)
                }
                bar = tqdm(as_completed(futures), total=len(tickers),
                           desc="  Scanning", unit="ticker", ncols=70, file=sys.stdout)
                for fut in bar:
                    ticker = futures[fut]
                    task = fut.result()
                    result = task.get("result") or {"ML": {"elapsed": 0.0, "success": False}}
                    all_results[ticker] = result
                    ok = task.get("error") is None and all(v['success'] for v in result.values())
                    _record_batch_completion(batch_progress, task, ok, result, bar=bar)
                    if not ok:
                        failed.append(ticker)
                    # (old try/except bar-update — replaced by _record_batch_completion)
                bar.close()
        else:
            for i, ticker in enumerate(tickers, 1):
                task = _run_timed_ticker_task(ticker, _run_ticker_batch, benchmark, run_name, i, len(tickers))
                print(
                    f"\n  [{i}/{len(tickers)}] Analyzing {ticker} "
                    f"(start {task['started_label']} | batch {_fmt_time(_batch_elapsed(batch_progress))})..."
                )
                all_results[ticker] = task.get("result") or {"ML": {"elapsed": 0.0, "success": False}}
                ok = task.get("error") is None and all(v['success'] for v in all_results[ticker].values())
                _record_batch_completion(batch_progress, task, ok, all_results[ticker])
                # (old inline print — replaced by _record_batch_completion)
                if not ok:
                    failed.append(ticker)

    print(f"\n  Finished {len(tickers)} tickers in {_fmt_time(time.time()-run_start)}")
    _print_batch_summary(batch_progress)
    if failed:
        print(f"  WARNING: errors in: {', '.join(sorted(set(failed)))}")

    for ticker in tickers:
        module_errors = {name: info.get("success", False) for name, info in all_results.get(ticker, {}).items()}
        if module_errors:
            try:
                build_single_dashboard(ticker, module_errors)
            except Exception:
                pass

    stock_data = {ticker: _load_stock_json(ticker) for ticker in tickers}
    quality_payload = {}
    elapsed_summary = _batch_elapsed_summary(batch_progress)
    if benchmark:
        gate_watchlist = benchmark_watchlist or watchlist_path
        if Path(gate_watchlist).resolve() == Path(watchlist_path).resolve():
            quality_payload = _evaluate_quality_gate(
                gate_watchlist, stock_data, elapsed_summary=elapsed_summary)
        else:
            quality_payload = _evaluate_quality_gate(
                gate_watchlist, elapsed_summary=elapsed_summary)
    else:
        quality_payload = _evaluate_quality_gate(
            watchlist_path, stock_data, elapsed_summary=elapsed_summary)

    portfolio_summary = _build_portfolio_summary(
        tickers,
        stock_data,
        optimizer=optimizer,
        watchlist_path=watchlist_path,
        quality_gate=quality_payload.get("quality_gate", {}),
        mode=run_type,
    )
    print("\n  Building portfolio dashboard...")
    path = build_portfolio_dashboard(tickers, failed, portfolio_summary, stock_data=stock_data)
    if failed:
        failed_log = Path("reports") / failed[0] / f"{failed[0]}_master.log"
        _notify_run_failure(run_name, log_path=failed_log if failed_log.exists() else TELEGRAM_FAILURE_LOG_PATH, progress_session=progress_session)
    elif benchmark:
        _notify_run_success("benchmark", progress_session=progress_session)
    elif run_type == "panel":
        _notify_run_success("panel", progress_session=progress_session)
    else:
        _notify_run_success("portfolio", progress_session=progress_session)
    _open_browser(path)
    print(f"\n  Dashboard: {path}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MEAN-VARIANCE ALLOCATION (NEW 20)
# ─────────────────────────────────────────────────────────────────────────────
def _load_stock_json(ticker: str) -> dict:
    _, _, load_report_bundle = _portfolio_tools()
    return load_report_bundle("reports", ticker)


def compute_mv_weights(tickers: list, stock_data: dict,
                       max_weight: float = 0.25,
                       min_weight: float = 0.02,
                       optimizer: str = DEFAULT_OPTIMIZER) -> dict:
    """Thin wrapper — delegates to portfolio_tools with the chosen optimizer."""
    _, compute_portfolio_weights, _ = _portfolio_tools()
    summary = compute_portfolio_weights(
        tickers,
        stock_data,
        optimizer=optimizer,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    return summary.get('weights', {})


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO DASHBOARD HTML BUILDER (NEW 19)
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio_dashboard(tickers: list, failed: list,
                              portfolio_summary: dict | None = None,
                              stock_data: dict | None = None) -> Path:
    out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
    # Reuse pre-loaded stock_data when available (avoids 3×N redundant disk reads).
    if stock_data is None:
        stock_data = {t: _load_stock_json(t) for t in tickers}
    portfolio_summary = portfolio_summary or _build_portfolio_summary(
        tickers, stock_data, optimizer=DEFAULT_OPTIMIZER, watchlist_path="watchlist.txt")
    optimizer_summary = portfolio_summary.get('optimizer', {}) or {}
    quality_gate = portfolio_summary.get('quality_gate', {}) or {}
    _requested_optimizer = (optimizer_summary.get('requested_method')
                            or optimizer_summary.get('method_used')
                            or DEFAULT_OPTIMIZER)
    weights = (optimizer_summary.get('weights', {})
               or compute_mv_weights(tickers, stock_data, optimizer=_requested_optimizer))
    expected_inputs = optimizer_summary.get('expected_returns', {}) or {}
    cash_weight = float(optimizer_summary.get('cash_weight', 0.0) or 0.0)
    allocation_status = optimizer_summary.get('allocation_status') or 'unknown'
    allocatable_tickers = list(optimizer_summary.get('allocatable_tickers', []) or [])

    # Build per-ticker summary
    rows = []
    for t in tickers:
        d       = stock_data[t]
        sig_raw = d.get('signal_data', {}); fund = d.get('fund_data', {}); mc = d.get('mc_data', {})
        sig_dict   = sig_raw.get('signal', {}) or {}
        signal_str = sig_dict.get('signal', 'N/A')  if isinstance(sig_dict, dict) else 'N/A'
        conf       = sig_dict.get('confidence', 0)   if isinstance(sig_dict, dict) else 0
        model_used = sig_dict.get('model_used', '')  if isinstance(sig_dict, dict) else ''
        bt         = sig_raw.get('backtest', {}) or {}
        wf_bt      = sig_raw.get('walkforward_backtest', {}) or {}
        cpcv       = sig_raw.get('cpcv', {}) or {}
        mc_risk    = mc.get('risk_summary', {}) or {}
        curr_price = float(mc.get('current_price', 0) or 0)
        mc_median  = sum(mc_risk.get(m, {}).get('median_1yr', curr_price) or curr_price
                         for m in ('GBM','Merton','Heston')) / 3 if mc_risk else 0
        mc_upside  = ((mc_median / curr_price) - 1) * 100 if curr_price > 0 else 0
        mc_var     = mc_risk.get('Stressed', {}).get('var95_63d', 0) or 0
        model_accs = sig_raw.get('model_accuracy', {}); best_acc = max(model_accs.values(), default=0)
        comp       = fund.get('composite', 'N/A')
        piotroski  = fund.get('piotroski', {}); pi_s = piotroski.get('score','N/A')
        pi_m       = piotroski.get('max_score', 9)
        fund_ctx   = fund.get('context', {})
        dl_diag    = sig_raw.get('dl_diagnostics', [])
        collapsed  = sum(1 for dd in dl_diag if dd.get('is_collapsed', False))
        regime_inf = sig_raw.get('regime', {})
        explain    = sig_raw.get('explainability', {}) or {}
        top_local  = (explain.get('top_local') or [{}])[0] if explain else {}
        alloc_info = expected_inputs.get(t, {}) or {}
        execution_status = sig_dict.get('execution_status', 'N/A')
        selection_status = sig_dict.get('selection_status', 'N/A')

        rows.append({
            'ticker':    t, 'signal': signal_str, 'conf': conf,
            'best_acc':  best_acc, 'sharpe': bt.get('strat_sharpe', 0),
            'wf_sharpe': wf_bt.get('wf_sharpe', None),
            'cpcv_p5':   cpcv.get('sharpe_p5', None),
            'comp':      comp, 'piotroski': pi_s, 'piotr_max': pi_m,
            'price':     curr_price, 'mc_upside': mc_upside, 'mc_var': mc_var,
            'weight':    weights.get(t, 0), 'failed': t in failed,
            'collapsed': collapsed,
            'horizon':   regime_inf.get('predict_days', '?'),
            'speed':     regime_inf.get('speed', '?'),
            'sector':    fund_ctx.get('sector', ''),
            'top_feature': top_local.get('feature', '—'),
            'execution_status': execution_status,
            'selection_status': selection_status,
            'allocatable': bool(alloc_info.get('allocatable', False)),
            'allocation_reason': alloc_info.get('allocation_reason'),
        })

    _order = {'BUY': 0, 'HOLD': 1, 'SELL': 2, 'N/A': 3}
    rows.sort(key=lambda r: (
        _order.get(r['signal'], 3),
        -(float(r['comp']) if str(r['comp']).replace('.','').isdigit() else 0)
    ))

    SC = {'BUY': '#00C853', 'SELL': '#D50000', 'HOLD': '#FFD600', 'N/A': '#888'}
    sig_counts = {}
    for r in rows: sig_counts[r['signal']] = sig_counts.get(r['signal'], 0) + 1

    total_long  = sum(r['weight'] for r in rows if r['signal'] == 'BUY')
    total_flat  = sum(r['weight'] for r in rows if r['signal'] == 'HOLD')
    actionable_count = sum(1 for r in rows if r['allocatable'])
    non_actionable_count = len(rows) - actionable_count

    # ── Table rows ────────────────────────────────────────────────────────────
    table_rows = ''
    for r in rows:
        sig_c  = SC.get(r['signal'], '#888')
        try:    comp_f = float(r['comp']); comp_c = '#00C853' if comp_f>=65 else ('#D50000' if comp_f<=35 else '#FFD600')
        except: comp_c = '#888'
        sh_c   = '#00C853' if r['sharpe']>1.0 else ('#FFD600' if r['sharpe']>0 else '#D50000')
        up_c   = '#00C853' if r['mc_upside']>10 else ('#D50000' if r['mc_upside']<-5 else '#888')
        wf_s   = f"{r['wf_sharpe']:+.2f}" if r['wf_sharpe'] is not None else '—'
        cp_s   = f"{r['cpcv_p5']:+.2f}"  if r['cpcv_p5']  is not None else '—'
        pi_s   = f"{r['piotroski']}/{r['piotr_max']}" if r['piotroski'] != 'N/A' else '—'
        coll   = f' <span style="color:#FFD600" title="{r["collapsed"]} DL collapsed">⚠</span>' if r['collapsed'] else ''

        table_rows += f"""
<tr class="{'row-fail' if r['failed'] else ''}">
  <td><a href="{r['ticker']}/{r['ticker']}_dashboard.html" class="ticker-link">{r['ticker']}</a>
      {coll}<br><span class="dim">{r['sector'][:28]}</span></td>
  <td><span class="sig-badge" style="color:{sig_c};border-color:{sig_c}40">{r['signal']}</span>
      <br><span class="dim">{r['conf']*100:.0f}% conf | {r['horizon']}d</span>
      <br><span class="dim">{r['execution_status']}</span>
      <br><span class="dim">{r['selection_status']}</span></td>
  <td class="num" style="color:{comp_c}">{r['comp']}</td>
  <td class="num">{pi_s}</td>
  <td class="num">{r['best_acc']*100:.1f}%</td>
  <td class="num" style="color:{sh_c}">{r['sharpe']:.2f}</td>
  <td class="num">{wf_s}</td>
  <td class="num">{cp_s}</td>
  <td class="num" style="color:{up_c}">{r['mc_upside']:+.1f}%</td>
  <td class="num" style="color:#D50000">{r['mc_var']:.1f}%</td>
  <td class="num">{r['top_feature']}</td>
  <td class="num" style="color:#58A6FF;font-weight:700">{r['weight']*100:.1f}%</td>
</tr>"""

    # ── Summary cards ─────────────────────────────────────────────────────────
    summary_html = ''
    for sig_name in ('BUY', 'HOLD', 'SELL'):
        cnt = sig_counts.get(sig_name, 0)
        if cnt == 0: continue
        c = SC[sig_name]
        summary_html += f"""<div class="sum-card" style="border-color:{c}40">
  <div class="sum-sig" style="color:{c}">{sig_name}</div>
  <div class="sum-count">{cnt}</div>
  <div class="sum-label">tickers</div></div>"""
    for label, val, color in [
        ('LONG', f'{total_long*100:.0f}%', '#58A6FF'),
        ('FLAT', f'{total_flat*100:.0f}%', '#888'),
        ('ACTIONABLE', str(actionable_count), '#00C853'),
        ('NON-ACT', str(non_actionable_count), '#FF9800'),
        ('CASH', f'{cash_weight*100:.0f}%', '#8B949E'),
    ]:
        summary_html += f"""<div class="sum-card" style="border-color:{color}40">
  <div class="sum-sig" style="color:{color}">{label}</div>
  <div class="sum-count">{val}</div>
  <div class="sum-label">suggested</div></div>"""

    # ── Allocation bar ────────────────────────────────────────────────────────
    alloc_bars = ''
    for r in rows:
        c = SC.get(r['signal'], '#888')
        alloc_bars += (f'<div class="alloc-seg" style="width:{r["weight"]*100:.1f}%;'
                       f'background:{c}" title="{r["ticker"]}: {r["weight"]*100:.1f}%">'
                       f'<span class="alloc-label">{r["ticker"]}</span></div>')
    if cash_weight > 0:
        alloc_bars += (
            f'<div class="alloc-seg" style="width:{cash_weight*100:.1f}%;background:#8B949E" '
            f'title="CASH: {cash_weight*100:.1f}%"><span class="alloc-label">CASH</span></div>'
        )

    error_notice = ''
    if failed:
        error_notice = (f'<div class="error-banner">⚠ {len(failed)} ticker(s) had module errors '
                        f'(partial data): {", ".join(failed)}</div>')
    portfolio_notice = ''
    if allocation_status == 'cash_no_actionable_names':
        portfolio_notice += (
            '<div class="error-banner">⚠ No deployable/actionable names passed the execution gate. '
            'Portfolio stays in CASH after rescue review.</div>'
        )
    elif cash_weight > 1e-9:
        portfolio_notice += (
            f'<div class="warning-banner compact">Risk allocation is constrained to '
            f'{(1.0 - cash_weight)*100:.1f}% because only {actionable_count}/{len(tickers)} names are actionable '
            f'and the risky sleeve cannot fully deploy capital under the current per-position cap.</div>'
        )
    if len(allocatable_tickers) > 0 and abs(cash_weight) <= 1e-9 and len(allocatable_tickers) == len(tickers) and \
            float(optimizer_summary.get('max_weight', 0.25) or 0.25) * len(allocatable_tickers) <= 1.0 + 1e-9:
        portfolio_notice += (
            '<div class="warning-banner compact">The optimizer is operating near a concentration-bound corner; '
            'weights may be structurally trivial under the current max-weight constraint.</div>'
        )
    gate_notice = ''
    if quality_gate:
        gate_color = {'PASS': '#00C853', 'MARGINAL': '#FFD600', 'FAIL': '#D50000'}.get(
            quality_gate.get('status'), '#58A6FF')
        metrics = quality_gate.get('metrics', {}) or {}
        coverage = quality_gate.get('coverage', {}) or {}
        reasons = quality_gate.get('primary_failure_reasons', []) or []
        actionable = (quality_gate.get('actionable_summary', {}) or {}).get('actionable_rate', 0.0)
        router_status = quality_gate.get('router_status_frequency', {}) or {}
        mc_summary = quality_gate.get('mc_reliability_summary', {}) or {}
        conformal_summary = quality_gate.get('conformal_usability_summary', {}) or {}
        per_ticker_gate = quality_gate.get('per_ticker', {}) or {}
        wf_fallback_count = sum(
            1 for row in per_ticker_gate.values()
            if ((row.get('evidence_sources', {}) or {}).get('wf_source') == 'tree_family_diagnostics')
        )
        cpcv_fallback_count = sum(
            1 for row in per_ticker_gate.values()
            if ((row.get('evidence_sources', {}) or {}).get('cpcv_source') == 'tree_family_diagnostics')
        )
        gate_notice = (
            f'<div class="card" style="border-color:{gate_color}40">'
            f'<h2>Portfolio Quality Gate</h2>'
            f'<div class="metric-row"><span class="metric-label">Status</span>'
            f'<span class="metric-val" style="color:{gate_color}">{quality_gate.get("status","N/A")}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Passed checks</span>'
            f'<span class="metric-val">{quality_gate.get("passed_checks",0)}/{quality_gate.get("total_checks",0)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Signal coverage</span>'
            f'<span class="metric-val">{(coverage.get("signal_json", {}) or {}).get("count",0)}/{metrics.get("n_tickers",0)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Diag coverage</span>'
            f'<span class="metric-val">{(coverage.get("diagnostics_json", {}) or {}).get("count",0)}/{metrics.get("n_tickers",0)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Median WF Sharpe</span>'
            f'<span class="metric-val">{metrics.get("median_wf_sharpe","N/A")}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Median ECE</span>'
            f'<span class="metric-val">{metrics.get("median_ece","N/A")}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Median BUY recall</span>'
            f'<span class="metric-val">{metrics.get("median_buy_recall","N/A")}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Actionable rate</span>'
            f'<span class="metric-val">{float(actionable or 0.0):.0%}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Actionable names</span>'
            f'<span class="metric-val">{actionable_count}/{len(tickers)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Failed names</span>'
            f'<span class="metric-val">{len(failed)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Router status</span>'
            f'<span class="metric-val">{", ".join(f"{k}={v}" for k, v in router_status.items()) if router_status else "n/a"}</span></div>'
            f'<div class="metric-row"><span class="metric-label">MC fallback-vol</span>'
            f'<span class="metric-val">{(mc_summary.get("fallback_vol_count", 0) if mc_summary else 0)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Conformal usable</span>'
            f'<span class="metric-val">{(conformal_summary.get("usable_count", 0) if conformal_summary else 0)}/{((conformal_summary.get("usable_count", 0) if conformal_summary else 0) + (conformal_summary.get("unusable_count", 0) if conformal_summary else 0))}</span></div>'
            f'<div class="metric-row"><span class="metric-label">WF source fallback</span>'
            f'<span class="metric-val">{wf_fallback_count}</span></div>'
            f'<div class="metric-row"><span class="metric-label">CPCV source fallback</span>'
            f'<span class="metric-val">{cpcv_fallback_count}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Main issues</span>'
            f'<span class="metric-val">{", ".join(reasons[:3]) if reasons else "none"}</span></div>'
            f'</div>'
        )
    optimizer_img = img_to_base64(out_dir / "portfolio_optimizer.png")
    benchmark_img = img_to_base64(out_dir / "benchmark_quality.png")
    panel_img = img_to_base64(out_dir / "panel_summary.png")
    visual_sections = ''
    if optimizer_img:
        visual_sections += f'<div class="card"><h2>Optimizer Overview</h2><img src="{optimizer_img}" alt="Portfolio Optimizer"></div>'
    if benchmark_img:
        visual_sections += f'<div class="card"><h2>Benchmark Quality</h2><img src="{benchmark_img}" alt="Benchmark Quality"></div>'
    if panel_img:
        visual_sections += f'<div class="card"><h2>Panel Summary</h2><img src="{panel_img}" alt="Panel Summary"></div>'

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Scanner Dashboard v13.0</title>
<style>
  :root{{--dark:#0D1117;--panel:#161B22;--border:#30363D;--text:#E6EDF3;--muted:#8B949E;
         --green:#00C853;--red:#D50000;--blue:#58A6FF;--gold:#FFD600;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--dark);color:var(--text);font-family:'JetBrains Mono',monospace;
        font-size:13px;padding:20px 28px;}}
  h1{{font-size:1.6rem;font-weight:800;margin-bottom:4px;}}
  h2{{font-size:0.82rem;font-weight:700;color:var(--muted);text-transform:uppercase;
      letter-spacing:.06em;margin-bottom:10px;}}
  .meta{{color:var(--muted);font-size:0.72rem;margin-bottom:20px;}}
  .card{{background:var(--panel);border:1px solid var(--border);border-radius:10px;
         padding:16px 18px;margin-bottom:14px;}}
  .sum-row{{display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap;}}
  .sum-card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;
             padding:12px 20px;text-align:center;min-width:90px;}}
  .sum-sig{{font-size:1.3rem;font-weight:900;}}
  .sum-count{{font-size:2rem;font-weight:800;}}
  .sum-label{{font-size:0.72rem;color:var(--muted);}}
  .alloc-bar{{display:flex;height:32px;border-radius:6px;overflow:hidden;
              margin-bottom:6px;border:1px solid var(--border);}}
  .alloc-seg{{display:flex;align-items:center;justify-content:center;overflow:hidden;}}
  .alloc-label{{font-size:0.65rem;font-weight:700;color:#0D1117;white-space:nowrap;padding:0 3px;}}
  img{{width:100%;border-radius:8px;border:1px solid var(--border);}}
  table{{width:100%;border-collapse:collapse;}}
  thead tr{{background:#0D1117;}}
  th{{padding:8px 6px;font-size:0.70rem;text-transform:uppercase;color:var(--muted);
      text-align:right;border-bottom:2px solid var(--border);}}
  th:first-child{{text-align:left;}}
  td{{padding:8px 6px;font-size:0.78rem;border-bottom:1px solid var(--border);color:var(--muted);vertical-align:middle;}}
  .num{{text-align:right;font-weight:700;color:var(--text);}}
  .dim{{font-size:0.68rem;color:var(--muted);}}
  .row-fail{{opacity:.55;}}
  tr:hover{{background:#1c2128;}}
  .sig-badge{{display:inline-block;font-size:0.72rem;font-weight:800;padding:2px 8px;
              border:1px solid;border-radius:10px;}}
  .ticker-link{{color:var(--blue);text-decoration:none;font-weight:700;font-size:0.85rem;}}
  .ticker-link:hover{{text-decoration:underline;}}
  .error-banner{{background:#3d0000;border:1px solid var(--red);border-radius:8px;
                 padding:8px 14px;margin-bottom:14px;font-size:0.78rem;color:#ff6b6b;}}
  .warning-banner{{background:#3d2300;border:1px solid #FF9800;border-radius:8px;
                   padding:8px 14px;margin-bottom:14px;font-size:0.78rem;color:#ffd699;}}
  .warning-banner.compact{{margin:0 0 12px 0;padding:8px 10px;font-size:0.75rem;}}
  .metric-row{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border);}}
  .metric-row:last-child{{border-bottom:none;}}
  .metric-label{{color:var(--muted);font-size:0.80rem;}}
  .metric-val{{font-weight:700;font-size:0.85rem;color:var(--text);}}
  .note{{font-size:0.70rem;color:var(--muted);margin-top:6px;}}
  .disclaimer{{background:#0d0d17;border:1px solid var(--border);border-radius:8px;
               padding:12px;font-size:0.72rem;color:var(--muted);margin-top:20px;}}
</style></head><body>

<h1>Portfolio <span style="color:var(--blue)">Scanner Dashboard</span></h1>
<p class="meta">v13.0 &nbsp;|&nbsp; {now} &nbsp;|&nbsp; {len(tickers)} tickers analysed</p>

{error_notice}
{portfolio_notice}
{gate_notice}

<div class="sum-row">{summary_html}</div>

<div class="card">
  <h2>Suggested Allocation — {optimizer_summary.get('method_used', DEFAULT_OPTIMIZER).replace('_', ' ').title()}</h2>
  <div class="alloc-bar">{alloc_bars}</div>
  <p class="note">
    Requested optimizer: {optimizer_summary.get('requested_method', DEFAULT_OPTIMIZER)}.
    Covariance source: {optimizer_summary.get('covariance_source', 'N/A')}.
    {f'Fallback: {optimizer_summary.get("fallback_reason")}. ' if optimizer_summary.get('fallback_reason') else ''}
    Allocation status: {allocation_status}. Cash sleeve: {cash_weight*100:.1f}%.
    Actionable universe: {actionable_count}/{len(tickers)}.
    Long-only · max {optimizer_summary.get('max_weight', 0.25)*100:.0f}% per position · min {optimizer_summary.get('min_weight', 0.02)*100:.0f}%.
  </p>
</div>

{visual_sections}

<div class="card">
  <h2>Portfolio Ranking — sorted: BUY → HOLD → SELL, then composite score</h2>
  <table>
    <thead><tr>
      <th style="text-align:left">Ticker / Sector</th>
      <th>Signal</th>
      <th>Score</th>
      <th>Piotroski</th>
      <th>Best Acc</th>
      <th>Sharpe</th>
      <th>WF Sharpe</th>
      <th>CPCV p5</th>
      <th>MC Upside</th>
      <th>Stressed VaR</th>
      <th>Top Driver</th>
      <th>Weight</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <p class="note">
    Click any ticker to open its individual deep-dive dashboard.
    ⚠ = one or more DL models collapsed during training.
    CPCV p5 &gt; 0 means skill is robust across all combinatorial CV paths.
  </p>
</div>

<div class="disclaimer">
  <strong>DISCLAIMER:</strong> Portfolio scanner is for educational and research purposes only.
  Allocation weights are a quantitative suggestion, not investment advice.
  Always consult a licensed financial advisor before investing.
</div>
</body></html>"""

    path = out_dir / "portfolio_dashboard.html"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Portfolio dashboard saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-STOCK DASHBOARD (unchanged v12 logic, v13 version string)
# ─────────────────────────────────────────────────────────────────────────────
def build_single_dashboard(ticker: str, module_errors: dict) -> Path:
    out_dir = Path("reports") / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(fname):
        p = out_dir / fname
        if not p.exists(): return {}
        try:
            with open(p, encoding='utf-8') as f: return json.load(f)
        except Exception as e:
            print(f"  ⚠  {fname} unreadable ({e})"); return {}

    signal    = _load(f"{ticker}_signal.json")
    fund_data = _load(f"{ticker}_fundamentals.json")
    mc_data   = _load(f"{ticker}_montecarlo.json")

    sig      = signal.get('signal', {}) if isinstance(signal.get('signal', {}), dict) else {}
    truth    = extract_dashboard_truth(signal, mc_data)
    sig_str  = truth.get('final_signal') or 'N/A'
    conf     = truth.get('confidence')
    model    = truth.get('deployment_model') or 'N/A'
    probs    = sig.get('probabilities', {}) if isinstance(sig, dict) else {}
    bt       = signal.get('backtest', {}) or {}
    bt_audit = signal.get('backtest_audit', {}) or {}
    gpu_info = signal.get('gpu_info') or 'N/A'
    runtime  = signal.get('total_runtime') or 'N/A'

    comp     = fund_data.get('composite', 'N/A')
    context  = fund_data.get('context', {})
    dcf      = fund_data.get('dcf', {})
    graham   = fund_data.get('graham', {})
    piotr    = fund_data.get('piotroski', {})

    conformal  = signal.get('conformal', {}) or {}
    wf_bt      = signal.get('walkforward_backtest', {}) or {}
    cpcv       = signal.get('cpcv', {}) or {}
    regime_inf = signal.get('regime', {})
    explain    = signal.get('explainability', {}) or {}
    router     = signal.get('router', {}) or {}
    selection  = signal.get('selection', {}) or {}
    execution_status = sig.get('execution_status') or 'N/A'
    execution_gate = sig.get('execution_gate') or 'N/A'
    execution_details = sig.get('execution_gate_details', {}) if isinstance(sig, dict) else {}
    deployment_eligible = bool(sig.get('deployment_eligible', False)) if isinstance(sig, dict) else False
    eligibility_failures = list(sig.get('eligibility_failures', []) or []) if isinstance(sig, dict) else []
    reference_model_used = truth.get('reference_model')
    deployment_model_used = truth.get('deployment_model')
    selection_status = truth.get('selection_status') or 'N/A'
    conformal_method = truth.get('conformal_method')
    conformal_target_coverage = truth.get('conformal_target_coverage')
    wf_sharpe = truth.get('walkforward_sharpe')
    cpcv_p5 = truth.get('cpcv_p5')
    mc_reliability_status = truth.get('mc_reliability_status')
    spec_growth = ((fund_data.get('fundamentals', {}) or {}).get('speculative_growth', {}) or
                   fund_data.get('speculative_growth', {}) or {})
    mc_reliability = mc_data.get('mc_reliability', {}) or {}
    mc_baseline_status = mc_reliability.get('baseline_reliability_status') or mc_reliability_status
    mc_scenario_status = mc_reliability.get('scenario_reliability_status')
    mc_primary_failures = (
        mc_reliability.get('baseline_reliability_failures')
        or mc_reliability.get('mc_reliability_failures')
        or []
    )
    evidence_scope = signal.get('evidence_scope', {}) or {}
    tree_family_diagnostics = signal.get('tree_family_diagnostics', {}) or {}
    selected_candidate_family = (
        evidence_scope.get('selected_candidate_family')
        or sig.get('deployment_family_used')
        or sig.get('reference_family_used')
    )
    wf_scope = evidence_scope.get('walkforward_scope') or ('selected_candidate' if selected_candidate_family == 'tree_family' else 'tree_family_diagnostics')
    cpcv_scope = evidence_scope.get('cpcv_scope') or ('selected_candidate' if selected_candidate_family == 'tree_family' else 'tree_family_diagnostics')
    tree_wf_bt = (tree_family_diagnostics.get('walkforward_backtest', {}) or {})
    tree_cpcv = (tree_family_diagnostics.get('cpcv', {}) or {})

    SC = {'BUY': '#00C853', 'SELL': '#D50000', 'HOLD': '#FFD600'}
    sig_color = SC.get(sig_str, '#aaa')
    execution_colors = {
        'ACTIONABLE': '#00C853',
        'HOLD_NEUTRAL': '#58A6FF',
        'ABSTAIN_NO_EDGE': '#FFD600',
        'ABSTAIN_UNCERTAIN': '#FF9800',
        'ABSTAIN_MODEL_UNRELIABLE': '#D50000',
    }
    exec_color = execution_colors.get(execution_status, '#8B949E')
    try:    comp_f = float(comp); comp_color = '#00C853' if comp_f>=65 else ('#D50000' if comp_f<=35 else '#FFD600')
    except: comp_color = '#aaa'

    _strat_cls = 'green' if bt.get('strat_return', 0) > 0 else 'red'
    _bh_cls    = 'green' if bt.get('bh_return',    0) > 0 else 'red'

    cat_rows     = ''.join(f'<li>&#10003; {c}</li>' for c in context.get('catalysts', []))
    risk_rows    = ''.join(f'<li>&#9888; {r}</li>'  for r in context.get('risks', []))
    analyst_rows = ''.join(f'<tr><td>{k}</td><td class="right">{v}</td></tr>'
                            for k, v in context.get('analyst_targets', {}).items())
    model_accs = signal.get('model_accuracy', {})
    model_rows = ''.join(f'<tr><td>{k}</td><td class="right acc">{v*100:.2f}%</td></tr>'
                          for k, v in sorted(model_accs.items(), key=lambda x: -x[1]))
    prob_bars  = ''.join(
        f'<div class="prob-row"><span class="prob-label">{k}</span>'
        f'<div class="prob-bar-wrap"><div class="prob-bar" style="width:{v*100:.0f}%;background:{SC.get(k,"#58A6FF")}"></div></div>'
        f'<span class="prob-val">{v*100:.1f}%</span></div>'
        for k, v in probs.items())

    def sig_class(s): return {'BUY':'green','SELL':'red','HOLD':'gold'}.get(s,'')
    all_sigs = []
    if signal.get('tree_signal'):
        ts = signal['tree_signal']
        all_sigs.append((ts.get('model_used','Tree'), ts.get('signal','?'), ts.get('confidence',0)))
    for ds in signal.get('dl_signals', []):
        all_sigs.append((ds.get('model_used','DL'), ds.get('signal','?'), ds.get('confidence',0)))
    if signal.get('meta_signal'):
        ms = signal['meta_signal']
        all_sigs.append((ms.get('model_used','Meta'), ms.get('signal','?'), ms.get('confidence',0)))
    sig_table_rows = ''.join(
        f'<tr><td>{n}</td><td class="{sig_class(s)}" style="font-weight:700">{s}</td>'
        f'<td class="right">{c*100:.1f}%</td></tr>'
        for n, s, c in all_sigs)

    mc_risk = mc_data.get('risk_summary', {}); mc_rows = ''
    if mc_risk:
        for mname, r in mc_risk.items():
            mc_rows += (f'<tr><td>{mname}</td><td class="right">${r.get("median_1yr",0):.2f}</td>'
                        f'<td class="right" style="color:#D50000">{r.get("var95_63d",0):.1f}%</td>'
                        f'<td class="right" style="color:#D50000">{r.get("cvar95_63d",0):.1f}%</td>'
                        f'<td class="right">{r.get("prob_loss_63d",0):.1f}%</td></tr>')

    dcf_section = ''
    if dcf.get('available'):
        mos = dcf.get('margin_of_safety', 0); color = '#00C853' if mos>10 else ('#D50000' if mos<-10 else '#FFD600')
        dcf_section = (f'<div class="metric-row"><span class="metric-label">DCF Base Value</span>'
                       f'<span class="metric-val">${dcf.get("intrinsic_base",0):.2f}</span></div>'
                       f'<div class="metric-row"><span class="metric-label">Margin of Safety</span>'
                       f'<span class="metric-val" style="color:{color}">{mos:+.1f}%&nbsp;{dcf.get("signal","")}</span></div>')

    graham_section = ''
    if graham.get('available'):
        mos = graham.get('margin_of_safety', 0); color = '#00C853' if mos>10 else ('#D50000' if mos<-10 else '#FFD600')
        graham_section = (f'<div class="metric-row"><span class="metric-label">Graham Number</span>'
                          f'<span class="metric-val">${graham.get("graham_number",0):.2f}</span></div>'
                          f'<div class="metric-row"><span class="metric-label">Graham Safety</span>'
                          f'<span class="metric-val" style="color:{color}">{mos:+.1f}%&nbsp;{graham.get("signal","")}</span></div>')

    piotr_section = ''
    if piotr.get('available'):
        sc = piotr.get('score',0); mx = piotr.get('max_score',9); pct = sc/mx*100 if mx>0 else 0
        color = '#00C853' if pct>=70 else ('#D50000' if pct<=35 else '#FFD600')
        piotr_section = (f'<div class="metric-row"><span class="metric-label">Piotroski F-Score</span>'
                         f'<span class="metric-val" style="color:{color}">{sc}/{mx}&nbsp;({piotr.get("label","")})</span></div>')

    error_banner = ''
    if any(not v for v in module_errors.values()):
        failed_mods = [k for k, v in module_errors.items() if not v]
        error_banner = (f'<div style="background:#3d0000;border:1px solid #D50000;border-radius:8px;'
                        f'padding:10px 16px;margin-bottom:16px;font-size:0.82rem;color:#ff6b6b">'
                        f'⚠ Module errors: {", ".join(failed_mods)}. Results may be partial.</div>')

    analysis_img = img_to_base64(out_dir / f"{ticker}_analysis.png")
    mc_img       = img_to_base64(out_dir / f"{ticker}_montecarlo.png")
    dl_img       = img_to_base64(out_dir / f"{ticker}_dl_models.png")
    sel_img      = img_to_base64(out_dir / f"{ticker}_selection_diagnostics.png")
    mc_vol_img   = img_to_base64(out_dir / f"{ticker}_montecarlo_volatility.png")
    mc_diag_img  = img_to_base64(out_dir / f"{ticker}_montecarlo_diagnostics.png")
    dcf_surface_img = img_to_base64(out_dir / f"{ticker}_dcf_surface_3d.png")
    img_tag_analysis = (f'<img src="{analysis_img}" alt="ML Analysis">'
                        if analysis_img else '<p style="color:var(--muted)">Chart not generated.</p>')
    img_tag_mc = (f'<img src="{mc_img}" alt="Monte Carlo">'
                  if mc_img else '<p style="color:var(--muted)">Chart not generated.</p>')
    extra_chart_cards = ''
    if dl_img or sel_img or mc_vol_img or mc_diag_img or dcf_surface_img:
        extra_cards = []
        if dl_img:
            extra_cards.append(f'<div class="card"><h2>DL Model Overview</h2><img src="{dl_img}" alt="DL Model Overview"></div>')
        if sel_img:
            extra_cards.append(f'<div class="card"><h2>Selection Diagnostics</h2><img src="{sel_img}" alt="Selection Diagnostics"></div>')
        if mc_vol_img:
            extra_cards.append(f'<div class="card"><h2>Volatility Model Overview</h2><img src="{mc_vol_img}" alt="Volatility Model Overview"></div>')
        if mc_diag_img:
            extra_cards.append(f'<div class="card"><h2>Monte Carlo Diagnostics</h2><img src="{mc_diag_img}" alt="Monte Carlo Diagnostics"></div>')
        if dcf_surface_img:
            extra_cards.append(f'<div class="card"><h2>DCF Surface</h2><img src="{dcf_surface_img}" alt="DCF Surface"></div>')
        extra_chart_cards = f'<div class="grid g2">{"".join(extra_cards)}</div>'

    mc_table_section = ''
    if mc_rows:
        mc_table_section = (f'<div style="margin-bottom:14px"><div class="card">'
                            f'<h2>Monte Carlo Risk Summary (5 Models, 63-day / 1-year)</h2><table>'
                            f'<tr style="color:var(--muted);font-size:0.72rem;text-transform:uppercase">'
                            f'<td>Model</td><td class="right">Median 1yr</td>'
                            f'<td class="right">VaR 95%</td><td class="right">CVaR 95%</td>'
                            f'<td class="right">Prob Loss</td></tr>{mc_rows}</table></div></div>')

    dashboard_generated_at = datetime.datetime.now().isoformat()
    conf_text = f'{float(conf)*100:.0f}% confidence' if conf is not None else 'N/A'
    conformal_method_label = str(conformal_method).upper() if conformal_method is not None else 'N/A'
    conformal_target_text = f'{float(conformal_target_coverage)*100:.1f}%' if conformal_target_coverage is not None else 'N/A'
    wf_sharpe_text = f'{float(wf_sharpe):.3f}' if wf_sharpe is not None else 'N/A'
    cpcv_p5_text = f'{float(cpcv_p5):+.3f}' if cpcv_p5 is not None else 'N/A'
    display_payload = {
        'final_signal': truth.get('final_signal'),
        'confidence': truth.get('confidence'),
        'selection_status': truth.get('selection_status'),
        'reference_model': truth.get('reference_model'),
        'deployment_model': truth.get('deployment_model'),
        'conformal_method': truth.get('conformal_method'),
        'conformal_target_coverage': truth.get('conformal_target_coverage'),
        'walkforward_sharpe': truth.get('walkforward_sharpe'),
        'cpcv_p5': truth.get('cpcv_p5'),
        'mc_reliability_status': truth.get('mc_reliability_status'),
    }
    consistency_report = validate_dashboard_payload(display_payload, signal, mc_data, dashboard_generated_at)
    write_dashboard_consistency_report(out_dir / f"{ticker}_dashboard_consistency.json", consistency_report)
    footer_ctx = build_debug_footer_context(signal, dashboard_generated_at)
    artifact_invariants = dict(signal.get('artifact_invariants', {}) or {})
    consistency_banner = ''
    if consistency_report.get('status') != 'OK':
        consistency_banner = (
            '<div class="warning-banner">'
            'Warning: dashboard display may be stale or inconsistent with source JSON.'
            '</div>'
        )
    invariant_banner = ''
    if artifact_invariants and artifact_invariants.get('status') != 'OK':
        issue_names = list(artifact_invariants.get('failure_names', []) or artifact_invariants.get('warning_names', []))
        invariant_banner = (
            '<div class="warning-banner">'
            'Warning: signal artifact invariants flagged issues: '
            + (', '.join(issue_names[:4]) if issue_names else 'artifact validation warning')
            + '.'
            '</div>'
        )

    def _fmt_pct(value, decimals=1, signed=False):
        if value is None:
            return 'N/A'
        try:
            number = float(value) * 100.0
        except Exception:
            return 'N/A'
        return f'{number:+.{decimals}f}%' if signed else f'{number:.{decimals}f}%'

    def _fmt_num(value, decimals=3, signed=False):
        if value is None:
            return 'N/A'
        try:
            number = float(value)
        except Exception:
            return 'N/A'
        return f'{number:+.{decimals}f}' if signed else f'{number:.{decimals}f}'

    def _fmt_int(value):
        if value is None:
            return 'N/A'
        try:
            return str(int(value))
        except Exception:
            return 'N/A'

    # Conformal / WF / CPCV panels
    conf_section = wf_section = cpcv_section = explain_section = ''
    execution_section = mc_rel_section = spec_growth_section = backtest_audit_section = ''
    failures_html = ', '.join(eligibility_failures[:4]) if eligibility_failures else 'none'
    margin_text = _fmt_num(execution_details.get("probability_margin"), decimals=3)
    execution_section = (
        f'<div class="card"><h2>Execution State</h2>'
        f'<div class="metric-row"><span class="metric-label">Status</span>'
        f'<span class="metric-val" style="color:{exec_color}">{execution_status}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Primary gate</span>'
        f'<span class="metric-val">{execution_gate}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Deployable</span>'
        f'<span class="metric-val" style="color:{"#00C853" if deployment_eligible else "#D50000"}">'
        f'{"YES" if deployment_eligible else "NO"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Router status</span>'
        f'<span class="metric-val">{router.get("router_status","N/A") or "N/A"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Probability margin</span>'
        f'<span class="metric-val">{margin_text}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Eligibility failures</span>'
        f'<span class="metric-val">{failures_html}</span></div></div>'
    )
    mc_rel_failures = ', '.join(list(mc_primary_failures)[:3]) or 'N/A'
    mc_baseline_status_text = mc_baseline_status if mc_baseline_status not in (None, "", "unknown") else "N/A"
    mc_scenario_status_text = mc_scenario_status if mc_scenario_status not in (None, "", "unknown") else "N/A"
    mc_rel_color = {
        'usable': '#00C853',
        'mild_miscalibration': '#FFD600',
        'miscalibrated': '#D50000',
    }.get(mc_baseline_status, '#8B949E')
    mc_scenario_color = {
        'usable': '#00C853',
        'mild_miscalibration': '#FFD600',
        'miscalibrated': '#D50000',
    }.get(mc_scenario_status, '#8B949E')
    mc_rel_section = (
        f'<div class="card"><h2>Monte Carlo Reliability</h2>'
        f'<div class="metric-row"><span class="metric-label">Primary status</span>'
        f'<span class="metric-val" style="color:{mc_rel_color}">{mc_baseline_status_text}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Scenario status</span>'
        f'<span class="metric-val" style="color:{mc_scenario_color}">{mc_scenario_status_text}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Fallback vol model</span>'
        f'<span class="metric-val">{mc_reliability.get("vol_model_fallback") or "N/A"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Primary failures</span>'
        f'<span class="metric-val">{mc_rel_failures}</span></div></div>'
    )
    if spec_growth:
        sg_color = '#FF9800' if spec_growth.get('speculative_growth_risk') else '#58A6FF'
        spec_growth_section = (
            f'<div class="card"><h2>Speculative Growth Profile</h2>'
            f'<div class="metric-row"><span class="metric-label">Risk flag</span>'
            f'<span class="metric-val" style="color:{sg_color}">'
            f'{"HIGH" if spec_growth.get("speculative_growth_risk") else "LOW"}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Score</span>'
            f'<span class="metric-val">{_fmt_num(spec_growth.get("speculative_growth_score"), decimals=2)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Haircut</span>'
            f'<span class="metric-val">{_fmt_num(spec_growth.get("speculative_growth_haircut"), decimals=2)}</span></div>'
            f'<div class="metric-row"><span class="metric-label">Confidence multiplier</span>'
            f'<span class="metric-val">{_fmt_num(spec_growth.get("fundamental_confidence_multiplier"), decimals=2)}</span></div></div>'
        )
    if conformal:
        sig_ps = sig.get('prediction_set', []); ps_str = ' / '.join(sig_ps) if sig_ps else '—'
        singleton = sig.get('is_conformal_singleton', False); cov_val = conformal.get('coverage', 0)
        ps_c   = '#00C853' if singleton else ('#FFD600' if len(sig_ps)==2 else '#D50000')
        cov_c  = '#00C853' if cov_val >= conformal.get('target_coverage', 0.9) else '#FFD600'
        conf_section = (f'<div class="card"><h2>Conformal Prediction (RAPS)</h2>'
                        f'<div class="metric-row"><span class="metric-label">Prediction Set</span>'
                        f'<span class="metric-val" style="color:{ps_c}">{ps_str}</span></div>'
                        f'<div class="metric-row"><span class="metric-label">Coverage (target 90%)</span>'
                        f'<span class="metric-val" style="color:{cov_c}">{cov_val*100:.1f}%</span></div>'
                        f'<div class="metric-row"><span class="metric-label">Avg Set Size</span>'
                        f'<span class="metric-val">{conformal.get("avg_set_size",0):.2f}</span></div>'
                        f'<div class="metric-row"><span class="metric-label">Singleton Rate</span>'
                        f'<span class="metric-val">{conformal.get("singleton_rate",0)*100:.1f}%</span></div>'
                        f'<div class="metric-row"><span class="metric-label">Usable for execution</span>'
                        f'<span class="metric-val" style="color:{"#00C853" if conformal.get("usable_for_execution", False) else "#D50000"}">'
                        f'{"YES" if conformal.get("usable_for_execution", False) else "NO"}</span></div></div>')
    if wf_bt:
        wf_ret = wf_bt.get('wf_return', 0); wf_sh = wf_bt.get('wf_sharpe', 0)
        wf_c   = '#00C853' if wf_ret>0 else '#D50000'
        wf_shc = '#00C853' if wf_sh>0.5 else ('#FFD600' if wf_sh>0 else '#D50000')
        wf_section = (f'<div class="card"><h2>Walk-Forward Backtest (OOS)</h2>'
                      f'<div class="metric-row"><span class="metric-label">OOS Return</span>'
                      f'<span class="metric-val" style="color:{wf_c}">{wf_ret*100:+.2f}%</span></div>'
                      f'<div class="metric-row"><span class="metric-label">Sharpe Ratio</span>'
                      f'<span class="metric-val" style="color:{wf_shc}">{wf_sh:.3f}</span></div>'
                      f'<div class="metric-row"><span class="metric-label">Max Drawdown</span>'
                      f'<span class="metric-val red">{wf_bt.get("wf_maxdd",0)*100:.2f}%</span></div>'
                      f'<div class="metric-row"><span class="metric-label">Win Rate</span>'
                      f'<span class="metric-val gold">{wf_bt.get("wf_winrate",0)*100:.1f}%</span></div></div>')
    if cpcv:
        sh_mean = cpcv.get('sharpe_mean', 0); sh_p5 = cpcv.get('sharpe_p5', 0)
        if sh_p5>0:     sl, sc2 = 'ROBUST',   '#00C853'
        elif sh_mean>0: sl, sc2 = 'MIXED',    '#FFD600'
        else:           sl, sc2 = 'NO SKILL', '#D50000'
        cpcv_section = (f'<div class="card"><h2>CPCV (Combinatorial Purged CV)</h2>'
                        f'<div class="metric-row"><span class="metric-label">Skill</span>'
                        f'<span class="metric-val" style="color:{sc2}">{sl}</span></div>'
                        f'<div class="metric-row"><span class="metric-label">Paths</span>'
                        f'<span class="metric-val">{cpcv.get("n_paths",0)}</span></div>'
                        f'<div class="metric-row"><span class="metric-label">Sharpe Mean</span>'
                        f'<span class="metric-val">{sh_mean:+.3f}</span></div>'
                        f'<div class="metric-row"><span class="metric-label">Sharpe p5</span>'
                        f'<span class="metric-val" style="color:{sc2}">{sh_p5:+.3f}</span></div>'
                        f'<div class="metric-row"><span class="metric-label">% Paths &gt; 0</span>'
                        f'<span class="metric-val">{cpcv.get("pct_positive",0)*100:.0f}%</span></div></div>')
    sig_ps = sig.get('prediction_set', []) if isinstance(sig, dict) else []
    ps_str = ' / '.join(sig_ps) if sig_ps else 'N/A'
    singleton = bool(sig.get('is_conformal_singleton', False)) if isinstance(sig, dict) else False
    cov_val = conformal.get('coverage')
    avg_set_size = conformal.get('avg_set_size')
    singleton_rate = conformal.get('singleton_rate')
    usable_for_execution = conformal.get('usable_for_execution')
    conformal_degenerate = conformal.get('degenerate_execution_conformal')
    conformal_degeneracy_reason = conformal.get('degeneracy_reason')
    set_size_dist = conformal.get('set_size_distribution', {}) or {}
    set_size_dist_text = ', '.join(
        f'{k}:{v}' for k, v in sorted(set_size_dist.items(), key=lambda item: str(item[0]))
    ) or 'N/A'
    ps_c = '#00C853' if singleton else ('#FFD600' if len(sig_ps) == 2 else '#D50000')
    cov_c = '#8B949E'
    if cov_val is not None and conformal_target_coverage is not None:
        cov_c = '#00C853' if float(cov_val) >= float(conformal_target_coverage) else '#FFD600'
    usable_text = 'N/A' if usable_for_execution is None else ('YES' if usable_for_execution else 'NO')
    usable_color = '#8B949E' if usable_for_execution is None else ('#00C853' if usable_for_execution else '#D50000')
    conf_section = (
        f'<div class="card"><h2>Conformal Prediction ({conformal_method_label})</h2>'
        f'<div class="metric-row"><span class="metric-label">Prediction Set</span>'
        f'<span class="metric-val" style="color:{ps_c}">{ps_str}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Coverage (target {conformal_target_text})</span>'
        f'<span class="metric-val" style="color:{cov_c}">{_fmt_pct(cov_val, decimals=1)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Avg Set Size</span>'
        f'<span class="metric-val">{_fmt_num(avg_set_size, decimals=2)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Singleton Rate</span>'
        f'<span class="metric-val">{_fmt_pct(singleton_rate, decimals=1)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Set Size Dist.</span>'
        f'<span class="metric-val">{set_size_dist_text}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Degeneracy</span>'
        f'<span class="metric-val">{("YES" if conformal_degenerate else "NO") if conformal_degenerate is not None else "N/A"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Degeneracy reason</span>'
        f'<span class="metric-val">{conformal_degeneracy_reason or "N/A"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Usable for execution</span>'
        f'<span class="metric-val" style="color:{usable_color}">{usable_text}</span></div></div>'
    )
    wf_payload = wf_bt if wf_scope == 'selected_candidate' else tree_wf_bt
    wf_ret = wf_payload.get('wf_return')
    wf_c = '#8B949E'
    if wf_ret is not None:
        wf_c = '#00C853' if float(wf_ret) > 0 else '#D50000'
    wf_shc = '#8B949E'
    wf_sharpe_display = wf_payload.get('wf_sharpe') if wf_payload else wf_sharpe
    if wf_sharpe_display is not None:
        wf_shc = '#00C853' if float(wf_sharpe_display) > 0.5 else ('#FFD600' if float(wf_sharpe_display) > 0 else '#D50000')
    wf_heading = 'Walk-Forward Backtest (OOS)' if wf_scope == 'selected_candidate' else 'Tree Family Diagnostics (WF OOS)'
    wf_scope_note = '' if wf_scope == 'selected_candidate' else (
        '<div style="color:var(--muted);font-size:.74rem;margin:-4px 0 10px 0">'
        'N/A for selected family. Showing tree-family proxy diagnostics only.</div>'
    )
    wf_section = (
        f'<div class="card"><h2>{wf_heading}</h2>{wf_scope_note}'
        f'<div class="metric-row"><span class="metric-label">OOS Return</span>'
        f'<span class="metric-val" style="color:{wf_c}">{_fmt_pct(wf_ret, decimals=2, signed=True)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Sharpe Ratio</span>'
        f'<span class="metric-val" style="color:{wf_shc}">{_fmt_num(wf_sharpe_display, decimals=3, signed=True)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Max Drawdown</span>'
        f'<span class="metric-val red">{_fmt_pct(wf_payload.get("wf_maxdd"), decimals=2)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Win Rate</span>'
        f'<span class="metric-val gold">{_fmt_pct(wf_payload.get("wf_winrate"), decimals=1)}</span></div></div>'
    )
    cpcv_payload = cpcv if cpcv_scope == 'selected_candidate' else tree_cpcv
    sh_mean = cpcv_payload.get('sharpe_mean')
    cpcv_p5_display = cpcv_payload.get('sharpe_p5') if cpcv_payload else cpcv_p5
    if cpcv_p5_display is not None and float(cpcv_p5_display) > 0:
        sl, sc2 = 'ROBUST', '#00C853'
    elif sh_mean is not None and float(sh_mean) > 0:
        sl, sc2 = 'MIXED', '#FFD600'
    elif cpcv_payload or cpcv_p5_display is not None:
        sl, sc2 = 'NO SKILL', '#D50000'
    else:
        sl, sc2 = 'N/A', '#8B949E'
    cpcv_heading = 'CPCV (Combinatorial Purged CV)' if cpcv_scope == 'selected_candidate' else 'Tree Family Diagnostics (CPCV)'
    cpcv_scope_note = '' if cpcv_scope == 'selected_candidate' else (
        '<div style="color:var(--muted);font-size:.74rem;margin:-4px 0 10px 0">'
        'N/A for selected family. Showing tree-family proxy diagnostics only.</div>'
    )
    cpcv_section = (
        f'<div class="card"><h2>{cpcv_heading}</h2>{cpcv_scope_note}'
        f'<div class="metric-row"><span class="metric-label">Skill</span>'
        f'<span class="metric-val" style="color:{sc2}">{sl}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Paths</span>'
        f'<span class="metric-val">{_fmt_int(cpcv_payload.get("n_paths"))}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Sharpe Mean</span>'
        f'<span class="metric-val">{_fmt_num(sh_mean, decimals=3, signed=True)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Sharpe p5</span>'
        f'<span class="metric-val" style="color:{sc2}">{_fmt_num(cpcv_p5_display, decimals=3, signed=True)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">% Paths &gt; 0</span>'
        f'<span class="metric-val">{_fmt_pct(cpcv_payload.get("pct_positive"), decimals=0)}</span></div></div>'
    )
    backtest_warning_html = ''
    backtest_secondary_note = ''
    backtest_card_class = 'card'
    backtest_source_name = str(bt_audit.get('strategy_return_source') or '')
    if backtest_source_name == 'selected_candidate_holdout_backtest':
        backtest_heading = 'Selected-Candidate Static Backtest (Holdout Audit)'
    elif backtest_source_name == 'static_holdout_backtest':
        backtest_heading = 'Static Backtest (Holdout Audit)'
    else:
        backtest_heading = 'Static Backtest (In-Sample Audit)'
    audit_flags = list(bt_audit.get('sanity_flags', []) or [])
    if bt_audit.get('sanity_status') == 'warning':
        backtest_card_class = 'card secondary-card'
        backtest_warning_html = (
            '<div class="warning-banner compact">'
            'Warning: static backtest metrics look extreme and should be treated as an audit surface, not deployment evidence.'
            '</div>'
        )
        backtest_secondary_note = (
            '<div style="color:var(--muted);font-size:.74rem;margin:-4px 0 10px 0">'
            'Primary robustness evidence is the walk-forward and CPCV cards below.'
            '</div>'
        )
    audit_flag_text = ', '.join(audit_flags) if audit_flags else 'none'
    backtest_audit_section = (
        f'<div class="card"><h2>Static Backtest Audit</h2>'
        f'<div class="metric-row"><span class="metric-label">Sanity status</span>'
        f'<span class="metric-val" style="color:{"#D50000" if bt_audit.get("sanity_status") == "warning" else "#00C853"}">'
        f'{bt_audit.get("sanity_status", "N/A").upper() if bt_audit.get("sanity_status") else "N/A"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Return source</span>'
        f'<span class="metric-val">{bt_audit.get("strategy_return_source") or "N/A"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Sizing method</span>'
        f'<span class="metric-val">{bt_audit.get("sizing_method") or "N/A"}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Trade density</span>'
        f'<span class="metric-val">{_fmt_pct(bt_audit.get("trade_density"), decimals=1)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Avg abs position</span>'
        f'<span class="metric-val">{_fmt_num(bt_audit.get("avg_abs_position"), decimals=3)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Strategy vs buy-hold</span>'
        f'<span class="metric-val">{_fmt_num(bt_audit.get("strategy_return_vs_buyhold_ratio"), decimals=2, signed=True)}</span></div>'
        f'<div class="metric-row"><span class="metric-label">Sanity flags</span>'
        f'<span class="metric-val">{audit_flag_text}</span></div></div>'
    )
    if explain:
        top_local = explain.get('top_local', []) or []
        top_global = explain.get('top_global', []) or []
        local_rows = ''.join(
            f'<tr><td>{row.get("feature","")}</td><td class="right">{row.get("value",0):+.4f}</td></tr>'
            for row in top_local[:5]
        )
        global_rows = ''.join(
            f'<tr><td>{row.get("feature","")}</td><td class="right">{row.get("abs_value",0):.4f}</td></tr>'
            for row in top_global[:5]
        )
        local_rows_html = local_rows or '<tr><td style="color:var(--muted)">No local explanation</td></tr>'
        global_rows_html = global_rows or '<tr><td style="color:var(--muted)">No global explanation</td></tr>'
        explain_section = (
            f'<div class="grid g2">'
            f'<div class="card"><h2>Explainability — Local Drivers</h2>'
            f'<div class="metric-row"><span class="metric-label">Method</span>'
            f'<span class="metric-val">{explain.get("method","N/A")}</span></div>'
            f'<table>{local_rows_html}</table></div>'
            f'<div class="card"><h2>Explainability — Global Features</h2>'
            f'<table>{global_rows_html}</table></div>'
            f'</div>'
        )

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{ticker} — ML Analysis Dashboard v13.0</title>
<style>
  :root{{--dark:#0D1117;--panel:#161B22;--border:#30363D;--text:#E6EDF3;--muted:#8B949E;
         --green:#00C853;--red:#D50000;--blue:#58A6FF;--gold:#FFD600;--orange:#FF9800;--purple:#CE93D8;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--dark);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px;padding:20px 28px;}}
  h1{{font-size:1.7rem;font-weight:800;margin-bottom:4px;}}
  h2{{font-size:0.88rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;}}
  .subtitle{{color:var(--muted);font-size:.8rem;margin-bottom:4px;}}
  .meta{{color:var(--muted);font-size:.72rem;margin-bottom:20px;}}
  .grid{{display:grid;gap:14px;margin-bottom:14px;}}
  .g2{{grid-template-columns:1fr 1fr;}} .g3{{grid-template-columns:1fr 1fr 1fr;}} .g4{{grid-template-columns:repeat(4,1fr);}}
  @media(max-width:900px){{.g4,.g3,.g2{{grid-template-columns:1fr;}}}}
  .card{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px 18px;}}
  .signal-card{{border-color:{sig_color}40;}}
  .signal-badge{{font-size:3rem;font-weight:900;color:{sig_color};letter-spacing:.04em;margin:8px 0;}}
  .conf{{font-size:.85rem;color:var(--muted);margin-bottom:6px;}}
  .tag{{display:inline-block;font-size:.68rem;padding:3px 8px;border-radius:12px;margin:3px 2px;font-weight:700;}}
  .model-tag{{background:#1c2a3a;color:var(--blue);border:1px solid var(--blue)40;}}
  .gpu-tag{{background:#1c2a1c;color:var(--green);border:1px solid var(--green)40;}}
  .metric-row{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border);}}
  .metric-row:last-child{{border-bottom:none;}}
  .metric-label{{color:var(--muted);font-size:.80rem;}}
  .metric-val{{font-weight:700;font-size:.85rem;color:var(--text);}}
  .metric-val.green{{color:var(--green);}} .metric-val.red{{color:var(--red);}}
  .metric-val.blue{{color:var(--blue);}}   .metric-val.gold{{color:var(--gold);}}
  .prob-row{{display:flex;align-items:center;gap:8px;margin:5px 0;}}
  .prob-label{{width:42px;font-size:.75rem;font-weight:700;}}
  .prob-bar-wrap{{flex:1;background:var(--dark);border-radius:4px;height:14px;}}
  .prob-bar{{height:14px;border-radius:4px;}}
  .prob-val{{width:44px;text-align:right;font-size:.75rem;color:var(--muted);}}
  ul{{padding-left:14px;}} li{{padding:4px 0;font-size:.82rem;border-bottom:1px solid var(--border);color:var(--muted);}}
  li:last-child{{border-bottom:none;}}
  table{{width:100%;border-collapse:collapse;}}
  td{{padding:6px 4px;font-size:.80rem;border-bottom:1px solid var(--border);color:var(--muted);}}
  .right{{text-align:right;font-weight:700;color:var(--gold);}}
  .acc{{color:var(--blue)!important;}}
  .green{{color:var(--green);}} .red{{color:var(--red);}} .gold{{color:var(--gold);}}
  img{{width:100%;border-radius:8px;border:1px solid var(--border);}}
  .score{{font-size:3.5rem;font-weight:800;}}
  .section-divider{{height:1px;background:var(--border);margin:8px 0 16px;}}
  .warning-banner{{background:#3d2300;border:1px solid #FF9800;border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:.82rem;color:#ffd699;}}
  .warning-banner.compact{{margin:0 0 12px 0;padding:8px 10px;font-size:.75rem;}}
  .secondary-card{{border-color:#FF980040;box-shadow:inset 0 0 0 1px #FF980020;}}
  .debug-footer{{margin-top:18px;padding:12px;border:1px solid var(--border);border-radius:8px;background:#11151d;color:var(--muted);font-size:.72rem;line-height:1.6;}}
  .disclaimer{{background:#0d0d17;border:1px solid var(--border);border-radius:8px;padding:12px;font-size:.75rem;color:var(--muted);margin-top:20px;}}
</style></head><body>
<h1>{ticker} <span style="color:{sig_color}">Analysis Dashboard</span></h1>
<p class="subtitle">ML + Modern DL + Monte Carlo + Fundamentals  v13.0</p>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Compute: {gpu_info} &nbsp;|&nbsp; Runtime: {runtime}</p>
{error_banner}
{consistency_banner}
{invariant_banner}
<div class="grid g4">
  <div class="card signal-card">
    <h2>Final ML Signal</h2>
    <div class="signal-badge">{sig_str}</div>
    <div class="conf">{conf_text}</div>
    <div class="tag model-tag">Deployment: {model}</div>
    <div class="tag" style="background:#161b22;color:{exec_color};border:1px solid {exec_color}40">{execution_status}</div>
    <div class="tag gpu-tag">&#9654; GPU Accelerated</div>
    <div class="tag" style="background:#1a1a2e;color:#CE93D8;border:1px solid #CE93D840">
      &#9200; {regime_inf.get('predict_days','?')}d &middot; {regime_inf.get('speed','?')}</div>
    <div class="metric-row" style="margin-top:12px"><span class="metric-label">Execution gate</span><span class="metric-val">{execution_gate}</span></div>
    <div class="metric-row"><span class="metric-label">Deployable</span><span class="metric-val" style="color:{"#00C853" if deployment_eligible else "#D50000"}">{"YES" if deployment_eligible else "NO"}</span></div>
    <div class="metric-row"><span class="metric-label">Selection status</span><span class="metric-val">{selection_status}</span></div>
    <div class="metric-row"><span class="metric-label">Reference model</span><span class="metric-val">{reference_model_used or "N/A"}</span></div>
    <div class="metric-row"><span class="metric-label">Deployment model</span><span class="metric-val">{deployment_model_used or "N/A"}</span></div>
    <div style="margin-top:14px">{prob_bars}</div>
  </div>
  <div class="{backtest_card_class}">
    <h2>{backtest_heading}</h2>
    {backtest_warning_html}
    {backtest_secondary_note}
    <div class="metric-row"><span class="metric-label">Strategy Return</span><span class="metric-val {_strat_cls}">{_fmt_pct(bt.get('strat_return'), decimals=2, signed=True)}</span></div>
    <div class="metric-row"><span class="metric-label">Buy &amp; Hold</span><span class="metric-val {_bh_cls}">{_fmt_pct(bt.get('bh_return'), decimals=2, signed=True)}</span></div>
    <div class="metric-row"><span class="metric-label">Sharpe Ratio</span><span class="metric-val blue">{_fmt_num(bt.get('strat_sharpe'), decimals=3)}</span></div>
    <div class="metric-row"><span class="metric-label">Sortino Ratio</span><span class="metric-val blue">{_fmt_num(bt.get('strat_sortino'), decimals=3)}</span></div>
    <div class="metric-row"><span class="metric-label">Max Drawdown</span><span class="metric-val red">{_fmt_pct(bt.get('strat_maxdd'), decimals=2)}</span></div>
    <div class="metric-row"><span class="metric-label">CVaR (95%)</span><span class="metric-val red">{_fmt_pct(bt.get('cvar95'), decimals=2)}</span></div>
    <div class="metric-row"><span class="metric-label">Calmar Ratio</span><span class="metric-val">{_fmt_num(bt.get('calmar'), decimals=3)}</span></div>
    <div class="metric-row"><span class="metric-label">Win Rate</span><span class="metric-val gold">{_fmt_pct(bt.get('win_rate'), decimals=1)}</span></div>
    <div class="metric-row"><span class="metric-label">Trades</span><span class="metric-val">{_fmt_int(bt.get('n_trades'))}</span></div>
  </div>
  <div class="card">
    <h2>All Model Signals</h2>
    <table><tr style="color:var(--muted);font-size:.72rem;text-transform:uppercase"><td>Model</td><td>Signal</td><td class="right">Conf</td></tr>
    {sig_table_rows}</table>
    <div class="section-divider"></div>
    <h2>Accuracy (Test Set)</h2>
    <table>{model_rows}</table>
  </div>
  <div class="card">
    <h2>Fundamental Score</h2>
    <div class="score" style="color:{comp_color}">{comp}</div>
    <div style="color:var(--muted);font-size:.78rem;margin-bottom:12px">/ 100</div>
    <div class="tag" style="background:#161b22;color:{"#FF9800" if spec_growth.get("speculative_growth_risk") else "#58A6FF"};border:1px solid {"#FF9800" if spec_growth.get("speculative_growth_risk") else "#58A6FF"}40">
      Speculative risk: {"HIGH" if spec_growth.get("speculative_growth_risk") else "LOW"}</div>
    <div class="section-divider"></div>
    {dcf_section}{graham_section}{piotr_section}
    <div style="font-size:.75rem;color:var(--muted);margin-top:10px">{context.get('thesis','')[:180]}…</div>
  </div>
</div>
<div class="grid g2">
  <div class="card"><h2>Technical + ML Backtest Charts</h2>{img_tag_analysis}</div>
  <div class="card"><h2>Monte Carlo Simulation (5 Models)</h2>{img_tag_mc}</div>
</div>
{extra_chart_cards}
<div class="grid g3">{execution_section}{conf_section}{wf_section}{cpcv_section}{mc_rel_section}{spec_growth_section}{backtest_audit_section}</div>
{explain_section}
{mc_table_section}
<div class="grid g3">
  <div class="card"><h2>Analyst Price Targets</h2><table>{analyst_rows or '<tr><td style="color:var(--muted)">No data</td></tr>'}</table></div>
  <div class="card"><h2>Key Catalysts</h2><ul>{cat_rows or '<li>No data</li>'}</ul></div>
  <div class="card"><h2>Risk Factors</h2><ul>{risk_rows or '<li>No data</li>'}</ul></div>
</div>
<div class="debug-footer">
  <div><strong>Signal JSON generated:</strong> {footer_ctx.get('signal_generated_at') or 'N/A'}</div>
  <div><strong>Dashboard generated:</strong> {footer_ctx.get('dashboard_generated_at') or 'N/A'}</div>
  <div><strong>Conformal method:</strong> {footer_ctx.get('conformal_method') or 'N/A'}</div>
  <div><strong>Reference model:</strong> {footer_ctx.get('reference_model') or 'N/A'}</div>
  <div><strong>Deployment model:</strong> {footer_ctx.get('deployment_model') or 'N/A'}</div>
  <div><strong>Schema version:</strong> {footer_ctx.get('schema_version') or 'N/A'}</div>
</div>
<div class="disclaimer"><strong>DISCLAIMER:</strong> For educational and research purposes only. Not financial advice.
ML models can and do fail. Past performance does not guarantee future results.</div>
</body></html>"""

    path = out_dir / f"{ticker}_dashboard.html"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  Dashboard saved --> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────────────────
def run_debug_benchmark(watchlist_path: str | None = None, n_workers: int = 1, gpu_jobs: int = 1):
    global TELEGRAM_FAILURE_LOG_PATH
    watchlist_path = watchlist_path or _default_benchmark_watchlist()
    effective_workers, scheduling_note = _resolve_benchmark_workers(n_workers, gpu_jobs)
    tickers = load_watchlist(watchlist_path)
    print()
    print("=" * 60)
    print("  DEBUG MODE  —  BENCHMARK")
    print("=" * 60)
    print(f"  Watchlist : {watchlist_path}")
    print(f"  Workers   : {effective_workers}  (requested={max(1, int(n_workers or 1))}, gpu_jobs={max(1, int(gpu_jobs or 1))})")
    if scheduling_note:
        print(f"  Scheduler : {scheduling_note}")
    print()

    results = {}
    failed = []
    TELEGRAM_FAILURE_LOG_PATH = Path("reports") / "run_all_error.log"
    progress_session = _progress_start_or_update(
        run_name="Benchmark run",
        phase_index=1,
        phase_total=3,
        phase_label="ML analyzer",
        completed=0,
        total=len(tickers),
        status="Starting",
    )
    batch_progress = _new_batch_progress(len(tickers), "benchmark")
    if effective_workers > 1:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(_run_timed_ticker_task, t, _run_ticker_analyzer_only, True, "Benchmark run", i, len(tickers)): t
                for i, t in enumerate(tickers, 1)
            }
            bar = tqdm(as_completed(futures), total=len(futures),
                       desc="  Benchmark", unit="ticker", ncols=70, file=sys.stdout)
            for fut in bar:
                ticker = futures[fut]
                task = fut.result()
                results[ticker] = task.get("result") or {"ML": {"elapsed": 0.0, "success": False}}
                ok = task.get("error") is None and all(v["success"] for v in results[ticker].values())
                _record_batch_completion(batch_progress, task, ok, results[ticker], bar=bar)
                if not ok:
                    failed.append(ticker)
                # (removed dead code: bar.set_postfix(last=ticker, ok='✓' if ok else '✗'))
            bar.close()
    else:
        for i, ticker in enumerate(tickers, 1):
            task = _run_timed_ticker_task(ticker, _run_ticker_analyzer_only, True, "Benchmark run", i, len(tickers))
            print(
                f"  [{i}/{len(tickers)}] Benchmarking {ticker} "
                f"(start {task['started_label']} | batch {_fmt_time(_batch_elapsed(batch_progress))})..."
            )
            results[ticker] = task.get("result") or {"ML": {"elapsed": 0.0, "success": False}}
            ok = task.get("error") is None and all(v["success"] for v in results[ticker].values())
            _record_batch_completion(batch_progress, task, ok, results[ticker])
            # (old inline print — replaced by _record_batch_completion)
            if not ok:
                failed.append(ticker)

    _print_batch_summary(batch_progress)
    stock_data = {ticker: _load_stock_json(ticker) for ticker in tickers}
    elapsed_summary = _batch_elapsed_summary(batch_progress)
    payload = _evaluate_quality_gate(watchlist_path, stock_data, elapsed_summary=elapsed_summary)
    gate = payload.get("quality_gate", {})
    metrics = gate.get("metrics", {})
    coverage = gate.get("coverage", {})
    print()
    print(f"  Status           : {gate.get('status', 'N/A')}")
    print(f"  Success rate     : {metrics.get('success_rate', 0):.0%}")
    print(f"  Signal coverage  : {(coverage.get('signal_json', {}) or {}).get('count', 0)}/{len(tickers)}")
    print(f"  Diag coverage    : {(coverage.get('diagnostics_json', {}) or {}).get('count', 0)}/{len(tickers)}")
    print(f"  Median WF Sharpe : {metrics.get('median_wf_sharpe', 'N/A')}")
    print(f"  Median ECE       : {metrics.get('median_ece', 'N/A')}")
    print(f"  Median BUY recall: {metrics.get('median_buy_recall', 'N/A')}")
    reasons = gate.get("primary_failure_reasons", []) or []
    if reasons:
        print(f"  Main issues      : {', '.join(reasons[:4])}")
    print(f"  Total elapsed    : {elapsed_summary.get('total_elapsed_human', 'N/A')}")
    print(f"  Artifact         : reports/benchmark_quality.json")
    if failed:
        print(f"  Failed tickers   : {', '.join(sorted(set(failed)))}")
    if failed:
        failed_log = Path("reports") / failed[0] / f"{failed[0]}_master.log"
        _notify_run_failure("benchmark", failed_log if failed_log.exists() else TELEGRAM_FAILURE_LOG_PATH, progress_session=progress_session)
    else:
        _notify_run_success("benchmark", progress_session=progress_session)


def run_debug_diagnostic(ticker: str):
    global TELEGRAM_FAILURE_LOG_PATH
    print()
    print("=" * 60)
    print("  DEBUG MODE  —  DIAGNOSTIC")
    print("=" * 60)
    _validate_ticker(ticker)
    log_path = Path("reports") / ticker / f"{ticker}_master.log"
    TELEGRAM_FAILURE_LOG_PATH = log_path
    progress_session = _progress_start_or_update(
        run_name=f"Diagnostic run: {ticker}",
        phase_index=1,
        phase_total=3,
        phase_label="ML analyzer",
        ticker=ticker,
        status="Starting",
    )
    elapsed, success = run_module("[ML-DEBUG] Analyzer diagnostics", "analyzer.py",
                                  ticker, log_path=log_path, silent=False,
                                  env=_progress_child_env(f"Diagnostic run: {ticker}", 1, 3, "ML analyzer", ticker))
    print()
    print(f"  Analyzer status : {'OK' if success else 'FAILED'}")
    print(f"  Elapsed         : {_fmt_time(elapsed)}")
    print(f"  Run log         : reports/{ticker}/{ticker}_run.log")
    print(f"  Signal JSON     : reports/{ticker}/{ticker}_signal.json")
    print(f"  Diagnostics JSON: reports/{ticker}/{ticker}_diagnostics.json")
    if success:
        _notify_run_success("single", ticker=ticker, progress_session=progress_session)
    else:
        _notify_run_failure(f"{ticker}-diagnostic", Path("reports") / ticker / f"{ticker}_master.log", progress_session=progress_session)


def run_debug_audit(reports_dir: str = "reports"):
    print()
    print("=" * 60)
    print("  DEBUG MODE  -  REPO AUDIT")
    print("=" * 60)
    print(f"  Reports dir: {reports_dir}")
    print()

    audit = build_repo_debug_audit(reports_dir)
    reports_root = Path(reports_dir)
    json_path = reports_root / "repo_debug_audit.json"
    md_path = reports_root / "repo_debug_audit.md"
    _write_json(json_path, audit)
    md_path.write_text(render_repo_debug_audit_markdown(audit), encoding="utf-8")

    summary = audit.get("summary", {}) or {}
    artifact_summary = audit.get("artifact_summary", {}) or {}
    print(f"  Reports scanned       : {summary.get('report_count', 0)}")
    print(f"  Defects logged        : {summary.get('defect_count', 0)}")
    print(f"  Latest success        : {audit.get('latest_successful_ticker') or 'N/A'}")
    print(f"  Focus tickers         : {', '.join(audit.get('focus_tickers', []) or []) or 'N/A'}")
    print(f"  Invariant statuses    : {artifact_summary.get('invariant_status_frequency', {})}")
    print(f"  Issue frequency       : {artifact_summary.get('issue_frequency', {})}")
    print(f"  JSON artifact         : {json_path}")
    print(f"  Markdown artifact     : {md_path}")


def ask_multi_mode() -> tuple:
    path = input("  Watchlist file [watchlist.txt]: ").strip() or "watchlist.txt"
    advanced = (input("  Advanced settings? [y/N]: ").strip().lower() == 'y')
    run_raw = "1"
    optimizer = DEFAULT_OPTIMIZER
    benchmark = False
    if advanced:
        run_raw = input("  Run type [1=standard, 2=panel]: ").strip() or "1"
        optimizer = input(f"  Optimizer [{DEFAULT_OPTIMIZER}]: ").strip() or DEFAULT_OPTIMIZER
        benchmark = (input("  Run benchmark quality gate? [y/N]: ").strip().lower() == 'y')
    else:
        print(f"  Using defaults: standard mode + {DEFAULT_OPTIMIZER} optimizer")
    return 'portfolio', {
        'watchlist_path': path,
        'run_type': 'panel' if run_raw == '2' else 'standard',
        'optimizer': optimizer,
        'benchmark': benchmark,
    }


def ask_debug_mode():
    while True:
        print()
        print("  [1] Benchmark mode")
        print("  [2] Diagnostic mode")
        print("  [3] Repo audit")
        print("  [0] Back")
        print()
        choice = input("  Select debug mode [1/2/3/0]: ").strip() or "0"
        if choice == "1":
            path = input(f"  Benchmark watchlist [{_default_benchmark_watchlist()}]: ").strip() or _default_benchmark_watchlist()
            workers = int(input("  Workers [1]: ").strip() or "1")
            return 'debug_benchmark', {'watchlist_path': path, 'n_workers': workers}
        if choice == "2":
            print()
            print("  Examples: AAPL  TSLA  NVDA  MSFT  AMD  SBUX  PLTR")
            ticker = input("  Enter ticker: ").strip().upper() or "AAPL"
            return 'debug_diagnostic', {'ticker': ticker}
        if choice == "3":
            return 'debug_audit', {'reports_dir': 'reports'}
        if choice == "0":
            return None, None


def ask_mode() -> tuple:
    while True:
        print()
        print("=" * 60)
        print("  Stock ML Analyzer  V.0.7.0")
        print("  ML + Modern DL + Monte Carlo + Fundamentals")
        print("=" * 60)
        print()
        print("  [1] Single-stock deep-dive")
        print("  [2] Multi-ticker scan (watchlist / portfolio)")
        print("  [3] Debug / development")
        print()
        choice = input("  Select mode [1/2/3]: ").strip() or "1"
        if choice == '1':
            print()
            print("  Examples: AAPL  TSLA  NVDA  MSFT  AMD  SBUX  PLTR")
            print()
            ticker = input("  Enter ticker: ").strip().upper() or "AAPL"
            _validate_ticker(ticker)
            return 'single', ticker
        if choice == '2':
            return ask_multi_mode()
        if choice == '3':
            mode, payload = ask_debug_mode()
            if mode is not None:
                return mode, payload


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.debug == 'benchmark':
        watchlist = args.benchmark_watchlist or args.portfolio or _default_benchmark_watchlist()
        run_debug_benchmark(watchlist, n_workers=args.parallel, gpu_jobs=args.gpu_jobs)
    elif args.debug == 'diagnostic':
        ticker = (args.ticker or "").strip().upper()
        if not ticker:
            ticker = input("Enter ticker for diagnostic mode: ").strip().upper() or "AAPL"
        run_debug_diagnostic(ticker)
    elif args.debug == 'audit':
        run_debug_audit("reports")
    elif args.portfolio:
        run_portfolio(
            args.portfolio,
            n_workers=args.parallel,
            run_type='panel' if args.panel else 'standard',
            optimizer=args.optimizer,
            benchmark=args.benchmark,
            benchmark_watchlist=args.benchmark_watchlist,
        )
    elif args.benchmark:
        watchlist = args.benchmark_watchlist or _default_benchmark_watchlist()
        run_debug_benchmark(watchlist, n_workers=args.parallel, gpu_jobs=args.gpu_jobs)
    elif args.ticker:
        _validate_ticker(args.ticker.strip().upper())
        run_single(args.ticker.strip().upper())
    else:
        mode, value = ask_mode()
        if mode == 'portfolio':
            run_portfolio(
                value['watchlist_path'],
                n_workers=1,
                run_type=value['run_type'],
                optimizer=value['optimizer'],
                benchmark=value.get('benchmark', False),
            )
        elif mode == 'debug_benchmark':
            run_debug_benchmark(
                value['watchlist_path'],
                n_workers=value.get('n_workers', 1),
                gpu_jobs=value.get('gpu_jobs', 1),
            )
        elif mode == 'debug_diagnostic':
            run_debug_diagnostic(value['ticker'])
        elif mode == 'debug_audit':
            run_debug_audit(value.get('reports_dir', 'reports'))
        else:
            run_single(value)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        err_path = _write_error_log(Path("reports") / "run_all_error.log", err)
        _notify_run_failure(TELEGRAM_RUN_NAME or "run_all", err_path, err, progress_session=TELEGRAM_PROGRESS_SESSION)
        raise
