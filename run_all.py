# -*- coding: utf-8 -*-
"""
Stock ML Analyzer  V.0.7.0  —  Master Runner
==========================================
Usage:
  python run_all.py                         interactive (asks single or portfolio)
  python run_all.py AAPL                    single stock deep-dive
  python run_all.py --portfolio watch.txt   portfolio scan
  python run_all.py --portfolio watch.txt --parallel 2
"""
import subprocess, sys, json, webbrowser, time, base64, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

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
        """
    )
    parser.add_argument('ticker', nargs='?', default=None,
                        help='Ticker symbol for single-stock mode')
    parser.add_argument('--portfolio', '-p', metavar='FILE',
                        help='Path to watchlist .txt file (one ticker per line)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Workers for parallel portfolio scan (default: 1)')
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


def run_module(label: str, script: str, ticker: str,
               master_bar=None, log_path: Path = None,
               silent: bool = False) -> tuple:
    """Run a single analysis module. Returns (elapsed_sec, success_bool)."""
    sep = "─" * 54
    msg = f"\n{sep}\n  {label}  [{ticker}]\n{sep}"
    if master_bar:   tqdm.write(msg)
    elif not silent: print(msg)

    t0 = time.time()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n{'='*54}\n  {label}  [{ticker}]\n{'='*54}\n")
            log_f.flush()
            proc = subprocess.Popen(
                [sys.executable, script, ticker],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1
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
            capture_output=silent
        )

    elapsed_sec = time.time() - t0
    success     = (result.returncode == 0)
    status = f"  Done in {_fmt_time(elapsed_sec)}" if success else \
             f"  WARNING: {script} exited with code {result.returncode}"
    if master_bar:   tqdm.write(status)
    elif not silent: print(status)
    return elapsed_sec, success


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


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-STOCK MODE
# ─────────────────────────────────────────────────────────────────────────────
def run_single(ticker: str):
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

    for short, label, script in master_bar:
        master_bar.set_postfix(current=short, refresh=True)
        elapsed_sec, success = run_module(label, script, ticker, master_bar, log_path)
        module_times[short]   = elapsed_sec
        module_success[short] = success

    master_bar.set_postfix(status="DONE", refresh=True)
    master_bar.close()
    _print_timing(module_times, module_success)

    path = build_single_dashboard(ticker, module_success)
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


def _run_ticker_batch(ticker: str) -> dict:
    """Run all 3 modules silently for one ticker. Returns per-module timing/success."""
    modules  = [("ML","analyzer.py"), ("FUND","fundamental.py"), ("MC","monte_carlo.py")]
    log_path = Path("reports") / ticker / f"{ticker}_master.log"
    results  = {}
    for short, script in modules:
        elapsed, success = run_module(f"[{short}] {ticker}", script,
                                      ticker, log_path=log_path, silent=True)
        results[short] = {'elapsed': elapsed, 'success': success}
    return results


def run_portfolio(watchlist_path: str, n_workers: int = 1):
    """Batch-process all tickers, then build the portfolio ranking dashboard."""
    tickers   = load_watchlist(watchlist_path)
    run_start = time.time()
    print()
    print("=" * 60)
    print(f"  Portfolio Scanner  v13.0  —  {len(tickers)} tickers")
    print("=" * 60)
    print(f"  Watchlist : {watchlist_path}")
    print(f"  Workers   : {n_workers}")
    print(f"  Output    : reports/portfolio_dashboard.html")
    print()

    all_results = {}; failed = []

    if n_workers > 1:
        print(f"  Running {len(tickers)} tickers in parallel ({n_workers} workers)...\n")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_ticker_batch, t): t for t in tickers}
            bar = tqdm(as_completed(futures), total=len(tickers),
                       desc="  Scanning", unit="ticker", ncols=70, file=sys.stdout)
            for fut in bar:
                t = futures[fut]
                try:
                    all_results[t] = fut.result()
                    ok = all(v['success'] for v in all_results[t].values())
                    bar.set_postfix(last=t, ok='✓' if ok else '✗')
                    if not ok: failed.append(t)
                except Exception:
                    failed.append(t)
                    bar.set_postfix(last=t, ok='✗')
            bar.close()
    else:
        for i, ticker in enumerate(tickers, 1):
            print(f"\n  [{i}/{len(tickers)}] Analyzing {ticker}...")
            all_results[ticker] = _run_ticker_batch(ticker)
            ok = all(v['success'] for v in all_results[ticker].values())
            ml_t = all_results[ticker].get('ML', {}).get('elapsed', 0)
            print(f"  {'✓' if ok else '✗'} {ticker:<6}  {ml_t/60:.1f}m")
            if not ok: failed.append(ticker)

    print(f"\n  Finished {len(tickers)} tickers in {_fmt_time(time.time()-run_start)}")
    if failed:
        print(f"  WARNING: errors in: {', '.join(failed)}")

    print("\n  Building portfolio dashboard...")
    path = build_portfolio_dashboard(tickers, failed)
    _open_browser(path)
    print(f"\n  Dashboard: {path}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MEAN-VARIANCE ALLOCATION (NEW 20)
# ─────────────────────────────────────────────────────────────────────────────
def _load_stock_json(ticker: str) -> dict:
    base = Path("reports") / ticker
    out  = {}
    for key, fn in [('signal_data',  f"{ticker}_signal.json"),
                    ('fund_data',    f"{ticker}_fundamentals.json"),
                    ('mc_data',      f"{ticker}_montecarlo.json")]:
        p = base / fn
        if p.exists():
            try:
                with open(p, encoding='utf-8') as f:
                    out[key] = json.load(f)
            except Exception:
                out[key] = {}
        else:
            out[key] = {}
    return out


def compute_mv_weights(tickers: list, stock_data: dict,
                       max_weight: float = 0.25,
                       min_weight: float = 0.02) -> dict:
    """
    Compute mean-variance optimal portfolio weights.

    Expected return per ticker:
      μ = 0.5 × (signal_direction × confidence)
        + 0.5 × (MC median 1yr / current_price − 1)
    Covariance: diagonal approximation using MC annual sigma.

    Falls back to signal-proportional equal allocation if scipy unavailable.
    """
    mus, vols, valid = [], [], []
    for t in tickers:
        d        = stock_data.get(t, {})
        sig_raw  = d.get('signal_data', {})
        mc_d     = d.get('mc_data', {})
        fund_d   = d.get('fund_data', {})

        sig_dict   = sig_raw.get('signal', {}) or {}
        signal_str = sig_dict.get('signal', 'HOLD') if isinstance(sig_dict, dict) else 'HOLD'
        conf       = float(sig_dict.get('confidence', 0.33) if isinstance(sig_dict, dict) else 0.33)
        direction  = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}.get(str(signal_str), 0.0)

        mc_risk = mc_d.get('risk_summary', {}) or {}
        current = float(mc_d.get('current_price', 1.0) or 1.0)
        mc_rets = [(mc_risk.get(m, {}).get('median_1yr', current) / current) - 1
                   for m in ('GBM', 'Merton', 'Heston')]
        mc_mu   = sum(mc_rets) / max(len(mc_rets), 1)

        comp     = float(fund_d.get('composite', 50) or 50)
        fund_adj = (comp - 50.0) / 100.0   # −0.5 to +0.5

        sigma = float(mc_d.get('params', {}).get('sigma', 0.30) or 0.30)

        # Combined expected return
        if direction < 0:
            mu = -abs(mc_mu) * 0.5
        elif direction > 0:
            mu = abs(mc_mu) * conf + 0.02 * fund_adj
        else:
            mu = mc_mu * 0.4

        mus.append(mu); vols.append(max(sigma, 0.05)); valid.append(t)

    if not valid:
        return {t: 1.0 / len(tickers) for t in tickers}

    n = len(valid)

    if HAS_SCIPY and HAS_NP:
        mu_arr   = np.array(mus)
        # NOTE: diagonal covariance (zero inter-asset correlations) is a
        # simplification.  A full sample covariance from historical returns
        # would require downloading price history for all tickers here.
        # The diagonal approximation still usefully penalises high-vol stocks
        # relative to each other; it just can't capture correlation benefits.
        cov_diag = np.diag(np.array(vols) ** 2)

        def neg_sharpe(w):
            w = np.array(w)
            ret = float(w @ mu_arr)
            var = float(w @ cov_diag @ w)
            return -(ret / (var ** 0.5 + 1e-8))

        w0 = np.clip([max(min_weight, abs(m) / (sum(abs(x) for x in mus) + 1e-8))
                      for m in mus], min_weight, max_weight)
        w0 = w0 / w0.sum()
        constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]
        bounds      = [(min_weight, max_weight)] * n
        try:
            res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'ftol':1e-9,'maxiter':500})
            weights = np.clip(res.x, min_weight, max_weight)
        except Exception:
            weights = w0
        # BUG FIX: always renormalize after clipping so weights sum to exactly 1.0
        weights = weights / weights.sum()
        return {t: round(float(w), 4) for t, w in zip(valid, weights)}
    else:
        pos   = [max(0.01, mu) for mu in mus]
        total = sum(pos) or 1.0          # BUG FIX: guard division-by-zero
        raw   = [min(max_weight, p / total) for p in pos]
        s     = sum(raw) or 1.0          # BUG FIX: renorm to sum exactly 1.0
        return {t: round(w / s, 4) for t, w in zip(valid, raw)}


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO DASHBOARD HTML BUILDER (NEW 19)
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio_dashboard(tickers: list, failed: list) -> Path:
    out_dir    = Path("reports"); out_dir.mkdir(exist_ok=True)
    stock_data = {t: _load_stock_json(t) for t in tickers}
    weights    = compute_mv_weights(tickers, stock_data)

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
      <br><span class="dim">{r['conf']*100:.0f}% conf · {r['horizon']}d</span></td>
  <td class="num" style="color:{comp_c}">{r['comp']}</td>
  <td class="num">{pi_s}</td>
  <td class="num">{r['best_acc']*100:.1f}%</td>
  <td class="num" style="color:{sh_c}">{r['sharpe']:.2f}</td>
  <td class="num">{wf_s}</td>
  <td class="num">{cp_s}</td>
  <td class="num" style="color:{up_c}">{r['mc_upside']:+.1f}%</td>
  <td class="num" style="color:#D50000">{r['mc_var']:.1f}%</td>
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

    error_notice = ''
    if failed:
        error_notice = (f'<div class="error-banner">⚠ {len(failed)} ticker(s) had module errors '
                        f'(partial data): {", ".join(failed)}</div>')

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
  .note{{font-size:0.70rem;color:var(--muted);margin-top:6px;}}
  .disclaimer{{background:#0d0d17;border:1px solid var(--border);border-radius:8px;
               padding:12px;font-size:0.72rem;color:var(--muted);margin-top:20px;}}
</style></head><body>

<h1>Portfolio <span style="color:var(--blue)">Scanner Dashboard</span></h1>
<p class="meta">v13.0 &nbsp;|&nbsp; {now} &nbsp;|&nbsp; {len(tickers)} tickers analysed</p>

{error_notice}

<div class="sum-row">{summary_html}</div>

<div class="card">
  <h2>Suggested Allocation — Mean-Variance Optimized (NEW 20)</h2>
  <div class="alloc-bar">{alloc_bars}</div>
  <p class="note">
    Weights from mean-variance optimization: ML signal direction × confidence,
    blended with Monte Carlo expected returns and historical volatility.
    Long-only · max 25% per position · min 2%.
  </p>
</div>

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
    print(f"  Portfolio dashboard saved → {path}")
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

    sig      = signal.get('signal', {})
    sig_str  = sig.get('signal', 'N/A')     if isinstance(sig, dict) else 'N/A'
    conf     = sig.get('confidence', 0)     if isinstance(sig, dict) else 0
    model    = sig.get('model_used', '')    if isinstance(sig, dict) else ''
    probs    = sig.get('probabilities', {}) if isinstance(sig, dict) else {}
    bt       = signal.get('backtest', {})
    gpu_info = signal.get('gpu_info', 'N/A')
    runtime  = signal.get('total_runtime', 'N/A')

    comp     = fund_data.get('composite', 'N/A')
    context  = fund_data.get('context', {})
    dcf      = fund_data.get('dcf', {})
    graham   = fund_data.get('graham', {})
    piotr    = fund_data.get('piotroski', {})

    conformal  = signal.get('conformal_prediction', {})
    wf_bt      = signal.get('walkforward_backtest', {})
    cpcv       = signal.get('cpcv', {})
    regime_inf = signal.get('regime', {})

    SC = {'BUY': '#00C853', 'SELL': '#D50000', 'HOLD': '#FFD600'}
    sig_color = SC.get(sig_str, '#aaa')
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
    img_tag_analysis = (f'<img src="{analysis_img}" alt="ML Analysis">'
                        if analysis_img else '<p style="color:var(--muted)">Chart not generated.</p>')
    img_tag_mc = (f'<img src="{mc_img}" alt="Monte Carlo">'
                  if mc_img else '<p style="color:var(--muted)">Chart not generated.</p>')

    mc_table_section = ''
    if mc_rows:
        mc_table_section = (f'<div style="margin-bottom:14px"><div class="card">'
                            f'<h2>Monte Carlo Risk Summary (5 Models, 63-day / 1-year)</h2><table>'
                            f'<tr style="color:var(--muted);font-size:0.72rem;text-transform:uppercase">'
                            f'<td>Model</td><td class="right">Median 1yr</td>'
                            f'<td class="right">VaR 95%</td><td class="right">CVaR 95%</td>'
                            f'<td class="right">Prob Loss</td></tr>{mc_rows}</table></div></div>')

    # Conformal / WF / CPCV panels
    conf_section = wf_section = cpcv_section = ''
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
                        f'<span class="metric-val">{conformal.get("singleton_rate",0)*100:.1f}%</span></div></div>')
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
  .disclaimer{{background:#0d0d17;border:1px solid var(--border);border-radius:8px;padding:12px;font-size:.75rem;color:var(--muted);margin-top:20px;}}
</style></head><body>
<h1>{ticker} <span style="color:{sig_color}">Analysis Dashboard</span></h1>
<p class="subtitle">ML + BiLSTM + Transformer + TFT + Monte Carlo + Fundamentals  v13.0</p>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Compute: {gpu_info} &nbsp;|&nbsp; Runtime: {runtime}</p>
{error_banner}
<div class="grid g4">
  <div class="card signal-card">
    <h2>Final ML Signal</h2>
    <div class="signal-badge">{sig_str}</div>
    <div class="conf">{conf*100:.0f}% confidence</div>
    <div class="tag model-tag">{model}</div>
    <div class="tag gpu-tag">&#9654; GPU Accelerated</div>
    <div class="tag" style="background:#1a1a2e;color:#CE93D8;border:1px solid #CE93D840">
      &#9200; {regime_inf.get('predict_days','?')}d &middot; {regime_inf.get('speed','?')}</div>
    <div style="margin-top:14px">{prob_bars}</div>
  </div>
  <div class="card">
    <h2>Backtest Performance</h2>
    <div class="metric-row"><span class="metric-label">Strategy Return</span><span class="metric-val {_strat_cls}">{bt.get('strat_return',0)*100:+.2f}%</span></div>
    <div class="metric-row"><span class="metric-label">Buy &amp; Hold</span><span class="metric-val {_bh_cls}">{bt.get('bh_return',0)*100:+.2f}%</span></div>
    <div class="metric-row"><span class="metric-label">Sharpe Ratio</span><span class="metric-val blue">{bt.get('strat_sharpe',0):.3f}</span></div>
    <div class="metric-row"><span class="metric-label">Sortino Ratio</span><span class="metric-val blue">{bt.get('strat_sortino',0):.3f}</span></div>
    <div class="metric-row"><span class="metric-label">Max Drawdown</span><span class="metric-val red">{bt.get('strat_maxdd',0)*100:.2f}%</span></div>
    <div class="metric-row"><span class="metric-label">CVaR (95%)</span><span class="metric-val red">{bt.get('cvar95',0)*100:.2f}%</span></div>
    <div class="metric-row"><span class="metric-label">Calmar Ratio</span><span class="metric-val">{bt.get('calmar',0):.3f}</span></div>
    <div class="metric-row"><span class="metric-label">Win Rate</span><span class="metric-val gold">{bt.get('win_rate',0)*100:.1f}%</span></div>
    <div class="metric-row"><span class="metric-label">Trades</span><span class="metric-val">{bt.get('n_trades',0)}</span></div>
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
    <div class="section-divider"></div>
    {dcf_section}{graham_section}{piotr_section}
    <div style="font-size:.75rem;color:var(--muted);margin-top:10px">{context.get('thesis','')[:180]}…</div>
  </div>
</div>
<div class="grid g2">
  <div class="card"><h2>Technical + ML Backtest Charts</h2>{img_tag_analysis}</div>
  <div class="card"><h2>Monte Carlo Simulation (5 Models)</h2>{img_tag_mc}</div>
</div>
<div class="grid g3">{conf_section}{wf_section}{cpcv_section}</div>
{mc_table_section}
<div class="grid g3">
  <div class="card"><h2>Analyst Price Targets</h2><table>{analyst_rows or '<tr><td style="color:var(--muted)">No data</td></tr>'}</table></div>
  <div class="card"><h2>Key Catalysts</h2><ul>{cat_rows or '<li>No data</li>'}</ul></div>
  <div class="card"><h2>Risk Factors</h2><ul>{risk_rows or '<li>No data</li>'}</ul></div>
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
def ask_mode() -> tuple:
    print()
    print("=" * 60)
    print("  Stock ML Analyzer  V.0.7.0")
    print("  ML + BiLSTM + TFT + Monte Carlo + Fundamentals")
    print("=" * 60)
    print()
    print("  [1] Single-stock deep-dive")
    print("  [2] Portfolio scanner (watchlist)")
    print()
    choice = input("  Select mode [1/2]: ").strip()
    if choice == '2':
        path = input("  Watchlist file [watchlist.txt]: ").strip() or "watchlist.txt"
        return 'portfolio', path
    else:
        print()
        print("  Examples: AAPL  TSLA  NVDA  MSFT  AMD  SBUX  PLTR")
        print()
        ticker = input("  Enter ticker: ").strip().upper() or "AAPL"
        _validate_ticker(ticker)
        return 'single', ticker


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.portfolio:
        run_portfolio(args.portfolio, n_workers=args.parallel)
    elif args.ticker:
        _validate_ticker(args.ticker.strip().upper())
        run_single(args.ticker.strip().upper())
    else:
        mode, value = ask_mode()
        if mode == 'portfolio': run_portfolio(value, n_workers=1)
        else:                   run_single(value)


if __name__ == '__main__':
    main()
