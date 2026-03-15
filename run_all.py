# -*- coding: utf-8 -*-
"""
Stock ML Analyzer  v0.6.0
Usage:
  python run_all.py           # interactive prompt
  python run_all.py AAPL      # pass ticker directly
"""
import subprocess, sys, json, webbrowser, time, base64
from pathlib import Path
import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
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


# ─────────────────────────────────────────────────────────────────────────────
# TICKER INPUT
# ─────────────────────────────────────────────────────────────────────────────
def ask_ticker() -> str:
    if len(sys.argv) > 1:
        ticker = sys.argv[1].strip().upper()
        print(f"\n  Ticker from command line: {ticker}")
        return ticker

    print()
    print("="*54)
    print("  Stock ML Analyzer  v12.0")
    print("  ML + LSTM + Transformer + Monte Carlo + Fundamentals")
    print("="*54)
    print()
    print("  Analyzes any stock using:")
    print("    - 220+ technical & regime features")
    print("    - RF / XGBoost / LightGBM / CatBoost (grid search)")
    print("    - PyTorch BiLSTM + Transformer (GPU accelerated)")
    print("    - Meta-stacking with isotonic calibration")
    print("    - 5-model Monte Carlo (GBM/Merton/Heston/Regime/Stressed)")
    print("    - DCF + Graham Number + Piotroski F-Score")
    print()
    print("  Examples: AAPL  TSLA  NVDA  MSFT  AMD  SPY")
    print("            AMZN  META  GOOGL  SBUX  PLTR  SOFI")
    print()

    ticker = input("  Enter ticker symbol: ").strip().upper()
    if not ticker:
        print("  No ticker entered — defaulting to CLPT")
        ticker = "CLPT"

    if HAS_YF:
        print(f"\n  Validating {ticker}...")
        try:
            info  = yf.Ticker(ticker).fast_info
            price = getattr(info, 'last_price', None)
            if price is None:
                raise ValueError("No price returned")
            name = getattr(info, 'quote_type', '')
            print(f"  OK: {ticker}  |  last price = ${price:.2f}  |  type = {name}")
        except Exception as e:
            print(f"  WARNING: Could not validate {ticker}: {e}")
            ans = input("  Continue anyway? (y/n): ").strip().lower()
            if ans != 'y':
                print("  Exiting."); sys.exit(0)

    return ticker


# ─────────────────────────────────────────────────────────────────────────────
# MODULE RUNNER — NEW 4: captures return code, continues on failure
# ─────────────────────────────────────────────────────────────────────────────
def run_module(label: str, script: str, ticker: str,
               master_bar=None, log_path: Path = None) -> tuple:
    """Run a single analysis module. Returns (elapsed_sec, success_bool)."""
    sep = "─"*54
    msg = f"\n{sep}\n  {label}  [{ticker}]\n{sep}"
    if master_bar:
        tqdm.write(msg)
    else:
        print(msg)

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
                sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
                sys.stdout.flush()
                log_f.write(line)
            proc.wait()
            result = proc
    else:
        result = subprocess.run(
            [sys.executable, script, ticker],
            capture_output=False
        )
    elapsed_sec = time.time() - t0
    success     = (result.returncode == 0)

    h, rem = divmod(int(elapsed_sec), 3600)
    m, s   = divmod(rem, 60)
    t_str  = f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"
    status = f"  Done in {t_str}" if success else f"  WARNING: {script} exited with code {result.returncode}"

    if master_bar: tqdm.write(status)
    else:          print(status)

    return elapsed_sec, success


# ─────────────────────────────────────────────────────────────────────────────
# NEW 1: Base64 image embedding helper
# ─────────────────────────────────────────────────────────────────────────────
def img_to_base64(path: Path) -> str:
    """Return a data URI string for an image file, or empty string if missing."""
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('ascii')
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_dashboard(ticker: str, module_errors: dict) -> Path:
    out_dir = Path("reports") / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load JSON files ───────────────────────────────────────────────────────
    signal_path = out_dir / f"{ticker}_signal.json"
    signal = {}
    if signal_path.exists():
        try:
            with open(signal_path, encoding='utf-8') as f:
                signal = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠  {ticker}_signal.json unreadable ({e}) — ML module may have crashed")

    fund_path = out_dir / f"{ticker}_fundamentals.json"
    fund_data = {}
    if fund_path.exists():
        try:
            with open(fund_path, encoding='utf-8') as f:
                fund_data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠  {ticker}_fundamentals.json unreadable ({e})")

    mc_path = out_dir / f"{ticker}_montecarlo.json"
    mc_data = {}
    if mc_path.exists():
        try:
            with open(mc_path, encoding='utf-8') as f:
                mc_data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠  {ticker}_montecarlo.json unreadable ({e})")

    # ── Signal data ───────────────────────────────────────────────────────────
    sig     = signal.get('signal', {})
    sig_str = sig.get('signal', 'N/A')     if isinstance(sig, dict) else 'N/A'
    conf    = sig.get('confidence', 0)     if isinstance(sig, dict) else 0
    model   = sig.get('model_used', '')    if isinstance(sig, dict) else ''
    probs   = sig.get('probabilities', {}) if isinstance(sig, dict) else {}
    bt      = signal.get('backtest', {})
    gpu_info= signal.get('gpu_info', 'N/A')
    runtime = signal.get('total_runtime', 'N/A')

    comp    = fund_data.get('composite', 'N/A')
    context = fund_data.get('context',  {})
    dcf     = fund_data.get('dcf',      {})
    graham  = fund_data.get('graham',   {})
    piotr   = fund_data.get('piotroski',{})

    # v12 data sections
    conformal  = signal.get('conformal_prediction', {})
    wf_bt      = signal.get('walkforward_backtest',  {})
    cpcv       = signal.get('cpcv',                  {})
    regime_inf = signal.get('regime',                {})

    # ── Colors ────────────────────────────────────────────────────────────────
    SIGNAL_COLORS = {'BUY': '#00C853', 'SELL': '#D50000', 'HOLD': '#FFD600'}
    sig_color   = SIGNAL_COLORS.get(sig_str, '#aaa')
    try:
        comp_f     = float(comp)
        comp_color = '#00C853' if comp_f >= 65 else ('#D50000' if comp_f <= 35 else '#FFD600')
    except Exception:
        comp_color = '#aaa'
    _strat_cls = 'green' if bt.get('strat_return', 0) > 0 else 'red'
    _bh_cls    = 'green' if bt.get('bh_return',    0) > 0 else 'red'

    # ── Sub-sections ─────────────────────────────────────────────────────────
    cat_rows  = ''.join(f'<li>&#10003; {c}</li>' for c in context.get('catalysts', []))
    risk_rows = ''.join(f'<li>&#9888; {r}</li>'  for r in context.get('risks', []))
    analyst_rows = ''.join(
        f'<tr><td>{k}</td><td class="right">{v}</td></tr>'
        for k, v in context.get('analyst_targets', {}).items())
    model_accs = signal.get('model_accuracy', {})
    model_rows = ''.join(
        f'<tr><td>{k}</td><td class="right acc">{v*100:.2f}%</td></tr>'
        for k, v in sorted(model_accs.items(), key=lambda x: -x[1]))
    prob_bars = ''.join(
        f'<div class="prob-row">'
        f'<span class="prob-label">{k}</span>'
        f'<div class="prob-bar-wrap">'
        f'<div class="prob-bar" style="width:{v*100:.0f}%;background:{SIGNAL_COLORS.get(k,"#58A6FF")}"></div>'
        f'</div>'
        f'<span class="prob-val">{v*100:.1f}%</span>'
        f'</div>'
        for k, v in probs.items())

    def sig_class(s): return {'BUY':'green','SELL':'red','HOLD':'gold'}.get(s,'')
    all_signals = []
    if signal.get('tree_signal'):
        ts = signal['tree_signal']
        all_signals.append((ts.get('model_used','Tree'), ts.get('signal','?'), ts.get('confidence',0)))
    for ds in signal.get('dl_signals', []):
        all_signals.append((ds.get('model_used','DL'), ds.get('signal','?'), ds.get('confidence',0)))
    if signal.get('meta_signal'):
        ms = signal['meta_signal']
        all_signals.append((ms.get('model_used','Meta'), ms.get('signal','?'), ms.get('confidence',0)))
    sig_table_rows = ''.join(
        f'<tr><td>{n}</td>'
        f'<td class="{sig_class(s)}" style="font-weight:700">{s}</td>'
        f'<td class="right">{c*100:.1f}%</td></tr>'
        for n, s, c in all_signals)

    # NEW 2: Monte Carlo table
    mc_risk = mc_data.get('risk_summary', {})
    mc_rows = ''
    if mc_risk:
        for mname, r in mc_risk.items():
            mc_rows += (
                f'<tr><td>{mname}</td>'
                f'<td class="right">${r.get("median_1yr", 0):.2f}</td>'
                f'<td class="right" style="color:#D50000">{r.get("var95_63d", 0):.1f}%</td>'
                f'<td class="right" style="color:#D50000">{r.get("cvar95_63d", 0):.1f}%</td>'
                f'<td class="right">{r.get("prob_loss_63d", 0):.1f}%</td>'
                f'</tr>'
            )

    # NEW 3: DCF / Graham / Piotroski rows
    dcf_section = ''
    if dcf.get('available'):
        mos   = dcf.get('margin_of_safety', 0)
        color = '#00C853' if mos > 10 else ('#D50000' if mos < -10 else '#FFD600')
        dcf_section = f"""
        <div class="metric-row">
          <span class="metric-label">DCF Base Value</span>
          <span class="metric-val">${dcf.get('intrinsic_base', 0):.2f}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Margin of Safety</span>
          <span class="metric-val" style="color:{color}">{mos:+.1f}%&nbsp;{dcf.get('signal','')}</span>
        </div>"""

    graham_section = ''
    if graham.get('available'):
        mos   = graham.get('margin_of_safety', 0)
        color = '#00C853' if mos > 10 else ('#D50000' if mos < -10 else '#FFD600')
        graham_section = f"""
        <div class="metric-row">
          <span class="metric-label">Graham Number</span>
          <span class="metric-val">${graham.get('graham_number', 0):.2f}</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Graham Safety</span>
          <span class="metric-val" style="color:{color}">{mos:+.1f}%&nbsp;{graham.get('signal','')}</span>
        </div>"""

    piotr_section = ''
    if piotr.get('available'):
        sc = piotr.get('score', 0); mx = piotr.get('max_score', 9)
        pct= sc / mx * 100 if mx > 0 else 0
        color = '#00C853' if pct >= 70 else ('#D50000' if pct <= 35 else '#FFD600')
        piotr_section = f"""
        <div class="metric-row">
          <span class="metric-label">Piotroski F-Score</span>
          <span class="metric-val" style="color:{color}">{sc}/{mx}&nbsp;({piotr.get('label','')})</span>
        </div>"""

    # Error banner
    error_banner = ''
    if any(not v for v in module_errors.values()):
        failed = [k for k, v in module_errors.items() if not v]
        error_banner = f"""
        <div style="background:#3d0000;border:1px solid #D50000;border-radius:8px;
                    padding:10px 16px;margin-bottom:16px;font-size:0.82rem;color:#ff6b6b">
          ⚠ The following modules reported errors: {', '.join(failed)}.
          Results may be partial. Check the .log file in reports/{ticker}/ for details.
        </div>"""

    # NEW 1: Embed images as base64
    analysis_img = img_to_base64(out_dir / f"{ticker}_analysis.png")
    mc_img       = img_to_base64(out_dir / f"{ticker}_montecarlo.png")

    img_tag_analysis = (
        f'<img src="{analysis_img}" alt="ML Analysis">'
        if analysis_img else
        '<p style="color:var(--muted)">Chart not generated — check ML module errors.</p>')
    img_tag_mc = (
        f'<img src="{mc_img}" alt="Monte Carlo">'
        if mc_img else
        '<p style="color:var(--muted)">Chart not generated — check MC module errors.</p>')

    # ── Conformal prediction section ──────────────────────────────────────────
    conformal_section = ''
    if conformal:
        sig_ps = sig.get('prediction_set', [])
        ps_str = ' / '.join(sig_ps) if sig_ps else '—'
        singleton = sig.get('is_conformal_singleton', False)
        ps_color  = '#00C853' if singleton else ('#FFD600' if len(sig_ps) == 2 else '#D50000')
        cov_val   = conformal.get('coverage', 0)
        cov_color = '#00C853' if cov_val >= conformal.get('target_coverage', 0.9) else '#FFD600'
        conformal_section = f"""
  <div class="card">
    <h2>Conformal Prediction (RAPS)</h2>
    <div class="metric-row">
      <span class="metric-label">Prediction Set</span>
      <span class="metric-val" style="color:{ps_color}">{ps_str}{"&nbsp;&#10003;" if singleton else ""}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Set Size</span>
      <span class="metric-val">{sig.get('set_size','—')}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Coverage (target 90%)</span>
      <span class="metric-val" style="color:{cov_color}">{cov_val*100:.1f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Singleton Rate</span>
      <span class="metric-val">{conformal.get('singleton_rate',0)*100:.1f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Avg Set Size</span>
      <span class="metric-val">{conformal.get('avg_set_size',0):.2f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">qhat</span>
      <span class="metric-val">{conformal.get('qhat',0):.4f}</span>
    </div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:8px">
      Singleton &#8594; act. &nbsp; Size-2 &#8594; reduce size. &nbsp; Full set &#8594; flat.
    </div>
  </div>"""

    # ── Walk-forward backtest section ─────────────────────────────────────────
    wf_section = ''
    if wf_bt:
        wf_ret   = wf_bt.get('wf_return', 0)
        wf_color = '#00C853' if wf_ret > 0 else '#D50000'
        wf_sh    = wf_bt.get('wf_sharpe', 0)
        wf_shc   = '#00C853' if wf_sh > 0.5 else ('#FFD600' if wf_sh > 0 else '#D50000')
        wf_section = f"""
  <div class="card">
    <h2>Walk-Forward Backtest (OOS)</h2>
    <div class="metric-row">
      <span class="metric-label">OOS Return</span>
      <span class="metric-val" style="color:{wf_color}">{wf_ret*100:+.2f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Annual Return</span>
      <span class="metric-val">{wf_bt.get('wf_annual',0)*100:+.2f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Sharpe Ratio</span>
      <span class="metric-val" style="color:{wf_shc}">{wf_sh:.3f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Max Drawdown</span>
      <span class="metric-val red">{wf_bt.get('wf_maxdd',0)*100:.2f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Win Rate</span>
      <span class="metric-val gold">{wf_bt.get('wf_winrate',0)*100:.1f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Trades</span>
      <span class="metric-val">{wf_bt.get('wf_trades',0)}</span>
    </div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:8px">
      Pure OOS — model re-fits every 63 days on expanding window.
    </div>
  </div>"""

    # ── CPCV section ──────────────────────────────────────────────────────────
    cpcv_section = ''
    if cpcv:
        sh_mean  = cpcv.get('sharpe_mean', 0)
        sh_p5    = cpcv.get('sharpe_p5',   0)
        pct_pos  = cpcv.get('pct_positive', 0)
        if sh_p5 > 0:     skill_label, skill_color = 'ROBUST', '#00C853'
        elif sh_mean > 0: skill_label, skill_color = 'MIXED',  '#FFD600'
        else:             skill_label, skill_color = 'NO SKILL','#D50000'
        cpcv_section = f"""
  <div class="card">
    <h2>CPCV (Combinatorial Purged CV)</h2>
    <div class="metric-row">
      <span class="metric-label">Skill Assessment</span>
      <span class="metric-val" style="color:{skill_color}">{skill_label}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Paths</span>
      <span class="metric-val">{cpcv.get('n_paths',0)}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Sharpe Mean</span>
      <span class="metric-val">{sh_mean:+.3f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Sharpe p5</span>
      <span class="metric-val" style="color:{skill_color}">{sh_p5:+.3f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">% Paths &gt; 0</span>
      <span class="metric-val">{pct_pos*100:.0f}%</span>
    </div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:8px">
      p5 Sharpe &gt; 0 &#8594; robust skill across all paths.
    </div>
  </div>"""

    mc_table_section = ''
    if mc_rows:
        mc_table_section = f"""
<div class="grid g1" style="margin-top:0">
  <div class="card">
    <h2>Monte Carlo Risk Summary (5 Models, 63-day / 1-year)</h2>
    <table>
      <tr style="color:var(--muted);font-size:0.72rem;text-transform:uppercase">
        <td>Model</td><td class="right">Median 1yr</td>
        <td class="right">VaR 95%</td><td class="right">CVaR 95%</td>
        <td class="right">Prob Loss</td>
      </tr>
      {mc_rows}
    </table>
  </div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{ticker} — ML Analysis Dashboard v12.0</title>
<style>
  :root {{
    --dark:#0D1117; --panel:#161B22; --border:#30363D;
    --text:#E6EDF3; --muted:#8B949E;
    --green:#00C853; --red:#D50000; --blue:#58A6FF;
    --gold:#FFD600; --orange:#FF9800; --purple:#CE93D8;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--dark); color:var(--text); font-family:'JetBrains Mono',monospace;
          font-size:13px; padding:20px 28px; }}
  h1 {{ font-size:1.7rem; font-weight:800; margin-bottom:4px; }}
  h2 {{ font-size:0.88rem; font-weight:700; color:var(--muted);
        text-transform:uppercase; letter-spacing:0.06em; margin-bottom:12px; }}
  .subtitle {{ color:var(--muted); font-size:0.8rem; margin-bottom:4px; }}
  .meta {{ color:var(--muted); font-size:0.72rem; margin-bottom:20px; }}
  .grid {{ display:grid; gap:14px; margin-bottom:14px; }}
  .g1  {{ grid-template-columns:1fr; }}
  .g2  {{ grid-template-columns:1fr 1fr; }}
  .g3  {{ grid-template-columns:1fr 1fr 1fr; }}
  .g4  {{ grid-template-columns:repeat(4, 1fr); }}
  @media(max-width:900px) {{ .g4,.g3,.g2 {{ grid-template-columns:1fr; }} }}
  .card {{ background:var(--panel); border:1px solid var(--border);
           border-radius:10px; padding:16px 18px; }}
  .signal-card {{ border-color:{sig_color}40; }}
  .signal-badge {{ font-size:3rem; font-weight:900; color:{sig_color};
                   letter-spacing:0.04em; margin:8px 0; }}
  .conf {{ font-size:0.85rem; color:var(--muted); margin-bottom:6px; }}
  .tag {{ display:inline-block; font-size:0.68rem; padding:3px 8px;
          border-radius:12px; margin:3px 2px; font-weight:700; }}
  .model-tag {{ background:#1c2a3a; color:var(--blue); border:1px solid var(--blue)40; }}
  .gpu-tag   {{ background:#1c2a1c; color:var(--green);border:1px solid var(--green)40; }}
  .metric-row {{ display:flex; justify-content:space-between; align-items:center;
                 padding:5px 0; border-bottom:1px solid var(--border); }}
  .metric-row:last-child {{ border-bottom:none; }}
  .metric-label {{ color:var(--muted); font-size:0.80rem; }}
  .metric-val   {{ font-weight:700; font-size:0.85rem; color:var(--text); }}
  .metric-val.green {{ color:var(--green); }}
  .metric-val.red   {{ color:var(--red);   }}
  .metric-val.blue  {{ color:var(--blue);  }}
  .metric-val.gold  {{ color:var(--gold);  }}
  .prob-row {{ display:flex; align-items:center; gap:8px; margin:5px 0; }}
  .prob-label {{ width:42px; font-size:0.75rem; font-weight:700; }}
  .prob-bar-wrap {{ flex:1; background:var(--dark); border-radius:4px; height:14px; }}
  .prob-bar {{ height:14px; border-radius:4px; transition:width 0.3s; }}
  .prob-val {{ width:44px; text-align:right; font-size:0.75rem; color:var(--muted); }}
  ul {{ padding-left:14px; }}
  li {{ padding:4px 0; font-size:0.82rem; border-bottom:1px solid var(--border); color:var(--muted); }}
  li:last-child {{ border-bottom:none; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:6px 4px; font-size:0.80rem; border-bottom:1px solid var(--border); color:var(--muted); }}
  .right {{ text-align:right; font-family:'JetBrains Mono',monospace; font-weight:700; color:var(--gold); }}
  .acc   {{ color:var(--blue) !important; }}
  .green {{ color:var(--green); }} .red {{ color:var(--red); }} .gold {{ color:var(--gold); }}
  img {{ width:100%; border-radius:8px; border:1px solid var(--border); }}
  .score {{ font-size:3.5rem; font-weight:800; }}
  .disclaimer {{ background:#0d0d17; border:1px solid var(--border); border-radius:8px;
                 padding:12px; font-size:0.75rem; color:var(--muted); margin-top:20px; }}
  .section-divider {{ height:1px; background:var(--border); margin:8px 0 16px; }}
</style>
</head>
<body>

<h1>{ticker} <span style="color:{sig_color}">Analysis Dashboard</span></h1>
<p class="subtitle">ML + BiLSTM + Transformer + TFT + Monte Carlo + Fundamentals  v12.0</p>
<p class="meta">Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
  &nbsp;|&nbsp; Compute: {gpu_info}
  &nbsp;|&nbsp; Runtime: {runtime}</p>

{error_banner}

<!-- Row 1: Signal / Backtest / Models / Score -->
<div class="grid g4">

  <div class="card signal-card">
    <h2>Final ML Signal</h2>
    <div class="signal-badge">{sig_str}</div>
    <div class="conf">{conf*100:.0f}% confidence</div>
    <div class="tag model-tag">{model}</div>
    <div class="tag gpu-tag">&#9654; GPU Accelerated</div>
    <div class="tag" style="background:#1a1a2e;color:#CE93D8;border:1px solid #CE93D840">
      &#9200; {regime_inf.get('predict_days','?')}d horizon &middot; {regime_inf.get('speed','?')} stock
    </div>
    <div style="margin-top:14px">{prob_bars}</div>
  </div>

  <div class="card">
    <h2>Backtest Performance</h2>
    <div class="metric-row">
      <span class="metric-label">Strategy Return</span>
      <span class="metric-val {_strat_cls}">{bt.get('strat_return',0)*100:+.2f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Buy &amp; Hold</span>
      <span class="metric-val {_bh_cls}">{bt.get('bh_return',0)*100:+.2f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Sharpe Ratio</span>
      <span class="metric-val blue">{bt.get('strat_sharpe',0):.3f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Sortino Ratio</span>
      <span class="metric-val blue">{bt.get('strat_sortino',0):.3f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Max Drawdown</span>
      <span class="metric-val red">{bt.get('strat_maxdd',0)*100:.2f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">CVaR (95%)</span>
      <span class="metric-val red">{bt.get('cvar95',0)*100:.2f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Calmar Ratio</span>
      <span class="metric-val">{bt.get('calmar',0):.3f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Win Rate</span>
      <span class="metric-val gold">{bt.get('win_rate',0)*100:.1f}%</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Trades</span>
      <span class="metric-val">{bt.get('n_trades',0)}</span>
    </div>
  </div>

  <div class="card">
    <h2>All Model Signals</h2>
    <table>
      <tr style="color:var(--muted);font-size:0.72rem;text-transform:uppercase">
        <td>Model</td><td>Signal</td><td class="right">Conf</td>
      </tr>
      {sig_table_rows}
    </table>
    <div class="section-divider"></div>
    <h2>Accuracy (Test Set)</h2>
    <table>{model_rows}</table>
  </div>

  <div class="card">
    <h2>Fundamental Score</h2>
    <div class="score" style="color:{comp_color}">{comp}</div>
    <div style="color:var(--muted);font-size:0.78rem;margin-bottom:12px">/ 100</div>
    <div class="section-divider"></div>
    {dcf_section}
    {graham_section}
    {piotr_section}
    <div style="font-size:0.75rem;color:var(--muted);margin-top:10px">
      {context.get('thesis','')[:180]}…
    </div>
  </div>
</div>

<!-- Row 2: Charts -->
<div class="grid g2">
  <div class="card">
    <h2>Technical + ML Backtest Charts</h2>
    {img_tag_analysis}
  </div>
  <div class="card">
    <h2>Monte Carlo Simulation (5 Models)</h2>
    {img_tag_mc}
  </div>
</div>

<!-- Row 3: Conformal / Walk-Forward / CPCV (v12 additions) -->
<div class="grid g3">
  {conformal_section}
  {wf_section}
  {cpcv_section}
</div>

<!-- Row 4: Monte Carlo table (NEW 2) -->
{mc_table_section}

<!-- Row 5: Analysts / Catalysts / Risks -->
<div class="grid g3">
  <div class="card">
    <h2>Analyst Price Targets</h2>
    <table>{analyst_rows or '<tr><td style="color:var(--muted)">No analyst data</td></tr>'}</table>
  </div>
  <div class="card">
    <h2>Key Catalysts</h2>
    <ul>{cat_rows  or '<li>No catalyst data available</li>'}</ul>
  </div>
  <div class="card">
    <h2>Risk Factors</h2>
    <ul>{risk_rows or '<li>No risk factor data available</li>'}</ul>
  </div>
</div>

<div class="disclaimer">
  <strong>DISCLAIMER:</strong> This dashboard is for
  <strong>educational and research purposes only</strong>.
  Nothing here constitutes financial advice. ML models can and do fail.
  Past performance does not guarantee future results. DCF and Graham Number
  valuations are simplistic estimates, not professional analysis.
  Always consult a licensed financial advisor before investing.
</div>
</body>
</html>"""

    path = out_dir / f"{ticker}_dashboard.html"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  Dashboard saved --> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    run_start = time.time()
    ticker    = ask_ticker()

    modules = [
        ("ML",   "[ML]    ML Analysis + BiLSTM + Transformer", "analyzer.py"),
        ("FUND", "[FUND]  Fundamental Analysis",               "fundamental.py"),
        ("MC",   "[MC]    Monte Carlo Simulation",             "monte_carlo.py"),
    ]

    print(f"\n  Starting full analysis for: {ticker}")
    print(f"  All outputs --> reports/{ticker}/")
    print(f"  {len(modules)} modules queued\n")

    master_bar = tqdm(
        modules,
        desc="  Overall",
        unit="module",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} modules [{elapsed}<{remaining}] {postfix}",
        file=sys.stdout,
        leave=True,
        position=0,
    )

    module_times   = {}
    module_success = {}
    # FIX 2: Use a SEPARATE master log so run_all.py's tee-copy doesn't
    # duplicate every line that each child script already writes to
    # {ticker}_run.log via its own FileHandler.
    log_path = Path("reports") / ticker / f"{ticker}_master.log"

    for short, label, script in master_bar:
        master_bar.set_postfix(current=short, refresh=True)
        elapsed_sec, success  = run_module(label, script, ticker, master_bar, log_path)
        module_times[short]   = elapsed_sec
        module_success[short] = success

    master_bar.set_postfix(status="DONE", refresh=True)
    master_bar.close()

    # ── Per-module timing ────────────────────────────────────────────────────
    print()
    print("  " + "─"*44)
    print("  Module timing:")
    for name, sec in module_times.items():
        h, rem = divmod(int(sec), 3600)
        m, s   = divmod(rem, 60)
        t_str  = f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"
        bar_w  = int(sec / max(module_times.values()) * 20) if module_times else 0
        ok     = "✓" if module_success.get(name) else "✗"
        print(f"    {ok} {name:<8}  {t_str:<12}  {'#'*bar_w}")
    total_wall = time.time() - run_start
    h, rem  = divmod(int(total_wall), 3600)
    m, s    = divmod(rem, 60)
    t_total = f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"
    print(f"  {'─'*44}")
    print(f"    TOTAL     {t_total}")
    print()

    path = build_dashboard(ticker, module_success)

    try:
        webbrowser.open(f"file://{path.resolve()}")
        print("  Opening dashboard in browser...")
    except Exception:
        print(f"  --> Open manually: {path.resolve()}")

    print()
    print("="*54)
    print(f"  Done!  Reports in: reports/{ticker}/")
    print("="*54)


if __name__ == '__main__':
    main()
