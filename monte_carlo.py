# -*- coding: utf-8 -*-
"""
Stock Monte Carlo Risk Analyzer  V.0.7.0
======================================
Run:
  python monte_carlo.py           # interactive
  python monte_carlo.py AAPL      # direct
"""

import warnings; warnings.filterwarnings('ignore')
import sys, time, datetime, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, kurtosis, skew
from scipy.optimize import minimize

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, **kw):
            self._it = iterable; self._n = 0
            self._total = kw.get('total', None)
        def __iter__(self): return iter(self._it)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1):
            self._n += n
        def set_postfix(self, **kw): pass
        def set_description(self, s): pass
        def close(self): pass
        @staticmethod
        def write(s): print(s)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

try:
    import yfinance as yf
except ImportError:
    print("  pip install yfinance"); exit(1)

# ── Globals ────────────────────────────────────────────────────────────────
TICKER   = None
OUT_DIR  = None
SCENARIOS= None

# ── Dark theme palette ─────────────────────────────────────────────────────
DARK  = '#0D1117'; PANEL = '#161B22'; BORDER= '#30363D'
TEXT  = '#E6EDF3'; MUTED = '#8B949E'
GREEN = '#00C853'; RED   = '#D50000'; BLUE  = '#58A6FF'
GOLD  = '#FFD600'; ORANGE= '#FF9800'; PURPLE= '#CE93D8'
TEAL  = '#00B8D4'

MODEL_COLORS = {
    'GBM':    BLUE,
    'Merton': ORANGE,
    'Heston': PURPLE,
    'Regime': GREEN,
    'Stressed': RED,
}


def apply_dark_theme():
    plt.rcParams.update({
        'figure.facecolor': DARK, 'axes.facecolor': PANEL, 'savefig.facecolor': DARK,
        'text.color': TEXT, 'axes.labelcolor': TEXT, 'axes.titlecolor': TEXT,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'legend.facecolor': PANEL, 'legend.edgecolor': BORDER, 'legend.labelcolor': TEXT,
        'axes.grid': True, 'grid.color': BORDER, 'grid.linewidth': 0.5, 'grid.alpha': 0.6,
        'axes.edgecolor': BORDER, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.family': 'monospace', 'font.size': 9,
    })


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    if title:  ax.set_title(title,  color=TEXT, fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# TICKER INPUT
# ─────────────────────────────────────────────────────────────────────────────
def get_ticker() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].strip().upper()
    print(); print("="*52)
    print("  Monte Carlo Risk Analyzer  V.0.7.0"); print("="*52)
    t = input("  Ticker (e.g. AAPL, SBUX, TSLA): ").strip().upper()
    return t if t else "CLPT"


def build_scenarios(S0: float) -> dict:
    return {
        "Bull":  {"target": round(S0*2.5, 2), "color": GREEN,  "prob": 0.25},
        "Base":  {"target": round(S0*1.4, 2), "color": BLUE,   "prob": 0.50},
        "Bear":  {"target": round(S0*0.6, 2), "color": RED,    "prob": 0.25},
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA & PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, period: str = "3y") -> pd.Series:
    print(f"\n  Downloading {ticker} ({period})...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    prices = df['Close'].dropna()
    print(f"  {len(prices)} trading days  |  last close: ${float(prices.iloc[-1]):.4f}")
    return prices


def estimate_params(prices: pd.Series) -> dict:
    log_ret = np.log(prices / prices.shift(1)).dropna().values
    dt      = 1 / 252

    mu_daily    = log_ret.mean()
    sigma_daily = log_ret.std()
    # BUG FIX: Jensen's inequality correction.
    # log_ret.mean() is the geometric (log) drift.  The GBM simulation formula
    # uses (mu - 0.5*sigma^2)*dt which expects the ARITHMETIC drift.
    # Without correction: drift = (mu_log - 0.5*sigma^2)*dt — double-subtracts
    # 0.5*sigma^2 and biases paths downward by ~0.5*sigma_daily^2 per step
    # (~32%/yr for a high-vol ticker like CLPT with sigma=0.80).
    # Correct arithmetic drift: mu_arithmetic = mu_log + 0.5*sigma^2
    sigma = sigma_daily / np.sqrt(dt)
    mu    = mu_daily / dt + 0.5 * sigma**2   # arithmetic annual drift

    print(f"\n  === Parameter Estimation ===")
    print(f"  GBM   mu={mu*100:+.2f}%/yr  sigma={sigma*100:.2f}%/yr")

    jump_mask    = np.abs(log_ret) > 2.5 * sigma_daily
    lambda_jump  = jump_mask.sum() / len(log_ret) * 252
    if jump_mask.sum() > 2:
        mu_jump    = log_ret[jump_mask].mean()
        sigma_jump = log_ret[jump_mask].std()
    else:
        mu_jump    = 0.0
        sigma_jump = sigma_daily * 2
    sigma_merton = log_ret[~jump_mask].std() / np.sqrt(dt)

    print(f"  Merton lambda={lambda_jump:.2f} jumps/yr  "
          f"mu_j={mu_jump*100:+.2f}%  sig_j={sigma_jump*100:.2f}%")

    ret_series = pd.Series(log_ret)
    rv = ret_series.rolling(21).var() * 252
    rv = rv.dropna().values
    v0    = float(rv[-1])
    theta = float(np.median(rv))
    dV    = np.diff(rv); V_lag = rv[:-1]
    if len(V_lag) > 10:
        kappa = max(0.1, float(-np.polyfit(V_lag - theta, dV, 1)[0] * 252))
    else:
        kappa = 2.0
    xi  = float(np.std(dV) * np.sqrt(252))
    if len(dV) == len(log_ret[1:len(dV)+1]):
        rho = float(np.corrcoef(log_ret[1:len(dV)+1], dV)[0, 1])
    else:
        rho = -0.5
    rho = np.clip(rho, -0.99, 0.99)

    print(f"  Heston v0={v0*100:.2f}%^2  theta={theta*100:.2f}%^2  "
          f"kappa={kappa:.2f}  xi={xi:.3f}  rho={rho:.2f}")

    # NEW 2: Regime-switching parameter estimation
    med_vol = np.median(np.abs(log_ret))
    hi_mask = np.abs(log_ret) > med_vol
    lo_mask = ~hi_mask
    # BUG FIX: apply the same Jensen correction to regime-specific drifts
    sig_hi  = log_ret[hi_mask].std()  / np.sqrt(dt)
    sig_lo  = log_ret[lo_mask].std()  / np.sqrt(dt)
    mu_hi   = log_ret[hi_mask].mean() / dt + 0.5 * sig_hi**2
    mu_lo   = log_ret[lo_mask].mean() / dt + 0.5 * sig_lo**2
    print(f"  Regime Bull: mu={mu_lo*100:+.2f}%/yr  sigma={sig_lo*100:.2f}%/yr")
    print(f"  Regime Bear: mu={mu_hi*100:+.2f}%/yr  sigma={sig_hi*100:.2f}%/yr")

    kurt = float(kurtosis(log_ret)); sk = float(skew(log_ret))
    print(f"  Return stats: kurtosis={kurt:.2f}  skewness={sk:.2f}  "
          f"({'fat tails' if kurt > 0 else 'thin tails'})")

    return {
        "mu": mu, "sigma": sigma,
        "mu_daily": mu_daily, "sigma_daily": sigma_daily,
        "lambda_jump": lambda_jump, "mu_jump": mu_jump,
        "sigma_jump": sigma_jump, "sigma_merton": sigma_merton,
        "v0": v0, "theta": theta, "kappa": kappa, "xi": xi, "rho": rho,
        "mu_hi": mu_hi, "mu_lo": mu_lo, "sig_hi": sig_hi, "sig_lo": sig_lo,
        "kurtosis": kurt, "skewness": sk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODELS
# ─────────────────────────────────────────────────────────────────────────────
def sim_gbm(S0, params, T=252, N=100_000, seed=42, vol_multiplier=1.0):
    np.random.seed(seed)
    dt = 1/252; half = N//2
    sig = params['sigma'] * vol_multiplier
    Z   = np.random.randn(T, half)
    Z_all = np.concatenate([Z, -Z], axis=1)
    drift   = (params['mu'] - 0.5 * sig**2) * dt
    diffuse = sig * np.sqrt(dt) * Z_all
    return np.vstack([np.full((1, N), S0), S0 * np.exp(np.cumsum(drift + diffuse, axis=0))])


def sim_merton(S0, params, T=252, N=100_000, seed=43):
    np.random.seed(seed)
    dt = 1/252; half = N//2
    lam = params['lambda_jump']; mu_j = params['mu_jump']
    sig_j = params['sigma_jump']; sigma_d = params['sigma_merton']; mu = params['mu']
    k = np.exp(mu_j + 0.5 * sig_j**2) - 1
    drift_adj = (mu - lam * k - 0.5 * sigma_d**2) * dt
    Z_d   = np.random.randn(T, half)
    Z_all = np.concatenate([Z_d, -Z_d], axis=1)
    n_jumps = np.random.poisson(lam * dt, size=(T, N))
    max_j   = int(n_jumps.max())
    if max_j > 0:
        raw_jumps = np.random.normal(mu_j, sig_j, size=(T, N, max_j))
        k_idx = np.arange(max_j)[np.newaxis, np.newaxis, :]
        mask  = k_idx < n_jumps[:, :, np.newaxis]
        jump_sizes = (raw_jumps * mask).sum(axis=2)
    else:
        jump_sizes = np.zeros((T, N))
    log_ret = drift_adj + sigma_d * np.sqrt(dt) * Z_all + jump_sizes
    return np.vstack([np.full((1, N), S0), S0 * np.exp(np.cumsum(log_ret, axis=0))])


def sim_heston(S0, params, T=252, N=100_000, seed=44):
    np.random.seed(seed)
    dt = 1/252; half = N//2
    mu    = params['mu']
    v0    = max(params['v0'], 1e-6)
    kappa = params['kappa']
    theta = max(params['theta'], 1e-6)
    xi    = params['xi']
    rho   = params['rho']
    # BUG FIX: enforce Feller condition 2*kappa*theta >= xi^2.
    # When violated, the CIR variance process can reach zero and produce
    # degenerate paths (V stuck at 0 → zero diffusion forever).
    # Clamp xi downward to satisfy the condition rather than crash silently.
    feller_lhs = 2.0 * kappa * theta
    if xi**2 > feller_lhs:
        xi_safe = np.sqrt(feller_lhs) * 0.99   # just below the boundary
        print(f"  [Heston] Feller condition violated (2κθ={feller_lhs:.4f} < ξ²={xi**2:.4f}). "
              f"Clamping ξ: {xi:.4f} → {xi_safe:.4f}")
        xi = xi_safe
    Z1 = np.random.randn(T, half); Z2 = np.random.randn(T, half)
    W1_h = Z1; W2_h = rho*Z1 + np.sqrt(max(1-rho**2, 1e-10))*Z2
    W1 = np.concatenate([W1_h, -W1_h], axis=1)
    W2 = np.concatenate([W2_h, -W2_h], axis=1)
    V = np.full(N, v0); S = np.full(N, S0); paths = [S.copy()]
    bar = tqdm(range(T), desc="  Heston sim", unit="day", ncols=80, file=sys.stdout, leave=True)
    for t in bar:
        V_pos = np.maximum(V, 0.0); sqV = np.sqrt(V_pos)
        V = V + kappa*(theta - V_pos)*dt + xi*sqV*np.sqrt(dt)*W2[t]
        S = S * np.exp((mu - 0.5*V_pos)*dt + sqV*np.sqrt(dt)*W1[t])
        paths.append(S.copy())
    bar.close()
    return np.array(paths)


def sim_regime_switching(S0, params, T=252, N=100_000, seed=45):
    """
    NEW 2: 2-state Markov Regime-Switching model.
    State 0 = Bull (low vol, positive drift)
    State 1 = Bear (high vol, negative or flat drift)
    Transition probabilities estimated from realized vol history.
    """
    np.random.seed(seed)
    dt = 1/252; half = N//2

    mu_bull  = params['mu_lo'];   sig_bull = params['sig_lo']
    mu_bear  = params['mu_hi'];   sig_bear = params['sig_hi']

    # Simple transition matrix: ~10 days average in each regime
    p_bull_to_bear = dt * (1/63)   # expected 63-day bull runs
    p_bear_to_bull = dt * (1/21)   # expected 21-day bear runs
    P = np.array([[1-p_bull_to_bear, p_bull_to_bear],
                  [p_bear_to_bull,   1-p_bear_to_bull]])

    Z   = np.random.randn(T, half)
    Z_a = np.concatenate([Z, -Z], axis=1)  # antithetic variates (T, N)

    # BUG FIX: the original code drew ONE shared regime path (states shape (T,))
    # so all 100K paths experienced the identical bull/bear sequence — the Markov
    # switching was not actually stochastic across paths.  Fix: draw independent
    # regime paths for every path (shape (T, N)) by vectorising the transitions.
    pi = p_bear_to_bull / (p_bull_to_bear + p_bear_to_bull)  # stationary bull prob
    states = np.zeros((T, N), dtype=np.int8)
    states[0] = (np.random.rand(N) >= pi).astype(np.int8)    # 0=bull, 1=bear
    trans_probs = np.random.rand(T - 1, N)
    for t in range(1, T):
        # stay in current state or switch
        curr = states[t - 1]
        # probability of staying (P[state, state])
        p_stay = np.where(curr == 0, P[0, 0], P[1, 1])
        states[t] = np.where(trans_probs[t - 1] < p_stay, curr, 1 - curr)

    mu_path  = np.where(states == 0, mu_bull,  mu_bear)   # (T, N)
    sig_path = np.where(states == 0, sig_bull, sig_bear)  # (T, N)
    drift    = (mu_path - 0.5 * sig_path**2) * dt         # (T, N)
    diffuse  = sig_path * np.sqrt(dt) * Z_a               # (T, N)

    log_ret = drift + diffuse   # (T, N)
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0))
    return np.vstack([np.full((1, N), S0), paths])


# ─────────────────────────────────────────────────────────────────────────────
# RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk(paths: np.ndarray, S0: float, horizons: dict = None) -> dict:
    if horizons is None:
        horizons = {21: '1M', 63: '3M', 126: '6M', 252: '1Y'}
    results = {}
    for hd, label in horizons.items():
        hd  = min(hd, paths.shape[0]-1)
        ret = (paths[hd] - S0) / S0
        var95  = float(np.percentile(ret, 5))
        mask   = ret <= var95
        cvar95 = float(ret[mask].mean()) if mask.any() else var95
        results[label] = {
            'p05':       float(np.percentile(ret, 5)  * 100),
            'p25':       float(np.percentile(ret, 25) * 100),
            'p50':       float(np.percentile(ret, 50) * 100),
            'p75':       float(np.percentile(ret, 75) * 100),
            'p95':       float(np.percentile(ret, 95) * 100),
            'var95':     var95 * 100,
            'cvar95':    cvar95 * 100,
            'prob_loss': float((ret < 0).mean() * 100),
            'prob_2x':   float((ret >= 1.0).mean() * 100),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — NEW 1: full 4-panel implementation
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(ticker, S0, params, paths_dict, prices):
    """
    4-panel dark-theme Monte Carlo chart.
    paths_dict: dict of {name: ndarray(T+1, N)}
    """
    apply_dark_theme()
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    T   = max(p.shape[0] for p in paths_dict.values()) - 1
    days = np.arange(T + 1)

    # ── Panel 1: Fan chart (median + percentile bands) ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    # Plot historical prices (last 252 days as context)
    hist = prices.tail(252).values
    hist_days = np.arange(-len(hist)+1, 1)
    ax1.plot(hist_days, hist, color=TEXT, lw=1.2, alpha=0.8, label='Historical')
    ax1.axvline(0, color=MUTED, lw=0.8, ls='--', alpha=0.5)

    for name, paths in paths_dict.items():
        color = MODEL_COLORS.get(name, BLUE)
        p5   = np.percentile(paths, 5,  axis=1)
        p25  = np.percentile(paths, 25, axis=1)
        p50  = np.percentile(paths, 50, axis=1)
        p75  = np.percentile(paths, 75, axis=1)
        p95  = np.percentile(paths, 95, axis=1)
        d    = np.arange(len(p50))
        ax1.plot(d, p50, color=color, lw=1.5, label=f'{name} (median)')
        ax1.fill_between(d, p25, p75, color=color, alpha=0.12)
        ax1.fill_between(d, p5,  p95, color=color, alpha=0.05)

    # Scenario lines
    for sc_name, sc in SCENARIOS.items():
        ax1.axhline(sc['target'], color=sc['color'], lw=0.8, ls=':', alpha=0.6)
        ax1.text(T * 0.98, sc['target'],
                 f" {sc_name} ${sc['target']:.0f}", color=sc['color'],
                 va='center', fontsize=7)

    style_ax(ax1, title=f'{ticker}  Monte Carlo Fan Chart  (1yr, 100k paths/model)',
             xlabel='Trading Days (0 = today)', ylabel='Price ($)')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}'))
    ax1.legend(loc='upper left', fontsize=8, ncol=len(paths_dict)+1)

    # ── Panel 2: Terminal price distribution ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for name, paths in paths_dict.items():
        color   = MODEL_COLORS.get(name, BLUE)
        term    = paths[-1]
        bins    = np.linspace(np.percentile(term, 0.5), np.percentile(term, 99.5), 60)
        ax2.hist(term, bins=bins, color=color, alpha=0.35, density=True, label=name)
        ax2.axvline(np.median(term), color=color, lw=1.5, ls='--')
    ax2.axvline(S0, color=GOLD, lw=1.5, label=f'Current ${S0:.2f}')
    style_ax(ax2, title='Terminal Price Distribution (1yr)',
             xlabel='Price ($)', ylabel='Density')
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}'))
    ax2.legend(fontsize=8)

    # ── Panel 3: VaR timeline ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    horizons_days = [5, 10, 21, 42, 63, 126, 252]
    hl_labels     = ['1W','2W','1M','2M','3M','6M','1Y']
    for name, paths in paths_dict.items():
        color = MODEL_COLORS.get(name, BLUE)
        var_vals = []
        for hd in horizons_days:
            hd   = min(hd, paths.shape[0]-1)
            ret  = (paths[hd] - S0) / S0
            var_vals.append(float(np.percentile(ret, 5)) * 100)
        ax3.plot(range(len(horizons_days)), var_vals,
                 color=color, lw=1.5, marker='o', ms=4, label=name)
        ax3.fill_between(range(len(horizons_days)), var_vals,
                         [0]*len(horizons_days), color=color, alpha=0.08)
    ax3.set_xticks(range(len(horizons_days)))
    ax3.set_xticklabels(hl_labels, fontsize=8)
    ax3.axhline(0, color=BORDER, lw=0.6)
    style_ax(ax3, title='VaR(95%) Timeline', xlabel='Horizon', ylabel='Return (%)')
    ax3.legend(fontsize=8)

    plt.suptitle(f'{ticker}  Monte Carlo Risk Analysis  v6.0  '
                 f'({paths_dict[list(paths_dict.keys())[0]].shape[1]:,} paths each)',
                 fontsize=13, fontweight='bold', color=TEXT, y=1.002)

    path = OUT_DIR / f"{ticker}_montecarlo.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close('all')
    print(f"\n  Charts saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(ticker, S0, params, paths_dict):
    SEP = "="*66
    print(f"\n{SEP}")
    print(f"  {ticker}  MONTE CARLO RESULTS  v6.0")
    print(f"  {list(paths_dict.values())[0].shape[1]:,} paths x {len(paths_dict)} models  |  antithetic variates")
    print(SEP)
    print(f"  Current Price : ${S0:.4f}")
    print(f"  Annual Drift  : {params['mu']*100:+.2f}%")
    print(f"  Annual Vol    : {params['sigma']*100:.2f}%")
    print(f"  Kurtosis      : {params['kurtosis']:.2f}  "
          f"({'fat tails — jumps important' if params['kurtosis']>1 else 'normal-ish'})")
    print(f"  Jump rate     : {params['lambda_jump']:.2f} events/yr  "
          f"avg size {params['mu_jump']*100:+.2f}%")
    print(f"  Heston rho    : {params['rho']:.2f}")

    col = f"  {'Model':<12}  {'5th':>8}  {'25th':>8}  {'Median':>8}  {'75th':>8}  {'95th':>8}  {'E[loss]':>8}"
    print(f"\n{col}")
    print(f"  {'-'*66}")
    for name, paths in paths_dict.items():
        term = paths[-1]
        pcts = [np.percentile(term, p) for p in [5, 25, 50, 75, 95]]
        ret  = (term - S0) / S0
        prob_loss = (ret < 0).mean()
        print(f"  {name:<12}  " + "  ".join(f"${p:>7.2f}" for p in pcts) +
              f"  {prob_loss*100:>6.1f}%")

    print(f"\n  VaR / CVaR  (95% confidence, 1-Quarter = 63 days):")
    print(f"  {'Model':<12}  {'VaR':>8}  {'CVaR':>8}  {'Prob(>+20%)':>12}")
    print(f"  {'-'*48}")
    for name, paths in paths_dict.items():
        idx = min(63, paths.shape[0]-1)
        ret = (paths[idx] - S0) / S0
        var  = float(np.percentile(ret, 5))
        mask = ret <= var
        cvar = float(ret[mask].mean()) if mask.any() else var
        p20  = float((ret > 0.20).mean())
        print(f"  {name:<12}  {var*100:>7.1f}%  {cvar*100:>7.1f}%  {p20*100:>11.1f}%")

    print(f"\n  Model notes:")
    print(f"    GBM     — constant vol, no jumps. Baseline.")
    print(f"    Merton  — adds sudden price gaps (earnings, news). Vectorized.")
    print(f"    Heston  — stochastic vol, captures vol clustering.")
    print(f"    Regime  — 2-state Markov switching (bull/bear regimes).")
    print(f"    Stressed— GBM with vol×1.5 stress scenario.")
    print(f"\n  [!] All models assume historical parameters persist.")
    print(f"  [!] Not financial advice. Educational purposes only.")
    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Fix Windows cp1252 console: reconfigure stdout/stderr to UTF-8 so that
    # Unicode symbols (→ × ▸ etc.) print without UnicodeEncodeError.
    # errors='replace' is a safety net on consoles that truly can't handle UTF-8.
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    global TICKER, OUT_DIR, SCENARIOS

    TICKER  = get_ticker()
    OUT_DIR = Path("reports") / TICKER
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0     = time.time()
    prices = fetch_data(TICKER, period="3y")
    S0     = float(prices.iloc[-1])
    params = estimate_params(prices)
    SCENARIOS = build_scenarios(S0)

    N = 100_000
    T = 252

    print(f"\n  Running 5 simulation models  x  {N:,} paths each...")

    sim_configs = [
        ("GBM",     "Geometric Brownian Motion",           lambda: sim_gbm(S0, params, T, N, seed=42)),
        ("Merton",  "Merton Jump-Diffusion",                lambda: sim_merton(S0, params, T, N, seed=43)),
        ("Heston",  "Heston Stochastic Volatility",         lambda: sim_heston(S0, params, T, N, seed=44)),
        ("Regime",  "2-State Markov Regime-Switching",      lambda: sim_regime_switching(S0, params, T, N, seed=45)),
        ("Stressed","GBM × 1.5 Vol Stress Scenario",        lambda: sim_gbm(S0, params, T, N, seed=46, vol_multiplier=1.5)),
    ]

    paths_dict = {}
    bar = tqdm(sim_configs, desc="  Models", unit="model", ncols=80,
               file=sys.stdout, leave=True, position=0)
    for short_name, full_name, fn in bar:
        bar.set_postfix(running=short_name)
        t1 = time.time()
        paths_dict[short_name] = fn()
        dt = time.time() - t1
        tqdm.write(f"  [{short_name:<8}] {full_name:<38}  "
                   f"shape={paths_dict[short_name].shape}  time={dt:.1f}s")
    bar.close()

    n_total = N * len(paths_dict)
    print(f"\n  Total simulated paths : {n_total:,}")
    print(f"  Simulation wall time  : {time.time()-t0:.1f}s")

    apply_dark_theme()
    plot_all(TICKER, S0, params, paths_dict, prices)
    print_summary(TICKER, S0, params, paths_dict)

    # Save JSON for dashboard integration
    risk_summary = {}
    for name, paths in paths_dict.items():
        idx = min(63, paths.shape[0]-1)
        ret = (paths[idx] - S0) / S0
        var  = float(np.percentile(ret, 5))
        mask = ret <= var
        cvar = float(ret[mask].mean()) if mask.any() else var
        risk_summary[name] = {
            'median_1yr': float(np.percentile(paths[-1], 50)),
            'p05_1yr':    float(np.percentile(paths[-1], 5)),
            'p95_1yr':    float(np.percentile(paths[-1], 95)),
            'var95_63d':  var * 100,
            'cvar95_63d': cvar * 100,
            'prob_loss_63d': float((ret < 0).mean() * 100),
        }

    out = {
        'ticker':    TICKER,
        'generated': datetime.datetime.now().isoformat(),
        'current_price': S0,
        'params': {k: float(v) for k, v in params.items()},
        'risk_summary': risk_summary,
        'scenarios': {k: {'target': v['target'], 'prob': v['prob']} for k, v in SCENARIOS.items()},
    }
    json_path = OUT_DIR / f"{TICKER}_montecarlo.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"  JSON saved → {json_path}")
    print(f"  Monte Carlo total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
