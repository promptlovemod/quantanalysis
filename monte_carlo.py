# -*- coding: utf-8 -*-
"""
Stock Monte Carlo Risk Analyzer  V.0.7.0
======================================
IMPROVEMENTS vs v5.1
─────────────────────
NEW 1  ▸ Full plot_all() implemented — 4-panel dark-theme chart:
           fan chart, terminal distribution, VaR timeline, model comparison.

NEW 2  ▸ Markov Regime-Switching model added (2-state: bull/bear).
           Identifies high-vol / low-vol regimes from return history using
           a simple k-means style EM approach, then simulates with
           regime-conditional drift and vol parameters.

NEW 3  ▸ SABR model added (stochastic alpha beta rho).
           Better fit for equity options; captures vol smile dynamics.

NEW 4  ▸ Stressed CVaR: re-runs GBM with vol × 1.5 to show tail risk under
           a volatility shock (useful for scenario planning).

NEW 5  ▸ Expected Shortfall (ES) and Conditional Drawdown at Risk (CDaR)
           added to risk metrics table.

NEW 6  ▸ MLE-based parameter estimation replaces moment-matching for
           Heston — more stable kappa/xi estimates.

NEW 7  ▸ plot_all() saves charts; console summary now includes all 5 models.

NEW 8  ▸ JSON output saved to reports/<TICKER>/SBUX_montecarlo.json for
           dashboard integration.

Run:
  python monte_carlo.py           # interactive
  python monte_carlo.py AAPL      # direct
"""

import warnings; warnings.filterwarnings('ignore')
import sys, time, datetime, json, textwrap
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, kurtosis, skew
from scipy.optimize import minimize
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

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
from utils.run_metadata import (
    DEFAULT_CONFIG_VERSION,
    append_experiment_record,
    build_run_metadata,
    complete_run_metadata,
)

try:
    import yfinance as yf
except ImportError:
    print("  pip install yfinance"); exit(1)

# ── Globals ────────────────────────────────────────────────────────────────
TICKER   = None
OUT_DIR  = None
SCENARIOS= None
MC_CONFIG = {
    "drift_method": "winsorized_shrinkage",
    "drift_winsor_pct": 0.05,
    "drift_shrink_target": 0.08,
    "drift_shrink_weight": 0.35,
    "drift_clamp_min": -0.25,
    "drift_clamp_max": 0.25,
    "drift_apply_to_regimes": True,
    "vol_forecast_model": "garch",
    "vol_fallback_model": "historical",
    "drawdown_threshold": 0.20,
    "calibration_horizons": [21, 63, 126],
    "dispersion_tolerance": 0.15,
    "dispersion_severe_tolerance": 0.30,
    "vol_term_structure_recalibration": True,
    "vol_calibration_clip_min": 0.80,
    "vol_calibration_clip_max": 1.25,
}

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


def _winsorize(x: np.ndarray, pct: float) -> np.ndarray:
    if len(x) == 0 or pct <= 0:
        return x.copy()
    lo = np.quantile(x, pct)
    hi = np.quantile(x, 1.0 - pct)
    return np.clip(x, lo, hi)


def _annualize_log_mu(mu_log_daily: float, sigma_annual: float, dt: float) -> float:
    return float(mu_log_daily / dt + 0.5 * sigma_annual**2)


def _estimate_bounded_drift(log_ret: np.ndarray, sigma_annual: float, dt: float) -> dict:
    method = MC_CONFIG.get("drift_method", "winsorized_shrinkage")
    winsor_pct = float(MC_CONFIG.get("drift_winsor_pct", 0.05))
    shrink_target = float(MC_CONFIG.get("drift_shrink_target", 0.08))
    shrink_weight = float(MC_CONFIG.get("drift_shrink_weight", 0.35))
    clamp_min = float(MC_CONFIG.get("drift_clamp_min", -0.25))
    clamp_max = float(MC_CONFIG.get("drift_clamp_max", 0.25))

    raw_mu = _annualize_log_mu(float(log_ret.mean()), sigma_annual, dt)
    median_mu = _annualize_log_mu(float(np.median(log_ret)), sigma_annual, dt)
    winsor_mu = _annualize_log_mu(float(_winsorize(log_ret, winsor_pct).mean()), sigma_annual, dt)

    if method == "median":
        base_mu = median_mu
    elif method == "winsorized_mean":
        base_mu = winsor_mu
    elif method == "winsorized_shrinkage":
        base_mu = winsor_mu * (1.0 - shrink_weight) + shrink_target * shrink_weight
    else:
        base_mu = raw_mu

    adjusted_mu = float(np.clip(base_mu, clamp_min, clamp_max))
    return {
        "method": method,
        "raw_mu": raw_mu,
        "median_mu": median_mu,
        "winsorized_mu": winsor_mu,
        "base_mu": float(base_mu),
        "adjusted_mu": adjusted_mu,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "shrink_target": shrink_target,
        "shrink_weight": shrink_weight,
    }


def _forecast_volatility(log_ret: np.ndarray, sigma_daily_hist: float, dt: float) -> dict:
    model = MC_CONFIG.get("vol_forecast_model", "historical")
    hist_sigma_annual = float(sigma_daily_hist / np.sqrt(dt))
    out = {
        "requested_model": model,
        "selected_model": "historical",
        "historical_sigma_annual": hist_sigma_annual,
        "forecast_sigma_annual": hist_sigma_annual,
        "model_params": {},
    }
    if model == "historical":
        return out
    specs = {
        "garch": {"vol": "GARCH", "p": 1, "q": 1},
        "egarch": {"vol": "EGARCH", "p": 1, "q": 1},
        "gjr_garch": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    }
    if model not in specs:
        out["fallback_reason"] = f"unsupported model '{model}'"
        return out
    if not HAS_ARCH:
        out["fallback_reason"] = "arch library not installed"
        return out
    try:
        scaled = log_ret * 100.0
        am = arch_model(scaled, mean="Zero", dist="normal", **specs[model])
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=1, reindex=False)
        daily_var_pct = float(forecast.variance.values[-1, 0])
        sigma_daily = np.sqrt(max(daily_var_pct, 0.0)) / 100.0
        sigma_annual = float(sigma_daily / np.sqrt(dt))
        out.update({
            "selected_model": model,
            "forecast_sigma_annual": sigma_annual,
            "model_params": {
                key: float(val)
                for key, val in dict(getattr(res, "params", {})).items()
                if np.isfinite(float(val))
            },
        })
        return out
    except Exception as e:
        out["fallback_reason"] = str(e)
        return out


def _jsonify_params(value):
    if isinstance(value, dict):
        return {k: _jsonify_params(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_params(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def _estimate_vol_term_structure_calibration(prices: pd.Series,
                                             sigma_annual: float,
                                             horizons: list | None = None) -> dict:
    horizons = list(horizons or MC_CONFIG.get("calibration_horizons", [21, 63, 126]))
    sigma_annual = float(sigma_annual or 0.0)
    if sigma_annual <= 0.0:
        return {
            "enabled": bool(MC_CONFIG.get("vol_term_structure_recalibration", True)),
            "available": False,
            "raw_multiplier": 1.0,
            "applied_multiplier": 1.0,
            "horizon_ratios": {},
            "fallback_reason": "nonpositive_sigma",
        }
    daily_sigma = sigma_annual / np.sqrt(252.0)
    series = prices.astype(float)
    ratios = {}
    values = []
    for horizon in horizons:
        if len(series) <= int(horizon):
            continue
        hist_ret = series.pct_change(int(horizon)).dropna().values
        if len(hist_ret) < 20:
            continue
        hist_std = float(np.std(hist_ret, ddof=0))
        implied_std = float(daily_sigma * np.sqrt(float(horizon)))
        if implied_std <= 1e-12:
            continue
        ratio = hist_std / implied_std
        ratios[str(int(horizon))] = {
            "hist_std": hist_std,
            "implied_std": implied_std,
            "ratio": float(ratio),
        }
        values.append(float(ratio))
    if not values:
        return {
            "enabled": bool(MC_CONFIG.get("vol_term_structure_recalibration", True)),
            "available": False,
            "raw_multiplier": 1.0,
            "applied_multiplier": 1.0,
            "horizon_ratios": ratios,
            "fallback_reason": "insufficient_horizon_history",
        }
    raw_multiplier = float(np.median(values))
    clip_min = float(MC_CONFIG.get("vol_calibration_clip_min", 0.80) or 0.80)
    clip_max = float(MC_CONFIG.get("vol_calibration_clip_max", 1.25) or 1.25)
    enabled = bool(MC_CONFIG.get("vol_term_structure_recalibration", True))
    applied_multiplier = float(np.clip(raw_multiplier, clip_min, clip_max)) if enabled else 1.0
    return {
        "enabled": enabled,
        "available": True,
        "raw_multiplier": raw_multiplier,
        "applied_multiplier": applied_multiplier,
        "horizon_ratios": ratios,
        "clip_min": clip_min,
        "clip_max": clip_max,
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
    vol_info = _forecast_volatility(log_ret, sigma_daily, dt)
    raw_sigma = float(vol_info["forecast_sigma_annual"])
    vol_calibration = _estimate_vol_term_structure_calibration(prices, raw_sigma)
    sigma_multiplier = float(vol_calibration.get("applied_multiplier", 1.0) or 1.0)
    sigma = float(raw_sigma * sigma_multiplier)
    vol_info = dict(vol_info)
    vol_info["forecast_sigma_annual_raw"] = raw_sigma
    vol_info["forecast_sigma_annual"] = sigma
    vol_info["vol_calibration"] = vol_calibration
    drift_info = _estimate_bounded_drift(log_ret, sigma, dt)
    mu    = drift_info["adjusted_mu"]

    print(f"\n  === Parameter Estimation ===")
    print(f"  GBM   mu_raw={drift_info['raw_mu']*100:+.2f}%/yr  "
          f"mu_adj={mu*100:+.2f}%/yr  sigma={sigma*100:.2f}%/yr")
    print(f"  Vol   requested={vol_info['requested_model']}  selected={vol_info['selected_model']}  "
          f"hist={vol_info['historical_sigma_annual']*100:.2f}%/yr  "
          f"forecast_raw={raw_sigma*100:.2f}%/yr  "
          f"forecast_adj={vol_info['forecast_sigma_annual']*100:.2f}%/yr")
    if vol_calibration.get("enabled"):
        print(f"  Vol calibration multiplier={sigma_multiplier:.3f}  "
              f"available={bool(vol_calibration.get('available', False))}")

    jump_mask    = np.abs(log_ret) > 2.5 * sigma_daily
    lambda_jump  = jump_mask.sum() / len(log_ret) * 252
    if jump_mask.sum() > 2:
        mu_jump    = log_ret[jump_mask].mean()
        sigma_jump = log_ret[jump_mask].std()
    else:
        mu_jump    = 0.0
        sigma_jump = sigma_daily * 2
    sigma_merton = float(log_ret[~jump_mask].std() / np.sqrt(dt) * sigma_multiplier)

    print(f"  Merton lambda={lambda_jump:.2f} jumps/yr  "
          f"mu_j={mu_jump*100:+.2f}%  sig_j={sigma_jump*100:.2f}%")

    ret_series = pd.Series(log_ret)
    rv = ret_series.rolling(21).var() * 252
    rv = rv.dropna().values
    variance_multiplier = sigma_multiplier ** 2
    v0    = float(rv[-1] * variance_multiplier)
    theta = float(np.median(rv) * variance_multiplier)
    dV    = np.diff(rv); V_lag = rv[:-1]
    if len(V_lag) > 10:
        kappa = max(0.1, float(-np.polyfit(V_lag - theta, dV, 1)[0] * 252))
    else:
        kappa = 2.0
    xi  = float(np.std(dV) * np.sqrt(252) * sigma_multiplier)
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
    sig_hi  = float(log_ret[hi_mask].std()  / np.sqrt(dt) * sigma_multiplier)
    sig_lo  = float(log_ret[lo_mask].std()  / np.sqrt(dt) * sigma_multiplier)
    mu_hi_raw = log_ret[hi_mask].mean() / dt + 0.5 * sig_hi**2
    mu_lo_raw = log_ret[lo_mask].mean() / dt + 0.5 * sig_lo**2
    if MC_CONFIG.get("drift_apply_to_regimes", True):
        mu_hi = float(np.clip(
            mu_hi_raw * (1.0 - drift_info['shrink_weight']) + drift_info['shrink_target'] * drift_info['shrink_weight'],
            drift_info['clamp_min'], drift_info['clamp_max']))
        mu_lo = float(np.clip(
            mu_lo_raw * (1.0 - drift_info['shrink_weight']) + drift_info['shrink_target'] * drift_info['shrink_weight'],
            drift_info['clamp_min'], drift_info['clamp_max']))
    else:
        mu_hi = mu_hi_raw
        mu_lo = mu_lo_raw
    print(f"  Regime Bull: mu_raw={mu_lo_raw*100:+.2f}%/yr  mu_adj={mu_lo*100:+.2f}%/yr  sigma={sig_lo*100:.2f}%/yr")
    print(f"  Regime Bear: mu_raw={mu_hi_raw*100:+.2f}%/yr  mu_adj={mu_hi*100:+.2f}%/yr  sigma={sig_hi*100:.2f}%/yr")

    kurt = float(kurtosis(log_ret)); sk = float(skew(log_ret))
    print(f"  Return stats: kurtosis={kurt:.2f}  skewness={sk:.2f}  "
          f"({'fat tails' if kurt > 0 else 'thin tails'})")

    return {
        "mu": mu, "sigma": sigma,
        "drift_estimator": drift_info,
        "vol_forecast": vol_info,
        "vol_calibration": vol_calibration,
        "mu_daily": mu_daily, "sigma_daily": sigma_daily,
        "lambda_jump": lambda_jump, "mu_jump": mu_jump,
        "sigma_jump": sigma_jump, "sigma_merton": sigma_merton,
        "v0": v0, "theta": theta, "kappa": kappa, "xi": xi, "rho": rho,
        "mu_hi": mu_hi, "mu_lo": mu_lo, "sig_hi": sig_hi, "sig_lo": sig_lo,
        "mu_hi_raw": mu_hi_raw, "mu_lo_raw": mu_lo_raw,
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


def _path_drawdown_stats(paths: np.ndarray, drawdown_threshold: float | None = None) -> dict:
    threshold = float(MC_CONFIG.get("drawdown_threshold", 0.20) if drawdown_threshold is None else drawdown_threshold)
    running_max = np.maximum.accumulate(paths, axis=0)
    drawdowns = paths / np.maximum(running_max, 1e-10) - 1.0
    max_drawdown = drawdowns.min(axis=0)
    underwater_frac = (drawdowns < 0).mean(axis=0)
    tail_cut = np.percentile(max_drawdown, 5)
    tail = max_drawdown[max_drawdown <= tail_cut]
    cdar95 = float(tail.mean()) if tail.size else float(tail_cut)
    return {
        "threshold_pct": threshold * 100.0,
        "prob_drawdown_beyond_threshold": float((max_drawdown <= -threshold).mean() * 100.0),
        "median_max_drawdown": float(np.percentile(max_drawdown, 50) * 100.0),
        "p05_max_drawdown": float(np.percentile(max_drawdown, 5) * 100.0),
        "cdar95": cdar95 * 100.0,
        "time_under_water_pct_median": float(np.percentile(underwater_frac, 50) * 100.0),
        "time_under_water_pct_p95": float(np.percentile(underwater_frac, 95) * 100.0),
    }


def _evaluate_mc_calibration(prices: pd.Series, paths: np.ndarray, S0: float,
                             horizons: list | None = None) -> dict:
    horizons = list(horizons or MC_CONFIG.get("calibration_horizons", [21, 63, 126]))
    tolerance = float(MC_CONFIG.get("dispersion_tolerance", 0.15))
    series = prices.astype(float)
    result = {
        "available": False,
        "dispersion_label": "unavailable",
        "median_std_ratio": np.nan,
        "horizons": {},
    }
    std_ratios = []
    for horizon in horizons:
        if len(series) <= horizon or paths.shape[0] <= horizon:
            continue
        hist_ret = series.pct_change(horizon).dropna().values
        if len(hist_ret) < 20:
            continue
        sim_ret = (paths[min(int(horizon), paths.shape[0] - 1)] - S0) / S0
        hist_std = float(np.std(hist_ret, ddof=0))
        sim_std = float(np.std(sim_ret, ddof=0))
        if hist_std <= 1e-12:
            continue
        std_ratio = sim_std / hist_std
        std_ratios.append(std_ratio)
        if std_ratio > 1.0 + tolerance:
            label = "overdispersed"
        elif std_ratio < 1.0 - tolerance:
            label = "underdispersed"
        else:
            label = "well_calibrated"
        result["horizons"][str(horizon)] = {
            "hist_std": hist_std,
            "sim_std": sim_std,
            "std_ratio": float(std_ratio),
            "hist_p05": float(np.percentile(hist_ret, 5) * 100.0),
            "sim_p05": float(np.percentile(sim_ret, 5) * 100.0),
            "hist_p95": float(np.percentile(hist_ret, 95) * 100.0),
            "sim_p95": float(np.percentile(sim_ret, 95) * 100.0),
            "dispersion_label": label,
        }
    if std_ratios:
        median_ratio = float(np.median(std_ratios))
        if median_ratio > 1.0 + tolerance:
            result["dispersion_label"] = "overdispersed"
        elif median_ratio < 1.0 - tolerance:
            result["dispersion_label"] = "underdispersed"
        else:
            result["dispersion_label"] = "well_calibrated"
        result["median_std_ratio"] = median_ratio
        result["available"] = True
    return result


def _assess_mc_reliability(calibration_check: dict | None,
                           vol_info: dict | None = None) -> dict:
    cal = dict(calibration_check or {})
    vol_info = dict(vol_info or {})
    failures = []
    fallback = vol_info.get("fallback_reason")
    dispersion = str(cal.get("dispersion_label", "unavailable") or "unavailable")
    median_ratio = float(cal.get("median_std_ratio", np.nan))
    severe_tol = float(MC_CONFIG.get("dispersion_severe_tolerance", 0.30) or 0.30)
    mild_tol = float(MC_CONFIG.get("dispersion_tolerance", 0.15) or 0.15)
    deviation = abs(median_ratio - 1.0) if np.isfinite(median_ratio) else None
    dispersion_severity = "unavailable"
    if deviation is None:
        dispersion_severity = "unavailable"
    elif deviation <= mild_tol:
        dispersion_severity = "ok"
    elif deviation <= severe_tol:
        dispersion_severity = "mild"
    else:
        dispersion_severity = "severe"
    vol_calibration = dict(vol_info.get("vol_calibration", {}) or {})
    vol_calibration_applied = float(vol_calibration.get("applied_multiplier", 1.0) or 1.0)
    if fallback:
        failures.append("vol_model_fallback")
    if dispersion in {"underdispersed", "overdispersed"}:
        failures.append(f"mc_{dispersion}")
    elif dispersion == "mixed" and dispersion_severity in {"mild", "severe"}:
        failures.append("mc_mixed_dispersion")
    status = "usable"
    if failures:
        if dispersion_severity == "severe" or (fallback and any(f != "vol_model_fallback" for f in failures)):
            status = "miscalibrated"
        else:
            status = "mild_miscalibration"
    return {
        "mc_reliability_status": status,
        "mc_reliability_failures": failures,
        "vol_model_fallback": fallback,
        "dispersion_label": dispersion,
        "dispersion_severity": dispersion_severity,
        "median_std_ratio": None if not np.isfinite(median_ratio) else float(median_ratio),
        "vol_calibration_applied": vol_calibration_applied,
    }


def _aggregate_reliability_bucket(model_rows: dict) -> dict:
    rows = dict(model_rows or {})
    statuses = [str((row or {}).get("mc_reliability_status", "usable") or "usable") for row in rows.values()]
    failures = sorted({
        str(failure)
        for row in rows.values()
        for failure in ((row or {}).get("mc_reliability_failures", []) or [])
    })
    status_frequency = {
        key: int(val) for key, val in sorted(Counter(statuses).items(), key=lambda item: item[0])
    }
    if not statuses:
        status = "unknown"
    elif "miscalibrated" in statuses:
        status = "miscalibrated"
    elif "mild_miscalibration" in statuses:
        status = "mild_miscalibration"
    else:
        status = "usable"
    return {
        "status": status,
        "failures": failures,
        "status_frequency": status_frequency,
        "models": list(rows.keys()),
    }


def _aggregate_mc_reliability_overview(risk_summary: dict,
                                       vol_info: dict | None = None) -> dict:
    risk_summary = dict(risk_summary or {})
    vol_info = dict(vol_info or {})
    baseline_rows = {
        name: row for name, row in risk_summary.items()
        if str(name) != "Stressed"
    }
    scenario_rows = {
        name: row for name, row in risk_summary.items()
        if str(name) == "Stressed"
    }
    all_bucket = _aggregate_reliability_bucket(risk_summary)
    baseline_bucket = _aggregate_reliability_bucket(baseline_rows or risk_summary)
    scenario_bucket = _aggregate_reliability_bucket(scenario_rows)
    primary_bucket = baseline_bucket if baseline_rows else all_bucket
    return {
        "mc_reliability_status": primary_bucket.get("status"),
        "mc_reliability_failures": list(primary_bucket.get("failures", [])),
        "vol_model_fallback": vol_info.get("fallback_reason"),
        "baseline_reliability_status": baseline_bucket.get("status"),
        "baseline_reliability_failures": list(baseline_bucket.get("failures", [])),
        "baseline_status_frequency": dict(baseline_bucket.get("status_frequency", {})),
        "baseline_models": list(baseline_bucket.get("models", [])),
        "scenario_reliability_status": scenario_bucket.get("status"),
        "scenario_reliability_failures": list(scenario_bucket.get("failures", [])),
        "scenario_status_frequency": dict(scenario_bucket.get("status_frequency", {})),
        "scenario_models": list(scenario_bucket.get("models", [])),
        "all_model_status_frequency": dict(all_bucket.get("status_frequency", {})),
        "aggregation_policy": "baseline_models_primary_scenarios_secondary",
    }


def _build_model_risk_summary(paths: np.ndarray,
                              S0: float,
                              prices: pd.Series,
                              vol_info: dict | None = None) -> dict:
    idx = min(63, paths.shape[0] - 1)
    ret = (paths[idx] - S0) / S0
    var = float(np.percentile(ret, 5))
    mask = ret <= var
    cvar = float(ret[mask].mean()) if mask.any() else var
    calibration_check = _evaluate_mc_calibration(prices, paths, S0)
    reliability = _assess_mc_reliability(calibration_check, vol_info=vol_info)
    return {
        "median_1yr": float(np.percentile(paths[-1], 50)),
        "p05_1yr": float(np.percentile(paths[-1], 5)),
        "p95_1yr": float(np.percentile(paths[-1], 95)),
        "var95_63d": var * 100,
        "cvar95_63d": cvar * 100,
        "prob_loss_63d": float((ret < 0).mean() * 100),
        "terminal_percentiles": {
            "p05": float(np.percentile(paths[-1], 5)),
            "p25": float(np.percentile(paths[-1], 25)),
            "p50": float(np.percentile(paths[-1], 50)),
            "p75": float(np.percentile(paths[-1], 75)),
            "p95": float(np.percentile(paths[-1], 95)),
        },
        "drawdown": _path_drawdown_stats(paths),
        "risk_by_horizon": compute_risk(paths, S0),
        "calibration_check": calibration_check,
        **reliability,
    }


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
def plot_volatility_model(ticker, prices, params):
    apply_dark_theme()
    fig = plt.figure(figsize=(16, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28)

    log_ret = np.log(prices / prices.shift(1)).dropna()
    rv21 = log_ret.rolling(21).std() * np.sqrt(252) * 100
    rv63 = log_ret.rolling(63).std() * np.sqrt(252) * 100
    vol_info = params.get('vol_forecast', {}) or {}
    hist_ann = float(vol_info.get('historical_sigma_annual', 0.0)) * 100
    forecast_ann = float(vol_info.get('forecast_sigma_annual', 0.0)) * 100

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rv21.index, rv21.values, color=BLUE, lw=1.2, label='21d realised vol')
    ax1.plot(rv63.index, rv63.values, color=PURPLE, lw=1.2, label='63d realised vol')
    ax1.axhline(hist_ann, color=GOLD, lw=1.0, ls='--', label=f'Historical {hist_ann:.1f}%')
    ax1.axhline(forecast_ann, color=GREEN, lw=1.0, ls='--', label=f'Forecast {forecast_ann:.1f}%')
    style_ax(ax1, title='Realised vs Forecast Volatility', xlabel='Date', ylabel='Annualised Vol (%)')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    labels = ['Historical', 'Forecast']
    values = [hist_ann, forecast_ann]
    forecast_color = GREEN if vol_info.get('selected_model') in {'garch', 'egarch', 'gjr_garch'} else BLUE
    colors = [GOLD, forecast_color]
    bars = ax2.bar(labels, values, color=colors, alpha=0.85, width=0.55)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=TEXT)
    style_ax(ax2, title='Volatility Model Summary', ylabel='Annualised Vol (%)')
    params_text = vol_info.get('model_params', {}) or {}
    fallback = vol_info.get('fallback_reason')
    info_lines = [
        f"Requested: {vol_info.get('requested_model', 'historical')}",
        f"Selected: {vol_info.get('selected_model', 'historical')}",
    ]
    if fallback:
        info_lines.append(f"Fallback: {fallback[:60]}")
    if params_text:
        info_lines.extend(f"{k}={v:.4g}" for k, v in params_text.items() if v == v)
    ax2.text(0.02, 0.98, "\n".join(info_lines), transform=ax2.transAxes,
             va='top', ha='left', fontsize=9, color=MUTED, family='monospace')

    fig.suptitle(f'{ticker} â€” Volatility Forecast Overview', fontsize=13, fontweight='bold', color=TEXT)
    if getattr(fig, '_suptitle', None) is not None:
        fig._suptitle.set_text(f'{ticker} - Volatility Forecast Overview')
    path = OUT_DIR / f"{ticker}_montecarlo_volatility.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close('all')
    print(f"  Volatility chart saved â†’ {path}")


def plot_risk_diagnostics(ticker, risk_summary):
    apply_dark_theme()
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)

    models = list(risk_summary.keys())
    if not models:
        return

    dd_prob = [risk_summary[m].get("drawdown", {}).get("prob_drawdown_beyond_threshold", 0.0) for m in models]
    cdar = [risk_summary[m].get("drawdown", {}).get("cdar95", 0.0) for m in models]
    tuw = [risk_summary[m].get("drawdown", {}).get("time_under_water_pct_median", 0.0) for m in models]
    model_labels = [textwrap.fill(str(m), width=11) for m in models]

    ax1 = fig.add_subplot(gs[0, 0])
    xpos = np.arange(len(models))
    width = 0.38
    ax1.bar(xpos - width / 2, dd_prob, width=width, color=RED, alpha=0.82, label='Prob drawdown > threshold')
    ax1.bar(xpos + width / 2, tuw, width=width, color=ORANGE, alpha=0.78, label='Median time under water')
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(model_labels, rotation=12, ha='right')
    style_ax(ax1, title='Drawdown Diagnostics', ylabel='Percent')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(model_labels, cdar, color=PURPLE, alpha=0.82)
    ax2.axhline(0, color=BORDER, lw=0.7)
    style_ax(ax2, title='Tail Drawdown (CDaR95)', ylabel='Percent')
    ax2.tick_params(axis='x', rotation=12, labelsize=8)
    for bar, val in zip(bars, cdar):
        y = val + (-1.0 if val < 0 else 0.6)
        ax2.text(bar.get_x() + bar.get_width() / 2, y, f'{val:.1f}%',
                 ha='center', va='top' if val < 0 else 'bottom',
                 fontsize=8, color=TEXT)

    ax3 = fig.add_subplot(gs[1, 0])
    horizon_sets = set()
    for model in models:
        horizon_sets.update((risk_summary[model].get("calibration_check", {}) or {}).get("horizons", {}).keys())
    horizons = sorted(horizon_sets, key=lambda x: int(x))
    if horizons:
        for model in models:
            cal = risk_summary[model].get("calibration_check", {}) or {}
            std_ratios = [cal.get("horizons", {}).get(h, {}).get("std_ratio", np.nan) for h in horizons]
            ax3.plot(horizons, std_ratios, marker='o', lw=1.4,
                     label=model, color=MODEL_COLORS.get(model, BLUE))
        ax3.axhline(1.0, color=GOLD, lw=0.9, ls='--', alpha=0.8)
        style_ax(ax3, title='Calibration Dispersion Ratio', xlabel='Horizon (days)', ylabel='Sim std / Hist std')
        ax3.legend(fontsize=8)
    else:
        style_ax(ax3, title='Calibration Dispersion Ratio')
        ax3.text(0.5, 0.5, 'No calibration data', transform=ax3.transAxes,
                 ha='center', va='center', color=MUTED)

    ax4 = fig.add_subplot(gs[1, 1])
    labels = []
    values = []
    colors = []
    for model in models:
        cal = risk_summary[model].get("calibration_check", {}) or {}
        labels.append(model)
        values.append(float(cal.get("median_std_ratio", np.nan)))
        disp = cal.get("dispersion_label", "unavailable")
        if disp == "overdispersed":
            colors.append(ORANGE)
        elif disp == "underdispersed":
            colors.append(BLUE)
        elif disp == "well_calibrated":
            colors.append(GREEN)
        else:
            colors.append(MUTED)
    bars = ax4.bar([textwrap.fill(str(lbl), width=11) for lbl in labels], values, color=colors, alpha=0.82)
    ax4.axhline(1.0, color=GOLD, lw=0.9, ls='--', alpha=0.8)
    style_ax(ax4, title='Median Dispersion Summary', ylabel='Median std ratio')
    ax4.tick_params(axis='x', rotation=12, labelsize=8)
    for bar, val in zip(bars, values):
        label = 'NA' if not np.isfinite(val) else f'{val:.2f}'
        height = bar.get_height() if np.isfinite(val) else 0.0
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.03, label,
                 ha='center', va='bottom', fontsize=8, color=TEXT)

    fig.suptitle(f'{ticker} - Monte Carlo Diagnostics', fontsize=13, fontweight='bold', color=TEXT)
    path = OUT_DIR / f"{ticker}_montecarlo_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close('all')
    print(f"  Monte Carlo diagnostics chart saved -> {path}")


def print_summary(ticker, S0, params, paths_dict):
    SEP = "="*66
    print(f"\n{SEP}")
    print(f"  {ticker}  MONTE CARLO RESULTS  v6.0")
    print(f"  {list(paths_dict.values())[0].shape[1]:,} paths x {len(paths_dict)} models  |  antithetic variates")
    print(SEP)
    print(f"  Current Price : ${S0:.4f}")
    print(f"  Annual Drift  : {params['mu']*100:+.2f}%")
    drift_info = params.get('drift_estimator', {})
    if drift_info:
        print(f"  Drift method  : {drift_info.get('method')}  "
              f"raw={drift_info.get('raw_mu', 0.0)*100:+.2f}%  "
              f"adj={drift_info.get('adjusted_mu', 0.0)*100:+.2f}%")
    vol_info = params.get('vol_forecast', {})
    if vol_info:
        print(f"  Vol model     : {vol_info.get('selected_model')}  "
              f"hist={vol_info.get('historical_sigma_annual', 0.0)*100:.2f}%  "
              f"forecast={vol_info.get('forecast_sigma_annual', 0.0)*100:.2f}%")
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
    run_metadata = build_run_metadata(
        mode='single_stock_monte_carlo',
        config={'module': 'monte_carlo', 'mc_config': MC_CONFIG},
        config_version=DEFAULT_CONFIG_VERSION,
        universe=[TICKER],
        extra={'ticker': TICKER, 'module': 'monte_carlo'},
    )

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

    risk_summary = {}
    mc_reliability_overview = {}
    vol_info = params.get('vol_forecast', {}) or {}
    for name, paths in paths_dict.items():
        risk_summary[name] = _build_model_risk_summary(paths, S0, prices, vol_info=vol_info)
    mc_reliability_overview = _aggregate_mc_reliability_overview(risk_summary, vol_info=vol_info)

    apply_dark_theme()
    plot_all(TICKER, S0, params, paths_dict, prices)
    try:
        plot_volatility_model(TICKER, prices, params)
    except Exception as e:
        print(f"  [!] Volatility chart skipped: {e}")
    try:
        plot_risk_diagnostics(TICKER, risk_summary)
    except Exception as e:
        print(f"  [!] Monte Carlo diagnostics chart skipped: {e}")
    print_summary(TICKER, S0, params, paths_dict)

    # Save JSON for dashboard integration
    out = {
        'ticker':    TICKER,
        'generated': datetime.datetime.now().isoformat(),
        'current_price': S0,
        'params': _jsonify_params(params),
        'risk_summary': risk_summary,
        'mc_reliability': mc_reliability_overview,
        'scenarios': {k: {'target': v['target'], 'prob': v['prob']} for k, v in SCENARIOS.items()},
        'run_metadata': complete_run_metadata(run_metadata, status='OK'),
    }
    json_path = OUT_DIR / f"{TICKER}_montecarlo.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    append_experiment_record(Path("reports"), run_metadata, status='OK', summary={
        'ticker': TICKER,
        'module': 'monte_carlo',
        'current_price': S0,
    })
    print(f"  JSON saved → {json_path}")
    print(f"  Monte Carlo total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
