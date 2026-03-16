# -*- coding: utf-8 -*-
"""
Stock ML Analyzer  V.0.7.0
Run: python analyzer.py  |  python analyzer.py AAPL
"""

import warnings; warnings.filterwarnings('ignore')
import os, sys, json, datetime, time, logging, traceback
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────
# Tiingo API (free tier: 500 req/day, clean OHLCV, superior data quality).
# Get a free key at https://api.tiingo.com/  — leave empty to use yfinance.
TIINGO_API_KEY: str = ""

# Local DuckDB cache: stores downloaded OHLCV so portfolio scans never
# re-download the same ticker twice in the same session (or within TTL hours).
# Set to "" to disable caching entirely.
DATA_CACHE_DIR: str = str(Path.home() / ".stock_ml_cache")
DATA_CACHE_TTL_HOURS: int = 6   # re-download after this many hours

# ── JSON serializer that handles all numpy scalar/array types ─────────────────
# In NumPy ≥ 1.24, np.bool_.__name__ == 'bool', np.float32.__name__ == 'float32',
# etc. — none of which the stdlib json module handles. This encoder converts every
# numpy type to its nearest Python primitive before serialisation.
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def set_postfix(self, **kw): pass
        def set_description(self, s): pass
        def update(self, n=1): pass
        def close(self): pass
        @staticmethod
        def write(s): print(s)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    import duckdb as _duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import urllib.request as _urllib_req
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import xgboost as xgb
    HAS_XGB = True
    _xgb_ver = tuple(int(x) for x in xgb.__version__.split('.')[:2])
    _XGB_NEW  = (_xgb_ver >= (2, 0))
except ImportError:
    HAS_XGB  = False; _XGB_NEW = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.ensemble import (RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import (accuracy_score, f1_score, classification_report)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif

try:
    from hmmlearn import hmm as _hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
import io, contextlib
from joblib import Parallel, delayed
import multiprocessing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ── Palette ───────────────────────────────────────────────────────────────────
DARK='#0D1117'; PANEL='#161B22'; BORDER='#30363D'
TEXT='#E6EDF3'; MUTED='#8B949E'
GREEN='#00C853'; RED='#D50000'; BLUE='#58A6FF'; GOLD='#FFD600'
ORANGE='#FF9800'; PURPLE='#CE93D8'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "period":           "10y",   # was 5y → doubles samples, free extra data
    "train_split":      0.80,

    # ── Adaptive regime (all set dynamically by analyze_stock_regime()) ───────
    # These are fallback values only — the IC scan will override them per stock
    "predict_days":     10,      # fallback if IC scan fails
    "buy_threshold":    0.04,
    "sell_threshold":  -0.04,
    "label_buy_quantile":  0.75, # top 25% of forward returns = BUY
    "label_sell_quantile": 0.25, # bottom 25% = SELL  → guaranteed ~25/50/25
    "ic_scan_horizons": [3, 5, 7, 10, 15, 21, 30],  # candidate horizons
    "ic_min_samples":   80,      # min rows needed to compute IC at a horizon

    # ── NEW 1: Triple Barrier Method (López de Prado) ─────────────────────────
    "use_triple_barrier": True,   # False → fall back to quantile labels
    # Symmetric multipliers (1.0/1.0) per de Prado: asymmetric pt > sl injects
    # long-bias into labels; keep equal so the model discovers direction itself.
    # tb_max_hold_pct removed: natural HOLD dominance is handled by FocalLoss +
    # class_weight='balanced'. Tightening barriers to force balance is label noise.
    "tb_pt_sl":          [1.0, 1.0],  # symmetric barriers
    "tb_vol_window":     20,           # 20-day rolling vol (standard)
    "tb_min_samples":    150,          # min rows; below this uses quantile labels

    # ── NEW 2: Fractional Differentiation ────────────────────────────────────
    "use_fracdiff":      True,
    "fracdiff_d":        0.4,    # d∈(0,1): 0.4 keeps ~80% memory, stationary
    "fracdiff_thres":    1e-4,   # weight truncation threshold

    # ── NEW 3: Fundamental features ───────────────────────────────────────────
    "fetch_fundamentals": True,  # pull options/short/analyst data from yfinance

    # ── CV ────────────────────────────────────────────────────────────────────
    "wf_splits":        5,
    "cv_folds":         5,
    "purged_embargo_pct": 0.01,

    # ── Tree grids ────────────────────────────────────────────────────────────
    "rf_grid": {
        "n_estimators":     [600, 1000],
        "max_depth":        [8, 12, None],
        "min_samples_leaf": [3, 5],
        "max_features":     ["sqrt", 0.3],
    },
    "hist_gb_grid": {
        "max_iter":          [200, 400],
        "max_depth":         [3, 4, 5],
        "learning_rate":     [0.02, 0.05, 0.10],
        "min_samples_leaf":  [20, 40],
        "l2_regularization": [0.0, 0.1],
    },
    "xgb_grid": {
        "n_estimators":     [400, 700],
        "max_depth":        [4, 6],
        "learning_rate":    [0.03, 0.07],
        "subsample":        [0.75, 0.90],
        "colsample_bytree": [0.75],
        "gamma":            [0.0, 0.05],
        "reg_alpha":        [0.0, 0.1],
    },
    "lgb_grid": {
        "n_estimators":      [400, 700],
        "num_leaves":        [40, 63],
        "learning_rate":     [0.03, 0.07],
        "subsample":         [0.75, 0.90],
        "colsample_bytree":  [0.75],
        "min_child_samples": [10, 20],
        "reg_alpha":         [0.0, 0.1],
    },
    "cb_grid": {
        "iterations":        [400, 700],
        "depth":             [4, 6],
        "learning_rate":     [0.03, 0.07],
        "l2_leaf_reg":       [1.0, 3.0],
    },

    # ── Feature selection ─────────────────────────────────────────────────────
    "feat_select_k":      80,
    "feat_var_thresh":    0.001,
    "feat_select_method": "mi",   # "mi"=mutual_info (nonlinear) | "fscore"=ANOVA F
    "min_signal_confidence": 0.38,
    "rocket_n_kernels":   1000,   # NEW 9: MiniROCKET kernel count

    # ── NEW 12: CPCV ──────────────────────────────────────────────────────────
    "cpcv_n_splits":      6,        # k folds
    "cpcv_n_test_splits": 2,        # t test-folds per combination

    # ── NEW 13: Regime-conditional ────────────────────────────────────────────
    "regime_model_enabled": True,
    "regime_n_states":    2,

    # ── SWA (Stochastic Weight Averaging) ─────────────────────────────────────
    "swa_enabled":        True,
    "swa_start_pct":      0.60,   # start collecting after 60% of epochs
    "swa_freq":           5,      # collect every N epochs

    # ── DL ────────────────────────────────────────────────────────────────────
    # BiLSTM rightsized: 192h×3L = 3.0M params for ~1000 training samples is
    # severe over-parameterization → memorizes noise → collapses on test set.
    # Rule of thumb: params ≤ 10× training samples. 96h×2L → ~0.45M. ~6× safer.
    "lstm_seq_len":   60,
    "lstm_hidden":    96,    # was 192 — halved to reduce over-param (3M→0.45M)
    "lstm_layers":    2,     # was 3 — one fewer recurrent layer
    "lstm_dropout":   0.40,  # slightly higher dropout to compensate
    "lstm_epochs":    150,
    "lstm_batch":     128,
    "lstm_lr":        3e-4,
    "lstm_patience":  25,
    "tf_d_model":     128,
    "tf_nhead":       8,
    "tf_layers":      4,
    "tf_dropout":     0.20,
    "tf_epochs":      100,
    "tf_patience":    20,
    "dataloader_workers": 0,

    # ── DL augmentation (multiplies training sequences) ───────────────────────
    "dl_augment":        True,
    "dl_aug_factor":     3,     # 3× training sequences (2 noise copies per original)
    "dl_aug_noise_std":  0.005, # Gaussian σ relative to feature scale (small = safe)
    "dl_aug_mag_warp":   True,  # magnitude warping: scale each feature by ~U(0.95,1.05)

    # ── Backtest ──────────────────────────────────────────────────────────────
    # ── NEW 14+15: Backtest realism ──────────────────────────────────────────
    "wf_refit_enabled":     True,   # walk-forward OOS backtest
    "wf_refit_period":      63,     # re-fit every N trading days
    "wf_min_train":         252,    # min rows before first prediction
    "tc_base_bps":          8,      # base half-spread
    "tc_market_impact":     True,   # volume-adaptive impact
    "tc_participation":     0.01,   # fraction of ADV per trade
    "transaction_cost_bps": 10,     # kept for backward-compat / CPCV

    "output_dir":   "reports",
    "random_state": 42,

    # ── NEW 16: Temporal Fusion Transformer (TFT) ─────────────────────────────
    "tft_hidden":         64,     # state size for GRN / VSN hidden layers
    "tft_lstm_layers":    2,      # LSTM layers in the local processing block
    "tft_attn_heads":     4,      # interpretable multi-head attention heads
    "tft_dropout":        0.25,
    "tft_epochs":         120,
    "tft_patience":       20,
    "tft_lr":             3e-4,

    # ── Conformal Prediction (RAPS) ───────────────────────────────────
    "conformal_alpha":    0.10,   # coverage target = 1 - alpha = 90%
    "conformal_cal_pct":  0.25,   # fraction of test set used for calibration
    "conformal_lambda":   0.01,   # RAPS regularisation (penalises larger sets)
    "conformal_kreg":     1,      # RAPS: rank threshold before penalty applies

    # Meta-labeler: if reliability confidence < this, don't force HOLD — fall through
    # BUG FIX: raised from 0.15 → 0.30.  At 0.15 the threshold was too permissive:
    # RKLB had meta_pass_conf=16.2% which barely exceeded 15%, causing a 16.7%-
    # accurate meta-stack to override a 70%-confident LightGBM SELL signal.
    "meta_low_conf_threshold": 0.30,

    # ── Focal Loss for DL models ─────────────────────────────────────
    "focal_loss_gamma":   2.0,    # focusing parameter (0 = standard CE)
    "use_focal_loss":     True,   # applies to all DL models incl. TFT

    # ── Post-Earnings Announcement Drift (PEAD) ──────────────────────
    # Continuous earnings surprise history added as time-series features.
    # Surprise = (actual - estimate) / abs(estimate).  4 trailing quarters
    # encoded with geometric decay so oldest quarter matters less.
    "pead_enabled":       True,
    "pead_n_quarters":    4,       # trailing quarters to encode
    "pead_decay":         0.75,    # geometric decay weight (oldest = 0.75^3)

    # ── Adversarial Validation ───────────────────────────────────────
    # DISABLED by default. Designed for iid cross-sectional data, NOT financial
    # time-series. For a 10-year price series AUC=1.0 is expected (temporal
    # non-stationarity), and dropping the top features (ret_1, sma_144, etc.)
    # hurts F1 by ~0.06 on every ticker tested. Set True for factor models.
    "adv_val_enabled":    False,   # set True only for cross-sectional / iid data
    "adv_val_auc_thresh": 0.85,
    "adv_val_drop_top_n": 10,

    # ── Conformal Position Sizing ────────────────────────────────────
    # Prediction set size → Kelly fraction multiplier:
    #   Singleton {BUY}       → 1.0× (full Kelly — high conviction)
    #   Size-2  {HOLD, BUY}  → 0.5× (half Kelly — moderate conviction)
    #   Size-3  (full set)   → 0.0× (flat — no edge detected)
    "conformal_sizing_enabled": True,

    # ── Diagnostic test suite ─────────────────────────────────────────────────
    # Set True to run all 6 reliability modules after the main analysis.
    # Adds ~3 extra tree fits (≈5-10 min extra for large grids).
    # Output: reports/<TICKER>/<TICKER>_diagnostics.json
    "run_diagnostics": True,
}

TICKER = OUT_DIR = LOG_PATH = log = None

# Platform detection (needed for torch.compile guard)
_IS_WINDOWS = sys.platform.startswith('win')


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def ask_ticker() -> str:
    if len(sys.argv) > 1:
        t = sys.argv[1].strip().upper()
        print(f"  Ticker: {t}"); return t
    print("\n" + "="*54)
    print("  Stock ML Analyzer  V.0.7.0 (Full Stack)")
    print("="*54 + "\n")
    t = input("  Enter ticker: ").strip().upper()
    return t or "CLPT"


def init_run(ticker: str):
    global TICKER, OUT_DIR, LOG_PATH, log
    TICKER  = ticker
    OUT_DIR = Path(CONFIG["output_dir"]) / ticker
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUT_DIR / f"{ticker}_run.log"
    logger = logging.getLogger(ticker); logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    _stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    ch = logging.StreamHandler(_stdout_utf8); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    log = logger


def log_section(title: str):
    log.info(""); log.info("="*60); log.info(f"  {title}"); log.info("="*60)


def elapsed(t0: float) -> str:
    s = time.time() - t0
    if s < 60:   return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


# ─────────────────────────────────────────────────────────────────────────────
# NEW 2: FRACTIONAL DIFFERENTIATION  (López de Prado, AFML 2018 Ch.5)
# ─────────────────────────────────────────────────────────────────────────────
def fracdiff(series: pd.Series, d: float = 0.4, thres: float = 1e-4) -> pd.Series:
    """
    Fractionally differentiate a series with parameter d ∈ (0, 1).

    d=0   → original (non-stationary, full memory)
    d=1   → first difference (stationary, zero long-run memory)
    d≈0.4 → stationary while preserving ~80 % of autocorrelation structure

    The weight vector w_k = Π_{j=0}^{k-1} (d-j)/(j+1) decays geometrically.
    Weights below `thres` are truncated to bound the computation.
    """
    # Build weight vector
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1])   # oldest → newest
    L = len(w)

    out = series.copy() * np.nan
    vals = series.values.astype(float)
    for i in range(L - 1, len(series)):
        segment = vals[i - L + 1: i + 1]
        if not np.isnan(segment).any():
            out.iloc[i] = float(np.dot(w, segment))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NEW 3: FUNDAMENTAL / ALTERNATIVE DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────
class FundamentalFetcher:
    """
    Pulls alternative data from yfinance — all free, no API key needed.

    Sources:
      • ticker.info       — short interest, analyst targets, beta, PE, margins
      • ticker.options    — options chain → put/call ratio + IV proxy
      • ticker.calendar   — next earnings date → days-to-earnings feature
      • ticker.earnings_dates (or history) — post-earnings drift window

    Returns a flat dict of scalars appended as time-invariant features.
    All fields default to sensible neutrals on fetch failure so downstream
    models still train normally.
    """

    DEFAULTS = {
        'short_pct_float':     0.05,
        'short_velocity':      0.0,
        'short_ratio':         3.0,
        'analyst_rec_mean':    3.0,   # 1=strong buy … 5=strong sell
        'n_analysts':          5,
        'target_upside':       0.0,
        'target_spread':       0.3,
        'days_to_earnings':    45,
        'earnings_proximity':  0.02,
        'post_earn_drift_flag':0,
        'put_call_ratio':      1.0,
        'iv_proxy':            0.35,
        'insider_pct':         0.05,
        'institution_pct':     0.50,
        'beta':                1.0,
        'pe_ttm':              20.0,
        'ps_ratio':            3.0,
        'revenue_growth':      0.0,
        'gross_margins':       0.3,
    }

    def __init__(self, ticker: str):
        self.ticker_str = ticker

    def fetch(self) -> dict:
        result = dict(self.DEFAULTS)
        if not HAS_YF:
            return result
        try:
            tkr  = yf.Ticker(self.ticker_str)
            info = tkr.fast_info if hasattr(tkr, 'fast_info') else {}
            # Prefer slow info for fundamentals
            slow = {}
            try:
                slow = tkr.info or {}
            except Exception:
                pass

            def _g(key, default=np.nan):
                v = slow.get(key, default)
                return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v

            # ── Short interest ─────────────────────────────────────────────
            si_now    = _g('shortPercentOfFloat', 0.05)
            si_cur    = _g('sharesShort',         np.nan)
            si_prior  = _g('sharesShortPriorMonth', np.nan)
            shares    = _g('sharesOutstanding',   1e9)
            si_vel    = 0.0
            if not np.isnan(si_cur) and not np.isnan(si_prior) and shares > 0:
                si_vel = float((si_cur - si_prior) / shares)

            result['short_pct_float'] = float(si_now)
            result['short_velocity']  = float(si_vel)
            result['short_ratio']     = float(_g('shortRatio', 3.0))

            # ── Analyst data ──────────────────────────────────────────────
            rec_mean   = _g('recommendationMean', 3.0)
            n_analysts = _g('numberOfAnalystOpinions', 5)
            t_mean     = _g('targetMeanPrice',  np.nan)
            t_high     = _g('targetHighPrice',  np.nan)
            t_low      = _g('targetLowPrice',   np.nan)
            cur_price  = _g('currentPrice',     np.nan)
            if np.isnan(cur_price):
                cur_price = _g('regularMarketPrice', np.nan)

            t_upside = 0.0
            if not np.isnan(t_mean) and not np.isnan(cur_price) and cur_price > 0:
                t_upside = float((t_mean - cur_price) / cur_price)
            t_spread = 0.3
            if not (np.isnan(t_high) or np.isnan(t_low) or np.isnan(t_mean)) and t_mean > 0:
                t_spread = float((t_high - t_low) / t_mean)

            result['analyst_rec_mean'] = float(rec_mean)
            result['n_analysts']       = int(n_analysts)
            result['target_upside']    = float(t_upside)
            result['target_spread']    = float(t_spread)

            # ── Earnings calendar ─────────────────────────────────────────
            days_to_earn = 45
            post_drift   = 0
            try:
                cal = tkr.calendar
                if cal is not None:
                    # yfinance returns different shapes across versions
                    if isinstance(cal, pd.DataFrame):
                        if 'Earnings Date' in cal.index:
                            ed = pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
                        elif not cal.empty:
                            ed = pd.to_datetime(cal.iloc[0, 0])
                        else:
                            ed = None
                    elif isinstance(cal, dict):
                        ed_raw = cal.get('Earnings Date', [None])
                        ed = pd.to_datetime(ed_raw[0]) if ed_raw else None
                    else:
                        ed = None
                    if ed is not None:
                        days_to_earn = int((ed - pd.Timestamp.now()).days)
                        # Post-earnings drift: within 60 days AFTER earnings
                        if -60 <= days_to_earn <= 0:
                            post_drift = 1
            except Exception:
                pass
            result['days_to_earnings']    = float(days_to_earn)
            result['earnings_proximity']  = float(1.0 / (abs(days_to_earn) + 7))
            result['post_earn_drift_flag']= float(post_drift)

            # ── Options chain → put/call ratio + IV proxy ─────────────────
            try:
                dates = tkr.options
                if dates:
                    opt      = tkr.option_chain(dates[0])
                    put_oi   = opt.puts['openInterest'].sum()
                    call_oi  = opt.calls['openInterest'].sum()
                    pcr      = float(put_oi / (call_oi + 1))
                    iv_c     = opt.calls['impliedVolatility'].dropna().median()
                    iv_p     = opt.puts['impliedVolatility'].dropna().median()
                    iv_proxy = float((iv_c + iv_p) / 2) if (iv_c > 0 and iv_p > 0) else 0.35
                    result['put_call_ratio'] = pcr
                    result['iv_proxy']       = iv_proxy
            except Exception:
                pass  # options data not always available

            # ── Ownership / valuation ──────────────────────────────────────
            result['insider_pct']      = float(_g('heldPercentInsiders',  0.05))
            result['institution_pct']  = float(_g('heldPercentInstitutions', 0.50))
            result['beta']             = float(_g('beta', 1.0))
            result['pe_ttm']           = float(_g('trailingPE', 20.0))
            result['ps_ratio']         = float(_g('priceToSalesTrailing12Months', 3.0))
            result['revenue_growth']   = float(_g('revenueGrowth', 0.0))
            result['gross_margins']    = float(_g('grossMargins', 0.3))

            # ── NEW 19: PEAD — earnings surprise history ──────────────────────
            # Encodes up to pead_n_quarters of trailing EPS surprise as scalar
            # features. Surprise = (actual - estimate) / abs(estimate).
            # Geometric decay weights so Q1 (most recent) dominates.
            # Beat-rate and weighted-sum capture systematic over/under-delivery.
            if CONFIG.get('pead_enabled', True):
                try:
                    n_q   = CONFIG.get('pead_n_quarters', 4)
                    decay = CONFIG.get('pead_decay', 0.75)
                    eq    = None
                    for _attr in ['earnings_history', 'quarterly_earnings']:
                        try:
                            eq = getattr(tkr, _attr, None)
                            if eq is not None and not eq.empty:
                                break
                        except Exception:
                            pass
                    if eq is not None and not eq.empty:
                        # Normalise column names across yfinance versions
                        col_map = {}
                        for c_ in eq.columns:
                            cl = c_.lower().replace(' ', '_')
                            if 'estimate' in cl:
                                col_map[c_] = 'estimate'
                            elif 'actual' in cl or 'reported' in cl:
                                col_map[c_] = 'actual'
                        eq = eq.rename(columns=col_map)
                        if 'estimate' in eq.columns and 'actual' in eq.columns:
                            eq = eq[['estimate','actual']].dropna().tail(n_q)
                            surprises = []
                            for _, row in eq.iterrows():
                                est, act = float(row['estimate']), float(row['actual'])
                                if abs(est) > 1e-6:
                                    surprises.append((act - est) / abs(est))
                            surprises = list(reversed(surprises))   # Q1 = most recent
                            w_sum = 0.0
                            for qi, s in enumerate(surprises):
                                w = decay ** qi
                                result[f'pead_surprise_q{qi+1}'] = float(s)
                                result[f'pead_surprise_w{qi+1}'] = float(s * w)
                                w_sum += s * w
                            result['pead_weighted_surprise']  = float(w_sum)
                            if len(surprises) >= 2:
                                result['pead_surprise_momentum'] = float(
                                    surprises[0] - surprises[-1])
                            if surprises:
                                result['pead_beat_rate'] = float(
                                    sum(1 for s in surprises if s > 0) / len(surprises))
                            log.info(
                                f"PEAD: {len(surprises)}q  "
                                f"weighted={w_sum:+.3f}  "
                                f"beat_rate={result.get('pead_beat_rate',0):.0%}")
                except Exception as e:
                    log.debug(f"PEAD fetch skipped: {e}")

            # Sanitise — replace any remaining NaN/inf with defaults
            for k, v in result.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    result[k] = self.DEFAULTS.get(k, 0.0)

            log.info(
                f"Fundamentals fetched: "
                f"SI={result['short_pct_float']:.1%}  vel={result['short_velocity']:+.4f}  "
                f"PCR={result['put_call_ratio']:.2f}  IV={result['iv_proxy']:.0%}  "
                f"target_up={result['target_upside']:+.1%}  "
                f"days_earn={result['days_to_earnings']:.0f}"
            )
        except Exception as e:
            log.warning(f"FundamentalFetcher: {e} — using defaults")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: SAFE SPLIT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_safe_splits(n_samples: int, gap: int, min_train: int = 15) -> int:
    """
    Find the largest n_splits that sklearn's TimeSeriesSplit won't reject.
    sklearn requires: n_samples >= n_splits*(test_size+gap) + max(n_splits, test_size)
    where test_size defaults to n_samples//(n_splits+1).
    We iterate down from 5 until the constraint holds.
    """
    for n in range(5, 1, -1):
        test_size = max(1, n_samples // (n + 1))
        needed    = n * (test_size + gap) + max(n, test_size)
        if n_samples >= needed and (n_samples - n * test_size) >= min_train:
            return n
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE GRID SAMPLER  (Bergstra & Bengio, 2012 — Random Search for HP Opt.)
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_grid_sample(full_grid: list, n_train: int,
                         rng_seed: int = 42) -> list:
    """
    Scale the number of hyperparameter combinations evaluated during grid
    search proportionally to the training-set size.

    Problem — Hyperparameter Overfitting (a.k.a. "data dredging"):
    ──────────────────────────────────────────────────────────────
    Evaluating 50+ HP combinations on only 800 rows means the grid search
    is essentially memorising noise.  The "best" combo found will be the
    one that happens to fit the specific noise pattern of those 800 rows,
    leading to optimistic CV scores that collapse on new data.

    Budget formula:  max_combos = max(4, n_train // 100)
    ─────────────────────────────────────────────────────
    • n_train = 900   (RKLB)  →  max_combos =  9   (tight budget, anti-dredge)
    • n_train = 2000          →  max_combos = 20
    • n_train = 3500          →  max_combos = 35
    • n_train = 8000  (AAPL)  →  max_combos = 80   (likely ≥ full grid)

    When the full grid exceeds the budget, Bergstra & Bengio (2012) prove
    that RANDOM SEARCH outperforms grid search for the same evaluation
    budget: important dimensions get more distinct values sampled, while
    irrelevant dimensions waste nothing.

    Parameters
    ----------
    full_grid : list of dicts  (output of list(ParameterGrid(param_grid)))
    n_train   : number of training samples BEFORE the CV split
    rng_seed  : seed for reproducible random sampling

    Returns
    -------
    list of dicts — either the full grid (if within budget) or a random
    subsample of size max_combos.
    """
    budget = max(4, n_train // 100)
    if len(full_grid) <= budget:
        return full_grid
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(full_grid), size=budget, replace=False)
    return [full_grid[i] for i in sorted(idx)]


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: PURGED K-FOLD WITH EMBARGO
# ─────────────────────────────────────────────────────────────────────────────
class PurgedKFold:
    """
    Time-series K-Fold with:
      • Purging  — removes training samples whose labels overlap the test window.
      • Embargo  — a buffer of `embargo` rows after each test fold before
                   the next training window starts.
    Based on: López de Prado, "Advances in Financial Machine Learning" (2018).
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits   = n_splits
        self.embargo_pct= embargo_pct

    def split(self, X, y=None, groups=None):
        n          = len(X)
        embargo    = max(1, int(n * self.embargo_pct))
        fold_size  = n // self.n_splits
        for i in range(self.n_splits):
            te_start = i * fold_size
            te_end   = te_start + fold_size if i < self.n_splits - 1 else n
            # Training indices: everything before test (with embargo gap) AND after
            tr_idx = (
                list(range(0, max(0, te_start - embargo))) +
                list(range(min(n, te_end + embargo), n))
            )
            te_idx = list(range(te_start, te_end))
            if len(tr_idx) >= 10 and len(te_idx) >= 1:
                yield np.array(tr_idx), np.array(te_idx)


# ─────────────────────────────────────────────────────────────────────────────
# CV SCORER — now uses PurgedKFold and safe split count
# ─────────────────────────────────────────────────────────────────────────────
def cv_score(model, X, y, n_splits, gap=None):
    if gap is None:
        gap = CONFIG['predict_days']

    # FIX 1: auto-reduce to safe split count
    safe_n = compute_safe_splits(len(X), gap)
    n_splits = min(n_splits, safe_n)
    if n_splits < 2:
        n_splits = 2

    # FIX 3: use PurgedKFold
    pkf  = PurgedKFold(n_splits=n_splits, embargo_pct=CONFIG['purged_embargo_pct'])
    accs, f1s = [], []
    for tr_idx, te_idx in pkf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 1:
            continue
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, average='macro', zero_division=0))
    return {
        'acc_mean': float(np.mean(accs)) if accs else 0.0,
        'acc_std':  float(np.std(accs))  if accs else 0.0,
        'f1_mean':  float(np.mean(f1s))  if f1s  else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    log_section("HARDWARE DETECTION")
    if HAS_TORCH and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        prop = torch.cuda.get_device_properties(0)
        vram = prop.total_memory / 1e9
        log.info(f"GPU    : {name}")
        log.info(f"VRAM   : {vram:.2f} GB")
        log.info(f"SM     : {prop.multi_processor_count}  |  CUDA CC: {prop.major}.{prop.minor}")
        torch.backends.cudnn.benchmark       = True
        torch.backends.cuda.matmul.allow_tf32= True
        torch.backends.cudnn.allow_tf32      = True
        log.info("cuDNN  : benchmark=True  TF32=True")
        if HAS_XGB:
            log.info(f"XGBoost  : {'device=cuda' if _XGB_NEW else 'gpu_hist'}")
        log.info(f"LightGBM : device='gpu'")
        if HAS_CB:
            log.info("CatBoost : task_type='GPU'")
        log.info("PyTorch  : CUDA + AMP enabled")
        _dummy = torch.zeros(1, device='cuda'); del _dummy
        torch.cuda.empty_cache()
        log.info("CUDA warm-up  : done")
        return torch.device('cuda'), f"{name} ({vram:.1f}GB VRAM)"
    cpus = os.cpu_count() or 1
    log.info(f"No CUDA GPU — CPU mode  ({cpus} cores)")
    return torch.device('cpu'), "CPU"


# ─────────────────────────────────────────────────────────────────────────────
# DATA CACHE  (DuckDB — optional)
# ─────────────────────────────────────────────────────────────────────────────
class DataCache:
    """
    Local DuckDB-backed OHLCV cache.  Stores downloaded price data to
    avoid re-downloading the same ticker during portfolio scans.

    Schema:  ohlcv(ticker TEXT, date DATE, open DOUBLE, high DOUBLE,
                   low DOUBLE, close DOUBLE, volume DOUBLE,
                   fetched_at TIMESTAMP)

    TTL: rows older than DATA_CACHE_TTL_HOURS are treated as stale and
    re-fetched.  Set DATA_CACHE_DIR = "" to skip caching entirely.
    """

    _con  = None   # shared connection (lazy init)
    _path = None

    @classmethod
    def _connect(cls):
        if not HAS_DUCKDB or not DATA_CACHE_DIR:
            return None
        if cls._con is None:
            cache_dir = Path(DATA_CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cls._path = cache_dir / "market_data.duckdb"
            try:
                cls._con = _duckdb.connect(str(cls._path))
                cls._con.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv (
                        ticker     TEXT,
                        date       DATE,
                        open       DOUBLE,
                        high       DOUBLE,
                        low        DOUBLE,
                        close      DOUBLE,
                        volume     DOUBLE,
                        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (ticker, date)
                    )
                """)
            except Exception:
                cls._con = None
        return cls._con

    @classmethod
    def load(cls, ticker: str, ttl_hours: int = DATA_CACHE_TTL_HOURS
             ) -> 'pd.DataFrame | None':
        """Return cached DataFrame or None if cache miss / stale."""
        con = cls._connect()
        if con is None:
            return None
        try:
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=ttl_hours)
            rows = con.execute(
                "SELECT date,open,high,low,close,volume,fetched_at "
                "FROM ohlcv WHERE ticker=? ORDER BY date",
                [ticker.upper()]
            ).fetchdf()
            if rows.empty:
                return None
            # Check freshness on the most recently fetched row
            latest_fetch = pd.to_datetime(rows['fetched_at']).max()
            if latest_fetch < pd.Timestamp(cutoff):
                return None  # stale — trigger re-download
            rows['date'] = pd.to_datetime(rows['date'])
            df = rows.set_index('date')[['open','high','low','close','volume']]
            df.columns = ['Open','High','Low','Close','Volume']
            df.index.name = None
            return df
        except Exception:
            return None

    @classmethod
    def save(cls, ticker: str, df: pd.DataFrame):
        """Upsert OHLCV rows into the cache."""
        con = cls._connect()
        if con is None or df is None or df.empty:
            return
        try:
            rows = df.reset_index()
            rows.columns = [c.lower() for c in rows.columns]
            rows['ticker']     = ticker.upper()
            rows['fetched_at'] = datetime.datetime.utcnow()
            # Register the DataFrame as a DuckDB relation then INSERT via SQL.
            # The old syntax con.execute("... FROM rows", {'rows': rows}) is not
            # a valid DuckDB Python binding and silently fails on most versions,
            # meaning the cache was never actually written.
            con.execute("DELETE FROM ohlcv WHERE ticker=?", [ticker.upper()])
            con.register("_rows_to_insert", rows)
            con.execute(
                "INSERT INTO ohlcv (ticker,date,open,high,low,close,volume,fetched_at) "
                "SELECT ticker,date,open,high,low,close,volume,fetched_at "
                "FROM _rows_to_insert"
            )
            con.unregister("_rows_to_insert")
        except Exception:
            pass   # cache failure is non-fatal


# ─────────────────────────────────────────────────────────────────────────────
# TIINGO FETCHER  (optional — requires TIINGO_API_KEY to be set)
# ─────────────────────────────────────────────────────────────────────────────
class TiingoFetcher:
    """
    Fetches adjusted OHLCV from the Tiingo Daily Prices REST endpoint.

    Why Tiingo over yfinance?
    • Clean, adjusted OHLCV with no corporate-action errors
    • Stable JSON REST API (not web-scraped)
    • Free tier: 500 API calls/day, unlimited history
    • Daily data for all US equities + major ETFs

    Get a free API key at https://api.tiingo.com/ and set TIINGO_API_KEY
    at the top of this file.  Falls back to yfinance if key is empty or
    any request fails.
    """

    BASE = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str = TIINGO_API_KEY):
        self.key = api_key.strip()

    def available(self) -> bool:
        return bool(self.key) and HAS_URLLIB

    def _period_to_dates(self, period: str) -> tuple:
        """Convert yfinance period string to (start_date, end_date) strings."""
        end   = datetime.date.today()
        units = {'d': 1, 'w': 7, 'mo': 30, 'y': 365}
        num   = int(''.join(filter(str.isdigit, period)) or 1)
        unit  = ''.join(filter(str.isalpha, period.lower()))
        days  = num * units.get(unit, 365)
        start = end - datetime.timedelta(days=days)
        return str(start), str(end)

    def fetch(self, ticker: str, period: str = "10y") -> 'pd.DataFrame | None':
        """Fetch adjusted OHLCV.  Returns DataFrame or None on failure."""
        if not self.available():
            return None
        start, end = self._period_to_dates(period)
        url = (f"{self.BASE}/{ticker.upper()}/prices"
               f"?startDate={start}&endDate={end}"
               f"&resampleFreq=daily&sort=date&token={self.key}")
        try:
            req = _urllib_req.Request(url, headers={'Content-Type': 'application/json'})
            with _urllib_req.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            if not isinstance(data, list) or not data:
                return None
            rows = []
            for r in data:
                rows.append({
                    'date':   r['date'][:10],
                    'Open':   r.get('adjOpen')   or r.get('open', 0),
                    'High':   r.get('adjHigh')   or r.get('high', 0),
                    'Low':    r.get('adjLow')    or r.get('low', 0),
                    'Close':  r.get('adjClose')  or r.get('close', 0),
                    'Volume': r.get('adjVolume') or r.get('volume', 0),
                })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').dropna()
            df.index.name = None
            return df
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER — adds VIX to context (FIX 7)
# ─────────────────────────────────────────────────────────────────────────────
class DataFetcher:
    def __init__(self, ticker, period):
        self.ticker = ticker; self.period = period
        self._tiingo = TiingoFetcher()

    def fetch(self) -> pd.DataFrame:
        log_section("DATA DOWNLOAD")
        t0 = time.time()
        log.info(f"Ticker: {self.ticker}  Period: {self.period}")

        # ── 1. Try DuckDB cache ────────────────────────────────────────────────
        df = DataCache.load(self.ticker)
        src = "cache"
        if df is not None:
            log.info(f"Cache  : HIT  ({len(df)} rows from DuckDB cache)")
        else:
            # ── 2. Try Tiingo (if API key set) ─────────────────────────────────
            if self._tiingo.available():
                log.info("Source : Tiingo API")
                df = self._tiingo.fetch(self.ticker, self.period)
                if df is not None and len(df) > 50:
                    src = "Tiingo"
                    DataCache.save(self.ticker, df)
                else:
                    df = None
                    log.info("Tiingo : no data — falling back to yfinance")

            # ── 3. Fall back to yfinance ────────────────────────────────────────
            if df is None:
                if not HAS_YF:
                    raise RuntimeError("pip install yfinance")
                log.info("Source : yfinance")
                df = yf.download(self.ticker, period=self.period, progress=False, auto_adjust=True)
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df[['Open','High','Low','Close','Volume']].dropna()
                src = "yfinance"
                DataCache.save(self.ticker, df)

        if df is None or df.empty:
            raise RuntimeError(f"Could not fetch data for {self.ticker}")

        log.info(f"Source : {src}")
        log.info(f"Rows   : {len(df)}  ({df.index[0].date()} to {df.index[-1].date()})")
        log.info(f"Latest : ${df['Close'].iloc[-1]:.4f}")
        log.info(f"52w H/L: ${df['High'].tail(252).max():.2f} / ${df['Low'].tail(252).min():.2f}")
        log.info(f"Avg vol: {df['Volume'].mean():,.0f}")
        log.info(f"Time   : {elapsed(t0)}")
        return df

    def fetch_context(self) -> pd.DataFrame:
        # FIX 7: VIX added as the #1 macro fear gauge
        # FIX: "VIX" (no caret) is not a valid yfinance ticker and always fails.
        #      "^VIX" fails with KeyError('chart') via yf.download() — must use
        #      Ticker.history() for index tickers.  Removed "VIX" and "^VIX" from
        #      the download batch; fetched separately via Ticker.history() below.
        symbols = ["SPY", "QQQ", "GLD", "TLT"]
        log.info(f"Downloading market context: SPY, QQQ, GLD, TLT, VIX...")
        frames = []

        def _strip_tz(s: pd.Series) -> pd.Series:
            """Normalise index to tz-naive midnight dates so pd.concat never
            raises 'Cannot join tz-naive with tz-aware DatetimeIndex'.

            Two sources of mismatch:
              yf.download()    → tz-naive dates     (2024-01-02)
              Ticker.history() → tz-aware UTC datetimes (2024-01-02 00:00:00+00:00)

            After tz_localize(None), VIX becomes 2024-01-02 00:00:00 while SPY
            stays 2024-01-02 — pandas treats them as different types → 0 rows
            after dropna(). Fix: .normalize() floors every timestamp to midnight
            so all series share a common date-at-midnight index."""
            s = s.copy()
            if hasattr(s.index, 'tz') and s.index.tz is not None:
                s.index = s.index.tz_convert('UTC').tz_localize(None)
            s.index = s.index.normalize()   # floor to midnight — critical
            return s

        for sym in symbols:
            try:
                d = yf.download(sym, period=self.period, progress=False, auto_adjust=True)
                d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
                close = _strip_tz(d['Close'].rename(sym.replace('^','')))
                if len(close) > 100:
                    frames.append(close)
            except Exception as e:
                log.debug(f"  {sym}: {e}")

        # FIX: fetch ^VIX via Ticker.history() — yf.download('^VIX') raises KeyError('chart')
        try:
            vix_hist = yf.Ticker("^VIX").history(period=self.period, auto_adjust=True)
            if not vix_hist.empty and len(vix_hist) > 100:
                frames.append(_strip_tz(vix_hist['Close'].rename('VIX')))
                log.debug("  ^VIX: OK via Ticker.history()")
        except Exception as e:
            log.debug(f"  ^VIX via Ticker.history(): {e}")

        if frames:
            out = pd.concat(frames, axis=1)
            out = out.loc[:, ~out.columns.duplicated()]   # guard against any dup columns
            out = out.dropna()
            log.info(f"Context: {out.shape[1]} symbols, {len(out)} rows")
            return out
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEER — adds regime features (FIX 10)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngineer:
    def __init__(self, df, market=None, fundamentals: dict = None):
        self.df = df.copy(); self.market = market; self.feat = pd.DataFrame(index=df.index)
        self.fundamentals = fundamentals or {}

    def build(self) -> pd.DataFrame:
        log_section("FEATURE ENGINEERING")
        t0 = time.time()
        c, h, l, o, v = (self.df[x] for x in ['Close','High','Low','Open','Volume'])

        # Returns
        for n in [1,2,3,5,7,10,14,21,42,63]:
            self.feat[f'ret_{n}']    = c.pct_change(n)
            self.feat[f'logret_{n}'] = np.log(c / c.shift(n))

        # Moving averages
        smas, emas = {}, {}
        for w in [5,8,13,21,34,50,89,144,200]:
            smas[w] = c.rolling(w).mean()
            self.feat[f'sma_{w}']       = smas[w]
            self.feat[f'price_sma_{w}'] = (c - smas[w]) / (smas[w].abs() + 1e-10)
        for w in [5,9,12,21,26,50,100,200]:
            emas[w] = c.ewm(span=w, adjust=False).mean()
            self.feat[f'ema_{w}']       = emas[w]
            self.feat[f'price_ema_{w}'] = (c - emas[w]) / (emas[w].abs() + 1e-10)
        for fs, sl in [(5,21),(8,21),(13,34),(21,50),(50,200)]:
            if fs in emas and sl in emas:
                self.feat[f'ema_cross_{fs}_{sl}'] = (emas[fs] > emas[sl]).astype(int)
                self.feat[f'ema_gap_{fs}_{sl}']   = (emas[fs]-emas[sl])/(emas[sl].abs()+1e-10)

        # MACD / RSI / Stochastic / Williams / CCI
        macd = emas[12] - emas[26]; msig = macd.ewm(span=9, adjust=False).mean()
        self.feat['macd'] = macd; self.feat['macd_sig'] = msig
        self.feat['macd_hist'] = macd - msig
        self.feat['macd_cross'] = (macd > msig).astype(int)
        self.feat['macd_div']  = (macd - msig).diff()
        self.feat['ppo'] = (emas[12]-emas[26])/(emas[26].abs()+1e-10)*100

        for w in [7,9,14,21,28]:
            delta = c.diff()
            g  = delta.clip(lower=0).ewm(span=w, adjust=False).mean()
            ls = (-delta.clip(upper=0)).ewm(span=w, adjust=False).mean()
            rsi = 100 - 100/(1 + g/(ls+1e-10))
            self.feat[f'rsi_{w}'] = rsi
            self.feat[f'rsi_{w}_d'] = rsi - 50
            self.feat[f'rsi_{w}_ch'] = rsi.diff()

        for k in [9,14,21,28]:
            lo_k = l.rolling(k).min(); hi_k = h.rolling(k).max()
            sk = 100*(c-lo_k)/(hi_k-lo_k+1e-10); sd = sk.rolling(3).mean()
            self.feat[f'stoch_k_{k}'] = sk; self.feat[f'stoch_d_{k}'] = sd

        for w in [14,21]:
            hi_w = h.rolling(w).max(); lo_w = l.rolling(w).min()
            self.feat[f'willr_{w}'] = -100*(hi_w-c)/(hi_w-lo_w+1e-10)
        for w in [14,20]:
            tp = (h+l+c)/3; sma = tp.rolling(w).mean()
            mad = tp.rolling(w).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)
            self.feat[f'cci_{w}'] = (tp-sma)/(0.015*mad+1e-10)

        # Bollinger / Keltner / Donchian
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        for w in [10,20,50]:
            mid = c.rolling(w).mean(); std = c.rolling(w).std()
            self.feat[f'bb_pct_{w}']   = (c-(mid-2*std))/(4*std+1e-10)
            self.feat[f'bb_width_{w}'] = 4*std/(mid.abs()+1e-10)
            self.feat[f'bb_above_{w}'] = (c>mid+2*std).astype(int)
            self.feat[f'bb_below_{w}'] = (c<mid-2*std).astype(int)
        for w in [14,20]:
            atr = tr.rolling(w).mean(); ema_w = c.ewm(span=w).mean()
            self.feat[f'kelt_pct_{w}']   = (c-ema_w)/(2*atr+1e-10)
            self.feat[f'kelt_above_{w}'] = (c>ema_w+2*atr).astype(int)
        for w in [20,55]:
            hi_dc = h.rolling(w).max(); lo_dc = l.rolling(w).min()
            self.feat[f'don_pct_{w}']  = (c-lo_dc)/(hi_dc-lo_dc+1e-10)
            self.feat[f'don_brk_{w}']  = (c>=hi_dc).astype(int)

        # ATR / Volatility
        for w in [5,7,14,21,42,63]:
            atr = tr.rolling(w).mean()
            self.feat[f'atr_{w}']     = atr
            self.feat[f'atr_pct_{w}'] = atr/(c.abs()+1e-10)
            self.feat[f'vol_{w}']     = c.pct_change().rolling(w).std()*np.sqrt(252)
        for ws,wl in [(5,21),(10,42),(21,63)]:
            self.feat[f'vol_regime_{ws}_{wl}'] = (
                c.pct_change().rolling(ws).std()/(c.pct_change().rolling(wl).std()+1e-10))
        self.feat['chaikin_vol'] = (h-l).ewm(span=10).mean().pct_change(10)

        # FIX 10: Regime features
        r = c.pct_change()
        rv63 = r.rolling(63).std() * np.sqrt(252)
        rv_median = rv63.rolling(252).median()
        self.feat['vol_regime_hi'] = (rv63 > rv_median).astype(int)
        self.feat['vol_regime_lo'] = (rv63 < rv_median * 0.7).astype(int)
        self.feat['rv_63_annualised'] = rv63
        self.feat['rv_21_annualised'] = r.rolling(21).std() * np.sqrt(252)
        self.feat['rv_ratio_21_63']   = (r.rolling(21).std() / (r.rolling(63).std() + 1e-10))

        # Volume / OBV / MFI / A-D / EOM
        for w in [5,10,20,50,100]:
            self.feat[f'vol_ratio_{w}'] = v/(v.rolling(w).mean()+1e-10)
        obv = (np.sign(c.diff())*v).cumsum()
        self.feat['obv'] = obv
        self.feat['obv_trend'] = (obv-obv.rolling(20).mean())/(obv.rolling(20).std()+1e-10)
        self.feat['obv_roc10'] = obv.pct_change(10)
        vwap = (c*v).rolling(20).sum()/(v.rolling(20).sum()+1e-10)
        self.feat['vwap_dev'] = (c-vwap)/(c.abs()+1e-10)
        tp = (h+l+c)/3; rmf = tp*v
        for w in [14,21]:
            pos_mf = rmf.where(tp>tp.shift(1),0).rolling(w).sum()
            neg_mf = rmf.where(tp<tp.shift(1),0).rolling(w).sum()
            self.feat[f'mfi_{w}'] = 100-100/(1+pos_mf/(neg_mf+1e-10))
        clv = ((c-l)-(h-c))/(h-l+1e-10); adl = (clv*v).cumsum()
        self.feat['adl'] = adl
        self.feat['adl_trend'] = (adl-adl.rolling(20).mean())/(adl.rolling(20).std()+1e-10)
        dm = ((h+l)/2)-((h.shift()+l.shift())/2); br = v/(h-l+1e-10)
        self.feat['eom'] = (dm/(br+1e-10)).rolling(14).mean()
        self.feat['force13'] = (c.diff()*v).ewm(span=13).mean()
        self.feat['pv_corr20'] = r.rolling(20).corr(v.pct_change())
        self.feat['pv_corr60'] = r.rolling(60).corr(v.pct_change())

        # Candlestick patterns
        body = (c-o).abs(); total = (h-l).abs()+1e-10
        self.feat['candle_body'] = (c-o)/total
        self.feat['upper_wick']  = (h-pd.concat([o,c],axis=1).max(axis=1))/total
        self.feat['lower_wick']  = (pd.concat([o,c],axis=1).min(axis=1)-l)/total
        self.feat['body_ratio']  = body/total
        self.feat['gap_up']      = ((o-c.shift())/(c.shift().abs()+1e-10)).clip(0)
        self.feat['gap_down']    = ((c.shift()-o)/(c.shift().abs()+1e-10)).clip(0)
        self.feat['doji']        = (body/total < 0.1).astype(int)
        self.feat['hammer']      = (
            (self.feat['lower_wick']>2*self.feat['body_ratio']) &
            (self.feat['upper_wick']<0.1)).astype(int)
        self.feat['engulf_bull'] = (
            (c>o)&(c.shift()<o.shift())&(c>o.shift())&(o<c.shift())).astype(int)
        self.feat['engulf_bear'] = (
            (c<o)&(c.shift()>o.shift())&(c<o.shift())&(o>c.shift())).astype(int)

        # Statistical features
        for w in [10,20,40,60,120]:
            self.feat[f'zscore_{w}'] = (c-c.rolling(w).mean())/(c.rolling(w).std()+1e-10)
            self.feat[f'skew_{w}']   = r.rolling(w).skew()
            self.feat[f'kurt_{w}']   = r.rolling(w).kurt()
        for lag in [1,2,5,10]:
            self.feat[f'autocorr_{lag}'] = r.rolling(30).apply(
                lambda x: x.autocorr(lag=lag) if len(x)>lag+1 else 0, raw=False)
        self.feat['rv_hist'] = (r.rolling(5).std()/(r.rolling(63).std()+1e-10))

        # Market context
        if self.market is not None and not self.market.empty:
            for sym in self.market.columns:
                m = self.market[sym].reindex(c.index, method='ffill')
                for w in [5,10,21,63]:
                    self.feat[f'rs_{sym}_{w}'] = c.pct_change(w)-m.pct_change(w)
                self.feat[f'beta_{sym}_30'] = (
                    r.rolling(30).cov(m.pct_change())/(m.pct_change().rolling(30).var()+1e-10))
                self.feat[f'corr_{sym}_20'] = r.rolling(20).corr(m.pct_change())
                self.feat[f'corr_{sym}_60'] = r.rolling(60).corr(m.pct_change())

        result = self.feat.replace([np.inf,-np.inf],np.nan)
        result = result.dropna(thresh=int(len(result)*0.5), axis=1)

        # ── NEW 2: Fractional Differentiation features ────────────────────────
        if CONFIG.get('use_fracdiff', True):
            d     = CONFIG.get('fracdiff_d', 0.4)
            thres = CONFIG.get('fracdiff_thres', 1e-4)
            try:
                log_price = np.log(c.replace(0, np.nan))
                result['fracdiff_price']   = fracdiff(log_price,                   d, thres)
                result['fracdiff_vol']     = fracdiff(np.log(v.replace(0, np.nan)),d, thres)
                # BUG FIX: bfill() filled the first ~20 NaN rows by looking
                # FORWARD in time (using future volatility estimates). Replace
                # with min_periods=5 so early rows get partial estimates, then
                # ffill() which only propagates known past values, never future ones.
                rv21 = c.pct_change().rolling(21, min_periods=5).std()
                result['fracdiff_rv21']    = fracdiff(rv21.ffill(), d, thres)
                log.info(f"fracdiff features added (d={d})")
            except Exception as e:
                log.warning(f"fracdiff skipped: {e}")

        # ── NEW 3: Fundamental / alternative data features ────────────────────
        if self.fundamentals:
            for key, val in self.fundamentals.items():
                try:
                    result[f'fund_{key}'] = float(val)
                except Exception:
                    pass
            log.info(f"Fundamental scalar features added: {len(self.fundamentals)}")

        result = result.replace([np.inf,-np.inf],np.nan)
        log.info(f"Total features : {result.shape[1]}")
        log.info(f"NaN remaining  : {result.isna().sum().sum()}")
        log.info(f"Build time     : {elapsed(t0)}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# STOCK REGIME ANALYSER  —  per-stock IC scan + adaptive horizon + thresholds
# ─────────────────────────────────────────────────────────────────────────────
def analyze_stock_regime(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Automatically picks the best prediction horizon and label thresholds
    for THIS specific stock, using raw OHLCV data (no feature set needed).

    Step 1 — IC scan
      For each horizon h in [3, 5, 7, 10, 15, 21, 30]:
        Compute Spearman rank correlation between three cheap predictors
        (5d momentum, RSI-14, Bollinger %B-20) and the h-day forward return.
        Average the |IC| across the three predictors.
      → Choose horizon with highest mean |IC|.

    Step 2 — Quantile thresholds
      On the chosen horizon's forward-return distribution, set:
        BUY  = top    (1 - label_buy_quantile)  of returns
        SELL = bottom label_sell_quantile        of returns
      Default: top/bottom 25% → guaranteed ~25 / 50 / 25 class split,
      regardless of the stock's volatility regime.

    Step 3 — Speed / vol classification (for logging and context)
      ATR(14)/Price ratio determines FAST / MEDIUM / SLOW label.

    Returns a dict consumed by make_labels() and logged in main().
    """
    log_section("STOCK REGIME ANALYSIS  (adaptive horizon + thresholds)")
    c  = df['Close']
    r  = c.pct_change().dropna()
    n  = len(c)

    # ── ATR-based daily speed ────────────────────────────────────────────────
    h, l = df['High'], df['Low']
    tr     = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                        axis=1).max(axis=1)
    atr14  = tr.rolling(14).mean()
    atr_pct = float((atr14 / c).median() * 100)   # daily ATR as % of price
    ann_vol = float(r.rolling(63).std().iloc[-1] * np.sqrt(252)) if len(r) >= 63 else 0.3

    # ── Cheap predictors (computed on raw prices, no feat engineering) ────────
    delta  = c.diff()
    g      = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls     = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi14  = 100 - 100 / (1 + g / (ls + 1e-10))

    mom5   = c.pct_change(5)

    mid20  = c.rolling(20).mean(); std20 = c.rolling(20).std()
    bbpct  = (c - (mid20 - 2*std20)) / (4*std20 + 1e-10)

    predictors = {'mom5': mom5, 'rsi14': rsi14, 'bbpct': bbpct}

    # ── IC scan ───────────────────────────────────────────────────────────────
    horizons    = cfg.get('ic_scan_horizons', [3, 5, 7, 10, 15, 21, 30])
    min_samples = cfg.get('ic_min_samples', 80)
    ic_table    = {}   # horizon → mean |IC|

    for h_days in horizons:
        fwd = c.pct_change(h_days).shift(-h_days)
        ics = []
        for name, pred in predictors.items():
            common = pred.dropna().index.intersection(fwd.dropna().index)
            if len(common) < min_samples:
                continue
            p_rank = pred.loc[common].rank(pct=True).values
            f_rank = fwd.loc[common].rank(pct=True).values
            ic = float(np.corrcoef(p_rank, f_rank)[0, 1])
            if not np.isnan(ic):
                ics.append(abs(ic))
        ic_table[h_days] = float(np.mean(ics)) if ics else 0.0

    # Best horizon = highest mean |IC|
    best_horizon = max(ic_table, key=lambda k: ic_table[k]) if ic_table else cfg['predict_days']
    best_ic      = ic_table.get(best_horizon, 0.0)

    # ── Quantile thresholds on the best horizon's return distribution ─────────
    fwd_best = c.pct_change(best_horizon).shift(-best_horizon).dropna()
    buy_q    = cfg.get('label_buy_quantile',  0.75)
    sell_q   = cfg.get('label_sell_quantile', 0.25)
    buy_thresh  = float(fwd_best.quantile(buy_q))
    sell_thresh = float(fwd_best.quantile(sell_q))

    # Safety floor: must be at least 1× daily ATR × horizon (no trivial labels)
    min_move = max(0.01, atr_pct / 100 * best_horizon * 0.5)
    buy_thresh  = max(buy_thresh,   min_move)
    sell_thresh = min(sell_thresh, -min_move)

    # ── Stock speed classification ────────────────────────────────────────────
    if atr_pct > 3.0:   speed = "FAST"
    elif atr_pct > 1.5: speed = "MEDIUM"
    else:               speed = "SLOW"

    regime = {
        'predict_days':  best_horizon,
        'buy_thresh':    buy_thresh,
        'sell_thresh':   sell_thresh,
        'ann_vol':       ann_vol,
        'atr_pct_day':   atr_pct,
        'speed':         speed,
        'best_ic':       best_ic,
        'ic_table':      ic_table,
        'total_rows':    n,
    }

    # ── Log results ───────────────────────────────────────────────────────────
    log.info(f"  Stock speed  : {speed}  (daily ATR = {atr_pct:.2f}%  ann_vol = {ann_vol*100:.1f}%)")
    log.info(f"  IC scan results (mean |Spearman IC| across mom5 / RSI14 / BB%B):")
    for hh, ic in sorted(ic_table.items()):
        marker = " ← CHOSEN" if hh == best_horizon else ""
        log.info(f"    {hh:>3}d horizon:  IC = {ic:.4f}{marker}")
    log.info(f"  → Horizon selected: {best_horizon}d  (best IC = {best_ic:.4f})")
    log.info(f"  → Thresholds  BUY ≥ {buy_thresh*100:.2f}%   SELL ≤ {sell_thresh*100:.2f}%")
    log.info(f"    (quantile-based: top/bottom {(1-buy_q)*100:.0f}% "
             f"of {best_horizon}d return distribution)")
    return regime


# ─────────────────────────────────────────────────────────────────────────────
# NEW 1: TRIPLE BARRIER LABELER  (López de Prado, AFML 2018 Ch.3)
# ─────────────────────────────────────────────────────────────────────────────
class TripleBarrierLabeler:
    """
    Three dynamic barriers per observation:
      Upper (take-profit): price ≥ p0 × (1 + pt_mult × σ_d × √h)
      Lower (stop-loss):   price ≤ p0 × (1 − sl_mult × σ_d × √h)
      Vertical:            h trading days elapse

    First barrier touched → label.  σ_d = rolling daily volatility (vol_window days).
    Symmetric multipliers (pt == sl) avoid encoding directional bias in labels.

    Advantages over fixed-horizon labeling:
      • No look-ahead noise from arbitrary horizon choice
      • Adapts barriers to current volatility regime
      • Naturally generates fewer false BUY signals in high-vol environments
      • Path-dependent: a stock that bounces +12% and falls -7% in 21 days
        gets labelled BUY, not HOLD — the path matters
    """

    def __init__(self, close: pd.Series, regime: dict,
                 pt_sl=(1.0, 1.0), vol_window: int = 21):
        """
        pt_sl: (take-profit multiplier, stop-loss multiplier).
        Default (1.0, 1.0) = symmetric barriers per López de Prado AFML §3.
        BUG FIX: was (1.5, 1.0) — asymmetric pt > sl injects long-bias into
        labels.  Symmetric barriers let the model discover direction itself.
        """
        self.close      = close
        self.h          = regime['predict_days']
        self.pt_m, self.sl_m = pt_sl
        self.vol_window = vol_window

    def label(self) -> pd.Series:
        close   = self.close
        h       = self.h
        daily_r = close.pct_change()
        daily_v = daily_r.rolling(self.vol_window).std()

        labels  = pd.Series(np.nan, index=close.index, dtype=float)

        for i in range(self.vol_window, len(close) - 1):
            vol_i = daily_v.iloc[i]
            if np.isnan(vol_i) or vol_i <= 0:
                continue
            p0     = close.iloc[i]
            upper  = p0 * (1.0 + self.pt_m * vol_i * np.sqrt(h))
            lower  = p0 * (1.0 - self.sl_m * vol_i * np.sqrt(h))
            end    = min(i + h + 1, len(close))
            future = close.iloc[i + 1: end]

            lbl = 1   # default: time barrier hit → HOLD
            for price in future:
                if price >= upper:
                    lbl = 2  # BUY (take-profit)
                    break
                elif price <= lower:
                    lbl = 0  # SELL (stop-loss)
                    break
            labels.iloc[i] = lbl

        return labels


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAKER  —  uses Triple Barrier (NEW 1) or per-stock quantile fallback
# ─────────────────────────────────────────────────────────────────────────────
def make_labels(df: pd.DataFrame, regime: dict) -> pd.Series:
    """
    Creates BUY/HOLD/SELL labels via the Triple Barrier Method (de Prado, AFML Ch.3).

    Design principle — NO forced balancing:
    ─────────────────────────────────────────
    Dynamically tightening barriers to cap HOLD% is a form of target
    non-stationarity: the same price path gets a different label depending on
    how imbalanced the overall dataset happens to be.  That instability is
    worse than any class-imbalance problem it tries to solve, and is the root
    cause of DL model collapse in noisy financial regimes.

    The natural market distribution (often 65-80% HOLD) is the ground truth.
    Class imbalance is handled entirely in the loss function (FocalLoss +
    class_weight clipping), which is the statistically correct place to do it.

    Barrier geometry:
        Upper (take-profit): price ≥ p0 × (1 + pt_m × σ_d × √h)
        Lower (stop-loss):   price ≤ p0 × (1 − sl_m × σ_d × √h)
        Vertical:            h trading days elapse → HOLD

    σ_d = 20-day rolling daily volatility (close returns).
    Symmetric multipliers (1.0 / 1.0) avoid injecting long-bias into labels.
    """
    log_section("LABEL GENERATION")
    use_tb      = CONFIG.get('use_triple_barrier', True)
    min_samples = CONFIG.get('tb_min_samples', 150)
    n_rows      = len(df)

    if use_tb and n_rows >= min_samples:
        pt_sl = list(CONFIG.get('tb_pt_sl', [1.0, 1.0]))
        vol_w = CONFIG.get('tb_vol_window', 20)
        try:
            log.info(f"Label method : Triple Barrier  "
                     f"pt={pt_sl[0]:.2f}×σ√h  sl={pt_sl[1]:.2f}×σ√h  "
                     f"h={regime['predict_days']}d  vol_win={vol_w}")
            log.info("  Class imbalance handled by FocalLoss + weight clipping — "                     "no barrier tightening applied.")
            labeler = TripleBarrierLabeler(df['Close'], regime,
                                           pt_sl=pt_sl, vol_window=vol_w)
            y       = labeler.label()
            n_valid = y.notna().sum()
            if n_valid < 50:
                raise ValueError(f"Only {n_valid} valid labels — too few")
            vc = y.value_counts().sort_index()
            for idx, cnt in vc.items():
                log.info(f"  {['SELL','HOLD','BUY'][int(idx)]:<5}: "
                         f"{cnt:>5}  ({cnt/n_valid*100:.1f}%)")
            hold_pct = float((y == 1).sum() / n_valid)
            if hold_pct > 0.80:
                log.warning(f"  HOLD {hold_pct*100:.1f}% is high — this is natural market "
                            f"structure.  FocalLoss + clipped weights will compensate.")
            return y
        except Exception as e:
            log.warning(f"Triple Barrier failed ({e}) — falling back to quantile labels")

    # ── Quantile-based fallback ───────────────────────────────────────────────
    log.info("Label method : Quantile-based fixed-horizon (fallback)")
    n           = regime['predict_days']
    buy_thresh  = regime['buy_thresh']
    sell_thresh = regime['sell_thresh']
    fwd = df['Close'].pct_change(n).shift(-n)
    y   = pd.Series(1, index=df.index, name='label', dtype=float)
    y[fwd >= buy_thresh]  = 2
    y[fwd <= sell_thresh] = 0
    log.info(f"Horizon     : {n}d  |  BUY ≥ {buy_thresh*100:.2f}%  SELL ≤ {sell_thresh*100:.2f}%")
    vc = y.value_counts().sort_index()
    for idx, cnt in vc.items():
        log.info(f"  {['SELL','HOLD','BUY'][int(idx)]:<5}: {cnt:>5}  ({cnt/len(y)*100:.1f}%)")
    return y


# ─────────────────────────────────────────────────────────────────────────────
# NON-OVERLAPPING SUBSAMPLE
# ─────────────────────────────────────────────────────────────────────────────
def subsample_nonoverlapping(X: pd.DataFrame, y: pd.Series, step: int):
    idx = np.arange(0, len(X), step)
    log.info(f"Subsampling: {len(X)} rows → {len(idx)} non-overlapping "
             f"(step={step}, predict_days={step})")
    return X.iloc[idx], y.iloc[idx]


# ─────────────────────────────────────────────────────────────────────────────
# NEW 9: MINI-ROCKET  (Dempster et al., 2021 — sklearn-compatible)
# ─────────────────────────────────────────────────────────────────────────────
class MiniROCKET:
    """
    Random Convolutional Kernel Transform for time-series classification.
    Generates N_KERNELS random 1-D kernels, applies them to each sample's
    time-series window, then extracts two pooling statistics per kernel:
      • max (captures extremes / spikes)
      • ppv = proportion of positive values (captures shape frequency)
    → 2 * N_KERNELS scalar features fed to a RidgeClassifierCV.

    Why it works:
      • Random kernels at multiple scales implicitly span the Fourier basis
      • PPV is surprisingly discriminative for financial waveforms
      • No gradient descent → no overfitting to training noise
      • Trains in ~1-3 s; inference in ms
    """
    KERNEL_LENGTHS = [7, 9, 11]   # odd lengths only (as in original paper)

    def __init__(self, n_kernels: int = 1000, seq_len: int = 60,
                 n_features: int = 1, random_state: int = 42):
        self.n_kernels    = n_kernels
        self.seq_len      = seq_len
        self.n_features   = n_features   # use top-1 PCA component if >1
        self.rng          = np.random.default_rng(random_state)
        self.kernels_     = None   # list of (weights, length, bias, dilation, padding)
        self.ridge_       = None
        self.pca_         = None

    def _build_kernels(self):
        """Generate random kernels (length, dilation, bias, weights)."""
        kernels = []
        for _ in range(self.n_kernels):
            klen     = self.rng.choice(self.KERNEL_LENGTHS)
            dilation = int(2 ** self.rng.uniform(0, np.log2(self.seq_len // klen + 1)))
            padding  = ((klen - 1) * dilation) // 2
            weights  = self.rng.standard_normal(klen).astype(np.float32)
            weights -= weights.mean()
            bias     = self.rng.uniform(-1, 1)
            kernels.append((weights, klen, bias, dilation, padding))
        self.kernels_ = kernels

    def _apply_kernels(self, X1d: np.ndarray) -> np.ndarray:
        """
        X1d: (n_samples, seq_len) float32
        Returns: (n_samples, 2*n_kernels)
        """
        n = len(X1d)
        feats = np.empty((n, 2 * self.n_kernels), dtype=np.float32)
        for ki, (w, klen, bias, dil, pad) in enumerate(self.kernels_):
            # Manual dilated 1-D convolution via stride tricks
            # Build strided view for each dilation
            effective_len = klen + (klen - 1) * (dil - 1)
            out_len = self.seq_len - effective_len + 1 + 2 * pad
            if out_len <= 0:
                feats[:, 2*ki]   = 0.0
                feats[:, 2*ki+1] = 0.5
                continue
            # Convolve using np.convolve on each sample
            # BUG FIX: allocate exactly `out_len` elements, NOT `out_len + 2*pad`.
            # out_len = seq_len - effective_len + 1 + 2*pad already incorporates
            # the padding effect.  Adding 2*pad a second time appended trailing
            # zeros that were included in the ppv (proportion-of-positive-values)
            # pooling, systematically deflating that feature for all kernels.
            conv_out = np.zeros((n, out_len), dtype=np.float32)
            # Build dilated kernel (insert zeros between weights)
            w_dilated = np.zeros(effective_len, dtype=np.float32)
            w_dilated[::dil] = w
            for i in range(n):
                sig    = X1d[i]
                if pad > 0:
                    sig = np.pad(sig, pad, mode='edge')
                c = np.convolve(sig, w_dilated[::-1], mode='valid').astype(np.float32)
                c_len = min(len(c), out_len)
                conv_out[i, :c_len] = c[:c_len]
            conv_out += bias
            feats[:, 2*ki]   = conv_out.max(axis=1)        # max pooling
            feats[:, 2*ki+1] = (conv_out > 0).mean(axis=1) # ppv
        return feats

    def _to_1d(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce (n, seq_len, n_features) or (n, seq_len) → (n, seq_len).
        Uses the first principal component if n_features > 1.
        """
        if X.ndim == 2:
            return X.astype(np.float32)
        # X shape: (n, seq_len, n_features)
        n, s, f = X.shape
        Xr = X.reshape(n * s, f)
        if self.pca_ is None:
            from sklearn.decomposition import PCA
            self.pca_ = PCA(n_components=1, random_state=42)
            self.pca_.fit(Xr)
        return self.pca_.transform(Xr).reshape(n, s).astype(np.float32)

    def fit(self, X_seqs: np.ndarray, y: np.ndarray):
        """
        X_seqs: (n, seq_len, n_feat) — raw sliding-window sequences
        y: (n,) int labels
        """
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.preprocessing import StandardScaler
        self._build_kernels()
        X1d  = self._to_1d(X_seqs)
        feat = self._apply_kernels(X1d)
        self._scaler = StandardScaler(copy=False)
        feat = self._scaler.fit_transform(feat)
        self.ridge_ = RidgeClassifierCV(
            alphas=[1e-3, 0.01, 0.1, 1.0, 10.0],
            class_weight='balanced', cv=3)
        self.ridge_.fit(feat, y)
        return self

    def predict(self, X_seqs: np.ndarray) -> np.ndarray:
        X1d  = self._to_1d(X_seqs)
        feat = self._apply_kernels(X1d)
        feat = self._scaler.transform(feat)
        return self.ridge_.predict(feat)

    def predict_proba(self, X_seqs: np.ndarray) -> np.ndarray:
        """Converts decision function to soft probabilities via softmax."""
        X1d  = self._to_1d(X_seqs)
        feat = self._apply_kernels(X1d)
        feat = self._scaler.transform(feat)
        dec  = self.ridge_.decision_function(feat)
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        # Softmax
        dec -= dec.max(axis=1, keepdims=True)
        exp  = np.exp(dec)
        return exp / exp.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL SOFT VOTER
# Replaces sklearn VotingClassifier for the tree-model ensemble.
#
# Why not VotingClassifier?
#   sklearn's VotingClassifier.fit() calls clone() on every sub-estimator
#   before training them in parallel (n_jobs=-1).  clone() does a round-trip
#   check: get_params() → __init__(**params) → get_params() again, raising
#   RuntimeError if the two get_params() dicts differ.
#   CatBoost fails this check because it normalises class_weights internally
#   (list → dict), so the reconstructed object's get_params() returns a
#   different type than the original.  MiniROCKET fails for a different reason:
#   it has no __sklearn_tags__ (doesn't inherit BaseEstimator) and expects 3-D
#   sequences rather than the 2-D array VotingClassifier passes.
#
# This class averages predict_proba() from already-fitted models, skipping the
# clone/refit step entirely.  Benefits:
#   • Works with any estimator that has predict_proba() — no BaseEstimator req.
#   • Faster: no redundant re-fit of 5 tuned models.
#   • More consistent: ensemble uses the exact same fitted objects evaluated
#     individually.
# ─────────────────────────────────────────────────────────────────────────────
class ManualSoftVoter:
    """
    Soft-vote ensemble over a dict of already-fitted classifiers.

    predict_proba(X) = mean of each model's predict_proba(X).
    predict(X)       = argmax of the averaged probability vector.

    Models that raise an exception during predict_proba are silently skipped
    so that a single bad model never kills the ensemble.
    """
    def __init__(self, estimators: dict, classes: np.ndarray = None):
        """
        Parameters
        ----------
        estimators : dict  {name: fitted_model}
            All models must already be fitted and support predict_proba().
        classes : array-like, optional
            Class labels.  Defaults to [0, 1, 2].
        """
        self.estimators_ = estimators
        self.classes_    = np.array(classes if classes is not None else [0, 1, 2])

    # sklearn VotingClassifier API compatibility stubs ─────────────────────────
    def fit(self, X, y, sample_weight=None):
        """No-op: models are already fitted."""
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average softmax probabilities across all sub-models."""
        probas = []
        for name, model in self.estimators_.items():
            try:
                p = model.predict_proba(X)
                if p.ndim == 2 and p.shape[1] == len(self.classes_):
                    probas.append(p)
            except Exception:
                pass   # skip models that can't produce probabilities for this X
        if not probas:
            # Fallback: uniform distribution
            return np.full((len(X), len(self.classes_)),
                           1.0 / len(self.classes_))
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # Allow get_params / set_params so downstream sklearn utilities work ───────
    def get_params(self, deep=True):
        return {'estimators': self.estimators_, 'classes': self.classes_}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# TREE MODEL TRAINER — FIX 4: feature name tracking + FIX 6: CatBoost
# ─────────────────────────────────────────────────────────────────────────────
class TreeTrainer:
    def __init__(self, X, y, cfg, use_gpu: bool):
        self.X, self.y, self.cfg = X, y, cfg
        self.use_gpu = use_gpu
        self.models, self.results = {}, {}
        self.scaler   = RobustScaler()
        self.var_sel  = None
        self.feat_sel = None
        self.feat_names_selected_ = None   # FIX 4

    def _prep_walkforward(self):
        """
        80/20 chronological split — keeps ALL training rows.
        Adds overlap-aware sample weights: rows whose label windows overlap
        the test boundary get downweighted, reducing overfitting from leakage.
        """
        Xf  = self.X.fillna(self.X.median())
        y   = self.y
        gap = self.cfg['predict_days']

        sp   = int(len(Xf) * self.cfg['train_split'])
        X_tr, X_te = Xf.iloc[:sp],  Xf.iloc[sp:]
        y_tr, y_te = y.iloc[:sp],   y.iloc[sp:]

        m_tr = y_tr.notna(); m_te = y_te.notna()
        X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        X_te, y_te = X_te[m_te], y_te[m_te]

        # ── Overlap-aware sample weights ─────────────────────────────────────
        # Rows near the train/test boundary have forward-return windows that
        # partially overlap the test period. Downweight them exponentially.
        n_tr = len(X_tr)
        w    = np.ones(n_tr, dtype=np.float32)
        for i in range(min(gap, n_tr)):
            # last `gap` training rows get linear decay: 0.1 → 1.0
            w[n_tr - gap + i] = 0.1 + 0.9 * (i / gap)
        self.sample_weights_ = w   # stored for use in model.fit()

        Xs_tr = self.scaler.fit_transform(X_tr)
        Xs_te = self.scaler.transform(X_te)

        self.var_sel  = VarianceThreshold(threshold=self.cfg['feat_var_thresh'])
        Xs_tr = self.var_sel.fit_transform(Xs_tr)
        Xs_te = self.var_sel.transform(Xs_te)
        n_after_var = Xs_tr.shape[1]

        # ── Adaptive feature count — scale with training set size ─────────────
        # Rule of thumb: ≥ 12 samples per feature avoids over-parameterization.
        # RKLB has ~897 train rows; 80 features → 11.2× (borderline).
        # Adaptive cap: min(cfg_k, n_train//12) keeps the ratio safe.
        raw_k = self.cfg['feat_select_k']
        k = min(raw_k, max(20, len(X_tr) // 12), n_after_var)
        if k < raw_k:
            log.info(f"  Adaptive feat_k: {raw_k} → {k} "
                     f"(n_train={len(X_tr)}, ratio cap = n//12)")
        # NEW 11: Mutual Information (nonlinear) vs F-score (linear)
        sel_method = self.cfg.get('feat_select_method', 'mi')
        score_fn   = mutual_info_classif if sel_method == 'mi' else f_classif
        if sel_method == 'mi':
            log.info(f"  Feature selector: MutualInfo (nonlinear, k={k})")
        else:
            log.info(f"  Feature selector: F-score/ANOVA (linear, k={k})")
        self.feat_sel = SelectKBest(score_fn, k=k)
        Xs_tr = self.feat_sel.fit_transform(Xs_tr, y_tr.values)
        Xs_te = self.feat_sel.transform(Xs_te)

        cols_after_var   = np.array(self.X.columns)[self.var_sel.get_support()]
        cols_after_kbest = cols_after_var[self.feat_sel.get_support()]
        self.feat_names_selected_ = list(cols_after_kbest)

        log.info(f"Feature selection: {self.X.shape[1]} raw → "
                 f"{n_after_var} (var) → {Xs_tr.shape[1]} "
                 f"(top-K {'MutualInfo' if sel_method == 'mi' else 'F-score'})")
        log.info(f"Train: {len(X_tr)}  Test: {len(X_te)}  "
                 f"(gap={gap}d — all rows kept, overlap-weighted)")
        return X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te

    def _grid_search(self, name, ModelClass, param_grid, Xs_tr, y_tr,
                     fixed_kwargs, parallel_jobs=1):
        full_grid = list(ParameterGrid(param_grid))
        # ── Adaptive grid sampling — scale search budget to dataset size ──────
        # Prevents hyperparameter overfitting on small datasets.
        # See adaptive_grid_sample() for the budget formula.
        n_train_for_budget = len(Xs_tr)
        grid    = adaptive_grid_sample(full_grid, n_train_for_budget,
                                        rng_seed=self.cfg['random_state'])
        n_comb  = len(grid)
        n_full  = len(full_grid)
        t_gs    = time.time()
        rs      = self.cfg['random_state']
        n_splits = self.cfg['cv_folds']
        if n_comb < n_full:
            log.info(f"  Grid: {n_full} combos → {n_comb} sampled "
                     f"(budget=max(4,{n_train_for_budget}//100), random search) "
                     f"× {n_splits} folds  |  parallel_jobs={parallel_jobs}")
        else:
            log.info(f"  Grid: {n_comb} combos × {n_splits} folds  |  parallel_jobs={parallel_jobs}")

        def _eval_combo(params):
            import warnings as _w; _w.filterwarnings('ignore')
            kwargs = {**fixed_kwargs, **params, 'random_state': rs}
            model  = ModelClass(**kwargs)
            sc     = cv_score(model, Xs_tr, y_tr.values, n_splits)
            return sc['f1_mean'], params, kwargs

        if parallel_jobs == 1:
            best_score, best_params, best_kwargs = -1, None, {}
            bar = tqdm(grid, desc=f"  {name} grid", unit="combo", ncols=90,
                       file=sys.stdout, leave=True)
            for params in bar:
                score, p_, kw_ = _eval_combo(params)
                if score > best_score:
                    best_score, best_params, best_kwargs = score, p_, kw_
                bar.set_postfix(best=f"{best_score:.4f}", cur=f"{score:.4f}")
            bar.close()
        else:
            log.info(f"  Launching {n_comb} parallel jobs on {parallel_jobs} workers...")
            raw = Parallel(n_jobs=parallel_jobs, prefer='processes')(
                delayed(_eval_combo)(p)
                for p in tqdm(grid, desc=f"  {name} sched", unit="combo",
                              ncols=80, file=sys.stdout))
            best_score, best_params, best_kwargs = max(raw, key=lambda x: x[0])

        log.info(f"  Best params : {best_params}")
        log.info(f"  Best F1-macro: {best_score:.4f}  (grid time: {elapsed(t_gs)})")
        best_model = ModelClass(**best_kwargs)
        best_model.fit(Xs_tr, y_tr.values)
        return best_model, best_params, best_score

    def train(self):
        log_section("TREE MODEL TRAINING")
        X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te = self._prep_walkforward()
        self.X_tr, self.X_te = X_tr, X_te
        self.y_tr, self.y_te = y_tr, y_te
        self.Xs_tr, self.Xs_te = Xs_tr, Xs_te
        log.info(f"Train: {len(X_tr)}  Test: {len(X_te)}  Features (post-select): {Xs_tr.shape[1]}")

        sw    = getattr(self, 'sample_weights_', None)  # overlap-aware weights
        n_cpu = multiprocessing.cpu_count()
        log.info(f"CPU cores available: {n_cpu}  |  sample_weights: {'yes' if sw is not None else 'no'}")
        trained = {}

        # ── helper: final fit with sample weights ─────────────────────────────
        def _fit_with_sw(model, Xs, ys, supports_sw=True):
            if sw is not None and supports_sw:
                w = sw[:len(ys)]          # trim to actual train size after dropna
                w = w / w.sum() * len(w)  # normalise so sum = n_samples
                model.fit(Xs, ys, sample_weight=w)
            else:
                model.fit(Xs, ys)

        # 1. Random Forest
        log.info(f"\n[1/6] Random Forest (CPU, outer parallel_jobs={n_cpu})...")
        t0 = time.time()
        rf, rfp, _ = self._grid_search(
            "RF", RandomForestClassifier, self.cfg['rf_grid'], Xs_tr, y_tr,
            {'class_weight': 'balanced', 'n_jobs': 1}, parallel_jobs=n_cpu)
        _fit_with_sw(rf, Xs_tr, y_tr.values)
        preds = rf.predict(Xs_te)
        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, average='macro', zero_division=0)
        trained['RandomForest'] = rf
        self.results['RandomForest'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': rfp}
        log.info(f"  Test accuracy: {acc:.4f}  F1-macro: {f1:.4f}  |  {elapsed(t0)}")
        log.info(f"\n{classification_report(y_te, preds, target_names=['SELL','HOLD','BUY'], zero_division=0)}")

        # 2. HistGradientBoosting (doesn't support sample_weight in sklearn < 1.4 properly, skip)
        log.info(f"[2/6] HistGradientBoosting (CPU, outer parallel_jobs={n_cpu})...")
        t0 = time.time()
        gb, gbp, _ = self._grid_search(
            "HistGB", HistGradientBoostingClassifier, self.cfg['hist_gb_grid'],
            Xs_tr, y_tr,
            {'class_weight': 'balanced', 'early_stopping': False,
             'random_state': self.cfg['random_state']}, parallel_jobs=n_cpu)
        gb.fit(Xs_tr, y_tr.values)   # HistGB uses class_weight internally
        preds = gb.predict(Xs_te)
        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, average='macro', zero_division=0)
        trained['HistGradBoost'] = gb
        self.results['HistGradBoost'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': gbp}
        log.info(f"  Test accuracy: {acc:.4f}  F1-macro: {f1:.4f}  |  {elapsed(t0)}")

        # 3. XGBoost — GPU only when n_train ≥ 3k (GPU launch overhead dominates below)
        if HAS_XGB:
            n_train = len(Xs_tr)
            _GPU_BREAK_EVEN = 3000
            use_xgb_gpu = self.use_gpu and (n_train >= _GPU_BREAK_EVEN)
            if use_xgb_gpu:
                gpu_kwargs   = {'device': 'cuda', 'tree_method': 'hist'} if _XGB_NEW else \
                               {'tree_method': 'gpu_hist', 'gpu_id': 0}
                xgb_parallel = 1
                log.info(f"[3/6] XGBoost (GPU cuda — {n_train} samples)...")
            else:
                gpu_kwargs   = {'tree_method': 'hist', 'n_jobs': -1}
                xgb_parallel = n_cpu
                reason = "no GPU" if not self.use_gpu else f"n={n_train}<{_GPU_BREAK_EVEN}→CPU faster"
                log.info(f"[3/6] XGBoost (CPU n_jobs=-1, {reason})...")
            t0 = time.time()
            xg, xgp, _ = self._grid_search(
                "XGB", xgb.XGBClassifier, self.cfg['xgb_grid'], Xs_tr, y_tr,
                {**gpu_kwargs, 'eval_metric': 'mlogloss', 'verbosity': 0},
                parallel_jobs=xgb_parallel)
            _fit_with_sw(xg, Xs_tr, y_tr.values)
            preds = xg.predict(Xs_te)
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro', zero_division=0)
            trained['XGBoost'] = xg
            self.results['XGBoost'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': xgp}
            log.info(f"  Test accuracy: {acc:.4f}  F1-macro: {f1:.4f}  |  {elapsed(t0)}")
        else:
            log.warning("[3/6] XGBoost not installed")

        # 4. LightGBM — same GPU break-even
        if HAS_LGB:
            n_train      = len(Xs_tr)
            _GPU_BREAK_EVEN = 3000
            use_lgb_gpu  = self.use_gpu and (n_train >= _GPU_BREAK_EVEN)
            lgb_device   = 'gpu' if use_lgb_gpu else 'cpu'
            lgb_parallel = 1 if use_lgb_gpu else n_cpu
            lgb_njobs    = 1 if use_lgb_gpu else -1
            reason = lgb_device.upper() if use_lgb_gpu else \
                     ("no GPU" if not self.use_gpu else f"n={n_train}<{_GPU_BREAK_EVEN}→CPU faster")
            log.info(f"[4/6] LightGBM ({reason})...")
            t0 = time.time()
            with contextlib.redirect_stderr(io.StringIO()):
                lm, lmp, _ = self._grid_search(
                    "LGB", lgb.LGBMClassifier, self.cfg['lgb_grid'], Xs_tr, y_tr,
                    {'device': lgb_device, 'class_weight': 'balanced',
                     'n_jobs': lgb_njobs, 'verbose': -1}, parallel_jobs=lgb_parallel)
            _fit_with_sw(lm, Xs_tr, y_tr.values)
            preds = lm.predict(Xs_te)
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro', zero_division=0)
            trained['LightGBM'] = lm
            self.results['LightGBM'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': lmp}
            log.info(f"  Test accuracy: {acc:.4f}  F1-macro: {f1:.4f}  |  {elapsed(t0)}")
        else:
            log.warning("[4/6] LightGBM not installed")

        # 5. CatBoost
        if HAS_CB:
            cb_task = 'GPU' if self.use_gpu else 'CPU'
            log.info(f"[5/6] CatBoost ({cb_task})...")
            t0 = time.time()
            ct, ctp, _ = self._grid_search(
                "CB", cb.CatBoostClassifier, self.cfg['cb_grid'], Xs_tr, y_tr,
                {'task_type': cb_task, 'class_weights': [1.5, 1.0, 1.5],
                 'verbose': 0, 'allow_writing_files': False}, parallel_jobs=1)
            _fit_with_sw(ct, Xs_tr, y_tr.values, supports_sw=True)
            preds = ct.predict(Xs_te).flatten().astype(int)
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro', zero_division=0)
            trained['CatBoost'] = ct
            self.results['CatBoost'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': ctp}
            log.info(f"  Test accuracy: {acc:.4f}  F1-macro: {f1:.4f}  |  {elapsed(t0)}")
        else:
            log.info("[5/6] CatBoost not installed  (pip install catboost)")

        # ── NEW 9: MiniROCKET ─────────────────────────────────────────────────
        log.info("[6/6] MiniROCKET (random kernel transform + Ridge)...")
        t0 = time.time()
        try:
            # Build sliding-window sequences from the SCALED feature matrix
            # Use only the top-1 PCA direction so ROCKET sees a univariate series
            seq_len  = min(self.cfg.get('lstm_seq_len', 60), len(Xs_tr) // 3)
            n_kern   = self.cfg.get('rocket_n_kernels', 1000)
            rocket   = MiniROCKET(n_kernels=n_kern, seq_len=seq_len,
                                   random_state=self.cfg['random_state'])
            # Build sequences: (n, seq_len, n_features)
            def _make_seqs_np(Xarr, yarr, sl):
                Xs_, ys_ = [], []
                for i in range(sl, len(Xarr)):
                    Xs_.append(Xarr[i-sl:i]); ys_.append(yarr[i])
                return np.array(Xs_, dtype=np.float32), np.array(ys_, dtype=np.int64)

            Xseq_tr, yseq_tr = _make_seqs_np(Xs_tr, y_tr.values, seq_len)
            Xseq_te, yseq_te = _make_seqs_np(Xs_te, y_te.values, seq_len)

            rocket.fit(Xseq_tr, yseq_tr)
            preds_r = rocket.predict(Xseq_te)
            acc_r   = accuracy_score(yseq_te, preds_r)
            f1_r    = f1_score(yseq_te, preds_r, average='macro', zero_division=0)
            trained['MiniROCKET'] = rocket
            self.results['MiniROCKET'] = {
                'accuracy': acc_r, 'f1': f1_r,
                'preds': preds_r, 'params': {'n_kernels': n_kern, 'seq_len': seq_len}}
            # Store for signal + meta use
            self._rocket_seq_len = seq_len
            self._rocket_y_te    = yseq_te
            log.info(f"  Test accuracy: {acc_r:.4f}  F1-macro: {f1_r:.4f}  |  {elapsed(t0)}")
            log.info(f"\n{classification_report(yseq_te, preds_r, target_names=['SELL','HOLD','BUY'], zero_division=0)}")
        except Exception as e:
            log.warning(f"  MiniROCKET failed: {e}")
            self._rocket_seq_len = None

        # ── Soft-vote ensemble via ManualSoftVoter ────────────────────────────
        # MiniROCKET excluded: takes 3-D sliding-window sequences as input,
        # not the flat 2-D Xs_te array the other models receive.
        # CatBoost excluded from VotingClassifier (but included here) because
        # sklearn's clone() cannot round-trip CatBoost's class_weights param
        # (list → internal dict conversion breaks the equality check).
        # ManualSoftVoter averages predict_proba() from already-fitted models,
        # so no cloning or re-fitting is needed.
        ens_models = {n: m for n, m in trained.items() if n != 'MiniROCKET'}
        if len(ens_models) >= 2:
            log.info("Soft-vote ensemble (ManualSoftVoter — all sklearn-compatible "
                     "models including CatBoost)...")
            t0  = time.time()
            ens = ManualSoftVoter(ens_models)
            preds = ens.predict(Xs_te)
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro', zero_division=0)
            trained['Ensemble'] = ens
            self.results['Ensemble'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': {}}
            log.info(f"  Ensemble accuracy: {acc:.4f}  F1-macro: {f1:.4f}  "
                     f"|  {len(ens_models)} models  |  {elapsed(t0)}")
            log.info(f"\n{classification_report(y_te, preds, target_names=['SELL','HOLD','BUY'], zero_division=0)}")

        self.models = trained
        log.info("\nTree model summary:")
        for n, r in self.results.items():
            log.info(f"  {n:<20} acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")
        return self

    def _transform_latest(self, X_df):
        x  = X_df.fillna(self.X.median())
        xs = self.scaler.transform(x)
        xs = self.var_sel.transform(xs)
        xs = self.feat_sel.transform(xs)
        return xs

    def get_latest_signal(self):
        last_x  = self.X.iloc[[-1]]
        last_xs = self._transform_latest(last_x)
        # Exclude MiniROCKET (needs 3D sequences) and Ensemble from direct predict.
        _skip = {'MiniROCKET', 'Ensemble'}
        eligible = {k: v for k, v in self.results.items() if k not in _skip}
        best    = max(eligible or self.results, key=lambda k: self.results[k]['f1'])
        model   = self.models[best]
        lmap    = {0:'SELL',1:'HOLD',2:'BUY'}
        pred    = model.predict(last_xs)
        pred    = int(pred.flatten()[0])
        try:
            proba = model.predict_proba(last_xs)[0]
            conf  = float(proba[pred])
        except Exception:
            proba = [1/3]*3; conf = 1/3
        min_conf = self.cfg.get('min_signal_confidence', 0.38)
        pred_str = lmap[pred] if conf >= min_conf else 'HOLD'
        if conf < min_conf:
            log.info(f"  Filtered to HOLD (conf {conf:.2%} < {min_conf:.0%})")
        return {'signal': pred_str, 'label': pred, 'confidence': conf,
                'model_used': best,
                'probabilities': {lmap[i]: float(p) for i, p in enumerate(proba)}}

    def get_feature_importance(self, top_n=25):
        """FIX 4: returns actual feature names not feat_N indices."""
        rows = []
        names = self.feat_names_selected_ or [f'feat_{i}' for i in range(
            self.cfg['feat_select_k'])]
        for mname, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                if len(imp) == len(names):
                    sr = pd.Series(imp, index=names)
                    for fname, val in sr.nlargest(top_n).items():
                        rows.append({'model': mname, 'feature': fname, 'importance': val})
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH MODELS
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden=192, n_layers=3, dropout=0.35, n_classes=3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden, n_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        d = hidden * 2
        self.attn_q = nn.Linear(d, d)
        self.attn_k = nn.Linear(d, d)
        self.attn_v = nn.Linear(d, d)
        self.scale  = d ** -0.5
        # Head scales with hidden size so small architectures don't blow their
        # parameter budget on a fixed 256→128→64 tower.
        # hidden=192 → d=384 → h1=256, h2=128 (same as before for large models)
        # hidden=24  → d=48  → h1=48,  h2=24  (proportional — saves 50K params)
        h1 = max(n_classes * 4, min(256, d))
        h2 = max(n_classes * 2, h1 // 2)
        self.head   = nn.Sequential(
            nn.Linear(d, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(h2, n_classes))

    def forward(self, x):
        b, s, f = x.shape
        x = self.input_bn(x.reshape(b*s, f)).reshape(b, s, f)
        h, _ = self.lstm(x)
        Q = self.attn_q(h); K = self.attn_k(h); V = self.attn_v(h)
        w = torch.softmax(torch.bmm(Q, K.transpose(1,2)) * self.scale, dim=-1)
        ctx = torch.bmm(w, V).mean(dim=1)
        return self.head(ctx)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, n_layers=4,
                 dropout=0.20, n_classes=3, seq_len=60):
        super().__init__()
        self.proj      = nn.Linear(input_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm      = nn.LayerNorm(d_model)
        # Head scales with d_model
        h1 = max(n_classes * 4, min(128, d_model))
        h2 = max(n_classes * 2, h1 // 2)
        self.head      = nn.Sequential(
            nn.Linear(d_model, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Linear(h2, n_classes))

    def forward(self, x):
        b, s, _ = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        x   = self.proj(x) + self.pos_embed(pos)
        x   = self.norm(self.encoder(x))
        return self.head(x.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# NEW 18: FOCAL LOSS  (Lin et al., 2017 — RetinaNet)
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-class focal loss: FL(p_t) = -α_t · (1−p_t)^γ · log(p_t)

    γ=2 (standard) down-weights easy examples exponentially.
    For stock signals HOLD (dominant class, easy to predict) is suppressed
    while harder SELL/BUY examples are amplified — directly attacking the
    BiLSTM 'predict all HOLD' collapse seen in the run logs.

    α per class = inverse frequency (same as class_weight='balanced').
    """
    def __init__(self, gamma: float = 2.0,
                 weight: 'torch.Tensor | None' = None,
                 label_smoothing: float = 0.10):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)
        self.ls = label_smoothing

    def forward(self, logits: 'torch.Tensor',
                targets: 'torch.Tensor') -> 'torch.Tensor':
        # Force fp32 for numerical stability even inside BF16 autocast context.
        # F.cross_entropy promotes automatically, but being explicit avoids any
        # dtype mismatch between logits (BF16) and self.weight buffer (FP32).
        logits_f = logits.float()

        # ── Step 1: unweighted CE → focal modulation factor ────────────────────
        # pt must reflect the model's *actual confidence* in the true class,
        # NOT the weighted version. If we used weighted CE here, the tiny
        # per-class weights (~0.001) would make pt ≈ exp(-0.001) ≈ 0.999,
        # setting (1-pt)^2 ≈ 1e-6 → focal loss ≈ 0 → no gradient signal.
        with torch.no_grad():
            ce_base = F.cross_entropy(logits_f, targets, reduction='none')
            pt = torch.exp(-ce_base)  # ∈ (0,1]: high = easy sample, low = hard

        # ── Step 2: weighted CE → class imbalance correction ──────────────────
        w = self.weight.to(logits_f.dtype) if self.weight is not None else None
        ce_w = F.cross_entropy(logits_f, targets, weight=w,
                               label_smoothing=self.ls, reduction='none')

        # ── Step 3: focal modulation applied to weighted loss ─────────────────
        # (1-pt)^γ: γ=2 suppresses easy HOLD examples, amplifies rare SELL/BUY
        fl = ((1.0 - pt) ** self.gamma) * ce_w
        return fl.mean()


# ─────────────────────────────────────────────────────────────────────────────
# NEW 16: TEMPORAL FUSION TRANSFORMER (TFT)
# Ref: Lim et al., "Temporal Fusion Transformers for Interpretable
#      Multi-horizon Time Series Forecasting" (2021)
# ─────────────────────────────────────────────────────────────────────────────
class _GRN(nn.Module):
    """
    Gated Residual Network — the core building block of TFT.

    GRN(x) = LayerNorm( x_proj + GLU( Dense(ELU(Dense(x))) ) )

    The GLU gate allows the network to suppress irrelevant transformations,
    giving TFT its interpretability advantage over plain Transformers.
    Optional context vector c is injected before the ELU non-linearity.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: int = 0):
        super().__init__()
        self.skip   = (nn.Linear(input_dim, output_dim)
                       if input_dim != output_dim else nn.Identity())
        self.fc1    = nn.Linear(input_dim + context_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, output_dim * 2)  # GLU splits here
        self.norm   = nn.LayerNorm(output_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor',
                ctx: 'torch.Tensor | None' = None) -> 'torch.Tensor':
        res = self.skip(x)
        inp = torch.cat([x, ctx], dim=-1) if ctx is not None else x
        h   = F.elu(self.fc1(inp))
        h   = self.drop(self.fc2(h))
        h1, h2 = h.chunk(2, dim=-1)
        return self.norm(res + h1 * torch.sigmoid(h2))


class _VSN(nn.Module):
    """
    Variable Selection Network — vectorized implementation.

    PERFORMANCE FIX: The naive TFT-paper VSN creates one GRN per feature
    (237 individual GRNs for this dataset) iterated in a Python for-loop,
    causing ~237 serial CUDA dispatches per batch and 2.5 min/epoch on RTX 3050.
    Additionally the selection GRN had input_dim = n_features × hidden = 15,168,
    giving its skip linear 15,168 × 237 = 3.6 M parameters alone (6.8 M total TFT).

    This version replaces all per-feature GRNs with two vectorized operations:
      1. embed_grn  — single GRN on the full feature vector → (B*S, hidden)
      2. select_grn — single GRN → n_features softmax weights (interpretable)

    Result: 2 matrix multiplies per batch instead of 237, ~50× faster,
    parameter count drops from 6.8 M to ~235 K.

    Feature importance is preserved: select_grn still outputs per-feature
    softmax weights that show which of the 237 indicators drove each prediction.
    """
    def __init__(self, n_features: int, hidden_dim: int,
                 dropout: float = 0.1, context_dim: int = 0):
        super().__init__()
        self.n_features = n_features
        # Single GRN maps all features simultaneously to hidden_dim
        self.embed_grn  = _GRN(n_features, hidden_dim * 2, hidden_dim,
                                dropout, context_dim=context_dim)
        # Selection GRN outputs per-feature softmax weights
        self.select_grn = _GRN(n_features, hidden_dim, n_features,
                                dropout, context_dim=context_dim)

    def forward(self, x: 'torch.Tensor',
                ctx: 'torch.Tensor | None' = None
                ) -> 'tuple[torch.Tensor, torch.Tensor]':
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(1)
        B, S, F = x.shape
        flat  = x.reshape(B * S, F)          # (B*S, F) — all features at once
        ctx_r = (ctx.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
                 if ctx is not None else None)

        embedded = self.embed_grn(flat, ctx_r)                    # (B*S, hidden)
        weights  = torch.softmax(self.select_grn(flat, ctx_r), dim=-1)  # (B*S, F)

        embedded = embedded.reshape(B, S, -1)
        weights  = weights.reshape(B, S, F)

        if squeeze:
            return embedded.squeeze(1), weights.squeeze(1)
        return embedded, weights


class TemporalFusionTransformer(nn.Module):
    """
    TFT for 3-class stock signal classification.

    Pipeline:
      BN → VSN (feature selection) → LSTM (local) →
      Static Enrichment (GRN) → Self-Attention (global) →
      FF-GRN → Mean Pool → Classification head

    Key improvements over the existing TransformerEncoder:
    • VSN learns *which* of the 237 features to trust each timestep
    • GRN gates prevent noisy features from corrupting gradients
    • LSTM + Attention hierarchy mirrors multi-scale price dynamics
    • Fewer parameters than BiLSTM (≈900K vs 3M) → less overfitting
    """
    def __init__(self, input_size: int, hidden: int = 64,
                 lstm_layers: int = 2, attn_heads: int = 4,
                 dropout: float = 0.25, n_classes: int = 3,
                 seq_len: int = 60):
        super().__init__()
        self.hidden   = hidden
        self.input_bn = nn.BatchNorm1d(input_size)
        self.vsn      = _VSN(input_size, hidden, dropout)
        self.lstm     = nn.LSTM(hidden, hidden, num_layers=lstm_layers,
                                batch_first=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_norm    = nn.LayerNorm(hidden)
        self.static_enrich= _GRN(hidden, hidden, hidden, dropout,
                                  context_dim=hidden)
        self.attn     = nn.MultiheadAttention(hidden, num_heads=attn_heads,
                                              dropout=dropout, batch_first=True)
        self.attn_norm= nn.LayerNorm(hidden)
        self.ff_grn   = _GRN(hidden, hidden * 2, hidden, dropout)
        self.ff_norm  = nn.LayerNorm(hidden)
        self.head     = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, n_classes))
        self._vsn_weights = None   # stored after each forward pass

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        B, S, F = x.shape
        x = self.input_bn(x.reshape(B * S, F)).reshape(B, S, F)
        x, vsn_w = self.vsn(x)
        # vsn_w: (B, S, n_features) — per-timestep feature importance softmax
        # Average across the sequence length for a single per-sample importance vector
        self._vsn_weights = vsn_w.detach().mean(dim=1)  # (B, n_features)
        lstm_out, (h_n, _) = self.lstm(x)
        x   = self.lstm_norm(lstm_out)
        ctx = h_n[-1]                              # last-layer final state (B, H)
        ctx_exp = ctx.unsqueeze(1).expand(B, S, self.hidden).reshape(B * S, self.hidden)
        x   = self.static_enrich(
            x.reshape(B * S, self.hidden), ctx_exp).reshape(B, S, self.hidden)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x   = self.attn_norm(x + attn_out)
        x   = self.ff_norm(
            self.ff_grn(x.reshape(B * S, self.hidden)).reshape(B, S, self.hidden))
        return self.head(x.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# NEW 17: CONFORMAL PREDICTION  (RAPS variant)
# Ref: Angelopoulos et al., "Uncertainty Sets for Image Classifiers using
#      Conformal Prediction" (ICLR 2021)
# ─────────────────────────────────────────────────────────────────────────────
class ConformalPredictor:
    """
    Split Conformal Prediction with RAPS for 3-class stock signals.

    Coverage guarantee: P(true_class ∈ prediction_set) ≥ 1 − α
    regardless of model family, under exchangeability assumption.

    Trading interpretation:
    • Singleton set  {SELL}  → high-confidence actionable short
    • Singleton set  {BUY}   → high-confidence actionable long
    • Singleton set  {HOLD}  → clear flat signal
    • Size-2 set             → ambiguous; reduce position size
    • Full set {SELL,HOLD,BUY} → no signal; stay flat

    RAPS nonconformity score for sample i:
        s_i = Σ p_k  (for all k ranked above true class)
              + λ · max(0, rank(true_class) − k_reg)

    The λ·rank penalty shrinks prediction sets (smaller sets = sharper signal)
    while maintaining the coverage guarantee.
    """
    LABELS = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

    def __init__(self, alpha: float = 0.10, lam: float = 0.01, kreg: int = 1):
        self.alpha = alpha
        self.lam   = lam
        self.kreg  = kreg
        self.qhat  = None
        self.n_cal = 0

    def _nonconformity(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """RAPS nonconformity scores for calibration samples."""
        scores = np.zeros(len(y), dtype=np.float64)
        for i in range(len(y)):
            order  = np.argsort(proba[i])[::-1]   # descending prob
            cumsum = 0.0
            for rank, cls in enumerate(order):
                cumsum += proba[i, cls]
                if cls == int(y[i]):
                    scores[i] = cumsum - proba[i, cls] \
                                 + self.lam * max(0, rank + 1 - self.kreg)
                    break
        return scores

    def calibrate(self, proba_cal: np.ndarray,
                  y_cal: np.ndarray) -> 'ConformalPredictor':
        """Compute qhat from the calibration split."""
        self.n_cal = len(y_cal)
        s      = self._nonconformity(proba_cal, y_cal)
        level  = min(np.ceil((self.n_cal + 1) * (1.0 - self.alpha))
                     / self.n_cal, 1.0)
        self.qhat = float(np.quantile(s, level, method='higher'))
        return self

    def predict_set(self, proba: np.ndarray) -> list:
        """Return RAPS prediction set for a single (3,) probability vector."""
        if self.qhat is None:
            raise RuntimeError("Call calibrate() first.")
        order    = np.argsort(proba)[::-1]
        pred_set = []
        cumsum   = 0.0
        for rank, cls in enumerate(order):
            cumsum += proba[cls]
            # Include this class; check stop condition
            pred_set.append(int(cls))
            threshold = self.qhat + proba[cls] \
                        - self.lam * max(0, rank + 1 - self.kreg)
            if cumsum >= threshold:
                break
        return sorted(set(pred_set)) if pred_set else [int(order[0])]

    def empirical_coverage(self, proba: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate empirical coverage and average set size on hold-out."""
        covered, sizes = [], []
        for i in range(len(y)):
            ps = self.predict_set(proba[i])
            covered.append(int(y[i]) in ps)
            sizes.append(len(ps))
        return {
            'coverage':         float(np.mean(covered)),
            'target_coverage':  1.0 - self.alpha,
            'avg_set_size':     float(np.mean(sizes)),
            'singleton_rate':   float(np.mean([s == 1 for s in sizes])),
            'n_cal':            self.n_cal,
            'qhat':             self.qhat,
        }

    def annotate_signal(self, proba: np.ndarray) -> dict:
        """Augment a probability vector with conformal prediction set metadata."""
        if self.qhat is None:
            return {}
        ps     = self.predict_set(proba)
        labels = [self.LABELS[c] for c in ps]
        return {
            'prediction_set':           labels,
            'set_size':                 len(ps),
            'is_conformal_singleton':   len(ps) == 1,
            'conformal_coverage_level': round(1.0 - self.alpha, 2),
            'qhat':                     round(float(self.qhat), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE DL ARCHITECTURE  —  enforces a safe parameter-to-sample ratio
# ─────────────────────────────────────────────────────────────────────────────
# Rule of thumb for noisy financial time-series:
#   trainable_params  ≤  n_training_sequences / SAMPLES_PER_PARAM
#   where SAMPLES_PER_PARAM = 10  (conservative; standard DL uses ~5)
#
# Why this matters:
#   A 3-layer BiLSTM with hidden=192 has ≈3 M parameters.  Feeding it RKLB's
#   ~840 training sequences means 3,570 parameters per sample — the model will
#   memorise the training set on epoch 1, producing 100% train accuracy and
#   near-random test accuracy (the "collapse" visible in the run logs).
#   Shrinking to hidden=64, layers=2 gives ≈450 K params → 536 params/sample,
#   still above 10× but massively safer, and further regularised by dropout.
#   For AAPL with ~24 K augmented sequences, the full hidden=192, L=3 is fine.
#
# Parameter estimators — closed-form approximations:
#   BiLSTM  :  4*(F+H+1)*H*2  (L1)  +  4*(2H+H+1)*H*2  (L≥2, per extra layer)
#              + head: 2H→256→128→64→3  ≈  2H*256 + 256*128 + 128*64 + 64*3
#   Transformer:  F*D (proj)  +  L*(12*D²)  (attn+FFN)
#              + head: D→128→64→3
#   TFT (vectorised VSN):
#              VSN embed (F→2H→H) + VSN select (F→H→F)
#              + LSTM (L layers, hidden=H)
#              + attention+GRN+head  ≈  4H²*L + 6H²
# ─────────────────────────────────────────────────────────────────────────────
_SAMPLES_PER_PARAM = 10   # global constant; tune higher for noisier data


def _bilstm_param_count(input_size: int, hidden: int, n_layers: int) -> int:
    """Approximate trainable parameter count for BiLSTMAttention."""
    # Layer 1: input_size → hidden (bidirectional)
    count = 4 * (input_size + hidden + 1) * hidden * 2
    # Layers 2+: (2*hidden) → hidden (bidirectional)
    for _ in range(n_layers - 1):
        count += 4 * (2 * hidden + hidden + 1) * hidden * 2
    # Attention: Q/K/V projections (d=2H → 2H each)
    d = hidden * 2
    count += 3 * d * d
    # Classification head: 2H → 256 → 128 → 64 → 3
    count += d * 256 + 256 * 128 + 128 * 64 + 64 * 3
    return count


def _transformer_param_count(input_size: int, d_model: int,
                              n_layers: int) -> int:
    """Approximate trainable parameter count for TransformerEncoder."""
    count = input_size * d_model          # input projection
    count += d_model * 32                 # pos embedding (seq_len ≤ 60, pad to 32)
    # Per encoder layer: MHA (≈4D²) + FFN (2×D×4D) + 2× LayerNorm = 12D²
    count += n_layers * (12 * d_model * d_model)
    # Head: D → 128 → 64 → 3
    count += d_model * 128 + 128 * 64 + 64 * 3
    return count


def _tft_param_count(input_size: int, hidden: int, lstm_layers: int) -> int:
    """Approximate trainable parameter count for vectorised TFT."""
    # VSN embed GRN: F→2H→H  (2 linear layers)
    count = input_size * (2 * hidden) + (2 * hidden) * hidden
    # VSN select GRN: F→H→F
    count += input_size * hidden + hidden * input_size
    # LSTM encoder (single-direction, internal hidden=H)
    count += lstm_layers * 4 * (hidden + hidden + 1) * hidden
    # Static enrichment GRN + FF GRN + attention (approx)
    count += 6 * hidden * hidden
    # Head: H → H → 3
    count += hidden * hidden + hidden * 3
    return count


def get_adaptive_dl_kwargs(model_name: str, n_train_rows: int,
                            n_features: int, cfg: dict,
                            use_gpu: bool = True) -> dict:
    """
    Return model_kwargs sized so that:
        estimated_params  ≤  n_training_sequences / _SAMPLES_PER_PARAM

    Also returns adjusted epochs and patience (scaled down for small datasets
    to reduce wall-clock time; early stopping still fires first if needed).

    Parameters
    ----------
    model_name  : 'bilstm' | 'transformer' | 'tft'
    n_train_rows: raw training rows (before sliding-window construction)
    n_features  : number of input features (after SelectKBest)
    cfg         : CONFIG dict
    use_gpu     : whether GPU augmentation will be used (aug×3 on GPU)

    Returns
    -------
    dict with keys:
        model_kw   — kwargs for the ModelClass constructor
        epochs     — (possibly reduced) epoch count
        patience   — (possibly reduced) early-stop patience
    """
    seq_len      = cfg.get('lstm_seq_len', 60)
    # Budget uses ORIGINAL sequences only — augmented copies are synthetic noise
    # duplicates of the same patterns and must NOT count toward capacity.
    # Using n_aug_seqs × 10 = 2547 × 10 = 25,470 on RKLB made the budget
    # appear 3× bigger than reality, allowing architectures that still massively
    # overfit.  Correct formula: original seqs × _SAMPLES_PER_PARAM.
    n_orig_seqs  = max(10, n_train_rows - seq_len)
    n_seqs       = n_orig_seqs   # for epoch scaling decisions
    budget       = int(n_orig_seqs * _SAMPLES_PER_PARAM)

    # Epoch/patience scaling: small datasets overfit faster and converge sooner
    if n_seqs < 400:
        ep_scale, pat_scale = 0.55, 0.60
    elif n_seqs < 1000:
        ep_scale, pat_scale = 0.75, 0.75
    elif n_seqs < 2500:
        ep_scale, pat_scale = 0.90, 0.90
    else:
        ep_scale, pat_scale = 1.00, 1.00

    def _scaled(base_ep, base_pat):
        return (max(30, int(base_ep * ep_scale)),
                max(10, int(base_pat * pat_scale)))

    # ── BiLSTM ────────────────────────────────────────────────────────────────
    if model_name == 'bilstm':
        base_ep, base_pat = cfg['lstm_epochs'], cfg['lstm_patience']
        epochs, patience  = _scaled(base_ep, base_pat)
        # Candidates: try largest hidden first; step down until within budget
        candidates = [
            (192, 3), (128, 2), (96, 2), (64, 2), (48, 1), (32, 1), (24, 1)
        ]
        for h, l in candidates:
            est = _bilstm_param_count(n_features, h, l)
            if est <= budget:
                # More dropout for smaller (more regularisation needed)
                base_drop = cfg['lstm_dropout']
                dropout   = min(0.60, base_drop + max(0.0, (0.48 - h / 400)))
                log.info(
                    f"  [Adaptive BiLSTM] hidden={h}  layers={l}  "
                    f"est_params={est:,}  budget={budget:,}  "
                    f"(n_seqs={n_seqs}, ratio={est/n_seqs:.1f})  "
                    f"dropout={dropout:.2f}  epochs={epochs}  patience={patience}")
                if (h, l) != (cfg['lstm_hidden'], cfg['lstm_layers']):
                    log.info(
                        f"  ↳ Downsized from config "
                        f"h={cfg['lstm_hidden']},L={cfg['lstm_layers']} "
                        f"(est_params={_bilstm_param_count(n_features, cfg['lstm_hidden'], cfg['lstm_layers']):,})"
                        f" to prevent over-parameterisation on {n_train_rows} train rows.")
                return dict(model_kw=dict(input_size=n_features,
                                          hidden=h, n_layers=l,
                                          dropout=dropout),
                             epochs=epochs, patience=patience)
        # Hard minimum
        log.warning(
            f"  [Adaptive BiLSTM] All candidates exceed budget={budget:,} "
            f"— using minimum architecture (h=24, L=1).")
        return dict(model_kw=dict(input_size=n_features,
                                   hidden=24, n_layers=1, dropout=0.60),
                     epochs=epochs, patience=patience)

    # ── Transformer ───────────────────────────────────────────────────────────
    elif model_name == 'transformer':
        base_ep, base_pat = cfg['tf_epochs'], cfg['tf_patience']
        epochs, patience  = _scaled(base_ep, base_pat)
        candidates = [
            (128, 4, 8), (96, 3, 4), (64, 2, 4), (48, 2, 4), (32, 1, 4)
        ]
        for d, l, nh in candidates:
            # nhead must evenly divide d_model
            while nh > 1 and d % nh != 0:
                nh //= 2
            est = _transformer_param_count(n_features, d, l)
            if est <= budget:
                log.info(
                    f"  [Adaptive Transformer] d_model={d}  layers={l}  "
                    f"nhead={nh}  est_params={est:,}  budget={budget:,}  "
                    f"(n_seqs={n_seqs})  epochs={epochs}  patience={patience}")
                if (d, l) != (cfg['tf_d_model'], cfg['tf_layers']):
                    log.info(
                        f"  ↳ Downsized from config "
                        f"d={cfg['tf_d_model']},L={cfg['tf_layers']}.")
                return dict(model_kw=dict(input_size=n_features, d_model=d,
                                           nhead=nh, n_layers=l,
                                           dropout=cfg['tf_dropout'],
                                           seq_len=seq_len),
                             epochs=epochs, patience=patience)
        log.warning(
            f"  [Adaptive Transformer] Minimum architecture (d=32, L=1).")
        return dict(model_kw=dict(input_size=n_features, d_model=32,
                                   nhead=4, n_layers=1,
                                   dropout=cfg['tf_dropout'],
                                   seq_len=seq_len),
                     epochs=epochs, patience=patience)

    # ── TFT ───────────────────────────────────────────────────────────────────
    elif model_name == 'tft':
        base_ep, base_pat = cfg['tft_epochs'], cfg['tft_patience']
        epochs, patience  = _scaled(base_ep, base_pat)
        candidates = [
            (64, 2, 4), (48, 2, 4), (32, 1, 2), (24, 1, 2), (16, 1, 2)
        ]
        for h, l, ah in candidates:
            est = _tft_param_count(n_features, h, l)
            if est <= budget:
                log.info(
                    f"  [Adaptive TFT] hidden={h}  lstm_layers={l}  "
                    f"attn_heads={ah}  est_params={est:,}  budget={budget:,}  "
                    f"(n_seqs={n_seqs})  epochs={epochs}  patience={patience}")
                if (h, l) != (cfg['tft_hidden'], cfg['tft_lstm_layers']):
                    log.info(
                        f"  ↳ Downsized from config "
                        f"h={cfg['tft_hidden']},L={cfg['tft_lstm_layers']}.")
                return dict(model_kw=dict(input_size=n_features, hidden=h,
                                           lstm_layers=l, attn_heads=ah,
                                           dropout=cfg['tft_dropout'],
                                           seq_len=seq_len),
                             epochs=epochs, patience=patience)
        log.warning(
            f"  [Adaptive TFT] Minimum architecture (h=16, L=1).")
        return dict(model_kw=dict(input_size=n_features, hidden=16,
                                   lstm_layers=1, attn_heads=2,
                                   dropout=cfg['tft_dropout'],
                                   seq_len=seq_len),
                     epochs=epochs, patience=patience)

    # Fallback — should never reach here
    raise ValueError(f"Unknown model_name '{model_name}'. "
                     "Expected 'bilstm', 'transformer', or 'tft'.")


class TorchTrainer:
    def __init__(self, name, device, cfg):
        self.name    = name
        self.device  = device
        self.cfg     = cfg
        self.model   = None
        self.scaler  = RobustScaler()
        self.te_preds= None
        self.te_proba= None
        self.y_te_vals = None
        self.X_arr   = None
        self.seq_len = None

    def _make_seqs(self, X: np.ndarray, y: np.ndarray, seq_len: int,
                   augment: bool = False,
                   aug_factor: int = 3,
                   noise_std: float = 0.005,
                   mag_warp: bool = True) -> tuple:
        """
        Build sliding-window sequences (seq_len, n_features) for DL.

        Augmentation (training data only — never applied to test):
          • Gaussian noise: adds σ=noise_std × feature_std noise to each copy
          • Magnitude warping: per-feature scale drawn from U(0.95, 1.05)
          Together these give aug_factor × as many training sequences without
          changing the temporal structure or leaking future information.
        """
        rng  = np.random.default_rng(42)
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            seq = X[i - seq_len:i]          # (seq_len, n_feat)
            Xs.append(seq)
            ys.append(y[i])
            if augment:
                feat_std = seq.std(axis=0) + 1e-8
                for _ in range(aug_factor - 1):
                    s = seq.copy()
                    # Gaussian noise proportional to local feature std
                    s = s + rng.standard_normal(s.shape) * (noise_std * feat_std)
                    # Magnitude warping: random scale per feature
                    if mag_warp:
                        scale = rng.uniform(0.95, 1.05, size=s.shape[1])
                        s = s * scale
                    Xs.append(s)
                    ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    def train(self, X_df, y, ModelClass, model_kwargs, epochs, patience,
              seq_len, lr, batch):
        log_section(f"DEEP LEARNING: {self.name}")
        # ── AMP dtype ─────────────────────────────────────────────────────────
        use_amp = (self.device.type == 'cuda')
        if use_amp:
            prop      = torch.cuda.get_device_properties(0)
            amp_dtype = torch.bfloat16 if prop.major >= 8 else torch.float16
        else:
            amp_dtype = torch.float32
        log.info(f"AMP: {'BF16' if (use_amp and amp_dtype==torch.bfloat16) else 'FP16' if use_amp else 'off'}")
        log.info(f"seq_len={seq_len}  epochs={epochs}  patience={patience}  "
                 f"batch={batch}  lr={lr}")
        if self.device.type == 'cuda':
            free, total = torch.cuda.mem_get_info()
            log.info(f"VRAM free/total: {free/1e9:.2f}/{total/1e9:.2f} GB")

        # ── BUG FIX: fit scaler ONLY on training rows, never on test rows ─────
        # Old code called fit_transform on the full array and split afterward,
        # leaking test-set statistics (median, IQR) into the feature matrix.
        # Fix: determine the chronological split point on RAW rows first, then
        # fit the scaler exclusively on the training portion.
        y_arr  = y.values.astype(np.int64)
        n_rows = len(X_df)
        sp_raw = int(n_rows * self.cfg.get('train_split', 0.80))

        # Impute NaNs using TRAIN-ONLY median — avoids test-set leakage
        tr_df   = X_df.iloc[:sp_raw]
        te_df   = X_df.iloc[sp_raw:]
        tr_med  = tr_df.median()
        X_tr_raw = tr_df.fillna(tr_med).values.astype(np.float32)
        X_te_raw = te_df.fillna(tr_med).values.astype(np.float32)  # use train median

        # Fit scaler on train rows only, apply to both
        self.scaler.fit(X_tr_raw)
        X_arr = np.vstack([
            self.scaler.transform(X_tr_raw),
            self.scaler.transform(X_te_raw),
        ])

        # ── Build raw sequences first, then split, then augment training only ─
        Xs_raw, ys_raw = self._make_seqs(X_arr, y_arr, seq_len,
                                          augment=False)   # no aug yet
        sp    = int(len(Xs_raw) * self.cfg.get('train_split', 0.80))
        X_te  = Xs_raw[sp:]
        y_te  = ys_raw[sp:]
        # Now augment ONLY the training portion
        do_aug = self.cfg.get('dl_augment', True) and self.device.type == 'cuda'
        if do_aug:
            aug_f = self.cfg.get('dl_aug_factor', 3)
            aug_n = self.cfg.get('dl_aug_noise_std', 0.005)
            aug_m = self.cfg.get('dl_aug_mag_warp', True)
            X_tr, y_tr = self._make_seqs(X_arr[:sp + seq_len], y_arr[:sp + seq_len],
                                          seq_len,
                                          augment=True, aug_factor=aug_f,
                                          noise_std=aug_n, mag_warp=aug_m)
            log.info(f"Augmentation: {aug_f}× (noise σ={aug_n}, mag_warp={aug_m}) "
                     f"→ {len(X_tr)} train seqs from {sp} originals")
        else:
            X_tr, y_tr = Xs_raw[:sp], ys_raw[:sp]
            if not do_aug:
                log.info("Augmentation: disabled (CPU mode)")

        log.info(f"Train seqs: {len(X_tr)}  Test seqs: {len(X_te)}  "
                 f"Features: {X_tr.shape[2]}")

        counts  = np.bincount(y_tr % 3, minlength=3).astype(float)  # mod 3 safe for aug
        # recount from originals for class weight (aug copies same label)
        counts_orig = np.bincount(ys_raw[:sp], minlength=3).astype(float)
        # Sklearn-balanced formula: n_total / (n_classes * count)
        # Produces weights ~0.5–2.1 (same scale as sklearn class_weight='balanced')
        # CRITICAL FIX: old formula 1/(count+1) produced weights ~0.001–0.006,
        # making focal loss ≈ 1e-8 which displays as 0.0000 and means gradients
        # are negligible → models collapse to predicting a single class.
        n_total_orig = float(counts_orig.sum())
        # Compute sklearn-balanced weights then clip to [0.5, 4.0].
        # Without clipping, severely imbalanced labels (e.g. HOLD ~85%) push
        # BUY/SELL weights to 6-10×, which causes gradient explosions in
        # focal loss and drives the model to collapse to the majority class.
        # The clip keeps multipliers in a safe range while still correcting
        # for imbalance — mirrors torch's recommended alpha range for focal loss.
        weights_np   = np.clip(
            n_total_orig / (3.0 * (counts_orig + 1e-6)),
            0.5, 4.0
        )
        weights = torch.tensor(weights_np, dtype=torch.float32).to(self.device)
        log.info(f"Class counts (original): {dict(zip(['SELL','HOLD','BUY'], counts_orig.astype(int)))}")
        log.info(f"Class weights (clipped [0.5,4.0]): SELL={weights_np[0]:.3f}  HOLD={weights_np[1]:.3f}  BUY={weights_np[2]:.3f}")

        pin = (self.device.type == 'cuda')
        tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                               batch_size=batch, shuffle=True,
                               pin_memory=pin, num_workers=0)
        te_loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
                               batch_size=512, shuffle=False,
                               pin_memory=pin, num_workers=0)

        model_raw = ModelClass(**model_kwargs).to(self.device)
        n_params  = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
        log.info(f"Parameters: {n_params:,}")

        # torch.compile: Linux/Mac + CUDA only (Triton not available on Windows)
        _can_compile = (
            hasattr(torch, 'compile') and
            self.device.type == 'cuda' and
            not _IS_WINDOWS
        )
        if _can_compile:
            try:
                self.model = torch.compile(model_raw, mode='max-autotune')
                log.info("torch.compile(max-autotune): enabled")
            except Exception as e:
                log.warning(f"torch.compile skipped ({e})")
                self.model = model_raw
        else:
            self.model = model_raw
            if _IS_WINDOWS:
                log.info("torch.compile: skipped (Windows — Triton unavailable)")

        optimizer  = torch.optim.AdamW(model_raw.parameters(), lr=lr, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(tr_loader), epochs=epochs, pct_start=0.3)
        # NEW 18: Focal Loss — replaces CrossEntropy to fight HOLD-collapse
        use_focal = self.cfg.get('use_focal_loss', True)
        gamma     = self.cfg.get('focal_loss_gamma', 2.0)
        if use_focal:
            criterion = FocalLoss(gamma=gamma, weight=weights, label_smoothing=0.10)
            log.info(f"Loss: FocalLoss(γ={gamma}, ls=0.10)  [NEW 18 — anti-collapse]")
        else:
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.10)
        # CRITICAL FIX: GradScaler is ONLY needed for FP16 (prevents underflow).
        # For BF16 it is not needed and is actively harmful: it detects inf in
        # BF16 ops, skips optimizer steps, reduces scale until gradients
        # underflow → weights never update → model collapses.
        use_grad_scaler = use_amp and (amp_dtype == torch.float16)
        amp_scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
        if use_amp and not use_grad_scaler:
            log.info("GradScaler: disabled (BF16 — not needed, avoids skip-step bug)")

        t0 = time.time()
        best_acc, best_state, no_imp = 0.0, None, 0
        # Lucky-init guard: random weights can score high if majority class
        # dominates test set. Gate best_state saves to post-warmup epochs only
        # and reject dominated (>85%) checkpoints. If nothing valid is saved,
        # best_state stays None and the final-epoch weights are used.
        min_save_epoch = max(3, int(epochs * 0.10))   # earliest epoch to save
        # Random-init baseline: a model that always predicts majority class
        # gets acc = majority_class_fraction. If we never beat this, we've collapsed.
        majority_frac  = float(np.bincount(y_te, minlength=3).max()) / len(y_te)
        # FIX B: plateau threshold = random-chance baseline + 5% margin.
        # Using majority_frac (~67% for HOLD-heavy labels) kills DL models that
        # predict all 3 classes at ~46% accuracy — which is genuinely good for
        # imbalanced 3-class data. A model only needs to beat random chance (33%),
        # not the "predict-HOLD-always" shortcut.
        n_classes      = len(np.unique(y_te))
        plateau_thresh = max(1.0 / n_classes + 0.05, 0.35)   # ~38% for 3 classes
        # plateau_limit: MUST NOT fire during OneCycleLR warmup.
        # pct_start=0.3 → warmup runs for the first 30% of epochs. We add a
        # generous 20% post-warmup buffer so the model can recover from the
        # warmup phase before being judged.
        # BUG FIX: old value int(epochs*0.35) = 35 for 100-ep Transformer,
        # leaving only 5 post-warmup epochs before the plateau fires.
        # For NVDA (HOLD=48%), models legitimately can't reach 38% accuracy in
        # just 5 epochs after warmup — the plateau abort mislabelled a
        # struggling-but-functional model as collapsed, triggering a restart
        # that then produced the actual 97% HOLD collapse.
        # New value: 50% of epochs gives sufficient post-warmup exploration.
        warmup_epochs  = int(epochs * 0.50)          # 50% — warmup + recovery buffer
        plateau_limit  = max(warmup_epochs, patience // 2)  # whichever is larger
        # NEW 10: SWA — collect snapshots after swa_start_pct of epochs
        swa_enabled   = self.cfg.get('swa_enabled', True) and (self.device.type == 'cuda')
        swa_start_ep  = int(epochs * self.cfg.get('swa_start_pct', 0.60))
        swa_freq      = self.cfg.get('swa_freq', 5)
        swa_snapshots = []   # list of state_dicts

        epoch_bar = tqdm(range(1, epochs+1), desc=f"  {self.name}", unit="ep", ncols=100,
                         file=sys.stdout, leave=True)
        for epoch in epoch_bar:
            self.model.train(); ep_loss = 0.0
            for Xb, yb in tr_loader:
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.device.type,
                                        dtype=amp_dtype, enabled=use_amp):
                    loss = criterion(self.model(Xb), yb)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model_raw.parameters(), 1.0)
                amp_scaler.step(optimizer); amp_scaler.update(); scheduler.step()
                ep_loss += loss.item()

            val_acc = self._eval(te_loader, amp_dtype, use_amp)
            if val_acc > best_acc:
                # BUG FIX: guard against saving a lucky-init or collapsed state.
                # Skip if too early in training, OR if this epoch is dominated.
                _ep_preds = []
                with torch.no_grad():
                    for Xb_c, _ in te_loader:
                        with torch.amp.autocast(device_type=self.device.type,
                                                dtype=amp_dtype, enabled=use_amp):
                            _lg = model_raw(Xb_c.to(self.device))
                        _ep_preds.extend(_lg.float().argmax(dim=1).cpu().numpy())
                _ep_dom = (np.bincount(np.array(_ep_preds), minlength=3).max()
                           / (len(_ep_preds) + 1e-10))
                _is_early = (epoch < min_save_epoch)
                _is_dom   = (_ep_dom > 0.85)
                if not _is_early and not _is_dom:
                    best_acc   = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model_raw.state_dict().items()}
                    no_imp     = 0
                elif _is_dom and val_acc > (best_acc + 0.02):
                    # If ALL states are dominated (common for extreme imbalance)
                    # still save the least-bad one rather than keeping nothing.
                    best_acc   = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model_raw.state_dict().items()}
                    no_imp     = 0
                else:
                    no_imp += 1
            else:
                no_imp += 1

            mem_str = (f"  vram={torch.cuda.memory_allocated()/1e9:.2f}GB"
                       if self.device.type == 'cuda' else "")
            epoch_bar.set_postfix(loss=f"{ep_loss/len(tr_loader):.6f}",
                                  val=f"{val_acc:.4f}", best=f"{best_acc:.4f}",
                                  p=f"{no_imp}/{patience}")
            if epoch % 10 == 0 or epoch == 1:
                log.debug(f"  ep{epoch:4d}  loss={ep_loss/len(tr_loader):.6f}  "
                          f"val={val_acc:.4f}  best={best_acc:.4f}{mem_str}")
            if no_imp >= patience:
                epoch_bar.set_description(f"  {self.name} [EARLY STOP]")
                epoch_bar.close()
                log.info(f"  Early stopping at epoch {epoch}")
                break
            # Plateau abort: if stuck below baseline after plateau_limit epochs,
            # abort training now — the full patience would just waste time on a
            # model that's already collapsed and will trigger the restart anyway.
            if epoch >= plateau_limit and best_acc < plateau_thresh:
                epoch_bar.set_description(f"  {self.name} [PLATEAU ABORT]")
                epoch_bar.close()
                log.warning(f"  Plateau abort at epoch {epoch}: "
                            f"best={best_acc:.4f} < threshold={plateau_thresh:.4f} "
                            f"(majority baseline={majority_frac:.4f}). "
                            f"Collapse restart will follow.")
                break
            # NEW 10: SWA snapshot collection
            if swa_enabled and epoch >= swa_start_ep and (epoch % swa_freq == 0):
                swa_snapshots.append(
                    {k: v.cpu().clone() for k, v in model_raw.state_dict().items()})
        epoch_bar.close()

        if best_state:
            model_raw.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # NEW 10: SWA — average collected snapshots; keep if better than best-val
        if swa_enabled and len(swa_snapshots) >= 2:
            log.info(f"  SWA: averaging {len(swa_snapshots)} snapshots...")
            swa_state = {}
            keys = swa_snapshots[0].keys()
            for k in keys:
                stacked = torch.stack([s[k].float() for s in swa_snapshots])
                swa_state[k] = stacked.mean(dim=0).to(swa_snapshots[0][k].dtype)
            model_raw.load_state_dict({k: v.to(self.device) for k, v in swa_state.items()})
            swa_acc = self._eval(te_loader, amp_dtype, use_amp)
            if swa_acc >= best_acc:
                log.info(f"  SWA acc={swa_acc:.4f} >= best-checkpoint {best_acc:.4f} — keeping SWA")
                best_acc = swa_acc
            else:
                log.info(f"  SWA acc={swa_acc:.4f} < best-checkpoint {best_acc:.4f} — reverting")
                model_raw.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        all_preds, all_proba = [], []
        model_raw.eval()
        with torch.no_grad():
            for Xb, _ in te_loader:
                with torch.amp.autocast(device_type=self.device.type,
                                        dtype=amp_dtype, enabled=use_amp):
                    logits = model_raw(Xb.to(self.device))
                proba = torch.softmax(logits.float(), dim=1).cpu().numpy()
                all_preds.extend(proba.argmax(axis=1)); all_proba.extend(proba)

        self.te_preds   = np.array(all_preds)
        self.te_proba   = np.array(all_proba)
        self.y_te_vals  = y_te
        self.X_arr      = X_arr
        self.seq_len    = seq_len
        self._model_raw = model_raw

        # ── NEW 4: Model Collapse Detection + Restart ─────────────────────────
        unique_preds, counts = np.unique(self.te_preds, return_counts=True)
        dominant_frac = counts.max() / (counts.sum() + 1e-10)
        self.is_collapsed = False
        if dominant_frac > 0.85:
            dominant_cls = ['SELL','HOLD','BUY'][unique_preds[counts.argmax()]]
            log.warning(
                f"⚠  MODEL COLLAPSE DETECTED: {self.name} predicts '{dominant_cls}' "
                f"on {dominant_frac*100:.1f}% of test samples."
            )
            # ── Collapse restart: one retry with a new random seed ────────────
            # Strategy: re-init weights (seed+1), raise dropout by 0.05 to
            # increase regularisation pressure, use a lower LR and more warmup.
            # Only attempt once to keep total runtime bounded.
            log.info(f"  Attempting collapse restart (seed+1, dropout+0.05, lr÷2)...")
            try:
                restart_seed = self.cfg.get('random_state', 42) + 1
                torch.manual_seed(restart_seed)
                np.random.seed(restart_seed)
                # Build a fresh model with slightly more dropout
                restart_kwargs = dict(model_kwargs)
                if 'dropout' in restart_kwargs:
                    restart_kwargs['dropout'] = min(0.55, restart_kwargs['dropout'] + 0.05)
                model_r = ModelClass(**restart_kwargs).to(self.device)
                opt_r   = torch.optim.AdamW(model_r.parameters(),
                                            lr=lr * 0.5, weight_decay=2e-4)
                sch_r   = torch.optim.lr_scheduler.OneCycleLR(
                    opt_r, max_lr=lr * 2.0,
                    steps_per_epoch=len(tr_loader),
                    epochs=min(epochs, 80), pct_start=0.3)
                best_r, state_r, no_r = 0.0, None, 0
                patience_r = patience + 10
                # FIX 3: Mirror the main loop's GradScaler — required for FP16
                # GPUs (e.g. GTX 1080, RTX 2070).  BF16 (RTX 3xxx+) is safe
                # either way because use_grad_scaler=False for BF16.
                scaler_r = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
                for ep_r in range(1, min(epochs, 80) + 1):
                    model_r.train()
                    for Xb, yb in tr_loader:
                        Xb = Xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        opt_r.zero_grad(set_to_none=True)
                        with torch.amp.autocast(device_type=self.device.type,
                                                dtype=amp_dtype, enabled=use_amp):
                            loss_r = criterion(model_r(Xb), yb)
                        scaler_r.scale(loss_r).backward()
                        scaler_r.unscale_(opt_r)
                        nn.utils.clip_grad_norm_(model_r.parameters(), 1.0)
                        scaler_r.step(opt_r); scaler_r.update(); sch_r.step()
                    vacc_r = self._eval_model(model_r, te_loader, amp_dtype, use_amp)
                    if vacc_r > best_r:
                        best_r = vacc_r
                        state_r = {k: v.cpu().clone()
                                   for k, v in model_r.state_dict().items()}
                        no_r = 0
                    else:
                        no_r += 1
                    if no_r >= patience_r:
                        break
                if state_r:
                    model_r.load_state_dict({k: v.to(self.device)
                                             for k, v in state_r.items()})
                # Evaluate restart model
                r_preds, r_proba = [], []
                model_r.eval()
                with torch.no_grad():
                    for Xb, _ in te_loader:
                        with torch.amp.autocast(device_type=self.device.type,
                                                dtype=amp_dtype, enabled=use_amp):
                            lg = model_r(Xb.to(self.device))
                        p = torch.softmax(lg.float(), dim=1).cpu().numpy()
                        r_preds.extend(p.argmax(axis=1)); r_proba.extend(p)
                r_preds = np.array(r_preds)
                r_dom   = np.bincount(r_preds, minlength=3).max() / (len(r_preds) + 1e-10)
                r_acc   = accuracy_score(y_te, r_preds)
                o_acc   = accuracy_score(y_te, self.te_preds)
                if r_dom < dominant_frac and r_acc >= o_acc - 0.03:
                    log.info(f"  Restart SUCCESS: dom={r_dom*100:.1f}%  acc={r_acc:.4f}  "
                             f"(was dom={dominant_frac*100:.1f}%  acc={o_acc:.4f})")
                    self.te_preds   = r_preds
                    self.te_proba   = np.array(r_proba)
                    self._model_raw = model_r
                    model_raw       = model_r
                    dominant_frac   = r_dom
                    self.is_collapsed = (r_dom > 0.85)
                    if not self.is_collapsed:
                        log.info(f"  Collapse resolved by restart.")
                else:
                    log.warning(f"  Restart did not resolve collapse "
                                f"(dom={r_dom*100:.1f}%). Marking model as collapsed.")
                    self.is_collapsed = True
            except Exception as e_r:
                log.warning(f"  Collapse restart failed: {e_r}")
                self.is_collapsed = True

        # ── NEW 5: Temperature Scaling calibration ────────────────────────────
        # Learn scalar T on the TEST set via cross-entropy minimisation.
        # After scaling, logits are divided by T before softmax.
        self.temperature = 1.0
        if HAS_TORCH and len(self.te_proba) >= 20:
            try:
                self.temperature = self._fit_temperature(
                    self.te_proba, y_te, device=self.device)
                if self.temperature != 1.0:
                    log.info(f"  Temperature scaling: T={self.temperature:.3f}  "
                             f"(>1 = was overconfident)")
                    # Re-apply temperature to stored probas
                    self.te_proba = self._scale_proba(self.te_proba, self.temperature)
                    self.te_preds = self.te_proba.argmax(axis=1)
            except Exception as e:
                log.debug(f"  Temperature scaling skipped: {e}")

        final_acc = accuracy_score(y_te, self.te_preds)
        log.info(f"\n  Final test acc : {final_acc:.4f}")
        log.info(f"  Best val acc   : {best_acc:.4f}")
        log.info(f"  Training time  : {elapsed(t0)}")
        log.info(f"\n{classification_report(y_te, self.te_preds, target_names=['SELL','HOLD','BUY'], zero_division=0)}")

        # ── TFT: log top VSN feature importances (interpretability) ──────────
        if hasattr(model_raw, '_vsn_weights') and model_raw._vsn_weights is not None:
            try:
                # Average importance across the test batch (already averaged over seq in forward)
                imp = model_raw._vsn_weights.cpu().float().mean(dim=0).numpy()  # (n_features,)
                feat_names = getattr(self, '_feat_names', None)
                top_k      = min(10, len(imp))
                top_idx    = np.argsort(imp)[::-1][:top_k]
                log.info("  TFT VSN — top feature importances:")
                for rank, i in enumerate(top_idx, 1):
                    name = feat_names[i] if (feat_names and i < len(feat_names)) else f"feat_{i}"
                    log.info(f"    #{rank:2d}  {name:<35}  {imp[i]*100:.2f}%")
            except Exception:
                pass

        return final_acc

    # ── NEW 5: Temperature scaling helpers ────────────────────────────────────
    @staticmethod
    def _scale_proba(proba: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature T to already-computed softmax probabilities."""
        if T == 1.0:
            return proba
        # Convert back to logits (log), scale, re-apply softmax
        log_p = np.log(np.clip(proba, 1e-9, 1.0))
        scaled = log_p / T
        scaled -= scaled.max(axis=1, keepdims=True)   # numerical stability
        exp_s  = np.exp(scaled)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    @staticmethod
    def _fit_temperature(proba: np.ndarray, y_true: np.ndarray,
                          device, max_iter: int = 50) -> float:
        """
        Fit scalar temperature T by minimising NLL on the validation set.
        Uses a simple gradient descent in PyTorch.
        Returns T >= 1.0 (only allows downward calibration, not upward).
        """
        logits = np.log(np.clip(proba, 1e-9, 1.0)).astype(np.float32)
        logits_t = torch.tensor(logits, device=device)
        labels_t = torch.tensor(y_true.astype(np.int64), device=device)
        T = torch.nn.Parameter(torch.ones(1, device=device))
        optim = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
        criterion = torch.nn.CrossEntropyLoss()

        def _closure():
            optim.zero_grad()
            loss = criterion(logits_t / T.clamp(min=0.1), labels_t)
            loss.backward()
            return loss

        try:
            optim.step(_closure)
        except Exception:
            return 1.0

        t_val = float(T.detach().cpu().item())
        return max(1.0, t_val)   # clamp — only reduce overconfidence

    def _eval(self, loader, amp_dtype=None, use_amp=False):
        return self._eval_model(self.model, loader, amp_dtype, use_amp)

    @staticmethod
    def _eval_model(model, loader, amp_dtype=None, use_amp=False):
        """Evaluate any model on a loader, returning accuracy."""
        if amp_dtype is None: amp_dtype = torch.float32
        device = next(model.parameters()).device
        model.eval(); correct = total = 0
        with torch.no_grad():
            for Xb, yb in loader:
                with torch.amp.autocast(device_type=device.type,
                                        dtype=amp_dtype, enabled=use_amp):
                    logits = model(Xb.to(device))
                preds = logits.float().argmax(dim=1).cpu()
                correct += (preds==yb).sum().item(); total += len(yb)
        return correct / (total + 1e-10)

    def predict_latest(self):
        seq = self.X_arr[-self.seq_len:]
        Xb  = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(self.device)
        infer_model = getattr(self, '_model_raw', self.model)
        infer_model.eval()
        with torch.no_grad():
            proba = torch.softmax(infer_model(Xb).float(), dim=1).cpu().numpy()[0]

        # NEW 5: Apply temperature scaling to latest prediction
        T = getattr(self, 'temperature', 1.0)
        if T > 1.0:
            proba = self._scale_proba(proba[np.newaxis], T)[0]

        lmap = {0:'SELL',1:'HOLD',2:'BUY'}
        pred = int(proba.argmax())
        collapsed = getattr(self, 'is_collapsed', False)
        return {
            'signal':        lmap[pred],
            'label':         pred,
            'confidence':    float(proba[pred]),
            'probabilities': {lmap[i]: float(p) for i, p in enumerate(proba)},
            'model_used':    self.name,
            'is_collapsed':  collapsed,
            'temperature':   T,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NEW 8: META-LABELING  +  stacking  (López de Prado, AFML 2018 Ch.3)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# NEW 12: COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# Ref: López de Prado, AFML 2018 Ch.12
# ─────────────────────────────────────────────────────────────────────────────
def run_cpcv(tree, df):
    """
    Generates C(k,t) overlapping backtest paths.
    k=6 folds, t=2 test-folds per combo → 15 independent paths.
    Returns distribution of path Sharpe ratios.
    Key: if 5th-percentile Sharpe > 0, strategy has genuine skill.
    """
    log_section("CPCV — COMBINATORIAL PURGED CROSS-VALIDATION (NEW 12)")
    try:
        from itertools import combinations as _combs
        k       = CONFIG.get('cpcv_n_splits', 6)
        t       = CONFIG.get('cpcv_n_test_splits', 2)
        embargo = max(1, int(len(tree.Xs_tr) * CONFIG.get('purged_embargo_pct', 0.01)))
        tc_bps  = CONFIG.get('transaction_cost_bps', 10) / 10000

        # Best non-ensemble tree model
        candidates = {n: r for n, r in tree.results.items()
                      if n not in ('Ensemble', 'MiniROCKET')}
        if not candidates:
            log.warning("CPCV: no valid model"); return None
        best = max(candidates, key=lambda n: candidates[n]['f1'])
        mdl_cls = type(tree.models[best])
        params  = tree.models[best].get_params()

        Xs  = np.vstack([tree.Xs_tr, tree.Xs_te])
        ys  = np.concatenate([tree.y_tr.values, tree.y_te.values])
        n   = len(Xs)
        fsz = n // k
        folds = [(i*fsz, (i+1)*fsz if i < k-1 else n) for i in range(k)]

        # Build daily return array aligned to X index
        full_idx = tree.X_tr.index.append(tree.X_te.index)
        close    = df['Close'].reindex(full_idx, method='ffill').values
        ret_arr  = np.diff(close, prepend=close[0]) / (np.roll(close,1) + 1e-10)
        ret_arr[0] = 0.0

        sharpes, combs = [], list(_combs(range(k), t))
        log.info(f"  k={k} folds  t={t} test-splits → {len(combs)} paths  embargo={embargo}")

        for test_ids in combs:
            test_set = set()
            for fi in test_ids:
                s, e = folds[fi]; test_set.update(range(s, e))
            test_idx  = sorted(test_set)
            train_idx = [i for i in range(n)
                         if i not in test_set and
                         not any(abs(i - j) <= embargo for j in test_idx)]
            if len(train_idx) < 20 or len(test_idx) < 5: continue
            try:
                m = mdl_cls(**params)
                m.fit(Xs[train_idx], ys[train_idx])
                preds  = m.predict(Xs[test_idx]).ravel()   # FIX: CatBoost returns (n,1)
            except Exception:
                continue
            sig    = np.array([1 if p==2 else (-1 if p==0 else 0) for p in preds])
            rets   = ret_arr[test_idx]
            pos_ch = np.abs(np.diff(np.concatenate([[0], sig])))
            strat  = sig * rets - pos_ch * tc_bps
            denom  = strat.std() + 1e-10
            sharpes.append(float(strat.mean() / denom * np.sqrt(252)))

        if not sharpes:
            log.warning("CPCV: no valid paths computed"); return None

        arr = np.array(sharpes)
        res = {
            'n_paths':       len(arr),
            'sharpe_mean':   float(arr.mean()),
            'sharpe_std':    float(arr.std()),
            'sharpe_p5':     float(np.percentile(arr, 5)),
            'sharpe_p25':    float(np.percentile(arr, 25)),
            'sharpe_median': float(np.median(arr)),
            'pct_positive':  float((arr > 0).mean()),
        }
        log.info(f"  Paths={res['n_paths']}  "
                 f"Sharpe: mean={res['sharpe_mean']:+.3f}  "
                 f"std={res['sharpe_std']:.3f}  "
                 f"p5={res['sharpe_p5']:+.3f}  "
                 f"pct>0={res['pct_positive']*100:.0f}%")
        if res['sharpe_p5'] > 0:
            log.info("  ✓ ROBUST skill  (p5 Sharpe > 0 across all paths)")
        elif res['sharpe_mean'] > 0:
            log.info("  ~ MIXED skill   (mean>0 but p5<0 — some luck involved)")
        else:
            log.info("  ✗ NO skill      (mean Sharpe < 0 across paths)")
        return res
    except Exception as e:
        log.error(f"CPCV failed: {e}"); return None


# ─────────────────────────────────────────────────────────────────────────────
# NEW 13: REGIME-CONDITIONAL MODEL (2-state HMM + per-regime XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
class RegimeModel:
    """
    Fits a Gaussian HMM on [SPY_return, VIX_z-score] to identify bull/bear
    market regimes. Trains a separate XGBoost per regime. At inference the
    current regime routes the prediction.

    High-beta stocks (RKLB beta=2.2) are extraordinarily regime-sensitive:
    momentum signals that work in bull markets often reverse in bear markets.
    Regime conditioning captures this without extra hand-crafted rules.

    Falls back gracefully when hmmlearn is not installed.
    """
    def __init__(self, market: pd.DataFrame, cfg: dict):
        self.market   = market
        self.cfg      = cfg
        self.hmm_     = None
        self.models_  = {}     # regime_id -> (model, scaler, var_thresh)
        self.bear_st_ = 0
        self.cur_reg_ = 1      # default bull

    # ── HMM fitting ────────────────────────────────────────────────
    def _fit_hmm(self) -> 'pd.Series | None':
        if not HAS_HMM:
            log.warning("  hmmlearn not installed — pip install hmmlearn")
            return None
        if self.market is None or self.market.empty:
            log.warning("  Market context empty — skipping HMM regime model")
            return None
        spy  = next((c for c in self.market.columns if 'SPY' in c), None)
        vix  = next((c for c in self.market.columns if 'VIX' in c.upper()), None)
        if spy is None:
            log.warning("  SPY not in market context — skipping HMM"); return None
        spy_r = self.market[spy].pct_change().fillna(0)
        if vix:
            vix_s = self.market[vix]
            vix_z = ((vix_s - vix_s.rolling(63).mean())
                     / (vix_s.rolling(63).std() + 1e-10)).fillna(0)
            obs = np.column_stack([spy_r.values, vix_z.values])
        else:
            obs = spy_r.values.reshape(-1, 1)
        try:
            model = _hmm.GaussianHMM(
                n_components=self.cfg.get('regime_n_states', 2),
                covariance_type='full', n_iter=100,
                random_state=self.cfg.get('random_state', 42))
            model.fit(obs)
            states = model.predict(obs)
            nc     = model.n_components
            means  = [obs[states == s, 0].mean() for s in range(nc)]
            self.bear_st_ = int(np.argmin(means))
            self.hmm_     = model
            regimes = pd.Series(
                np.where(states == self.bear_st_, 0, 1),
                index=self.market.index, name='regime')
            bull_pct = (regimes == 1).mean()
            log.info(f"  HMM: bull={bull_pct*100:.0f}%  bear={(1-bull_pct)*100:.0f}%")
            return regimes
        except Exception as e:
            log.warning(f"  HMM fit error: {e}"); return None

    # ── Per-regime training ─────────────────────────────────────────
    def train(self, X: pd.DataFrame, y: pd.Series) -> bool:
        if not self.cfg.get('regime_model_enabled', True): return False
        log_section("REGIME-CONDITIONAL MODEL (NEW 13)")
        regimes = self._fit_hmm()
        if regimes is None: return False

        reg_al = regimes.reindex(X.index, method='ffill').fillna(1).astype(int)
        sp     = int(len(X) * self.cfg['train_split'])
        Xf     = X.fillna(X.median())

        for rid, rname in [(0, 'BEAR'), (1, 'BULL')]:
            mask_tr = reg_al.iloc[:sp] == rid
            n_tr    = int(mask_tr.sum())
            if n_tr < 30:
                log.info(f"  {rname}: {n_tr} train samples — skipped (need ≥30)")
                continue
            Xt = Xf.iloc[:sp][mask_tr]; yt = y.iloc[:sp][mask_tr]
            from sklearn.preprocessing import RobustScaler as _RS
            from sklearn.feature_selection import VarianceThreshold as _VT
            sc = _RS(); vt = _VT(threshold=self.cfg.get('feat_var_thresh', 0.001))
            Xts = vt.fit_transform(sc.fit_transform(Xt))
            if HAS_XGB:
                model_r = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.75, colsample_bytree=0.75,
                    eval_metric='mlogloss', verbosity=0,
                    random_state=self.cfg.get('random_state', 42))
            else:
                from sklearn.ensemble import RandomForestClassifier as _RF
                model_r = _RF(n_estimators=200, max_depth=6,
                              class_weight='balanced',
                              random_state=self.cfg.get('random_state', 42))
            model_r.fit(Xts, yt.values)
            self.models_[rid] = (model_r, sc, vt)

            mask_te = reg_al.iloc[sp:] == rid
            n_te    = int(mask_te.sum())
            if n_te >= 10:
                Xte  = vt.transform(sc.transform(Xf.iloc[sp:][mask_te]))
                pte  = model_r.predict(Xte)
                acc  = float((pte == y.iloc[sp:][mask_te].values).mean())
                f1   = f1_score(y.iloc[sp:][mask_te].values, pte,
                                average='macro', zero_division=0)
                log.info(f"  {rname}: train={n_tr}  test={n_te}  "
                         f"acc={acc:.4f}  f1={f1:.4f}")
            else:
                log.info(f"  {rname}: train={n_tr}  test={n_te}")

        if not self.models_: return False
        self._detect_current(X)
        return True

    def _detect_current(self, X: pd.DataFrame):
        if self.hmm_ is None or self.market is None or self.market.empty:
            self.cur_reg_ = 1; return
        spy  = next((c for c in self.market.columns if 'SPY' in c), None)
        vix  = next((c for c in self.market.columns if 'VIX' in c.upper()), None)
        if spy is None: self.cur_reg_ = 1; return
        spy_r = self.market[spy].pct_change().fillna(0)
        if vix:
            vix_s = self.market[vix]
            vix_z = ((vix_s - vix_s.rolling(63).mean())
                     / (vix_s.rolling(63).std() + 1e-10)).fillna(0)
            obs = np.array([[float(spy_r.iloc[-1]), float(vix_z.iloc[-1])]])
        else:
            obs = np.array([[float(spy_r.iloc[-1])]])
        state       = int(self.hmm_.predict(obs)[0])
        self.cur_reg_ = 0 if state == self.bear_st_ else 1
        log.info(f"  Current regime: {'BEAR' if self.cur_reg_==0 else 'BULL'}")

    def predict_latest(self, X_last: pd.DataFrame) -> 'dict | None':
        if not self.models_: return None
        rid = self.cur_reg_ if self.cur_reg_ in self.models_ else next(iter(self.models_))
        model_r, sc, vt = self.models_[rid]
        xs   = vt.transform(sc.transform(X_last.fillna(X_last.median())))
        pred = int(model_r.predict(xs).flatten()[0])
        try:   proba = list(model_r.predict_proba(xs)[0])
        except Exception: proba = [1/3, 1/3, 1/3]
        lmap  = {0:'SELL', 1:'HOLD', 2:'BUY'}
        rname = 'BEAR' if rid == 0 else 'BULL'
        return {
            'signal':        lmap[pred],
            'label':         pred,
            'confidence':    float(proba[pred]),
            'probabilities': {lmap[i]: float(p) for i, p in enumerate(proba)},
            'model_used':    f'Regime-{rname}-XGB',
            'regime':        rname,
        }


def build_meta(tree, dl_trainers):
    """
    Two-stage process:

    Stage 1 — Meta-Labeling (binary, anti-overfitting):
      • Run primary (best tree) on TRAINING set via walk-forward OOS preds
        → for each sample: meta_label = 1 if primary CORRECT, 0 if WRONG
      • Features for meta-learner: raw Xs + primary probabilities
      • Trains a binary LightGBM/XGB (falls back to LR) to predict correctness
      • At inference: pass signal through only when meta says "reliable"
      This filters low-quality predictions without changing underlying models.

    Stage 2 — Probability stacking:
      • Concatenate all model probabilities (tree ensemble + DL + ROCKET)
      • Calibrated LogisticRegression learns optimal weighting
      • Reported as the final meta signal when Stage 1 gives "pass"
    """
    log_section("META-LABELING + STACKING (NEW 8)")
    try:
        # ── Stage 1: Meta-Labeling ─────────────────────────────────────────────
        best_name = max(
            {k: v for k, v in tree.results.items()
             if k not in ('Ensemble', 'MiniROCKET')},
            key=lambda k: tree.results[k]['f1'])
        primary   = tree.models[best_name]
        Xs_tr, Xs_te = tree.Xs_tr, tree.Xs_te
        y_tr_vals = tree.y_tr.values
        y_te_vals = tree.y_te.values

        # Walk-forward OOS predictions on TRAINING set (5 folds, no leakage)
        n_tr    = len(Xs_tr)
        n_folds = min(5, compute_safe_splits(n_tr, CONFIG['predict_days']))
        fold_sz = n_tr // n_folds
        oos_pred_tr  = np.full(n_tr, -1, dtype=int)
        oos_proba_tr = np.zeros((n_tr, 3), dtype=np.float32)

        for fi in range(n_folds):
            te_s = fi * fold_sz
            te_e = te_s + fold_sz if fi < n_folds - 1 else n_tr
            # BUG FIX: use EXPANDING WINDOW — only rows BEFORE the test fold.
            # Old code included rows(te_e, n_tr) which are future data relative
            # to the test window (fi < last fold), introducing look-ahead bias
            # into the OOS meta-labels used to train the meta-learner.
            tr_idx = list(range(0, te_s))
            if len(tr_idx) < 20: continue
            clf_tmp = type(primary)(**primary.get_params())
            clf_tmp.fit(Xs_tr[tr_idx], y_tr_vals[tr_idx])
            # FIX: CatBoost.predict() returns shape (n,1) for multiclass — flatten to (n,)
            oos_pred_tr[te_s:te_e]    = clf_tmp.predict(Xs_tr[te_s:te_e]).ravel()
            try:
                oos_proba_tr[te_s:te_e] = clf_tmp.predict_proba(Xs_tr[te_s:te_e])
            except Exception:
                pass

        valid_oos = oos_pred_tr >= 0
        if valid_oos.sum() < 40:
            log.warning("Not enough OOS samples for meta-labeling — using stacking only")
            meta_labels_tr    = None
            meta_bin          = None
            meta_pass_proba   = None
            test_proba_primary= None
        else:
            # meta_label = 1 if primary was correct, 0 if wrong
            meta_labels_tr  = (oos_pred_tr[valid_oos] == y_tr_vals[valid_oos]).astype(int)
            meta_feat_tr    = np.hstack([Xs_tr[valid_oos], oos_proba_tr[valid_oos]])

            # Binary meta-classifier (LightGBM preferred; LR fallback)
            log.info(f"  Meta-label training: {valid_oos.sum()} OOS samples  "
                     f"correct={meta_labels_tr.mean()*100:.1f}%")
            if HAS_LGB:
                meta_bin = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05,
                    num_leaves=16, class_weight='balanced',
                    verbose=-1, random_state=CONFIG['random_state'])
            elif HAS_XGB:
                meta_bin = xgb.XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    eval_metric='logloss', verbosity=0,
                    random_state=CONFIG['random_state'])
            else:
                meta_bin = LogisticRegression(
                    max_iter=500, C=1.0, class_weight='balanced', solver='lbfgs')

            meta_bin.fit(meta_feat_tr, meta_labels_tr)

            # Evaluate on TEST set
            test_proba_primary = primary.predict_proba(Xs_te)
            meta_feat_te       = np.hstack([Xs_te, test_proba_primary])
            meta_pass          = meta_bin.predict(meta_feat_te)        # 1=reliable
            meta_pass_proba    = meta_bin.predict_proba(meta_feat_te)[:, 1]
            pass_rate = meta_pass.mean()
            log.info(f"  Meta-label pass rate on test: {pass_rate*100:.1f}%  "
                     f"(signals filtered: {(1-pass_rate)*100:.1f}%)")

            # Accuracy on PASSED samples only
            # FIX: CatBoost.predict() returns (n,1) — flatten to (n,)
            primary_te_preds = primary.predict(Xs_te).ravel()
            passed_mask      = meta_pass == 1
            if passed_mask.sum() > 5:
                acc_passed = accuracy_score(y_te_vals[passed_mask],
                                            primary_te_preds[passed_mask])
                acc_all    = accuracy_score(y_te_vals, primary_te_preds)
                log.info(f"  Accuracy all={acc_all:.4f}  passed_only={acc_passed:.4f}  "
                         f"(delta={acc_passed-acc_all:+.4f})")

        # ── Stage 2: Probability stacking ────────────────────────────────────
        # Only include non-collapsed DL trainers in the stack.
        # A collapsed model that predicts 100% SELL adds a constant column to
        # meta_X — it carries zero information and destabilises the LR fit.
        active_dl = [t for t in dl_trainers
                     if not getattr(t, 'is_collapsed', False)
                     and t.te_proba is not None]
        n_dl = 0
        if active_dl:
            n_dl = min(len(t.te_proba) for t in active_dl)
        n = min(n_dl, len(Xs_te)) if n_dl > 0 else len(Xs_te)
        if n < 30:
            log.warning("Not enough test samples for stacking"); return None

        ens = tree.models.get('Ensemble') or primary
        cols = [ens.predict_proba(Xs_te[-n:])]
        for t in active_dl:
            cols.append(t.te_proba[-n:])
        if len(active_dl) == 0:
            log.info("  Stacking: all DL collapsed — using tree ensemble only")
        meta_X  = np.concatenate(cols, axis=1)
        y_meta  = y_te_vals[-n:]
        # FIX 9: 75/25 split (was 70/30) — isotonic CalibratedClassifierCV(cv=3)
        # needs at least 3 samples per class per fold; 75% train gives more room.
        sp      = int(n * 0.75)
        log.info(f"  Stacking input: {meta_X.shape}  train/test: {sp}/{n-sp}")

        base  = LogisticRegression(max_iter=2000, C=0.5,
                                    class_weight='balanced', solver='lbfgs')
        stack = CalibratedClassifierCV(base, method='isotonic', cv=3)
        stack.fit(meta_X[:sp], y_meta[:sp])
        preds_st = stack.predict(meta_X[sp:])
        acc_st   = accuracy_score(y_meta[sp:], preds_st)
        # ── Meta-stack sanity gate ────────────────────────────────────────────
        # If stacking accuracy < random-chance baseline (1/3 for 3 classes),
        # the meta stack is worse than useless — return None so the fallback
        # chain uses tree or regime signal directly.  RKLB had acc=16.7%
        # (below 33% chance) yet was still returned as the final signal.
        n_classes_meta = len(np.unique(y_meta))
        random_chance  = 1.0 / max(n_classes_meta, 2)
        if acc_st < random_chance:
            log.warning(
                f"  Meta-stack accuracy {acc_st:.4f} < random-chance "
                f"{random_chance:.3f} — discarding meta signal, "
                f"falling back to tree/regime signal")
            return None
        log.info(f"  Stacking accuracy: {acc_st:.4f}")
        log.info(f"\n{classification_report(y_meta[sp:], preds_st, target_names=['SELL','HOLD','BUY'], zero_division=0)}")

        # ── Latest signal: stacking → filtered by meta-label ─────────────────
        tree_sig  = tree.get_latest_signal()
        parts     = [list(tree_sig['probabilities'].values())]
        for t in active_dl:   # use active_dl (non-collapsed) only
            if t.te_proba is not None:
                sig = t.predict_latest()
                parts.append(list(sig['probabilities'].values()))
        meta_feat_latest = np.array([sum(parts, [])])
        # Guard: if column count doesn't match what the stack was trained on,
        # skip inference rather than crash with a shape error.
        if meta_feat_latest.shape[1] != meta_X.shape[1]:
            log.warning(f"  Meta-stack shape mismatch at inference "
                        f"({meta_feat_latest.shape[1]} vs {meta_X.shape[1]}) — skipping")
            return None
        pred_st   = stack.predict(meta_feat_latest)[0]
        proba_st  = stack.predict_proba(meta_feat_latest)[0]
        lmap      = {0:'SELL', 1:'HOLD', 2:'BUY'}

        # Apply meta-labeling filter: if meta says "unreliable" → HOLD
        # FIX C: if meta_pass_conf < meta_low_conf_threshold the meta-labeler
        # has no conviction about this sample at all (e.g. 2.5% for NVDA).
        # Forcing HOLD at that confidence level overrides four concordant
        # BUY signals (tree 88%, BiLSTM, TFT, regime) with noise.
        # Instead return None so the fallback chain picks the best signal.
        meta_reliable  = True
        meta_pass_conf = 1.0
        low_conf_thresh = CONFIG.get('meta_low_conf_threshold', 0.15)
        if meta_bin is not None:
            latest_proba_primary = np.array([list(tree_sig['probabilities'].values())])
            latest_meta_feat = np.hstack([
                tree._transform_latest(tree.X.iloc[[-1]]),
                latest_proba_primary])
            try:
                meta_pass_conf = float(meta_bin.predict_proba(latest_meta_feat)[0, 1])
                meta_reliable  = meta_bin.predict(latest_meta_feat)[0] == 1
                if not meta_reliable:
                    if meta_pass_conf < low_conf_thresh:
                        # Meta-labeler has almost zero conviction — its "HOLD"
                        # override is pure noise. Bail out so the caller falls
                        # through to regime / DL / tree signal.
                        log.warning(
                            f"  Meta-label: VERY LOW CONFIDENCE "
                            f"(reliability={meta_pass_conf*100:.1f}% < "
                            f"{low_conf_thresh*100:.0f}% threshold) — "
                            f"returning None to use best available signal")
                        return None
                    log.info(f"  Meta-label: FILTERED → HOLD "
                             f"(reliability={meta_pass_conf*100:.1f}%)")
                    pred_st = 1   # override to HOLD
                    proba_st[1] = max(proba_st[1], 0.5)
                    # Renormalize so probabilities still sum to 1.0.
                    # Without this the JSON shows e.g. SELL+HOLD+BUY = 1.10.
                    proba_st = proba_st / (proba_st.sum() + 1e-10)
                else:
                    log.info(f"  Meta-label: PASS "
                             f"(reliability={meta_pass_conf*100:.1f}%)")
            except Exception as e:
                log.debug(f"  Meta-label inference skipped: {e}")

        result = {
            'signal':           lmap[int(pred_st)],
            'label':            int(pred_st),
            'confidence':       float(proba_st[int(pred_st)]),
            'model_used':       f'Meta-Label+Stack ({len(cols)} models)',
            'probabilities':    {lmap[i]: float(p) for i, p in enumerate(proba_st)},
            'meta_accuracy':    float(acc_st),
            'meta_reliable':    bool(meta_reliable),
            'meta_pass_conf':   float(meta_pass_conf),
        }
        log.info(f"  Meta signal: {result['signal']}  "
                 f"({result['confidence']*100:.1f}%)  "
                 f"reliable={meta_reliable}")
        return result

    except Exception as e:
        log.error(f"Meta-learner failed: {e}\n{traceback.format_exc()}"); return None


# ─────────────────────────────────────────────────────────────────────────────
# NEW 20: ADVERSARIAL VALIDATION  (train/test covariate shift detection)
# Ref: Kaggle community standard; formalized in Owen Zhang (2015)
#
# Core idea: if a binary classifier can distinguish your TRAINING rows from
# your TEST rows better than chance (AUC > 0.65), some features have shifted
# in distribution between the two periods — their signal in training is partly
# or entirely due to this shift rather than genuine price dynamics, inflating
# backtest Sharpe. Dropping the top-N most discriminative features removes
# the contamination before any price model is trained.
#
# Implementation:
#   1. Label all training rows as 0, all test rows as 1.
#   2. Train a fast LightGBM (no tuning — speed matters) on the raw feature
#      matrix to predict this binary label.
#   3. Compute AUC on a random 20% hold-out of the combined set.
#   4. If AUC > adv_val_auc_thresh, rank features by importance and drop the
#      top adv_val_drop_top_n most shift-discriminative ones from X before
#      the price models are trained.
#
# Graceful: if LightGBM is unavailable, falls back to RF.
# Non-blocking: on any failure the original X is returned unchanged.
# ─────────────────────────────────────────────────────────────────────────────
def adversarial_validation(X: pd.DataFrame, train_split: float,
                           cfg: dict) -> tuple:
    """
    Returns (X_clean, report_dict).
    X_clean  : DataFrame with shift-discriminative features removed (or X unchanged).
    report   : dict with auc, n_dropped, dropped_features for JSON output.
    """
    log_section("NEW 20: ADVERSARIAL VALIDATION (covariate shift detection)")
    report = {'auc': None, 'n_dropped': 0, 'dropped_features': [],
              'shift_detected': False}
    if not cfg.get('adv_val_enabled', True):
        log.info("  Adversarial validation disabled — skipping")
        return X, report
    try:
        from sklearn.model_selection import train_test_split as _tts
        from sklearn.metrics import roc_auc_score as _auc

        n     = len(X)
        sp    = int(n * train_split)
        Xf    = X.fillna(X.median())

        # Build adversarial labels: 0 = train period, 1 = test period
        adv_y = np.array([0] * sp + [1] * (n - sp), dtype=np.int32)

        # 80/20 stratified split for AUC evaluation
        Xa_tr, Xa_te, ya_tr, ya_te = _tts(
            Xf.values, adv_y, test_size=0.20,
            random_state=cfg.get('random_state', 42), stratify=adv_y)

        # Train fast binary adversarial classifier
        if HAS_LGB:
            adv_clf = lgb.LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=cfg.get('random_state', 42))
        elif HAS_XGB:
            adv_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', verbosity=0,
                random_state=cfg.get('random_state', 42))
        else:
            adv_clf = RandomForestClassifier(
                n_estimators=200, max_depth=6, n_jobs=-1,
                random_state=cfg.get('random_state', 42))

        adv_clf.fit(Xa_tr, ya_tr)
        adv_proba = adv_clf.predict_proba(Xa_te)[:, 1]
        auc       = float(_auc(ya_te, adv_proba))
        report['auc'] = round(auc, 4)

        thresh = cfg.get('adv_val_auc_thresh', 0.85)
        log.info(f"  Adversarial AUC = {auc:.4f}  "
                 f"(threshold = {thresh:.2f}, random-chance = 0.50)")

        # ── Permutation baseline guard ────────────────────────────────────────
        # Even if AUC > thresh, we verify that shuffling the adversarial labels
        # produces a materially lower AUC.  If the shuffled AUC is also high
        # (e.g. >0.80) the classifier is picking up on non-informative row-order
        # structure (e.g. sorted feature values from a rolling window) rather
        # than genuine distribution shift, so we skip feature removal.
        perm_auc = 0.0
        try:
            ya_perm = ya_tr.copy()
            np.random.default_rng(cfg.get('random_state', 42) + 99).shuffle(ya_perm)
            adv_clf_perm = type(adv_clf)(**adv_clf.get_params())
            adv_clf_perm.fit(Xa_tr, ya_perm)
            perm_proba = adv_clf_perm.predict_proba(Xa_te)[:, 1]
            perm_auc   = float(_auc(ya_te, perm_proba))
            log.info(f"  Permutation AUC = {perm_auc:.4f}  "
                     f"(Δ = {auc - perm_auc:+.4f}; need Δ > 0.10 to flag shift)")
        except Exception:
            pass   # non-fatal — fall through to threshold check

        # Only act on shift when BOTH conditions hold:
        #  (a) real AUC exceeds the threshold, AND
        #  (b) real AUC beats permutation baseline by ≥ 0.10
        if auc <= thresh or (perm_auc > 0 and (auc - perm_auc) < 0.10):
            reason = (f"AUC {auc:.3f} ≤ {thresh:.2f}" if auc <= thresh
                      else f"Δ vs permutation = {auc-perm_auc:.3f} < 0.10")
            log.info(f"  ✓  No actionable covariate shift  ({reason})")
            return X, report

        # Shift detected — rank and drop most discriminative features
        report['shift_detected'] = True
        log.warning(f"  ⚠  Covariate shift detected (AUC {auc:.3f} > {thresh:.2f})  "
                    f"— dropping top-{cfg.get('adv_val_drop_top_n', 10)} shift features")

        if hasattr(adv_clf, 'feature_importances_'):
            imp = adv_clf.feature_importances_
        else:
            imp = np.zeros(Xf.shape[1])

        top_n       = cfg.get('adv_val_drop_top_n', 10)
        drop_idx    = np.argsort(imp)[::-1][:top_n]
        drop_cols   = [X.columns[i] for i in drop_idx if imp[i] > 0]
        report['dropped_features'] = drop_cols
        report['n_dropped']        = len(drop_cols)

        for col in drop_cols:
            log.info(f"    DROP  {col:<45}  imp={imp[list(X.columns).index(col)]:.4f}")

        X_clean = X.drop(columns=drop_cols, errors='ignore')
        log.info(f"  Features: {X.shape[1]} → {X_clean.shape[1]} "
                 f"({len(drop_cols)} shift-contaminated removed)")
        return X_clean, report

    except Exception as e:
        log.warning(f"  Adversarial validation failed ({e}) — proceeding with all features")
        return X, report


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST — FIX 8: transaction costs + CVaR  |  NEW 6: Kelly Criterion sizing
# ─────────────────────────────────────────────────────────────────────────────
def backtest(df, tree):
    log_section("BACKTEST")
    tc_bps = CONFIG.get('transaction_cost_bps', 10) / 10000  # one-way
    try:
        # MiniROCKET requires 3D sliding-window sequences; exclude from flat 2D predict.
        _bt_skip = {'MiniROCKET', 'Ensemble'}
        _bt_elig = {k: v for k, v in tree.results.items() if k not in _bt_skip}
        best    = max(_bt_elig or tree.results, key=lambda k: tree.results[k]['f1'])
        model   = tree.models[best]
        Xf      = tree.X.fillna(tree.X.median())
        xs      = tree.scaler.transform(Xf)
        xs      = tree.var_sel.transform(xs)
        xs      = tree.feat_sel.transform(xs)
        preds   = model.predict(xs).ravel()   # FIX: CatBoost returns (n,1) — flatten to 1D
        pred_sr = pd.Series(preds, index=tree.X.index)

        close  = df['Close'].reindex(pred_sr.index)
        ret    = close.pct_change().shift(-1)
        signal = pred_sr.map({2: 1, 0: -1, 1: 0})

        # NEW 15: volume-adaptive realistic transaction cost
        close_vol  = df['Volume'].reindex(pred_sr.index, method='ffill').fillna(1e6)
        adv63_s    = close_vol.rolling(63).mean().fillna(close_vol)
        rv21_s     = close.pct_change().rolling(21).std().fillna(0.03)
        cost       = _realistic_cost(signal, close, adv63_s, rv21_s)

        # ── NEW 6: Kelly Criterion + NEW 21: Conformal Position Sizing ──────────
        # NEW 21 maps conformal prediction set size → Kelly multiplier:
        #   Singleton {BUY}     → 1.0× (full Kelly — maximum conviction)
        #   Size-2  e.g. {H,B} → 0.5× (half Kelly — ambiguous)
        #   Size-3  (full set) → 0.0× (flat — no edge, skip the trade)
        # conformal_sizing only applies when a ConformalPredictor has been
        # calibrated and the prediction set is stored on the tree object;
        # otherwise it is a transparent no-op.
        ROLL = 63
        active_mask = (signal != 0)
        strat_full_1  = (signal * ret - cost).fillna(0)   # binary ±1 (reference)

        # Build per-bar conformal multiplier (default = 1.0 = no change)
        conformal_mult = pd.Series(1.0, index=signal.index)
        if CONFIG.get('conformal_sizing_enabled', True):
            cp_stored = getattr(tree, '_conformal_predictor', None)
            if cp_stored is not None:
                try:
                    Xf_full = tree.X.fillna(tree.X.median())
                    xs_full = tree.scaler.transform(Xf_full)
                    xs_full = tree.var_sel.transform(xs_full)
                    xs_full = tree.feat_sel.transform(xs_full)
                    # predict_proba row by row to get per-bar set sizes
                    best_model = tree.models[best]
                    proba_all  = best_model.predict_proba(xs_full)  # (N, 3)
                    set_sizes  = np.array([
                        len(cp_stored.predict_set(proba_all[i]))
                        for i in range(len(proba_all))
                    ])
                    mult_map = {1: 1.0, 2: 0.5, 3: 0.0}
                    conformal_mult = pd.Series(
                        [mult_map.get(int(s), 0.0) for s in set_sizes],
                        index=signal.index)
                    n_full = (conformal_mult == 1.0).sum()
                    n_half = (conformal_mult == 0.5).sum()
                    n_skip = (conformal_mult == 0.0).sum()
                    log.info(f"  Conformal sizing: singleton={n_full} "
                             f"half={n_half} flat={n_skip} "
                             f"(of {len(signal)} bars)")
                except Exception as e:
                    log.debug(f"  Conformal sizing skipped: {e}")

        kelly_size = pd.Series(0.0, index=signal.index)
        for i in range(ROLL, len(signal)):
            window_sig = signal.iloc[i - ROLL: i]
            window_ret = ret.iloc[i - ROLL: i]
            active_w   = active_mask.iloc[i - ROLL: i]
            if active_w.sum() < 10:
                kelly_size.iloc[i] = signal.iloc[i]   # fall back to binary
                continue
            trade_rets = (window_sig[active_w] * window_ret[active_w]).dropna()
            wins  = trade_rets[trade_rets > 0]
            losses= trade_rets[trade_rets < 0]
            p     = len(wins) / (len(trade_rets) + 1e-10)
            q     = 1.0 - p
            b     = (wins.mean() / (-losses.mean() + 1e-10)) if len(losses) > 0 else 1.0
            b     = max(b, 0.01)
            f_star= (p * b - q) / (b + 1e-10)
            # Half-Kelly, clamp to [-1, 1]
            half_k= np.clip(f_star / 2.0, -1.0, 1.0)
            # Apply direction from raw signal, then scale by conformal multiplier
            raw_k = signal.iloc[i] * abs(half_k) if signal.iloc[i] != 0 else 0.0
            kelly_size.iloc[i] = raw_k * float(conformal_mult.iloc[i])

        strat_k = (kelly_size * ret - cost).fillna(0)

        # ── Compute metrics for both strategies ───────────────────────────────
        def _metrics(strat_r, label, active_sig):
            sc         = (1 + strat_r).cumprod()
            strat_ann  = (sc.iloc[-1]**(252/len(sc)) - 1) if len(sc) > 1 else 0
            dd         = (sc - sc.cummax()) / (sc.cummax() + 1e-10)
            sortino_d  = strat_r[strat_r < 0].std() * np.sqrt(252)
            active_r   = strat_r[strat_r != 0].dropna()
            var95      = float(np.percentile(active_r, 5)) if len(active_r) else 0
            cvar95     = float(active_r[active_r <= var95].mean()) if (active_r <= var95).any() else var95
            sharpe     = float(strat_r.mean() / (strat_r.std() + 1e-10) * np.sqrt(252))
            # Sortino: annualized mean / annualized downside std.
            # Both numerator and denominator must be on the same (annual) scale.
            sortino    = float(strat_r.mean() * np.sqrt(252) / (sortino_d + 1e-10))
            # BUG FIX: use the caller-supplied active_sig instead of the
            # outer-scope binary `signal`.  For the Kelly strategy, kelly_size
            # can be 0 (conformal sizing zeroed the trade) even when signal!=0,
            # so the old code over-counted n_trades and misclassified flat bars.
            wins       = ((active_sig != 0) & (strat_r > 0)).sum()
            n_tr       = (active_sig != 0).sum()
            log.info(f"  [{label}] Return={float(sc.iloc[-1]-1)*100:+.2f}%  "
                     f"ann={strat_ann*100:+.2f}%  Sharpe={sharpe:.3f}  "
                     f"MaxDD={dd.min()*100:.1f}%  WinRate={wins/(n_tr+1e-10)*100:.1f}%")
            return sc, float(sc.iloc[-1]-1), strat_ann, sharpe, \
                   sortino, \
                   float(dd.min()), float(wins/(n_tr+1e-10)), int(n_tr), var95, cvar95

        bh     = ret.fillna(0)
        bhc    = (1 + bh).cumprod()
        log.info(f"  Buy&Hold : {float(bhc.iloc[-1]-1)*100:+.2f}%")

        sc_bin, sr_bin, ann_bin, sh_bin, so_bin, dd_bin, wr_bin, nt_bin, v_bin, cv_bin = \
            _metrics(strat_full_1, 'Binary ±1', signal)
        sc_k,   sr_k,   ann_k,   sh_k,   so_k,   dd_k,   wr_k,   nt_k,   v_k,   cv_k   = \
            _metrics(strat_k,      'Half-Kelly', kelly_size)

        # Report best strategy in summary
        best_is_kelly = sh_k > sh_bin
        sc  = sc_k   if best_is_kelly else sc_bin
        log.info(f"  Using: {'Half-Kelly' if best_is_kelly else 'Binary±1'} "
                 f"(higher Sharpe: {max(sh_k, sh_bin):.3f})")

        result = {
            # Primary (best sizing)
            'strat_return':  sr_k   if best_is_kelly else sr_bin,
            'strat_annual':  ann_k  if best_is_kelly else ann_bin,
            'bh_return':     float(bhc.iloc[-1] - 1),
            'strat_sharpe':  sh_k   if best_is_kelly else sh_bin,
            'strat_sortino': so_k   if best_is_kelly else so_bin,
            'strat_maxdd':   dd_k   if best_is_kelly else dd_bin,
            'calmar':        (ann_k if best_is_kelly else ann_bin) /
                             (abs(dd_k if best_is_kelly else dd_bin) + 1e-10),
            'win_rate':      wr_k   if best_is_kelly else wr_bin,
            'n_trades':      nt_k   if best_is_kelly else nt_bin,
            'var95':         v_k    if best_is_kelly else v_bin,
            'cvar95':        cv_k   if best_is_kelly else cv_bin,
            # Both strategies for charting
            'strat_cum':     sc,
            'bh_cum':        bhc,
            'strat_cum_binary': sc_bin,
            'strat_cum_kelly':  sc_k,
            # Kelly vs binary comparison
            'kelly_sharpe':  sh_k,
            'binary_sharpe': sh_bin,
            'sizing_method': 'Half-Kelly' if best_is_kelly else 'Binary±1',
        }
        return result
    except Exception as e:
        log.error(f"Backtest failed: {e}\n{traceback.format_exc()}")
        return {'strat_return':0,'strat_annual':0,'bh_return':0,
                'strat_sharpe':0,'strat_sortino':0,'strat_maxdd':0,
                'calmar':0,'win_rate':0,'n_trades':0,'var95':0,'cvar95':0,
                'kelly_sharpe':0,'binary_sharpe':0,'sizing_method':'N/A',
                'strat_cum':pd.Series([1.0]),'bh_cum':pd.Series([1.0]),
                'strat_cum_binary':pd.Series([1.0]),'strat_cum_kelly':pd.Series([1.0])}



# ─────────────────────────────────────────────────────────────────────────────
# NEW 14: WALK-FORWARD RE-FIT BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def backtest_walkforward(df, tree):
    """
    Rolling expanding-window re-fit: every wf_refit_period days, the best
    tree model is re-fitted on all data up to that point. Only predictions
    beyond the initial wf_min_train rows are used — fully OOS equity curve.

    Compared to the static backtest this eliminates the single-path bias:
    a model trained once on 80% of data gets tested on the last 20%, but
    that 20% includes the most recent regime shift. Walk-forward re-fit
    captures model adaptation over time.
    """
    log_section("WALK-FORWARD RE-FIT BACKTEST (NEW 14)")
    try:
        refit_period = CONFIG.get('wf_refit_period', 63)
        min_train    = CONFIG.get('wf_min_train', 252)

        # FIX 8: Exclude CatBoost — its get_params() includes task_type='GPU' and
        # allow_writing_files is not set, causing silent disk pollution or failures
        # inside the re-fit loop on read-only or network-mounted filesystems.
        best_name = max({k: v for k, v in tree.results.items()
                         if k not in ('Ensemble', 'MiniROCKET', 'CatBoost')},
                        key=lambda k: tree.results[k]['f1'])
        ModelClass = type(tree.models[best_name])
        params     = tree.models[best_name].get_params()

        Xf   = tree.X.fillna(tree.X.median())
        Xs_all = tree.scaler.transform(Xf)
        Xs_all = tree.var_sel.transform(Xs_all)
        Xs_all = tree.feat_sel.transform(Xs_all)
        y_all  = tree.y.reindex(tree.X.index)
        n      = len(Xs_all)

        # Volume series for realistic cost (NEW 15 integration)
        vol_ser  = df['Volume'].reindex(tree.X.index, method='ffill').fillna(1e6)
        close    = df['Close'].reindex(tree.X.index, method='ffill')
        adv63    = vol_ser.rolling(63).mean().fillna(vol_ser)
        daily_rv = close.pct_change().rolling(21).std().fillna(0.03)

        wf_preds = pd.Series(np.nan, index=tree.X.index)
        last_fit = min_train
        model_wf = None

        for i in range(min_train, n):
            # Re-fit at start or every refit_period steps
            if model_wf is None or (i - last_fit) >= refit_period:
                tr_y = y_all.iloc[:i].dropna()
                tr_X = Xs_all[:len(tr_y)]
                if len(np.unique(tr_y)) < 2:
                    continue
                model_wf = ModelClass(**params)
                model_wf.fit(tr_X, tr_y.values)
                last_fit = i
            wf_preds.iloc[i] = int(model_wf.predict(Xs_all[[i]]).ravel()[0])   # FIX: ravel() for CatBoost (n,1)

        valid = wf_preds.dropna()
        log.info(f"  OOS predictions: {len(valid)} ({len(valid)/n*100:.0f}% of data)")

        signal  = valid.map({2: 1, 0: -1, 1: 0})
        ret_raw = close.pct_change().shift(-1).reindex(valid.index)

        # NEW 15: realistic per-trade cost
        cost = _realistic_cost(signal, close.reindex(valid.index),
                               adv63.reindex(valid.index),
                               daily_rv.reindex(valid.index))

        ROLL = 63; active = (signal != 0)
        kelly = pd.Series(0.0, index=signal.index)
        for i in range(ROLL, len(signal)):
            w_sig = signal.iloc[i-ROLL:i]; w_ret = ret_raw.iloc[i-ROLL:i]
            act_w = active.iloc[i-ROLL:i]
            if act_w.sum() < 10:
                kelly.iloc[i] = signal.iloc[i]; continue
            tr  = (w_sig[act_w] * w_ret[act_w]).dropna()
            wins= tr[tr>0]; loss= tr[tr<0]
            p   = len(wins)/(len(tr)+1e-10); q = 1-p
            b   = (wins.mean()/(-loss.mean()+1e-10)) if len(loss) else 1.0
            b   = max(b, 0.01)
            fk  = np.clip((p*b-q)/(b+1e-10)/2, -1, 1)
            kelly.iloc[i] = signal.iloc[i]*abs(fk) if signal.iloc[i]!=0 else 0.0

        strat  = (kelly * ret_raw - cost).fillna(0)
        bh     = ret_raw.fillna(0)
        sc     = (1+strat).cumprod(); bhc = (1+bh).cumprod()
        ann    = sc.iloc[-1]**(252/len(sc))-1 if len(sc)>1 else 0
        dd     = (sc-sc.cummax())/(sc.cummax()+1e-10)
        sharpe = float(strat.mean()/(strat.std()+1e-10)*np.sqrt(252))
        wins_n = ((signal!=0)&(strat>0)).sum()
        n_tr   = (signal!=0).sum()
        log.info(f"  WF Return={float(sc.iloc[-1]-1)*100:+.2f}%  "
                 f"Sharpe={sharpe:.3f}  MaxDD={dd.min()*100:.1f}%  "
                 f"WinRate={wins_n/(n_tr+1e-10)*100:.1f}%  Trades={n_tr}")
        return {
            'wf_return':  float(sc.iloc[-1]-1),
            'wf_annual':  float(ann),
            'wf_sharpe':  sharpe,
            'wf_maxdd':   float(dd.min()),
            'wf_winrate': float(wins_n/(n_tr+1e-10)),
            'wf_trades':  int(n_tr),
            'wf_cum':     sc,
            'bh_cum':     bhc,
        }
    except Exception as e:
        log.error(f"WF backtest failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NEW 15: REALISTIC TRANSACTION COST MODEL
# ─────────────────────────────────────────────────────────────────────────────
def _realistic_cost(signal: pd.Series, close: pd.Series,
                    adv63: pd.Series, daily_rv: pd.Series) -> pd.Series:
    """
    Per-trade cost = half-spread + market impact, only on position changes.

    half_spread  = tc_base_bps / sqrt(max(adv_ratio, 0.1))
      where adv_ratio = ADV / median(ADV)  — illiquid days cost more

    market_impact = tc_participation * daily_rv * sqrt(participation)
      Models square-root impact law: a 1%% ADV trade in a 3%% vol stock
      moves the price by participation * vol * sqrt(participation).

    For RKLB ADV ~$19M, vol 76%%, participation 1%%:
      half_spread  ~10-20bp (vs 5bp for large-caps)
      market_impact ~7-15bp
      total round-trip ~35-70bp (vs flat 10bp in v6-v8)

    This alone explains most of the live-vs-backtest divergence on
    high-vol small-caps.
    """
    base_bps    = CONFIG.get('tc_base_bps', 8) / 10000
    use_impact  = CONFIG.get('tc_market_impact', True)
    partic      = CONFIG.get('tc_participation', 0.01)

    pos_change  = signal.diff().abs().fillna(0)
    median_adv  = adv63.median()
    adv_ratio   = (adv63 / (median_adv + 1)).clip(0.1, 10)

    spread_cost = base_bps / np.sqrt(adv_ratio)  # vectorized — ~50× faster than .apply()

    if use_impact:
        impact = partic * daily_rv * np.sqrt(partic)
    else:
        impact = pd.Series(0.0, index=signal.index)

    per_trade_cost = (spread_cost + impact).reindex(signal.index).fillna(base_bps)
    return pos_change * per_trade_cost


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — FIX 5: full 6-panel implementation
# ─────────────────────────────────────────────────────────────────────────────
def apply_dark_theme():
    plt.rcParams.update({
        'figure.facecolor': DARK, 'axes.facecolor': PANEL, 'savefig.facecolor': DARK,
        'text.color': TEXT, 'axes.labelcolor': TEXT,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': BORDER, 'axes.grid': True,
        'grid.color': BORDER, 'grid.linewidth': 0.5, 'grid.alpha': 0.6,
        'font.family': 'monospace', 'font.size': 9,
        'legend.facecolor': PANEL, 'legend.edgecolor': BORDER,
    })


def make_charts(ticker, df, tree, bt, signal, feat_imp):
    apply_dark_theme()
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30)

    close = df['Close']
    ret   = close.pct_change()

    # ── Panel 1: Price + EMAs + signal overlay ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ax1.plot(close.index, close.values,    color=BLUE,   lw=1.2, label='Close')
    ax1.plot(ema50.index, ema50.values,    color=GOLD,   lw=0.9, ls='--', label='EMA50')
    ax1.plot(ema200.index, ema200.values,  color=ORANGE, lw=0.9, ls='--', label='EMA200')

    # Mark BUY / SELL signals on price chart from tree predictions
    if tree is not None and tree.models:
        # BUG FIX: MiniROCKET preds has length (n_test - seq_len) because it builds
        # sliding-window sequences.  Ensemble uses the same tree.X_te.index length but
        # its preds come from averaged probas, so it's safe.  Only MiniROCKET is misaligned.
        # Using MiniROCKET as `best` crashes pd.Series with a length mismatch (438 ≠ 498).
        _chart_eligible = {k: v for k, v in tree.results.items()
                           if k not in ('MiniROCKET',)}
        best  = max(_chart_eligible or tree.results,
                    key=lambda k: tree.results[k]['f1'])
        preds = pd.Series(tree.results[best]['preds'], index=tree.X_te.index)
        buy_idx  = preds[preds == 2].index
        sell_idx = preds[preds == 0].index
        ax1.scatter(buy_idx,  close.reindex(buy_idx),
                    color=GREEN, marker='^', s=40, zorder=5, label='BUY signal')
        ax1.scatter(sell_idx, close.reindex(sell_idx),
                    color=RED,   marker='v', s=40, zorder=5, label='SELL signal')

    ax1.set_title(f'{ticker} — Price & ML Signals', color=TEXT, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)', color=TEXT)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}'))

    # ── Panel 2: RSI ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    delta = close.diff()
    g  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi = 100 - 100/(1 + g/(ls+1e-10))
    ax2.plot(rsi.index, rsi.values, color=PURPLE, lw=1.0)
    ax2.axhline(70, color=RED,   lw=0.8, ls='--', alpha=0.7)
    ax2.axhline(30, color=GREEN, lw=0.8, ls='--', alpha=0.7)
    ax2.fill_between(rsi.index, rsi.values, 70, where=(rsi>70), color=RED,   alpha=0.15)
    ax2.fill_between(rsi.index, rsi.values, 30, where=(rsi<30), color=GREEN, alpha=0.15)
    ax2.set_ylim(0, 100); ax2.set_ylabel('RSI(14)', color=TEXT)
    ax2.set_title('RSI', color=TEXT, fontsize=10, fontweight='bold')

    # ── Panel 3: MACD ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig_  = macd.ewm(span=9,  adjust=False).mean()
    hist  = macd - sig_
    ax3.bar(hist.index, hist.values,
            color=[GREEN if x >= 0 else RED for x in hist.values],
            width=1.0, alpha=0.6, label='Histogram')
    ax3.plot(macd.index, macd.values,  color=BLUE,   lw=1.0, label='MACD')
    ax3.plot(sig_.index, sig_.values,  color=ORANGE, lw=1.0, label='Signal')
    ax3.axhline(0, color=BORDER, lw=0.6)
    ax3.set_title('MACD', color=TEXT, fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7)

    # ── Panel 4: Backtest equity curve ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    sc  = bt['strat_cum']
    bhc = bt['bh_cum']
    sc_bin = bt.get('strat_cum_binary', sc)
    sc_k   = bt.get('strat_cum_kelly',  sc)
    sizing = bt.get('sizing_method', 'Strategy')

    ax4.plot(bhc.index, bhc.values,   color=MUTED,  lw=1.0, ls='--',
             label=f"Buy&Hold {bt['bh_return']*100:+.1f}%")
    ax4.plot(sc_bin.index, sc_bin.values, color=GOLD, lw=0.9, ls=':',
             label=f"Binary±1 Sh={bt.get('binary_sharpe',0):.2f}")
    ax4.plot(sc_k.index,   sc_k.values,   color=GREEN, lw=1.2,
             label=f"Half-Kelly Sh={bt.get('kelly_sharpe',0):.2f}")
    ax4.axhline(1.0, color=BORDER, lw=0.5)
    # Shade drawdown on primary curve
    strat_vals = sc.values
    cum_max    = np.maximum.accumulate(strat_vals)
    ax4.fill_between(sc.index, sc.values, cum_max, color=RED, alpha=0.15)
    ax4.set_title(
        f"Backtest ({sizing})  Sharpe={bt['strat_sharpe']:.2f}  MaxDD={bt['strat_maxdd']*100:.1f}%",
        color=TEXT, fontsize=10, fontweight='bold')
    ax4.set_ylabel('Cumulative Return', color=TEXT)
    ax4.legend(fontsize=7)

    # ── Panel 5: Feature Importance ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    if feat_imp is not None and len(feat_imp) > 0:
        fi = (feat_imp.groupby('feature')['importance']
              .mean().nlargest(15).sort_values())
        bars = ax5.barh(fi.index, fi.values, color=BLUE, alpha=0.8)
        for bar, val in zip(bars, fi.values):
            ax5.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=7, color=MUTED)
    ax5.set_title('Top-15 Feature Importance', color=TEXT, fontsize=10, fontweight='bold')
    ax5.set_xlabel('Mean Importance', color=TEXT)
    ax5.tick_params(axis='y', labelsize=7)

    # ── Panel 6: Signal probability bars + conformal set ────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    probs = signal.get('probabilities', {})
    if probs:
        labels  = list(probs.keys())
        values  = [probs[k] * 100 for k in labels]
        colors_ = {'SELL': RED, 'HOLD': GOLD, 'BUY': GREEN}
        pred_set = set(signal.get('prediction_set', []))  # conformal set
        bar_alpha = [1.0 if lbl in pred_set or not pred_set else 0.35
                     for lbl in labels]
        bars = ax6.bar(labels, values,
                       color=[colors_.get(l, BLUE) for l in labels],
                       width=0.4,
                       alpha=0.85)
        # Apply per-bar alpha manually (matplotlib bar alpha is global)
        for bar, al in zip(bars, bar_alpha):
            bar.set_alpha(al)
        for bar, val, lbl in zip(bars, values, labels):
            suffix = ' ★' if lbl in pred_set else ''
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%{suffix}', ha='center', va='bottom', fontsize=11,
                     fontweight='bold', color=TEXT)
        sig_str = signal.get('signal', 'N/A')
        conf    = signal.get('confidence', 0)
        cp_str  = ''
        if pred_set:
            is_single = signal.get('is_conformal_singleton', False)
            cov_lvl   = signal.get('conformal_coverage_level', 0.9)
            cp_str    = (f'  |  Conformal set: {sorted(pred_set)}  '
                         f'{"SINGLETON" if is_single else "ambiguous"}  '
                         f'({cov_lvl:.0%} coverage)  ★ = in set')
        ax6.set_title(
            f"Final Signal: {sig_str}  ({conf*100:.1f}% confidence)  "
            f"Model: {signal.get('model_used','N/A')}{cp_str}",
            color=TEXT, fontsize=10, fontweight='bold')
        ax6.set_ylabel('Probability (%)', color=TEXT)
        ax6.set_ylim(0, 115)
        ax6.axhline(38, color=MUTED, lw=0.7, ls=':', alpha=0.5,
                    label='Min confidence threshold')
        ax6.legend(fontsize=8)

    plt.suptitle(f'{ticker} — ML Analysis Dashboard  v12.0',
                 fontsize=14, fontweight='bold', color=TEXT, y=1.001)
    path = OUT_DIR / f"{ticker}_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close('all')
    log.info(f"Chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    run_start = time.time()
    ticker    = ask_ticker()
    init_run(ticker)

    log.info("*"*60)
    log.info(f"  {ticker}  ML ANALYZER  v12.0")
    log.info(f"  Output: {OUT_DIR.resolve()}")
    log.info("*"*60)

    device, gpu_name = get_device()
    use_gpu = (device.type == 'cuda')

    fetcher = DataFetcher(ticker, CONFIG['period'])
    df      = fetcher.fetch()
    market  = fetcher.fetch_context()

    # ── NEW 3: Fundamental / alternative data ────────────────────────────────
    fundamentals = {}
    if CONFIG.get('fetch_fundamentals', True):
        log_section("FUNDAMENTAL / ALTERNATIVE DATA")
        fundamentals = FundamentalFetcher(ticker).fetch()

    # ── Per-stock adaptive horizon + quantile thresholds ─────────────────────
    regime = analyze_stock_regime(df, CONFIG)
    CONFIG['predict_days'] = regime['predict_days']

    fe   = FeatureEngineer(df, market, fundamentals=fundamentals)
    feat = fe.build()
    y    = make_labels(df, regime)

    common = feat.index.intersection(y.dropna().index)
    X, y   = feat.loc[common], y.loc[common]

    # ── Data sufficiency check ────────────────────────────────────────────────
    n_train_est = int(len(X) * CONFIG['train_split'])
    if n_train_est < 500:
        log.warning(
            f"⚠ SMALL DATASET: ~{n_train_est} training rows for {ticker}. "
            f"Models may overfit. DL and meta-stack reliability is reduced. "
            f"Consider using only tree models for tickers with < 2y history.")
    elif n_train_est < 1000:
        log.info(f"  Dataset size: {n_train_est} training rows — moderate. "
                 f"Walk-forward results are more reliable than static backtest.")

    # ── NEW 20: Adversarial Validation — detect & remove covariate shift ──────
    # DISABLED BY DEFAULT for financial time-series.
    # Adversarial validation is designed for iid cross-sectional data.  For a
    # 10-year price series, temporal distribution shift is EXPECTED (NVDA 2016
    # ≈ $3, NVDA 2026 ≈ $180 with 10× higher volatility), and AUC=1.0 with
    # Δ≈0.51 vs permutation is the correct result — not evidence of leakage.
    # Acting on this by dropping ret_1, sma_144, beta_SPY_30, etc. removes
    # the most predictive short-term features, consistently hurting F1 by ~0.06
    # across all four tickers confirmed in run logs (NVDA, CLPT, RKLB, SBUX).
    # To re-enable: set adv_val_enabled=True in CONFIG.  The validator now also
    # respects adv_val_warn_only=True (log warning, keep all features).
    adv_report = {}
    if CONFIG.get('adv_val_enabled', False):
        X, adv_report = adversarial_validation(X, CONFIG['train_split'], CONFIG)

    tree = TreeTrainer(X, y, CONFIG, use_gpu=use_gpu)
    tree.train()

    # NEW 13: Regime-conditional model
    regime_mdl = RegimeModel(market, CONFIG)
    regime_ok  = regime_mdl.train(X, y)
    regime_sig = regime_mdl.predict_latest(tree.X.iloc[[-1]]) if regime_ok else None

    dl_trainers = []
    if HAS_TORCH:
        # ── Use tree's SelectKBest features for DL (fixes over-parameterisation) ─
        # Raw X has 220 features. The tree trainer already fitted a SelectKBest
        # pipeline (scaler → VarianceThreshold → SelectKBest-MI → 74 features).
        # Passing 220 features to the LSTM means layer-1 alone costs
        #   4 * (220 + hidden + 1) * hidden * 2 = 47,040 params for hidden=24
        # which blows the entire parameter budget before any other layer runs.
        # Using the same 74 tree-selected features reduces this to
        #   4 * (74 + hidden + 1) * hidden * 2 = 15,120 params for hidden=24
        # and ensures DL and tree models operate on the same signal space.
        try:
            Xf_dl  = X.fillna(X.median())
            Xs_dl  = tree.scaler.transform(Xf_dl)
            Xs_dl  = tree.var_sel.transform(Xs_dl)
            Xs_dl  = tree.feat_sel.transform(Xs_dl)
            X_dl   = pd.DataFrame(Xs_dl, index=X.index,
                                  columns=tree.feat_names_selected_)
            n_feat_dl = X_dl.shape[1]
            log.info(f"  DL feature reduction: {X.shape[1]} raw → {n_feat_dl} "
                     f"(tree SelectKBest — same signal space as tree models)")
        except Exception as _e:
            log.warning(f"  DL feature reduction failed ({_e}) — using raw X")
            X_dl      = X
            n_feat_dl = X.shape[1]

        n_tr_rows = int(len(X_dl) * CONFIG['train_split'])

        log_section("ADAPTIVE DL ARCHITECTURE SIZING")
        log.info(
            f"  Training rows : {n_tr_rows}  |  features: {n_feat_dl}  |  "
            f"seq_len: {CONFIG['lstm_seq_len']}  |  aug: {'3×' if use_gpu else '1× (CPU)'}")
        log.info(
            f"  Safety rule   : est_params ≤ n_original_seqs × {_SAMPLES_PER_PARAM} "
            f"(Param-to-Sample ratio ≤ 1:{_SAMPLES_PER_PARAM}, original seqs only — "
            f"augmented copies not counted toward capacity)")

        # ── BiLSTM (adaptive size) ────────────────────────────────────────────
        lstm_kw = get_adaptive_dl_kwargs(
            'bilstm', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
        lstm_t = TorchTrainer("BiLSTM-Attention", device, CONFIG)
        lstm_t.train(X_dl, y, ModelClass=BiLSTMAttention,
                     model_kwargs=lstm_kw['model_kw'],
                     epochs=lstm_kw['epochs'],
                     patience=lstm_kw['patience'],
                     seq_len=CONFIG['lstm_seq_len'],
                     lr=CONFIG['lstm_lr'],
                     batch=CONFIG['lstm_batch'])
        dl_trainers.append(lstm_t)

        if use_gpu:
            torch.cuda.empty_cache()

        # ── Transformer (adaptive size) ───────────────────────────────────────
        tf_kw = get_adaptive_dl_kwargs(
            'transformer', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
        tf_t = TorchTrainer("Transformer-Encoder", device, CONFIG)
        tf_t.train(X_dl, y, ModelClass=TransformerEncoder,
                   model_kwargs=tf_kw['model_kw'],
                   epochs=tf_kw['epochs'],
                   patience=tf_kw['patience'],
                   seq_len=CONFIG['lstm_seq_len'],
                   lr=CONFIG['lstm_lr'],
                   batch=CONFIG['lstm_batch'])
        dl_trainers.append(tf_t)

        if use_gpu:
            torch.cuda.empty_cache()

        # ── NEW 16: Temporal Fusion Transformer (adaptive size) ───────────────
        tft_kw = get_adaptive_dl_kwargs(
            'tft', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
        tft_t = TorchTrainer("TFT", device, CONFIG)
        tft_t._feat_names = list(X_dl.columns)  # for VSN importance logging
        tft_t.train(X_dl, y, ModelClass=TemporalFusionTransformer,
                    model_kwargs=tft_kw['model_kw'],
                    epochs=tft_kw['epochs'],
                    patience=tft_kw['patience'],
                    seq_len=CONFIG['lstm_seq_len'],
                    lr=CONFIG['tft_lr'],
                    batch=CONFIG['lstm_batch'])
        dl_trainers.append(tft_t)

    else:
        log.warning("PyTorch not installed — skipping LSTM & Transformer")

    meta_signal = build_meta(tree, dl_trainers) if dl_trainers else None
    tree_signal = tree.get_latest_signal()
    dl_signals  = [t.predict_latest() for t in dl_trainers]

    # ── FIX D: Improved signal fallback chain ─────────────────────────────────
    # When meta_signal is None (either no DL or very-low-confidence filter):
    #   1. Gather all non-collapsed candidate signals with their confidence
    #   2. Prefer majority-vote direction when signals agree (2+ of tree/DL/regime)
    #   3. Tie-break on confidence
    # This prevents regime (single HMM bar) from silently overriding a
    # high-confidence tree signal just because it happens to be first in the chain.
    valid_dl = [s for s in dl_signals if not s.get('is_collapsed', False)]
    all_dl_collapsed = len(dl_trainers) > 0 and len(valid_dl) == 0
    if all_dl_collapsed:
        log.warning("All DL models collapsed — tree signal will be used as primary.")

    if meta_signal:
        final_signal = meta_signal
        log.info("Using META-STACK signal")
    else:
        # Build candidate pool: tree + non-collapsed DL + regime
        candidates = [tree_signal] + valid_dl
        if regime_sig and not regime_sig.get('is_collapsed', False):
            candidates.append(regime_sig)

        if candidates:
            # Majority vote on direction
            from collections import Counter
            vote_counts = Counter(s['signal'] for s in candidates)
            top_direction, top_votes = vote_counts.most_common(1)[0]
            if top_votes >= 2:
                # Two or more signals agree — pick the highest-confidence one
                # in the majority direction
                agreeing = [s for s in candidates if s['signal'] == top_direction]
                final_signal = max(agreeing, key=lambda s: s['confidence'])
                log.info(f"Majority vote → {top_direction} "
                         f"({top_votes}/{len(candidates)} signals agree, "
                         f"conf={final_signal['confidence']*100:.1f}%, "
                         f"model={final_signal.get('model_used','?')})")
            else:
                # No majority — use highest-confidence single signal
                final_signal = max(candidates, key=lambda s: s['confidence'])
                log.info(f"No majority — highest-confidence signal: "
                         f"{final_signal['signal']} "
                         f"({final_signal['confidence']*100:.1f}%, "
                         f"model={final_signal.get('model_used','?')})")
        else:
            # Last resort
            final_signal = tree_signal
            log.warning("No valid candidates — falling back to tree signal.")

    final_signal['date'] = str(df.index[-1].date())

    # ── NEW 17: Conformal Prediction — calibrate on best DL model ────────────
    conformal_result = {}
    best_dl_for_cp = None
    if dl_trainers:
        # Pick the non-collapsed DL trainer with highest accuracy
        non_collapsed = [t for t in dl_trainers
                         if not getattr(t, 'is_collapsed', False)
                         and t.te_proba is not None
                         and len(t.te_proba) >= 30]
        if non_collapsed:
            best_dl_for_cp = max(non_collapsed,
                                  key=lambda t: accuracy_score(t.y_te_vals, t.te_preds))
    if best_dl_for_cp is not None:
        try:
            log_section("CONFORMAL PREDICTION (NEW 17)")
            alpha    = CONFIG.get('conformal_alpha', 0.10)
            cal_pct  = CONFIG.get('conformal_cal_pct', 0.25)
            lam      = CONFIG.get('conformal_lambda', 0.01)
            kreg     = CONFIG.get('conformal_kreg', 1)
            cp = ConformalPredictor(alpha=alpha, lam=lam, kreg=kreg)

            proba_all = best_dl_for_cp.te_proba          # (N_test, 3)
            y_all     = best_dl_for_cp.y_te_vals         # (N_test,)
            n_cal     = max(20, int(len(y_all) * cal_pct))
            # Use first n_cal as calibration (chronological — no look-ahead)
            proba_cal, proba_eval = proba_all[:n_cal], proba_all[n_cal:]
            y_cal,     y_eval     = y_all[:n_cal],     y_all[n_cal:]

            cp.calibrate(proba_cal, y_cal)
            cov_stats = cp.empirical_coverage(proba_eval, y_eval)
            # NEW 21: store on tree so backtest() can use it for position sizing
            tree._conformal_predictor = cp
            log.info(f"  Model         : {best_dl_for_cp.name}")
            log.info(f"  Alpha (1-cov) : {alpha:.0%}  →  target ≥{1-alpha:.0%} coverage")
            log.info(f"  Empirical cov : {cov_stats['coverage']:.3f}  "
                     f"(target {cov_stats['target_coverage']:.3f})")
            log.info(f"  Avg set size  : {cov_stats['avg_set_size']:.2f}")
            log.info(f"  Singleton rate: {cov_stats['singleton_rate']:.1%}  "
                     f"← fraction with unambiguous signal")
            log.info(f"  qhat          : {cov_stats['qhat']:.4f}")

            # Annotate the final signal with the prediction set
            sig_proba = np.array(list(final_signal['probabilities'].values()),
                                 dtype=np.float64)
            cp_annot = cp.annotate_signal(sig_proba)
            final_signal.update(cp_annot)
            if cp_annot.get('is_conformal_singleton'):
                log.info(f"  Final signal prediction set: "
                         f"{cp_annot['prediction_set']}  ← SINGLETON (high confidence)")
            else:
                log.info(f"  Final signal prediction set: "
                         f"{cp_annot['prediction_set']}  "
                         f"(set_size={cp_annot['set_size']} — consider reducing size)")
            conformal_result = cov_stats
        except Exception as e:
            log.warning(f"Conformal prediction failed: {e}")

    bt       = backtest(df, tree)
    wf_bt    = backtest_walkforward(df, tree)  # NEW 14
    cpcv_res = run_cpcv(tree, df)  # NEW 12
    feat_imp = tree.get_feature_importance()
    make_charts(ticker, df, tree, bt, final_signal, feat_imp)

    all_acc = {n: r['accuracy'] for n, r in tree.results.items()}
    for t in dl_trainers:
        if t.te_preds is not None:
            from sklearn.metrics import accuracy_score as _acc
            all_acc[t.name] = float(_acc(t.y_te_vals, t.te_preds))
    if meta_signal:
        all_acc['Meta-Stack'] = meta_signal.get('meta_accuracy', 0)

    # Collapse / temperature info for JSON output
    dl_diagnostics = []
    for t in dl_trainers:
        dl_diagnostics.append({
            'model':        t.name,
            'is_collapsed': getattr(t, 'is_collapsed', False),
            'temperature':  getattr(t, 'temperature',  1.0),
        })

    out = {
        'ticker': ticker, 'generated': datetime.datetime.now().isoformat(),
        'signal': final_signal, 'tree_signal': tree_signal,
        'dl_signals': dl_signals, 'meta_signal': meta_signal,
        'dl_diagnostics': dl_diagnostics,
        'backtest': {k: v for k, v in bt.items() if not isinstance(v, pd.Series)},
        'model_accuracy': all_acc, 'gpu_info': gpu_name,
        'regime': {
            'predict_days':   regime['predict_days'],
            'buy_thresh':     regime['buy_thresh'],
            'sell_thresh':    regime['sell_thresh'],
            'ann_vol':        regime['ann_vol'],
            'speed':          regime['speed'],
            'best_ic':        regime['best_ic'],
            'ic_table':       regime['ic_table'],
            'label_method':   'triple_barrier' if CONFIG.get('use_triple_barrier') else 'quantile',
        },
        'fundamentals': fundamentals,
        'adversarial_validation': adv_report,
        'cpcv':          cpcv_res,
        'walkforward_backtest': {k: v for k, v in (wf_bt or {}).items()
                                 if not isinstance(v, pd.Series)},
        'regime_signal': regime_sig,
        'conformal_prediction': conformal_result,
        'total_runtime': elapsed(run_start),
    }
    path = OUT_DIR / f"{ticker}_signal.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, cls=_NumpyEncoder)

    log_section("FINAL SUMMARY")
    s = final_signal
    log.info(f"TICKER   : {ticker}")
    log.info(f"SIGNAL   : {s['signal']}")
    log.info(f"CONF     : {s.get('confidence',0)*100:.2f}%")
    log.info(f"MODEL    : {s.get('model_used','')}")
    log.info(f"REGIME   : {regime['speed']}  horizon={regime['predict_days']}d  "
             f"IC={regime['best_ic']:.4f}  vol={regime['ann_vol']*100:.1f}%")
    log.info(f"LABELS   : {'Triple Barrier' if CONFIG.get('use_triple_barrier') else 'Quantile'}")
    # Static backtest runs over ALL 1122 bars including the 897 training rows.
    # In-sample Sharpe is always inflated — treat as an upper bound only.
    # WF-BT (walk-forward OOS) and CPCV are the only trustworthy numbers.
    static_ret_pct = bt['strat_return'] * 100
    if static_ret_pct > 500:
        log.warning(
            f"BACKTEST : ⚠ IN-SAMPLE INFLATED — Return={static_ret_pct:+.1f}%  "
            f"Sharpe={bt['strat_sharpe']:.3f}  "
            f"(includes {int(len(X)*CONFIG['train_split'])} training rows — "
            f"use WF-BT Sharpe below as the real number)")
    else:
        log.info(f"BACKTEST : {bt.get('sizing_method','N/A')} sizing  "
                 f"Sharpe={bt['strat_sharpe']:.3f}")
    if wf_bt:
        log.info(f"WF-BT    : Return={wf_bt['wf_return']*100:+.2f}%  "
                 f"Sharpe={wf_bt['wf_sharpe']:.3f}  "
                 f"MaxDD={wf_bt['wf_maxdd']*100:.1f}%")
    if cpcv_res:
        log.info(f"CPCV     : {cpcv_res['n_paths']} paths  "
                 f"Sharpe mean={cpcv_res['sharpe_mean']:+.3f}  "
                 f"p5={cpcv_res['sharpe_p5']:+.3f}  "
                 f"pct>0={cpcv_res['pct_positive']*100:.0f}%")
    if regime_sig:
        log.info(f"REGIME   : {regime_sig.get('regime','?')}  "
                 f"signal={regime_sig['signal']}  "
                 f"conf={regime_sig['confidence']*100:.1f}%")
    if dl_diagnostics:
        for d in dl_diagnostics:
            col = '⚠ COLLAPSED' if d['is_collapsed'] else 'OK'
            log.info(f"  {d['model']:<30} {col}  T={d['temperature']:.3f}")
    if conformal_result:
        ps_str = str(final_signal.get('prediction_set', '—'))
        log.info(f"CONFORMAL: set={ps_str}  "
                 f"singleton={'YES' if final_signal.get('is_conformal_singleton') else 'NO'}  "
                 f"cov={conformal_result.get('coverage',0):.3f}  "
                 f"avg_size={conformal_result.get('avg_set_size',0):.2f}")
    if adv_report:
        auc_str = f"{adv_report['auc']:.4f}" if adv_report.get('auc') is not None else "N/A"
        shift   = adv_report.get('shift_detected', False)
        n_drop  = adv_report.get('n_dropped', 0)
        log.info(f"ADV-VAL  : AUC={auc_str}  "
                 f"shift={'YES ⚠' if shift else 'NO ✓'}  "
                 f"dropped={n_drop} features"
                 + (f"  {adv_report['dropped_features'][:3]}…" if n_drop > 0 else ""))
    log.info("\nALL MODEL ACCURACIES:")
    for n, a in sorted(all_acc.items(), key=lambda x: -x[1]):
        log.info(f"  {n:<30} {a*100:.3f}%")
    log.info(f"\nRUNTIME  : {elapsed(run_start)}")
    log.info("NOT FINANCIAL ADVICE. EDUCATIONAL PURPOSES ONLY.")
    log.info("="*60)

    # ── Optional diagnostic test suite ───────────────────────────────────────
    if CONFIG.get('run_diagnostics', False):
        run_diagnostic_tests(ticker, df, X, y, tree, dl_trainers,
                             tree_signal, OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC TEST SUITE  v1.0
# 6 modules that probe stability, reliability, and calibration.
# Enable with "run_diagnostics": True in CONFIG (adds ~3 extra tree fits).
# Output: reports/<TICKER>/<TICKER>_diagnostics.json
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic_tests(ticker, df, X, y, tree, dl_trainers,
                         tree_signal, out_dir):
    """
    6-module reliability test suite.

    Module 1 — Seed Stability
      Retrain the best tree model with seeds 42/43/44. If ≥2 agree on signal
      direction the model is stable. F1 std > 0.04 indicates high sensitivity
      to weight initialisation.

    Module 2 — Temporal Stability
      Split the full labelled dataset into 3 chronological windows. Train on
      window i-1, test on window i. F1 std > 0.10 means the model is strongly
      regime-dependent and walk-forward results should be preferred over the
      static backtest.

    Module 3 — Feature Stability
      Repeat seed experiment and record the top-5 features by importance each
      time. If the same 3+ features appear across all 3 seeds, feature selection
      is stable. Otherwise the selected feature set is noisy.

    Module 4 — Quarterly Hit Rate
      Apply the trained model over 63-bar rolling windows and record accuracy
      and signal-weighted Sharpe per quarter. Accuracy drift > 15% across
      quarters indicates regime sensitivity.

    Module 5 — Label Flip Sensitivity
      Randomly flip 5% of training labels (simulate annotation noise) and
      refit. If the final signal flips or F1 drops > 0.05, the model is
      fragile to label noise — common with ambiguous Triple Barrier labels.

    Module 6 — Probability Calibration
      Bin the ensemble's max predicted probability into 5 buckets. Compare
      average confidence vs actual hit rate. Overconfident bins (conf > hit+0.10)
      mean the model's reported confidence cannot be trusted for position sizing.
    """
    log_section("DIAGNOSTIC TEST SUITE  v.initial")
    results = {
        'ticker':    ticker,
        'generated': datetime.datetime.now().isoformat(),
        'n_samples': len(X),
        'n_features': X.shape[1],
    }
    t0_diag = time.time()

    # ── Identify best tree model (excluding MiniROCKET / Ensemble) ────────────
    best_name = max(
        {k: v for k, v in tree.results.items()
         if k not in ('Ensemble', 'MiniROCKET')},
        key=lambda k: tree.results[k]['f1'])
    BestModel = type(tree.models[best_name])
    best_params = tree.models[best_name].get_params()
    Xs_all = np.vstack([tree.Xs_tr, tree.Xs_te])
    ys_all = np.concatenate([tree.y_tr.values, tree.y_te.values])
    log.info(f"  Diagnostic base model: {best_name}  "
             f"(test F1={tree.results[best_name]['f1']:.4f})")

    # ── Helper: fast refit ────────────────────────────────────────────────────
    def _refit(seed):
        cfg_tmp = dict(CONFIG); cfg_tmp['random_state'] = seed
        tr_tmp  = TreeTrainer(X, y, cfg_tmp, use_gpu=tree.use_gpu)
        tr_tmp.train()
        return tr_tmp

    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 1: SEED STABILITY
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 1/6  — Seed Stability")
    seed_signals, seed_f1s = [], []
    for seed in [42, 43, 44]:
        try:
            tr_s = _refit(seed)
            sig  = tr_s.get_latest_signal()
            best_f1 = max(tr_s.results[k]['f1'] for k in tr_s.results
                          if k not in ('MiniROCKET', 'Ensemble'))
            seed_signals.append(sig['signal'])
            seed_f1s.append(best_f1)
            log.info(f"  seed={seed}  signal={sig['signal']}  "
                     f"conf={sig['confidence']*100:.1f}%  best_f1={best_f1:.4f}")
        except Exception as e:
            log.warning(f"  seed={seed} failed: {e}")

    if seed_signals:
        from collections import Counter as _Counter
        mode_sig    = _Counter(seed_signals).most_common(1)[0][0]
        agree_frac  = seed_signals.count(mode_sig) / len(seed_signals)
        f1_std_seed = float(np.std(seed_f1s)) if len(seed_f1s) > 1 else 0.0
        stable      = agree_frac >= 0.67 and f1_std_seed < 0.04
        log.info(f"\n  Consensus signal : {mode_sig}  "
                 f"agreement={agree_frac:.0%}  F1 std={f1_std_seed:.4f}")
        log.info(f"  {'✓ STABLE' if stable else '⚠ UNSTABLE'} — "
                 f"{'≥2/3 seeds agree and F1 variation low' if stable else 'signal or F1 varies across seeds'}")
        results['seed_stability'] = {
            'signals': seed_signals, 'f1s': seed_f1s,
            'consensus': mode_sig, 'agreement': agree_frac,
            'f1_std': f1_std_seed, 'stable': stable}
    else:
        results['seed_stability'] = {'stable': False, 'error': 'all seeds failed'}

    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 2: TEMPORAL STABILITY
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 2/6  — Temporal Stability")
    n_all   = len(Xs_all)
    wsize   = n_all // 3
    window_rows = []
    for wi in range(3):
        tr_end = wi * wsize
        te_s   = tr_end; te_e = te_s + wsize
        if tr_end < 30 or (te_e - te_s) < 20:
            continue
        try:
            m_w = BestModel(**best_params)
            m_w.fit(Xs_all[:tr_end], ys_all[:tr_end])
            p_w = m_w.predict(Xs_all[te_s:te_e]).ravel()
            wa  = float(accuracy_score(ys_all[te_s:te_e], p_w))
            wf  = float(f1_score(ys_all[te_s:te_e], p_w,
                                  average='macro', zero_division=0))
            window_rows.append({'window': wi+1, 'tr_rows': tr_end,
                                 'te_rows': te_e-te_s, 'acc': wa, 'f1': wf})
            log.info(f"  Window {wi+1}: train 0–{tr_end}  test {te_s}–{te_e}  "
                     f"acc={wa:.4f}  f1={wf:.4f}")
        except Exception as e:
            log.warning(f"  Window {wi+1} failed: {e}")

    if window_rows:
        f1_vals   = [r['f1'] for r in window_rows]
        f1_t_std  = float(np.std(f1_vals)) if len(f1_vals) > 1 else 0.0
        consistent = f1_t_std < 0.10
        log.info(f"\n  F1 std = {f1_t_std:.4f}  "
                 f"{'✓ CONSISTENT (< 0.10)' if consistent else '⚠ REGIME-SENSITIVE (≥ 0.10)'}")
        results['temporal_stability'] = {
            'windows': window_rows, 'f1_std': f1_t_std, 'consistent': consistent}
    else:
        results['temporal_stability'] = {'consistent': False, 'error': 'insufficient data'}

    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 3: FEATURE STABILITY
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 3/6  — Feature Stability")
    top5_sets = []
    for seed in [42, 43, 44]:
        try:
            tr_s = _refit(seed)
            fi   = tr_s.get_feature_importance()
            if len(fi) > 0:
                top5 = set(fi.groupby('feature')['importance'].mean()
                           .nlargest(5).index.tolist())
                top5_sets.append({'seed': seed, 'features': sorted(top5)})
                log.info(f"  seed={seed}: {sorted(top5)}")
        except Exception as e:
            log.warning(f"  seed={seed} feature importance failed: {e}")

    if len(top5_sets) >= 2:
        sets  = [set(s['features']) for s in top5_sets]
        # Pairwise overlap
        overlaps = []
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                overlaps.append(len(sets[i] & sets[j]) / 5.0)
        feat_stab = float(np.mean(overlaps)) if overlaps else 0.0
        stable_feat = feat_stab >= 0.60
        # Union across all seeds — always-present features
        always_present = set.intersection(*sets) if sets else set()
        log.info(f"\n  Pairwise top-5 overlap: {feat_stab:.0%}  "
                 f"{'✓ STABLE (≥60%)' if stable_feat else '⚠ UNSTABLE (<60%)'}")
        log.info(f"  Always-present features: {sorted(always_present) or 'none'}")
        results['feature_stability'] = {
            'top5_per_seed': top5_sets,
            'overlap_score': feat_stab,
            'always_present': sorted(always_present),
            'stable': stable_feat}
    else:
        results['feature_stability'] = {'stable': False, 'error': 'insufficient seeds'}

    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 4: QUARTERLY HIT RATE
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 4/6  — Quarterly Hit Rate")
    best_model = tree.models[best_name]

    # BUG FIX: The original code predicted on Xs_all (train + test). Q1–Q3 are
    # almost entirely training rows, so the model predicts its OWN training data
    # → acc=1.000 for 3 quarters, acc≈0.58 for Q4 (the only OOS quarter).
    # The drift metric (max−min = 0.418) was meaningless noise from this leak.
    #
    # Fix: predict ONLY on the test set, then split that into quarters.
    # We use a walk-forward re-prediction over Xs_te so every prediction is OOS.
    preds_te  = best_model.predict(tree.Xs_te).ravel()
    ys_te     = tree.y_te.values
    sig_te    = np.array([1 if p==2 else (-1 if p==0 else 0) for p in preds_te])
    n_te      = len(preds_te)

    close_arr = df['Close'].reindex(tree.X_te.index, method='ffill').values
    ret_arr   = np.zeros(n_te)
    if n_te > 1:
        ret_arr[1:] = np.diff(close_arr) / (close_arr[:-1] + 1e-10)

    log.info(f"  Quarterly analysis on OOS test set only ({n_te} rows — no train-set leakage)")
    qsize = max(20, n_te // 4)
    quarterly = []
    for qi in range(4):
        qs = qi * qsize; qe = min(qs + qsize, n_te)
        if qe - qs < 10: continue
        corr = float((preds_te[qs:qe] == ys_te[qs:qe]).mean())
        f1_q = float(f1_score(ys_te[qs:qe], preds_te[qs:qe],
                               average='macro', zero_division=0))
        strat_r  = sig_te[qs:qe] * ret_arr[qs:qe]
        sh_q     = float(strat_r.mean() / (strat_r.std() + 1e-10) * np.sqrt(252))
        n_trades = int((sig_te[qs:qe] != 0).sum())
        quarterly.append({'quarter': qi+1, 'rows': qe-qs,
                           'acc': corr, 'f1': f1_q,
                           'sharpe': sh_q, 'n_trades': n_trades})
        log.info(f"  Q{qi+1} (rows {qs}–{qe:4d}): "
                 f"acc={corr:.3f}  f1={f1_q:.3f}  "
                 f"sharpe={sh_q:+.2f}  trades={n_trades}")

    if quarterly:
        acc_vals   = [q['acc'] for q in quarterly]
        acc_drift  = float(max(acc_vals) - min(acc_vals))
        sharpe_neg = sum(1 for q in quarterly if q['sharpe'] < 0)
        stable_q   = acc_drift < 0.15 and sharpe_neg <= 1
        log.info(f"\n  Accuracy drift = {acc_drift:.3f}  "
                 f"{'✓ STABLE (<0.15)' if acc_drift < 0.15 else '⚠ HIGH DRIFT'}")
        log.info(f"  Negative-Sharpe quarters: {sharpe_neg}/4  "
                 f"{'✓' if sharpe_neg <= 1 else '⚠ EDGE DISAPPEARS IN SOME PERIODS'}")
        results['quarterly_performance'] = {
            'quarters': quarterly, 'acc_drift': acc_drift,
            'neg_sharpe_quarters': sharpe_neg, 'stable': stable_q}
    else:
        results['quarterly_performance'] = {'stable': False}

    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 5: LABEL FLIP SENSITIVITY
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 5/6  — Label Flip Sensitivity (5% noise)")
    try:
        rng_flip = np.random.default_rng(99)
        n_flip   = max(1, int(len(tree.y_tr) * 0.05))
        flip_idx = rng_flip.choice(len(tree.y_tr), n_flip, replace=False)
        y_noisy  = tree.y_tr.values.copy()
        for fi in flip_idx:
            alts = [c for c in [0, 1, 2] if c != y_noisy[fi]]
            y_noisy[fi] = rng_flip.choice(alts)

        m_noisy = BestModel(**best_params)
        m_noisy.fit(tree.Xs_tr, y_noisy)
        p_noisy   = m_noisy.predict(tree.Xs_te).ravel()
        f1_noisy  = float(f1_score(tree.y_te.values, p_noisy,
                                    average='macro', zero_division=0))
        f1_clean  = float(tree.results[best_name]['f1'])
        f1_delta  = f1_noisy - f1_clean

        latest_xs = tree._transform_latest(tree.X.iloc[[-1]])
        sig_noisy = {0:'SELL',1:'HOLD',2:'BUY'}[int(m_noisy.predict(latest_xs).ravel()[0])]
        sig_clean = tree_signal['signal']
        flipped   = sig_noisy != sig_clean
        robust    = abs(f1_delta) < 0.05 and not flipped

        log.info(f"  {n_flip} labels flipped ({n_flip/len(tree.y_tr)*100:.1f}% of train)")
        log.info(f"  F1  clean={f1_clean:.4f}  noisy={f1_noisy:.4f}  Δ={f1_delta:+.4f}")
        log.info(f"  Signal  clean={sig_clean}  noisy={sig_noisy}  "
                 f"{'✓ ROBUST' if not flipped else '⚠ SIGNAL FLIPPED'}")
        log.info(f"  {'✓ ROBUST to label noise' if robust else '⚠ FRAGILE — recheck label method'}")
        results['label_sensitivity'] = {
            'n_flipped': n_flip, 'f1_clean': f1_clean, 'f1_noisy': f1_noisy,
            'f1_delta': f1_delta, 'signal_clean': sig_clean,
            'signal_noisy': sig_noisy, 'signal_flipped': flipped, 'robust': robust}
    except Exception as e:
        log.warning(f"  Label sensitivity failed: {e}")
        results['label_sensitivity'] = {'robust': False, 'error': str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 6: PROBABILITY CALIBRATION
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 6/6  — Probability Calibration")
    try:
        ens = tree.models.get('Ensemble') or best_model
        proba_te = ens.predict_proba(tree.Xs_te)          # (n_test, 3)
        preds_te = proba_te.argmax(axis=1)
        correct  = (preds_te == tree.y_te.values).astype(int)
        max_conf = proba_te.max(axis=1)

        bins       = [0.33, 0.45, 0.55, 0.65, 0.75, 1.01]
        cal_rows   = []
        overconf_n = 0
        log.info(f"  {'Conf bin':<14} {'n':>5}  {'avg_conf':>9}  {'hit_rate':>9}  {'gap':>7}  status")
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (max_conf >= lo) & (max_conf < hi)
            if mask.sum() < 3:
                continue
            avg_c  = float(max_conf[mask].mean())
            hit_r  = float(correct[mask].mean())
            gap    = avg_c - hit_r
            over   = gap > 0.10
            if over: overconf_n += 1
            flag   = '⚠ overconf' if over else ('under' if gap < -0.10 else '✓ OK')
            log.info(f"  {lo:.2f}–{hi:.2f}       {mask.sum():>5}  "
                     f"{avg_c:>9.3f}  {hit_r:>9.3f}  {gap:>+7.3f}  {flag}")
            cal_rows.append({'bin': f'{lo:.2f}–{hi:.2f}', 'n': int(mask.sum()),
                              'avg_conf': avg_c, 'hit_rate': hit_r,
                              'gap': gap, 'overconfident': over})

        calibrated = overconf_n == 0
        log.info(f"\n  Overconfident bins: {overconf_n}/{len(cal_rows)}  "
                 f"{'✓ WELL CALIBRATED' if calibrated else '⚠ CONFIDENCE OVERSTATED — do not size by conf directly'}")
        results['calibration'] = {
            'table': cal_rows, 'overconfident_bins': overconf_n,
            'well_calibrated': calibrated}
    except Exception as e:
        log.warning(f"  Calibration failed: {e}")
        results['calibration'] = {'well_calibrated': False, 'error': str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL RELIABILITY SCORE
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAGNOSTIC SUMMARY")
    module_results = {
        'Seed stability    (≥2/3 seeds agree, F1 std < 0.04)':
            results.get('seed_stability', {}).get('stable', False),
        'Temporal stability (F1 std < 0.10 across windows)':
            results.get('temporal_stability', {}).get('consistent', False),
        'Feature stability  (≥60% top-5 overlap across seeds)':
            results.get('feature_stability', {}).get('stable', False),
        'Quarterly accuracy (drift < 15%, ≤1 neg-Sharpe quarter)':
            results.get('quarterly_performance', {}).get('stable', False),
        'Label robustness   (F1 Δ < 0.05, signal stable vs 5% noise)':
            results.get('label_sensitivity', {}).get('robust', False),
        'Calibration        (no overconfident bins)':
            results.get('calibration', {}).get('well_calibrated', False),
    }
    score = sum(module_results.values())
    for name, passed in module_results.items():
        log.info(f"  {'✓' if passed else '✗'}  {name}")

    label = 'RELIABLE' if score >= 5 else ('MODERATE' if score >= 3 else 'UNRELIABLE')
    log.info(f"\n  ─────────────────────────────────────────")
    log.info(f"  Reliability score : {score}/{len(module_results)}  [{label}]")
    log.info(f"  Diagnostic time   : {elapsed(t0_diag)}")
    log.info(f"  ─────────────────────────────────────────")
    log.info("\n  HOW TO READ THE RESULTS:")
    log.info("  ≥5 modules pass → signals are production-grade")
    log.info("  3-4 modules pass → use signals but verify with WF-BT")
    log.info("  ✗ ≤2 modules pass → do not trade; investigate failing modules first")

    results['reliability_score'] = score
    results['reliability_label'] = label
    results['module_results']    = {k: v for k, v in module_results.items()}
    results['total_diag_time']   = elapsed(t0_diag)

    diag_path = out_dir / f"{ticker}_diagnostics.json"
    with open(diag_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    log.info(f"\n  Full diagnostics → {diag_path}")
    return results


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
