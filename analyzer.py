# -*- coding: utf-8 -*-
"""
Stock ML Analyzer  v12.0
Run: python analyzer.py  |  python analyzer.py AAPL
"""

import warnings; warnings.filterwarnings('ignore')
import os, sys, json, datetime, time, logging, traceback, random, textwrap
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path

from utils.fundamental_utils import (
    build_dcf_state,
    compute_speculative_growth_profile,
    compute_dilution_metrics,
    dcf_surface_analysis,
    dcf_valuation,
    latest_statement_value,
    load_statement_frame,
    reverse_dcf_analysis,
    safe_float,
)
from utils.debug_audit import validate_signal_artifact
from utils.model_explainability import build_explainability
from utils.run_metadata import (
    DEFAULT_CONFIG_VERSION,
    append_experiment_record,
    build_run_metadata,
    complete_run_metadata,
)
from utils.telegram_notifier import (
    create_progress_session,
    notify_failure,
    notify_success,
    progress_enabled as telegram_progress_enabled,
    result_delay_seconds as telegram_result_delay_seconds,
    send_chat_action,
)

SIGNAL_SCHEMA_VERSION = "signal-json/v1"

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────
# Tiingo API (free tier: 500 req/day, clean OHLCV, superior data quality).
# Preferred: set the TIINGO_API_KEY environment variable so the key is never
# stored in source code.  The literal below is used only as a last-resort
# fallback (leave it empty to force env-only).
# Get a free key at https://api.tiingo.com/
TIINGO_API_KEY: str = os.environ.get("TIINGO_API_KEY", "") or ""

# Local DuckDB cache: stores downloaded OHLCV so portfolio scans never
# re-download the same ticker twice in the same session (or within TTL hours).
# Set to "" to disable caching entirely.
DATA_CACHE_DIR: str = str(Path(__file__).resolve().parent / "data_cache")
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
    class _TorchDeviceStub(str):
        @property
        def type(self):
            return str(self)
    class _TorchStub:
        @staticmethod
        def device(name):
            return _TorchDeviceStub(name)
    class _TorchNNStub:
        Module = object
    torch = _TorchStub()
    nn = _TorchNNStub()
    F = None
    DataLoader = None
    TensorDataset = None

from sklearn.ensemble import (RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             precision_recall_fscore_support,
                             average_precision_score,
                             log_loss)
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
    "use_triple_barrier": True,   # kept for backward-compat; overridden by label_method
    # label_method: 'rar' (default) | 'triple_barrier' | 'quantile'
    # 'rar' = Risk-Adjusted Return labels — guaranteed 25/50/25 balance, handles trending stocks.
    # 'triple_barrier' = de Prado path-dependent barriers — better for large / mean-reverting stocks.
    "label_method":      "rar",
    # Symmetric multipliers (1.0/1.0) per de Prado: asymmetric pt > sl injects
    # long-bias into labels; keep equal so the model discovers direction itself.
    # tb_max_hold_pct removed: natural HOLD dominance is handled by FocalLoss +
    # class_weight='balanced'. Tightening barriers to force balance is label noise.
    "tb_pt_sl":          [1.0, 1.0],  # symmetric barriers
    "tb_vol_window":     20,           # 20-day rolling vol (standard)
    "tb_min_samples":    150,          # min rows; below this uses quantile labels
    "rar_rank_window":   252,          # RAR: rolling window for percentile ranking

    # ── NEW 2: Fractional Differentiation ────────────────────────────────────
    "use_fracdiff":      True,
    "fracdiff_d":        0.4,    # d∈(0,1): 0.4 keeps ~80% memory, stationary
    "fracdiff_thres":    1e-4,   # weight truncation threshold

    # ── NEW 3: Fundamental features ───────────────────────────────────────────
    "fetch_fundamentals": True,  # pull options/short/analyst data from yfinance
    "fundamentals_reverse_dcf_enabled": True,
    "fundamentals_dcf_surface_enabled": True,
    "fundamentals_dilution_enabled": True,
    "fundamentals_country_risk_premium": 0.0,
    "fundamentals_dcf_terminal_growth": 0.025,
    "fundamentals_dcf_years": 5,
    "fundamentals_dcf_growth_grid": [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "fundamentals_dcf_discount_grid": [0.08, 0.10, 0.12, 0.14, 0.16],

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
    "deployment_ranking_metric": "robust_score",
    "active_tree_models": ["RandomForest", "CatBoost", "XGBoost"],
    "active_dl_models": ["TiDE", "PatchTST"],
    "active_stack_models": ["Meta-Stack"],
    "challenger_models_enabled": False,
    "router_enabled": True,
    "router_confidence_floor": 0.55,
    "router_block_size": 21,
    "router_min_blocks": 6,
    "threshold_objective": "wf_sharpe_net_cost",
    "threshold_buy_grid": [0.45, 0.50, 0.55, 0.60],
    "threshold_sell_grid": [0.45, 0.50, 0.55, 0.60],
    "threshold_margin_grid": [0.03, 0.05, 0.08, 0.12],
    "threshold_tuning_min_rows": 40,
    "calibration_tree_method": "auto",
    "calibration_dl_method": "temperature",
    "global_family_calibration_min_rows": 80,
    "per_regime_calibration_min_rows": 150,
    "tree_calibration_isotonic_min_rows": 120,
    "tree_calibration_isotonic_min_class_rows": 20,
    "robust_score_ece_cap": 0.25,
    "decision_margin_threshold": 0.08,
    "decision_edge_cost_multiplier": 1.0,
    "decision_trade_on_singleton_only": False,
    "deployment_min_non_hold_recall": 0.05,
    "deployment_min_macro_f1": 0.18,
    "deployment_min_macro_pr_auc": 0.18,
    "deployment_max_dominant_frac": 0.88,
    "deployment_min_predicted_classes": 3,
    "conformal_min_singleton_rate": 0.05,
    "conformal_min_class_singleton_rate": 0.02,
    "conformal_max_avg_set_size": 2.6,
    "conformal_max_full_set_rate": 0.70,
    "conformal_min_coverage_ratio": 0.80,
    "conformal_tuning_enabled": True,
    "conformal_tuning_alpha_grid": [0.08, 0.10, 0.12],
    "conformal_tuning_lambda_grid": [0.0, 0.01, 0.03, 0.06, 0.10],
    "conformal_tuning_kreg_grid": [0, 1, 2, 3],
    "conformal_tuning_method_grid": ["aps", "raps"],
    "conformal_mondrian_min_class_rows": 24,
    "conformal_degeneracy_probe_min": 8,
    "conformal_degeneracy_singleton_floor": 0.01,
    "conformal_degeneracy_full_set_floor": 0.85,
    "conformal_rescue_enabled": True,
    "conformal_rescue_min_confidence": 0.33,
    "conformal_rescue_min_margin": 0.07,
    "selection_non_hold_recall_target": 0.12,
    "selection_macro_pr_auc_target": 0.22,
    "selection_zero_actionable_recall_penalty": 0.30,
    "selection_low_non_hold_penalty_weight": 0.90,
    "selection_low_pr_auc_penalty_weight": 0.40,
    "selection_predicted_class_penalty": 0.20,
    "selection_dominant_frac_soft_cap": 0.80,
    "selection_dominant_frac_penalty_weight": 0.75,
    "selection_hold_dominance_soft_cap": 0.65,
    "selection_hold_dominance_penalty_weight": 1.10,
    "stack_directional_penalty_multiplier": 1.50,
    "stack_zero_actionable_extra_penalty": 0.20,
    "stack_low_non_hold_extra_penalty_weight": 0.60,
    "stack_hold_dominance_penalty_weight": 0.80,
    "static_backtest_sharpe_gap_warn": 1.50,
    "static_backtest_return_gap_warn": 1.00,
    "llm_generation_temperature": 0.0,

    # ── NEW 12: CPCV ──────────────────────────────────────────────────────────
    "cpcv_n_splits":      6,        # k folds
    "cpcv_n_test_splits": 2,        # t test-folds per combination

    # ── NEW 13: Regime-conditional ────────────────────────────────────────────
    "regime_model_enabled": True,
    "regime_n_states":    2,
    "regime_min_rows":    252,

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
    "market_ffill_limit":   5,      # bounded context forward-fill
    "market_min_aligned_rows": 63,  # minimum aligned rows before market features
    "market_min_coverage_ratio": 0.50,
    "feature_volatility_regime_enabled": True,
    "feature_market_residual_enabled": True,
    "feature_trend_quality_enabled": True,
    "feature_liquidity_enhanced_enabled": True,
    "feature_anchored_vwap_enabled": True,

    "output_dir":   "reports",
    "random_state": 42,
    "seed_everything": True,
    "torch_deterministic": True,
    "torch_deterministic_warn_only": False,
    "torch_cublas_workspace_config": ":4096:8",
    "structured_stage_errors": True,
    "stage_error_traceback": True,
    "config_version": DEFAULT_CONFIG_VERSION,
    "experiment_tracking_enabled": True,
    "explainability_enabled": True,
    "explainability_top_n": 10,
    "panel_mode_enabled": False,
    "panel_include_sector": True,
    "panel_include_market_cap": True,
    "tree_grid_parallel_backend": "threading" if sys.platform.startswith("win") else "processes",
    "tree_grid_max_workers": 4,
    "tree_grid_fallback_to_sequential": True,

    # ── NEW 16: Temporal Fusion Transformer (TFT) ─────────────────────────────
    "tft_hidden":         64,     # state size for GRN / VSN hidden layers
    "tft_lstm_layers":    2,      # LSTM layers in the local processing block
    "tft_attn_heads":     4,      # interpretable multi-head attention heads
    "tft_dropout":        0.25,
    "tft_epochs":         120,
    "tft_patience":       20,
    "tft_lr":             3e-4,
    "tide_enabled":       True,
    "tide_hidden":        96,
    "tide_layers":        3,
    "tide_dropout":       0.20,
    "tide_epochs":        90,
    "tide_patience":      18,
    "tide_lr":            3e-4,
    "patchtst_enabled":   True,
    "patchtst_patch_len": 8,
    "patchtst_stride":    4,
    "patchtst_d_model":   96,
    "patchtst_layers":    3,
    "patchtst_nhead":     4,
    "patchtst_dropout":   0.20,
    "patchtst_epochs":    100,
    "patchtst_patience":  20,
    "patchtst_lr":        3e-4,

    # ── Conformal Prediction (RAPS) ───────────────────────────────────
    "conformal_alpha":    0.10,   # coverage target = 1 - alpha = 90%
    "conformal_cal_pct":  0.25,   # fraction of test set used for calibration
    "conformal_lambda":   0.01,   # RAPS regularisation (penalises larger sets)
    "conformal_kreg":     1,      # RAPS: rank threshold before penalty applies
    "conformal_method":   "aps",
    "conformal_mondrian_enabled": True,
    "cross_conformal_benchmark": True,
    "conformal_on_final_signal": True,
    "conformal_min_cal_rows": 20,

    # Meta-labeler: if reliability confidence < this, don't force HOLD — fall through
    # BUG FIX: raised from 0.15 → 0.30.  At 0.15 the threshold was too permissive:
    # RKLB had meta_pass_conf=16.2% which barely exceeded 15%, causing a 16.7%-
    # accurate meta-stack to override a 70%-confident LightGBM SELL signal.
    "meta_low_conf_threshold": 0.30,
    "meta_stack_min_f1": 0.34,
    "meta_stack_max_f1_underperform": 0.03,

    # ── Focal Loss for DL models ─────────────────────────────────────
    "focal_loss_gamma":   2.0,    # focusing parameter (0 = standard CE)
    "use_focal_loss":     True,   # applies to all DL models incl. TFT
    "dl_selection_metric": "f1_macro",
    "dl_stack_min_f1":    0.35,
    "dl_collapse_hard_threshold": 0.85,
    "dl_collapse_soft_threshold": 0.75,
    "dl_collapse_soft_f1_margin": 0.02,
    "dl_collapse_min_classes": 2,

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
    "panel_adv_val_drop_features": False,
    "adv_val_confidence_penalty_enabled": True,
    "adv_val_confidence_penalty": 0.15,

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
    "run_diagnostics": False,
    "multi_seed_enabled": True,
    "multi_seed_seeds": [42, 43, 44],
    "multi_seed_confidence_penalty_enabled": True,
    "multi_seed_disagreement_penalty": 0.25,
    "calibration_bins": 5,
    "ensemble_weighting": "validation_f1",
    "ensemble_weight_power": 2.0,
    "ensemble_min_weight": 0.05,
}

TICKER = OUT_DIR = LOG_PATH = log = None
RUN_METADATA = None
_EXPERIMENT_RECORDED = False
TELEGRAM_PROGRESS_SESSION = None

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
    print("  Stock ML Analyzer  v12.0 (Full Stack)")
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


def _enabled_model_names(cfg: dict) -> list:
    names = []
    challenger_enabled = bool(cfg.get("challenger_models_enabled", False))
    active_tree = set(cfg.get("active_tree_models", []) or [])
    active_dl = set(cfg.get("active_dl_models", []) or [])
    active_stack = set(cfg.get("active_stack_models", []) or [])

    def _include(name: str, family: str, available: bool = True) -> bool:
        if not available:
            return False
        if challenger_enabled:
            return True
        active = {
            'tree': active_tree,
            'dl': active_dl,
            'stack': active_stack,
        }.get(family, set())
        return not active or name in active

    for name, available in (
        ("RandomForest", True),
        ("HistGradBoost", True),
        ("MiniROCKET", True),
        ("XGBoost", HAS_XGB),
        ("LightGBM", HAS_LGB),
        ("CatBoost", HAS_CB),
    ):
        if _include(name, 'tree', available):
            names.append(name)

    if _has_enabled_dl_models(cfg):
        for name, enabled in (
            ("BiLSTM-Attention", True),
            ("Transformer-Encoder", True),
            ("TFT", True),
            ("TiDE", cfg.get("tide_enabled", True)),
            ("PatchTST", cfg.get("patchtst_enabled", True)),
        ):
            if _include(name, 'dl', enabled):
                names.append(name)

    if cfg.get("regime_model_enabled", True):
        names.append("Regime-Conditional")
    if _include("Meta-Stack", "stack", True):
        names.append("Meta-Stack")
    return names


def _has_enabled_dl_models(cfg: dict) -> bool:
    if not HAS_TORCH:
        return False
    challenger_enabled = bool(cfg.get("challenger_models_enabled", False))
    active_dl = set(cfg.get("active_dl_models", []) or [])
    dl_candidates = (
        ("BiLSTM-Attention", True),
        ("Transformer-Encoder", True),
        ("TFT", True),
        ("TiDE", cfg.get("tide_enabled", True)),
        ("PatchTST", cfg.get("patchtst_enabled", True)),
    )
    if challenger_enabled:
        return any(enabled for _, enabled in dl_candidates)
    if not active_dl:
        return any(enabled for _, enabled in dl_candidates)
    return any(enabled and name in active_dl for name, enabled in dl_candidates)


def _model_family(name: str) -> str:
    name_u = str(name or '').upper()
    if name_u.startswith('META'):
        return 'stack_family'
    if name_u.startswith('REGIME'):
        return 'regime_family'
    if name_u in {'BILSTM-ATTENTION', 'TRANSFORMER-ENCODER', 'TFT', 'TIDE', 'PATCHTST'}:
        return 'dl_family'
    return 'tree_family'


def _is_model_enabled(name: str, cfg: dict) -> bool:
    enabled = set(_enabled_model_names(cfg))
    family = _model_family(name)
    if family == 'regime_family':
        return bool(cfg.get("regime_model_enabled", True))
    return name in enabled


def _record_experiment(status: str, summary: dict | None = None):
    global _EXPERIMENT_RECORDED
    if _EXPERIMENT_RECORDED or not CONFIG.get("experiment_tracking_enabled", True):
        return
    if RUN_METADATA is None:
        return
    try:
        append_experiment_record(
            Path(CONFIG.get("output_dir", "reports")),
            RUN_METADATA,
            status=status,
            summary=summary or {},
        )
        _EXPERIMENT_RECORDED = True
    except Exception as exc:
        if log is not None:
            log.warning(f"Experiment tracking failed ({exc})")


def log_section(title: str):
    log.info(""); log.info("="*60); log.info(f"  {title}"); log.info("="*60)


def elapsed(t0: float) -> str:
    s = time.time() - t0
    if s < 60:   return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def _ensure_cublas_workspace_config():
    workspace = str(CONFIG.get('torch_cublas_workspace_config', '') or '').strip()
    if not workspace:
        return None
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "").strip()
    if current:
        if current != workspace and log is not None:
            log.info(f"CUBLAS workspace already set: {current} (config requested {workspace})")
        return current
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    if log is not None:
        log.info(f"CUBLAS workspace set: {workspace}")
    return workspace


def _env_truthy(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _apply_runtime_overrides() -> dict:
    overrides = {}
    if _env_truthy("ANALYZER_FORCE_DIAGNOSTICS"):
        CONFIG["run_diagnostics"] = True
        overrides["run_diagnostics_forced"] = True
        overrides["run_context"] = os.environ.get("ANALYZER_RUN_CONTEXT", "external")
    else:
        overrides["run_diagnostics_forced"] = False
    return overrides


def _telegram_notifications_enabled_here() -> bool:
    return _env_truthy("TELEGRAM_ENABLED") and not _env_truthy("TELEGRAM_SUPPRESS_CHILD")


def _telegram_progress_parent_message_id() -> int | None:
    try:
        raw = str(os.environ.get("TELEGRAM_PROGRESS_MESSAGE_ID", "") or "").strip()
        return int(raw) if raw else None
    except Exception:
        return None


def _analyzer_progress_session(run_name: str):
    global TELEGRAM_PROGRESS_SESSION
    if TELEGRAM_PROGRESS_SESSION is not None:
        return TELEGRAM_PROGRESS_SESSION
    parent_message_id = _telegram_progress_parent_message_id()
    if parent_message_id is not None:
        TELEGRAM_PROGRESS_SESSION = create_progress_session(run_name, message_id=parent_message_id)
        return TELEGRAM_PROGRESS_SESSION
    if telegram_progress_enabled():
        TELEGRAM_PROGRESS_SESSION = create_progress_session(run_name)
        return TELEGRAM_PROGRESS_SESSION
    TELEGRAM_PROGRESS_SESSION = create_progress_session(run_name)
    return TELEGRAM_PROGRESS_SESSION


def _progress_env_text() -> dict:
    return {
        "run_name": str(os.environ.get("TELEGRAM_PROGRESS_RUN_NAME", TICKER or "Analyzer") or "Analyzer"),
        "phase_index": int(str(os.environ.get("TELEGRAM_PROGRESS_PHASE_INDEX", "1") or "1")),
        "phase_total": int(str(os.environ.get("TELEGRAM_PROGRESS_PHASE_TOTAL", "1") or "1")),
        "phase_label": str(os.environ.get("TELEGRAM_PROGRESS_PHASE_LABEL", "ML analyzer") or "ML analyzer"),
        "ticker": str(os.environ.get("TELEGRAM_PROGRESS_TICKER", TICKER or "") or (TICKER or "")),
        "ticker_index": str(os.environ.get("TELEGRAM_PROGRESS_TICKER_INDEX", "") or "").strip(),
        "ticker_total": str(os.environ.get("TELEGRAM_PROGRESS_TICKER_TOTAL", "") or "").strip(),
    }


def _render_progress_message(stage_label: str | None = None,
                             stage_pos: int | None = None,
                             stage_total: int | None = None,
                             status: str | None = None) -> str:
    ctx = _progress_env_text()
    lines = [f"{ctx['run_name']}"]
    lines.append(f"Phase {ctx['phase_index']}/{ctx['phase_total']}: {ctx['phase_label']}")
    ticker = ctx.get("ticker")
    if ticker:
        ticker_line = f"Ticker: {ticker}"
        if ctx.get("ticker_index") and ctx.get("ticker_total"):
            ticker_line += f" ({ctx['ticker_index']}/{ctx['ticker_total']})"
        lines.append(ticker_line)
    if stage_label:
        stage_line = "Analyzer stage"
        if stage_pos is not None and stage_total is not None:
            stage_line += f" {stage_pos}/{stage_total}"
        stage_line += f": {stage_label}"
        lines.append(stage_line)
    if status:
        lines.append(f"Status: {status}")
    return "\n".join(lines)


class ProgressTracker:
    def __init__(self, run_name: str, session, enabled_stages: list[tuple[str, str]]):
        self.run_name = str(run_name or "Analyzer")
        self.session = session
        self.stage_order = [key for key, _ in enabled_stages]
        self.labels = {key: label for key, label in enabled_stages}
        self.total_steps = len(self.stage_order)
        self.done = set()
        self.current = None

    def start(self):
        if self.session is None or not getattr(self.session, "enabled", False):
            return False
        text = _render_progress_message(
            stage_label="Queued",
            stage_pos=0,
            stage_total=self.total_steps,
            status="Starting",
        )
        if getattr(self.session, "message_id", None) is not None:
            return self.session.update(text)
        return self.session.start(text)

    def _stage_position(self, stage_key: str) -> int:
        try:
            return self.stage_order.index(stage_key) + 1
        except ValueError:
            return max(1, len(self.done) + 1)

    def stage_started(self, stage_key: str, label: str | None = None):
        if stage_key not in self.labels:
            return
        self.current = stage_key
        text = _render_progress_message(
            stage_label=str(label or self.labels.get(stage_key, stage_key)),
            stage_pos=self._stage_position(stage_key),
            stage_total=self.total_steps,
            status="Running",
        )
        if self.session is not None:
            self.session.update(text)

    def stage_done(self, stage_key: str):
        if stage_key not in self.labels:
            return
        self.done.add(stage_key)
        text = _render_progress_message(
            stage_label=str(self.labels.get(stage_key, stage_key)),
            stage_pos=self._stage_position(stage_key),
            stage_total=self.total_steps,
            status="Done",
        )
        if self.session is not None:
            self.session.update(text)

    def stage_failed(self, stage_key: str, err):
        if stage_key not in self.labels:
            return
        text = _render_progress_message(
            stage_label=str(self.labels.get(stage_key, stage_key)),
            stage_pos=self._stage_position(stage_key),
            stage_total=self.total_steps,
            status=f"Failed ({type(err).__name__}: {err})",
        )
        if self.session is not None:
            self.session.update(text)


def _resolve_success_pngs(ticker: str) -> list[Path]:
    if OUT_DIR is None:
        return []
    out = []
    for candidate in [
        OUT_DIR / f"{ticker}_analysis.png",
        OUT_DIR / f"{ticker}_selection_diagnostics.png",
        OUT_DIR / f"{ticker}_dl_models.png",
        OUT_DIR / f"{ticker}_montecarlo.png",
        OUT_DIR / f"{ticker}_montecarlo_volatility.png",
        OUT_DIR / f"{ticker}_montecarlo_diagnostics.png",
        OUT_DIR / f"{ticker}_dcf_surface_3d.png",
    ]:
        if candidate.exists():
            out.append(candidate)
    return out


def _tail_log_text(path: Path | None, max_chars: int = 3200) -> str:
    if path is None or not Path(path).exists():
        return ""
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')[-max_chars:]
    except Exception:
        return ""


def _notify_analyzer_outcome(ticker: str):
    if not _telegram_notifications_enabled_here():
        return False
    session = TELEGRAM_PROGRESS_SESSION if TELEGRAM_PROGRESS_SESSION is not None else _analyzer_progress_session(ticker)
    signal_path = (OUT_DIR / f"{ticker}_signal.json") if OUT_DIR is not None else None
    if signal_path is None or not signal_path.exists():
        if session is not None and getattr(session, "enabled", False):
            session.mark_failed("Failed")
        return notify_failure(ticker, _tail_log_text(LOG_PATH), log_path=str(LOG_PATH) if LOG_PATH else None)
    try:
        payload = json.loads(signal_path.read_text(encoding='utf-8'))
    except Exception:
        if session is not None and getattr(session, "enabled", False):
            session.mark_failed("Failed")
        return notify_failure(ticker, _tail_log_text(LOG_PATH), log_path=str(LOG_PATH) if LOG_PATH else None)
    pipeline_ok = str(payload.get('pipeline_status', 'FAILED')).upper() == 'OK'
    sig = (payload.get('signal', {}) or {})
    if pipeline_ok:
        photo_paths = _resolve_success_pngs(ticker)
        if not photo_paths:
            return False
        if session is not None and getattr(session, "enabled", False):
            session.mark_finalizing("Finalizing result...")
            send_chat_action("upload_photo")
            time.sleep(telegram_result_delay_seconds())
        caption = (
            f"{ticker} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"{sig.get('model_used', 'model')} | {sig.get('signal', 'N/A')} | "
            f"{float(sig.get('confidence', 0.0) or 0.0):.2f}"
        )
        return notify_success([str(path) for path in photo_paths], caption=caption)
    if session is not None and getattr(session, "enabled", False):
        session.mark_failed("Failed")
    tail = _tail_log_text(LOG_PATH)
    if not tail:
        errors = payload.get('pipeline_errors', []) or []
        tail = json.dumps(errors[-2:], indent=2, cls=_NumpyEncoder)
    return notify_failure(ticker, tail, log_path=str(LOG_PATH) if LOG_PATH else None)


def seed_everything(seed: int, deterministic_torch: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        if deterministic_torch:
            _ensure_cublas_workspace_config()
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        if deterministic_torch:
            try:
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass
            try:
                warn_only = bool(CONFIG.get('torch_deterministic_warn_only', False))
                if hasattr(torch, "use_deterministic_algorithms"):
                    try:
                        torch.use_deterministic_algorithms(True, warn_only=warn_only)
                    except TypeError:
                        torch.use_deterministic_algorithms(True)
            except Exception as exc:
                if log is not None:
                    log.warning(f"PyTorch deterministic setup failed ({exc})")
        else:
            try:
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                if hasattr(torch, "use_deterministic_algorithms"):
                    torch.use_deterministic_algorithms(False)
            except Exception:
                pass
    if log is not None:
        log.info(f"Random seed: {seed}  deterministic_torch={'ON' if deterministic_torch else 'OFF'}")


def _record_stage_error(errors: list, stage: str, exc: Exception, fatal: bool = False):
    tb = traceback.format_exc() if CONFIG.get('stage_error_traceback', True) else ""
    payload = {
        'stage': stage,
        'fatal': bool(fatal),
        'error_type': type(exc).__name__,
        'message': str(exc),
        'traceback': tb,
    }
    errors.append(payload)
    if log is not None:
        level = log.error if fatal else log.warning
        msg = f"{stage} failed: {type(exc).__name__}: {exc}"
        if tb:
            level(f"{msg}\n{tb}")
        else:
            level(msg)
    return payload


def _fixed_horizon_event_frame(index, horizon: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    n = len(idx)
    if n == 0:
        return pd.DataFrame(columns=['event_start', 'event_end'], index=idx)
    horizon = max(1, int(horizon))
    end_pos = np.minimum(np.arange(n) + horizon, n - 1)
    event_end = idx.take(end_pos)
    return pd.DataFrame({'event_start': idx, 'event_end': event_end}, index=idx)


def _attach_label_event_meta(y: pd.Series, event_meta: 'pd.DataFrame | None',
                             label_method: str) -> pd.Series:
    if event_meta is None:
        return y
    meta = event_meta.reindex(y.index)
    y.attrs['event_meta'] = meta
    y.attrs['label_method'] = label_method
    return y


def _slice_event_meta(event_meta: 'pd.DataFrame | None', index) -> 'pd.DataFrame | None':
    if event_meta is None:
        return None
    idx = pd.Index(index)
    if len(idx) == 0:
        return event_meta.iloc[0:0].copy()
    return event_meta.reindex(idx)


def _write_failure_signal_json(ticker: str, stage_errors: list,
                               partial: 'dict | None' = None) -> 'Path | None':
    if OUT_DIR is None:
        return None
    payload = {
        'ticker': ticker,
        'generated': datetime.datetime.now().isoformat(),
        'schema_version': SIGNAL_SCHEMA_VERSION,
        'signal': {
            'signal': 'HOLD',
            'label': 1,
            'confidence': 0.0,
            'probabilities': {'SELL': 0.0, 'HOLD': 1.0, 'BUY': 0.0},
            'model_used': 'PIPELINE-ERROR',
            'execution_status': 'ABSTAIN_MODEL_UNRELIABLE',
            'execution_gate': 'pipeline_error',
            'execution_gate_details': {'pipeline_errors': len(stage_errors)},
            'deployment_eligible': False,
            'eligibility_failures': ['pipeline_error'],
        },
        'pipeline_errors': stage_errors,
        'pipeline_status': 'FAILED',
    }
    if RUN_METADATA is not None:
        payload['run_metadata'] = complete_run_metadata(RUN_METADATA, status='FAILED')
    if partial:
        payload.update(partial)
    payload['artifact_invariants'] = validate_signal_artifact(payload)
    path = OUT_DIR / f"{ticker}_signal.json"
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, cls=_NumpyEncoder)
        if log is not None:
            log.info(f"Failure signal JSON saved → {path}")
        _record_experiment('FAILED', {
            'ticker': ticker,
            'pipeline_errors': len(stage_errors),
        })
    except Exception as e:
        if log is not None:
            log.error(f"Failed to write partial signal JSON: {e}")
    return path


def _derive_ensemble_weights(results: dict, model_names: list, cfg: dict) -> dict:
    strategy = cfg.get('ensemble_weighting', 'equal')
    min_weight = float(cfg.get('ensemble_min_weight', 0.05))
    power = float(cfg.get('ensemble_weight_power', 2.0))
    if strategy == 'equal':
        raw = {name: 1.0 for name in model_names}
    else:
        raw = {}
        for name in model_names:
            score = float(results.get(name, {}).get('f1', 0.0))
            raw[name] = max(min_weight, score ** power)
    total = sum(raw.values()) or 1.0
    weights = {name: float(raw[name] / total) for name in model_names}
    if log is not None:
        desc = ", ".join(f"{name}={weights[name]:.3f}" for name in model_names)
        log.info(f"  Ensemble weighting ({strategy}): {desc}")
    return weights


def _compute_calibration_diagnostics(proba: np.ndarray, y_true: np.ndarray,
                                     n_bins: int = 5) -> dict:
    proba = np.asarray(proba, dtype=np.float64)
    y_true = np.asarray(y_true)
    if proba.ndim != 2 or len(proba) == 0:
        raise ValueError("probability matrix is empty")

    n_classes = proba.shape[1]
    preds = proba.argmax(axis=1)
    max_conf = proba.max(axis=1)
    correct = (preds == y_true).astype(float)

    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    rows = []
    ece = 0.0
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (max_conf >= lo) & (max_conf <= hi)
        else:
            mask = (max_conf >= lo) & (max_conf < hi)
        if not mask.any():
            continue
        avg_conf = float(max_conf[mask].mean())
        hit_rate = float(correct[mask].mean())
        gap = avg_conf - hit_rate
        n_bin = int(mask.sum())
        ece += abs(gap) * n_bin / len(y_true)
        rows.append({
            'bin': f'{lo:.2f}-{hi:.2f}',
            'n': n_bin,
            'avg_conf': avg_conf,
            'hit_rate': hit_rate,
            'gap': gap,
            'overconfident': gap > 0.10,
        })

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=list(range(n_classes)), zero_division=0
    )
    labels = ['SELL', 'HOLD', 'BUY'][:n_classes]
    classwise = {
        labels[i]: {
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
        for i in range(n_classes)
    }

    try:
        pr_auc_macro = float(average_precision_score(one_hot, proba, average='macro'))
    except Exception:
        pr_auc_macro = None

    return {
        'brier_score': brier,
        'ece': float(ece),
        'reliability_bins': rows,
        'classwise': classwise,
        'pr_auc_macro': pr_auc_macro,
        'overconfident_bins': int(sum(1 for row in rows if row['overconfident'])),
        'well_calibrated': bool(all(not row['overconfident'] for row in rows)),
    }


def _prediction_summary(y_true, preds, n_classes: int = 3) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if len(y_true) == 0 or len(preds) == 0:
        return {
            'acc': 0.0,
            'f1': 0.0,
            'dominant_frac': 0.0,
            'pred_counts': {},
            'predicted_classes': 0,
        }
    counts = np.bincount(preds, minlength=max(int(n_classes), int(preds.max()) + 1))
    labels = ['SELL', 'HOLD', 'BUY']
    pred_counts = {
        labels[i] if i < len(labels) else str(i): int(counts[i])
        for i in range(len(counts))
    }
    return {
        'acc': float(accuracy_score(y_true, preds)),
        'f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'dominant_frac': float(counts.max() / (counts.sum() + 1e-10)),
        'pred_counts': pred_counts,
        'predicted_classes': int((counts > 0).sum()),
    }


def _clip_prob_matrix(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.clip(arr, 1e-9, 1.0)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return arr / row_sum


def _multiclass_nll(proba: np.ndarray, y_true: np.ndarray) -> float:
    proba = _clip_prob_matrix(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    labels = np.arange(proba.shape[1])
    return float(log_loss(y_true, proba, labels=labels))


def _prob_margin(proba_row: np.ndarray) -> float:
    row = np.sort(np.asarray(proba_row, dtype=np.float64))
    if len(row) < 2:
        return 0.0
    return float(row[-1] - row[-2])


class ProbabilityCalibrator:
    """Per-class multiclass probability calibrator with row renormalization."""

    def __init__(self, method: str = 'sigmoid'):
        self.method = str(method or 'sigmoid').lower()
        self.models_ = []
        self.n_classes_ = 0

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> 'ProbabilityCalibrator':
        proba = _clip_prob_matrix(proba)
        y_true = np.asarray(y_true, dtype=np.int64)
        self.n_classes_ = proba.shape[1]
        self.models_ = []
        for cls in range(self.n_classes_):
            target = (y_true == cls).astype(int)
            if target.min() == target.max():
                self.models_.append(('constant', float(target[0])))
                continue
            if self.method == 'isotonic':
                model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                model.fit(proba[:, cls], target)
                self.models_.append(('isotonic', model))
            else:
                model = LogisticRegression(max_iter=300, solver='lbfgs')
                model.fit(proba[:, [cls]], target)
                self.models_.append(('sigmoid', model))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        proba = _clip_prob_matrix(proba)
        if not self.models_:
            return proba
        cols = []
        for cls, (kind, model) in enumerate(self.models_):
            if kind == 'constant':
                cols.append(np.full(len(proba), float(model), dtype=np.float64))
            elif kind == 'isotonic':
                cols.append(np.asarray(model.predict(proba[:, cls]), dtype=np.float64))
            else:
                cols.append(np.asarray(model.predict_proba(proba[:, [cls]])[:, 1], dtype=np.float64))
        return _clip_prob_matrix(np.column_stack(cols))


def _build_calibration_report(pre_proba: np.ndarray,
                              post_proba: np.ndarray,
                              y_true: np.ndarray,
                              calibrator_type: str,
                              learned_temperature: float | None = None) -> dict:
    pre_proba = _clip_prob_matrix(pre_proba)
    post_proba = _clip_prob_matrix(post_proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    pre_diag = _compute_calibration_diagnostics(pre_proba, y_true, n_bins=CONFIG.get('calibration_bins', 5))
    post_diag = _compute_calibration_diagnostics(post_proba, y_true, n_bins=CONFIG.get('calibration_bins', 5))
    return {
        'calibrator_type': calibrator_type,
        'pre_calibration_nll': _multiclass_nll(pre_proba, y_true),
        'post_calibration_nll': _multiclass_nll(post_proba, y_true),
        'pre_calibration_ece': float(pre_diag.get('ece', 0.0)),
        'post_calibration_ece': float(post_diag.get('ece', 0.0)),
        'multiclass_brier': float(post_diag.get('brier_score', 0.0)),
        'learned_temperature': None if learned_temperature is None else float(learned_temperature),
        'calibration_split_size': int(len(y_true)),
        'reliability_bins': post_diag.get('reliability_bins', []),
        'pre_reliability_bins': pre_diag.get('reliability_bins', []),
        'classwise': post_diag.get('classwise', {}),
        'pr_auc_macro': post_diag.get('pr_auc_macro'),
        'overconfident_bins': int(post_diag.get('overconfident_bins', 0)),
        'ece': float(post_diag.get('ece', 0.0)),
        'brier_score': float(post_diag.get('brier_score', 0.0)),
        'calibration_improved': bool(
            (_multiclass_nll(post_proba, y_true) <= _multiclass_nll(pre_proba, y_true) + 1e-9)
            or (float(post_diag.get('ece', 0.0)) <= float(pre_diag.get('ece', 0.0)) + 1e-9)
        ),
    }


def _fit_probability_calibrator(proba: np.ndarray,
                                y_true: np.ndarray,
                                method: str = 'auto',
                                min_rows: int = 80) -> tuple:
    proba = _clip_prob_matrix(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    method = str(method or 'auto').lower()
    baseline_report = _build_calibration_report(proba, proba, y_true, calibrator_type='identity')
    class_counts = np.bincount(y_true, minlength=proba.shape[1]) if len(y_true) else np.zeros(proba.shape[1], dtype=int)
    min_class_rows = int(class_counts.min()) if len(class_counts) else 0
    isotonic_min_rows = int(CONFIG.get('tree_calibration_isotonic_min_rows', max(min_rows, 120)) or max(min_rows, 120))
    isotonic_min_class_rows = int(CONFIG.get('tree_calibration_isotonic_min_class_rows', 20) or 20)
    best_proba = proba
    best_report = None
    best_calibrator = None

    methods = ['sigmoid']
    if method == 'isotonic':
        methods = ['isotonic']
    elif method == 'auto':
        methods = ['sigmoid']
        if len(y_true) >= max(int(min_rows), isotonic_min_rows) and min_class_rows >= isotonic_min_class_rows:
            methods.append('isotonic')
        else:
            baseline_report['calibrator_fallback_reason'] = (
                'small_calibration_fold'
                if len(y_true) < max(int(min_rows), isotonic_min_rows)
                else 'imbalanced_calibration_fold'
            )

    for current in methods:
        try:
            calibrator = ProbabilityCalibrator(current).fit(proba, y_true)
            cal_proba = calibrator.transform(proba)
            report = _build_calibration_report(proba, cal_proba, y_true, calibrator_type=current)
            report['min_class_rows'] = int(min_class_rows)
            report['class_counts'] = [int(x) for x in class_counts.tolist()]
            report['calibrator_fallback_reason'] = baseline_report.get('calibrator_fallback_reason')
            if best_report is None:
                best_calibrator = calibrator
                best_proba = cal_proba
                best_report = report
            else:
                improved = (
                    report['post_calibration_nll'] < best_report['post_calibration_nll'] - 1e-9
                    or (
                        abs(report['post_calibration_nll'] - best_report['post_calibration_nll']) <= 1e-9
                        and report['post_calibration_ece'] < best_report['post_calibration_ece'] - 1e-9
                    )
                )
                if improved:
                    best_calibrator = calibrator
                    best_proba = cal_proba
                    best_report = report
        except Exception as exc:
            if log is not None:
                log.debug(f"Probability calibrator {current} skipped ({exc})")
    if best_report is None:
        best_report = baseline_report
    best_report.setdefault('min_class_rows', int(min_class_rows))
    best_report.setdefault('class_counts', [int(x) for x in class_counts.tolist()])
    best_report.setdefault('calibrator_fallback_reason', baseline_report.get('calibrator_fallback_reason'))
    return best_calibrator, best_proba, best_report


def _compute_classification_metrics(y_true: np.ndarray,
                                    preds: np.ndarray,
                                    n_classes: int = 3) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    labels = ['SELL', 'HOLD', 'BUY'][:int(max(1, n_classes))]
    if len(y_true) == 0 or len(preds) == 0:
        return {
            'classwise': {label: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0} for label in labels},
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
        }
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=list(range(len(labels))), zero_division=0
    )
    return {
        'classwise': {
            labels[i]: {
                'precision': float(prec[i]),
                'recall': float(rec[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            }
            for i in range(len(labels))
        },
        'macro_precision': float(np.mean(prec)) if len(prec) else 0.0,
        'macro_recall': float(np.mean(rec)) if len(rec) else 0.0,
        'macro_f1': float(np.mean(f1)) if len(f1) else 0.0,
    }


def _class_recall(classwise: dict, label: str) -> float:
    return float(((classwise or {}).get(label, {}) or {}).get('recall', 0.0) or 0.0)


def _build_class_edge_bins(proba: np.ndarray,
                           y_true: np.ndarray,
                           forward_returns: np.ndarray,
                           n_bins: int = 4) -> dict:
    proba = _clip_prob_matrix(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    rets = np.asarray(forward_returns, dtype=np.float64)
    if len(proba) == 0 or len(y_true) == 0 or len(rets) == 0:
        return {}
    labels = ['SELL', 'HOLD', 'BUY'][:proba.shape[1]]
    out = {}
    bins = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    for cls, label in enumerate(labels):
        cls_prob = proba[:, cls]
        if label == 'SELL':
            directional = -rets
        elif label == 'BUY':
            directional = rets
        else:
            directional = -np.abs(rets)
        rows = []
        for i in range(len(bins) - 1):
            lo = float(bins[i])
            hi = float(bins[i + 1])
            if i == len(bins) - 2:
                mask = (cls_prob >= lo) & (cls_prob <= hi)
            else:
                mask = (cls_prob >= lo) & (cls_prob < hi)
            if not np.any(mask):
                continue
            rows.append({
                'bin': f'{lo:.2f}-{hi:.2f}',
                'count': int(mask.sum()),
                'avg_prob': float(np.mean(cls_prob[mask])),
                'avg_realized_return': float(np.mean(rets[mask])),
                'avg_directional_edge': float(np.mean(directional[mask])),
                'hit_rate': float(np.mean(y_true[mask] == cls)),
            })
        out[label] = rows
    return out


def _estimate_class_edge_map(y_true: np.ndarray, forward_returns: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    rets = np.asarray(forward_returns, dtype=np.float64)
    edge_buy = float(np.nanmean(rets[y_true == 2])) if np.any(y_true == 2) else float(np.nanmean(np.maximum(rets, 0.0)))
    edge_sell = float(np.nanmean(-rets[y_true == 0])) if np.any(y_true == 0) else float(np.nanmean(np.maximum(-rets, 0.0)))
    return {
        'SELL': max(edge_sell, 0.0),
        'HOLD': 0.0,
        'BUY': max(edge_buy, 0.0),
    }


def _trade_signal_from_policy(proba_row: np.ndarray,
                              thresholds: dict,
                              edge_map: dict,
                              tx_cost: float,
                              uncertainty_ok: bool = True) -> tuple:
    proba_row = np.asarray(proba_row, dtype=np.float64)
    buy_p, hold_p, sell_p = float(proba_row[2]), float(proba_row[1]), float(proba_row[0])
    margin = _prob_margin(proba_row)
    buy_edge = buy_p * float(edge_map.get('BUY', 0.0))
    sell_edge = sell_p * float(edge_map.get('SELL', 0.0))
    min_edge = float(thresholds.get('cost_gate', tx_cost))
    min_margin = float(thresholds.get('margin_threshold', 0.0))
    if not uncertainty_ok:
        return 1, 'conformal_gate'
    if margin < min_margin:
        return 1, 'margin_gate'
    buy_ok = buy_p >= float(thresholds.get('buy_threshold', 0.50)) and buy_edge > min_edge
    sell_ok = sell_p >= float(thresholds.get('sell_threshold', 0.50)) and sell_edge > min_edge
    if buy_ok and sell_ok:
        return (2, None) if buy_edge >= sell_edge else (0, None)
    if buy_ok:
        return 2, None
    if sell_ok:
        return 0, None
    return 1, 'edge_gate'


def _strategy_metrics_from_predictions(pred_labels: np.ndarray,
                                       returns: np.ndarray,
                                       tx_cost: float) -> dict:
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    returns = np.asarray(returns, dtype=np.float64)
    signal = np.where(pred_labels == 2, 1.0, np.where(pred_labels == 0, -1.0, 0.0))
    pos_change = np.abs(np.diff(np.concatenate([[0.0], signal])))
    strat = np.nan_to_num(signal * returns - pos_change * tx_cost, nan=0.0)
    sharpe = float(strat.mean() / (strat.std() + 1e-10) * np.sqrt(252)) if len(strat) else 0.0
    n_trades = int(np.sum(pos_change > 0))
    return {
        'signal_series': signal,
        'strategy_returns': strat,
        'sharpe': sharpe,
        'n_trades': n_trades,
        'annualized_trade_rate': float(n_trades / max(len(strat), 1) * 252),
    }


def _conformal_summary_from_sets(pred_sets: list[list[int]],
                                 y_true: np.ndarray,
                                 n_classes: int,
                                 alpha: float,
                                 method: str,
                                 qhat: float | None = None,
                                 n_cal: int = 0) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(pred_sets) == 0 or len(y_true) == 0:
        return {}
    sizes = [len(ps) for ps in pred_sets]
    covered = [int(int(y_true[i]) in pred_sets[i]) for i in range(min(len(pred_sets), len(y_true)))]
    class_conditional_singleton = {}
    class_conditional_coverage = {}
    for cls in range(int(n_classes)):
        mask = (y_true == cls)
        if not np.any(mask):
            continue
        cls_sets = [pred_sets[i] for i in np.where(mask)[0]]
        class_conditional_singleton[ConformalPredictor.LABELS.get(cls, str(cls))] = float(np.mean([len(ps) == 1 for ps in cls_sets]))
        class_conditional_coverage[ConformalPredictor.LABELS.get(cls, str(cls))] = float(np.mean([cls in ps for ps in cls_sets]))
    avg_size = float(np.mean(sizes))
    singleton_rate = float(np.mean([s == 1 for s in sizes]))
    size_distribution = {
        str(size): int(count)
        for size, count in sorted(Counter(int(s) for s in sizes).items(), key=lambda item: item[0])
    }
    set_pattern_distribution = Counter()
    for pred_set in pred_sets:
        labels = [ConformalPredictor.LABELS.get(int(cls), str(cls)) for cls in sorted(pred_set)]
        set_pattern_distribution['/'.join(labels)] += 1
    full_set_rate = float(np.mean([s >= int(n_classes) for s in sizes]))
    class_singleton_values = [float(val) for val in class_conditional_singleton.values()]
    degenerate_full_set_mode = bool(
        singleton_rate <= float(CONFIG.get('conformal_degeneracy_singleton_floor', 0.01) or 0.01)
        and full_set_rate >= float(CONFIG.get('conformal_degeneracy_full_set_floor', 0.99) or 0.99)
        and avg_size >= float(n_classes)
    )
    return {
        'coverage': float(np.mean(covered)),
        'target_coverage': 1.0 - float(alpha),
        'avg_set_size': avg_size,
        'singleton_rate': singleton_rate,
        'full_set_rate': full_set_rate,
        'class_conditional_singleton_rate': class_conditional_singleton,
        'min_class_singleton_rate': float(min(class_singleton_values)) if class_singleton_values else 0.0,
        'mean_class_singleton_rate': float(np.mean(class_singleton_values)) if class_singleton_values else 0.0,
        'class_conditional_coverage': class_conditional_coverage,
        'set_size_distribution': size_distribution,
        'set_pattern_distribution': {
            key: int(val) for key, val in sorted(set_pattern_distribution.items(), key=lambda item: item[0])
        },
        'unique_set_patterns': int(len(set_pattern_distribution)),
        'degenerate_full_set_mode': degenerate_full_set_mode,
        'abstain_rate': float(np.mean([s > 1 for s in sizes])),
        'sharpness': _compute_conformal_sharpness(avg_size, singleton_rate, n_classes),
        'n_cal': int(n_cal),
        'qhat': None if qhat is None else float(qhat),
        'method': str(method or 'aps').lower(),
    }


def _conformal_nonconformity_summary(proba: np.ndarray,
                                     y_true: np.ndarray,
                                     *,
                                     method: str,
                                     lam: float,
                                     kreg: int) -> dict:
    proba = _clip_prob_matrix(np.asarray(proba, dtype=np.float64))
    y_true = np.asarray(y_true, dtype=np.int64)
    if proba.ndim != 2 or len(proba) == 0 or len(y_true) == 0:
        return {}
    cp = ConformalPredictor(alpha=0.10, lam=lam, kreg=kreg, method=method)
    scores = np.asarray(cp._nonconformity(proba, y_true), dtype=np.float64)
    summary = {
        'n_samples': int(len(scores)),
        'min': float(np.min(scores)) if len(scores) else 0.0,
        'median': float(np.median(scores)) if len(scores) else 0.0,
        'mean': float(np.mean(scores)) if len(scores) else 0.0,
        'max': float(np.max(scores)) if len(scores) else 0.0,
    }
    by_class = {}
    for cls in range(int(proba.shape[1])):
        mask = y_true == cls
        if not np.any(mask):
            continue
        cls_scores = scores[mask]
        label = ConformalPredictor.LABELS.get(cls, str(cls))
        by_class[label] = {
            'n_samples': int(len(cls_scores)),
            'min': float(np.min(cls_scores)),
            'median': float(np.median(cls_scores)),
            'mean': float(np.mean(cls_scores)),
            'max': float(np.max(cls_scores)),
        }
    summary['by_true_class'] = by_class
    return summary


def _conformal_mondrian_supported(proba: np.ndarray,
                                  min_rows: int | None = None) -> tuple[bool, dict]:
    proba = _clip_prob_matrix(proba)
    min_rows = int(min_rows or CONFIG.get('conformal_mondrian_min_class_rows', 24) or 24)
    counts = Counter(int(x) for x in np.argmax(proba, axis=1).tolist())
    label_counts = {
        ConformalPredictor.LABELS.get(cls, str(cls)): int(counts.get(cls, 0))
        for cls in range(proba.shape[1])
    }
    supported = all(counts.get(cls, 0) >= min_rows for cls in range(proba.shape[1]))
    return bool(supported), label_counts


def _fit_execution_conformal(proba: np.ndarray,
                             y_true: np.ndarray,
                             alpha: float,
                             lam: float,
                             kreg: int,
                             method: str = 'aps',
                             mondrian: bool = False) -> dict | None:
    proba = _clip_prob_matrix(np.asarray(proba, dtype=np.float64))
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(y_true) < int(CONFIG.get('conformal_min_cal_rows', 20)):
        return None

    global_cp = ConformalPredictor(alpha=alpha, lam=lam, kreg=kreg, method=method)
    global_cp.calibrate(proba, y_true)

    use_mondrian = False
    class_predictors = {}
    class_support = {}
    mondrian_supported, predicted_class_counts = _conformal_mondrian_supported(
        proba,
        min_rows=int(CONFIG.get('conformal_mondrian_min_class_rows', 24) or 24),
    )
    fallback_reason = None
    if mondrian:
        if not mondrian_supported:
            fallback_reason = 'insufficient_mondrian_class_rows'
        else:
            use_mondrian = True
            pred_cal = np.argmax(proba, axis=1)
            for cls in range(proba.shape[1]):
                cls_mask = pred_cal == cls
                class_support[ConformalPredictor.LABELS.get(cls, str(cls))] = int(np.sum(cls_mask))
                class_predictors[cls] = ConformalPredictor(
                    alpha=alpha,
                    lam=lam,
                    kreg=kreg,
                    method=method,
                ).calibrate(proba[cls_mask], y_true[cls_mask])
    return {
        'global_predictor': global_cp,
        'class_predictors': class_predictors,
        'alpha': float(alpha),
        'lam': float(lam),
        'kreg': int(kreg),
        'method': str(method or 'aps').lower(),
        'mondrian': bool(use_mondrian),
        'mondrian_requested': bool(mondrian),
        'mondrian_supported': bool(mondrian_supported),
        'mondrian_fallback_reason': fallback_reason,
        'predicted_class_counts': predicted_class_counts,
        'class_support': class_support,
    }


def _execution_conformal_predict_set(execution_cp: dict | None,
                                     proba: np.ndarray) -> tuple[list[int], dict]:
    if execution_cp is None:
        return [], {}
    row = np.asarray(proba, dtype=np.float64)
    label_idx = int(np.argmax(row))
    cp = execution_cp.get('global_predictor')
    scope = 'global'
    if execution_cp.get('mondrian'):
        cp = execution_cp.get('class_predictors', {}).get(label_idx, cp)
        if cp is not execution_cp.get('global_predictor'):
            scope = 'mondrian'
    if cp is None:
        return [], {}
    pred_set = cp.predict_set(row)
    annotation = {
        'prediction_set': [ConformalPredictor.LABELS[c] for c in pred_set],
        'set_size': len(pred_set),
        'is_conformal_singleton': len(pred_set) == 1,
        'conformal_coverage_level': round(1.0 - float(execution_cp.get('alpha', 0.10)), 2),
        'qhat': round(float(cp.qhat), 4),
        'conformal_method': str(execution_cp.get('method', 'aps') or 'aps'),
        'conformal_scope': scope,
    }
    return pred_set, annotation


def _conformal_coverage_threshold(alpha: float | None = None) -> float:
    alpha = float(alpha if alpha is not None else CONFIG.get('conformal_alpha', 0.10) or 0.10)
    return (1.0 - alpha) * float(CONFIG.get('conformal_min_coverage_ratio', 0.80) or 0.80)


def _is_degenerate_conformal_stats(stats: dict | None, n_classes: int = 3) -> bool:
    stats = dict(stats or {})
    singleton_rate = float(stats.get('singleton_rate', 0.0) or 0.0)
    full_set_rate = float(stats.get('full_set_rate', 0.0) or 0.0)
    avg_set_size = float(stats.get('avg_set_size', 0.0) or 0.0)
    return bool(
        singleton_rate <= float(CONFIG.get('conformal_degeneracy_singleton_floor', 0.01) or 0.01)
        and full_set_rate >= float(CONFIG.get('conformal_degeneracy_full_set_floor', 0.99) or 0.99)
        and avg_set_size >= float(n_classes)
    )


def _directional_selection_penalty(candidate: dict) -> float:
    cfg = CONFIG
    evaluation = dict(candidate.get('evaluation', {}) or {})
    classwise = dict(evaluation.get('classwise', {}) or {})
    pred_counts = dict(evaluation.get('pred_counts', {}) or {})
    buy_recall = _class_recall(classwise, 'BUY')
    sell_recall = _class_recall(classwise, 'SELL')
    non_hold_min = float(evaluation.get('non_hold_recall_min', min(buy_recall, sell_recall)) or min(buy_recall, sell_recall))
    macro_pr_auc = float(evaluation.get('macro_pr_auc', 0.0) or 0.0)
    dominant_frac = float(evaluation.get('dominant_frac', 0.0) or 0.0)
    predicted_classes = int(evaluation.get('predicted_classes', 0) or 0)
    pred_total = max(int(sum(int(v or 0) for v in pred_counts.values())), 1)
    hold_frac = float(pred_counts.get('HOLD', 0) or 0.0) / float(pred_total)
    family = str(candidate.get('family', '') or '')

    penalty = 0.0
    zero_penalty = float(cfg.get('selection_zero_actionable_recall_penalty', 0.30) or 0.30)
    if buy_recall <= 0.0:
        penalty += zero_penalty
    if sell_recall <= 0.0:
        penalty += zero_penalty
    penalty += max(0.0, float(cfg.get('selection_non_hold_recall_target', 0.12) or 0.12) - non_hold_min) * float(
        cfg.get('selection_low_non_hold_penalty_weight', 0.90) or 0.90
    )
    penalty += max(0.0, float(cfg.get('selection_macro_pr_auc_target', 0.22) or 0.22) - macro_pr_auc) * float(
        cfg.get('selection_low_pr_auc_penalty_weight', 0.40) or 0.40
    )
    penalty += max(0, int(cfg.get('deployment_min_predicted_classes', 3) or 3) - predicted_classes) * float(
        cfg.get('selection_predicted_class_penalty', 0.20) or 0.20
    )
    dominant_cap = float(cfg.get('selection_dominant_frac_soft_cap', 0.80) or 0.80)
    if dominant_frac > dominant_cap:
        penalty += (dominant_frac - dominant_cap) * float(cfg.get('selection_dominant_frac_penalty_weight', 0.75) or 0.75)
    hold_cap = float(cfg.get('selection_hold_dominance_soft_cap', 0.65) or 0.65)
    if hold_frac > hold_cap:
        penalty += (hold_frac - hold_cap) * float(cfg.get('selection_hold_dominance_penalty_weight', 1.10) or 1.10)
    if family == 'stack_family':
        if buy_recall <= 0.0 or sell_recall <= 0.0:
            penalty += float(cfg.get('stack_zero_actionable_extra_penalty', 0.20) or 0.20)
        penalty += max(0.0, float(cfg.get('selection_non_hold_recall_target', 0.12) or 0.12) - non_hold_min) * float(
            cfg.get('stack_low_non_hold_extra_penalty_weight', 0.60) or 0.60
        )
        if hold_frac > hold_cap:
            penalty += (hold_frac - hold_cap) * float(cfg.get('stack_hold_dominance_penalty_weight', 0.80) or 0.80)
        penalty *= float(cfg.get('stack_directional_penalty_multiplier', 1.50) or 1.50)
    return float(max(penalty, 0.0))


def _selection_rank_score(candidate: dict) -> float:
    selection = dict(candidate.get('selection', {}) or {})
    robust_score = float(selection.get('robust_score', float('-inf')) or float('-inf'))
    if not np.isfinite(robust_score):
        return float('-inf')
    return float(robust_score - _directional_selection_penalty(candidate))


def _raw_candidate_robust_score(candidate: dict) -> float:
    selection = dict(candidate.get('selection', {}) or {})
    robust_score = float(selection.get('robust_score', float('-inf')) or float('-inf'))
    return robust_score if np.isfinite(robust_score) else float('-inf')


def _candidate_sort_key(candidate: dict) -> tuple:
    evaluation = dict(candidate.get('evaluation', {}) or {})
    conformal = dict(candidate.get('conformal', {}) or {})
    return (
        _selection_rank_score(candidate),
        float(evaluation.get('macro_pr_auc', 0.0) or 0.0),
        float(evaluation.get('non_hold_recall_min', 0.0) or 0.0),
        float(conformal.get('sharpness', 0.0) or 0.0),
        str(candidate.get('name', '') or ''),
    )


def _candidate_public_summary(candidate: dict | None) -> dict | None:
    if not candidate:
        return None
    selection = dict(candidate.get('selection', {}) or {})
    evaluation = dict(candidate.get('evaluation', {}) or {})
    conformal = dict(candidate.get('conformal', {}) or {})
    return {
        'model': candidate.get('name'),
        'family': candidate.get('family'),
        'robust_score': float(selection.get('robust_score', 0.0) or 0.0),
        'selection_rank_score': _selection_rank_score(candidate),
        'directional_penalty': _directional_selection_penalty(candidate),
        'deployment_eligible': bool(candidate.get('deployment_eligible', False)),
        'eligibility_failures': list(candidate.get('eligibility_failures', []) or []),
        'macro_pr_auc': float(evaluation.get('macro_pr_auc', 0.0) or 0.0),
        'non_hold_recall_min': float(evaluation.get('non_hold_recall_min', 0.0) or 0.0),
        'conformal_sharpness': float(conformal.get('sharpness', 0.0) or 0.0),
        'degenerate_execution_conformal': bool(conformal.get('degenerate_execution_conformal', False)),
    }


def _candidate_family_counts(candidates: list[dict]) -> dict:
    counts = Counter(str(cand.get('family', '') or 'unknown') for cand in candidates)
    return {key: int(val) for key, val in sorted(counts.items(), key=lambda item: item[0])}


def _candidate_rejection_counts(candidates: list[dict], directional_only: bool = False) -> dict:
    directional_reasons = {
        'zero_buy_recall',
        'zero_sell_recall',
        'min_non_hold_recall',
        'predicted_class_coverage',
        'dominant_class_excess',
        'dl_zero_actionable_recall',
        'dl_predicted_classes',
        'dl_dominant_class',
    }
    counts = Counter()
    for cand in candidates:
        for reason in list(cand.get('eligibility_failures', []) or []):
            if directional_only and reason not in directional_reasons:
                continue
            counts[str(reason)] += 1
    return {key: int(val) for key, val in sorted(counts.items(), key=lambda item: item[0])}


def _assess_conformal_usability(conf_stats: dict | None, n_classes: int = 3) -> dict:
    stats = dict(conf_stats or {})
    failures = []
    if not stats:
        failures.append('missing_conformal_stats')
    coverage = float(stats.get('coverage', 0.0) or 0.0)
    target = float(stats.get('target_coverage', 1.0 - float(CONFIG.get('conformal_alpha', 0.10) or 0.10)) or 0.90)
    singleton_rate = float(stats.get('singleton_rate', 0.0) or 0.0)
    avg_set_size = float(stats.get('avg_set_size', float(n_classes)) or float(n_classes))
    full_set_rate = float(stats.get('full_set_rate', 1.0 if avg_set_size >= float(n_classes) else 0.0) or 0.0)
    class_singleton = {
        str(label): float(val)
        for label, val in (stats.get('class_conditional_singleton_rate', {}) or {}).items()
    }
    min_class_singleton = float(stats.get('min_class_singleton_rate', min(class_singleton.values()) if class_singleton else 0.0) or 0.0)
    degenerate = bool(stats.get('degenerate_execution_conformal', False)) or _is_degenerate_conformal_stats(stats, n_classes=n_classes)
    if coverage < target * float(CONFIG.get('conformal_min_coverage_ratio', 0.80)):
        failures.append('coverage_below_target')
    if singleton_rate < float(CONFIG.get('conformal_min_singleton_rate', 0.05)):
        failures.append('singleton_rate_too_low')
    if avg_set_size > float(CONFIG.get('conformal_max_avg_set_size', 2.6)):
        failures.append('set_size_too_wide')
    if full_set_rate > float(CONFIG.get('conformal_max_full_set_rate', 0.70)):
        failures.append('full_set_rate_too_high')
    min_class_floor = float(CONFIG.get('conformal_min_class_singleton_rate', 0.02) or 0.02)
    if class_singleton and min_class_singleton < min_class_floor:
        failures.append('class_singleton_rate_too_low')
    if degenerate:
        failures.append('degenerate_conformal_sets')
    stats['usable_for_execution'] = not failures
    stats['usability_failures'] = failures
    stats['coverage_floor'] = float(target * float(CONFIG.get('conformal_min_coverage_ratio', 0.80) or 0.80))
    stats['full_set_rate'] = full_set_rate
    stats['min_class_singleton_rate'] = min_class_singleton
    stats['degenerate_execution_conformal'] = bool(degenerate)
    stats['conformal_block_reason_counts'] = {
        reason: int(count) for reason, count in Counter(failures).items()
    }
    return stats


def _assess_candidate_deployment(candidate: dict) -> tuple[bool, list]:
    cfg = CONFIG
    evaluation = dict(candidate.get('evaluation', {}) or {})
    classwise = (evaluation.get('classwise') or {}).copy()
    calibration = dict(candidate.get('calibration', {}) or {})
    conformal = dict(candidate.get('conformal', {}) or {})
    buy_recall = _class_recall(classwise, 'BUY')
    sell_recall = _class_recall(classwise, 'SELL')
    non_hold_min = min(buy_recall, sell_recall)
    macro_f1 = float(evaluation.get('macro_f1', 0.0) or 0.0)
    macro_pr_auc = float(evaluation.get('macro_pr_auc', 0.0) or 0.0)
    dominant_frac = float(evaluation.get('dominant_frac', 0.0) or 0.0)
    predicted_classes = int(evaluation.get('predicted_classes', 0) or 0)
    failures = []

    if buy_recall <= 0.0:
        failures.append('zero_buy_recall')
    if sell_recall <= 0.0:
        failures.append('zero_sell_recall')
    if non_hold_min < float(cfg.get('deployment_min_non_hold_recall', 0.05)):
        failures.append('min_non_hold_recall')
    if macro_f1 < float(cfg.get('deployment_min_macro_f1', 0.18)):
        failures.append('macro_f1_floor')
    if macro_pr_auc < float(cfg.get('deployment_min_macro_pr_auc', 0.18)):
        failures.append('macro_pr_auc_floor')
    if dominant_frac > float(cfg.get('deployment_max_dominant_frac', 0.88)):
        failures.append('dominant_class_excess')
    if predicted_classes < int(cfg.get('deployment_min_predicted_classes', 3)):
        failures.append('predicted_class_coverage')
    if calibration and not bool(calibration.get('calibration_improved', True)):
        failures.append('calibration_worsened')
    if conformal and not bool(conformal.get('usable_for_execution', True)):
        failures.append('conformal_unusable')
    if conformal and float(conformal.get('singleton_rate', 0.0) or 0.0) < float(cfg.get('conformal_min_singleton_rate', 0.05)):
        failures.append('conformal_singleton_floor')

    if candidate.get('family') == 'dl_family':
        if bool(candidate.get('is_collapsed', False)):
            failures.append('dl_collapse')
        if predicted_classes < 2:
            failures.append('dl_predicted_classes')
        if buy_recall <= 0.0 or sell_recall <= 0.0:
            failures.append('dl_zero_actionable_recall')
        if dominant_frac > float(cfg.get('dl_collapse_soft_threshold', 0.75)):
            failures.append('dl_dominant_class')

    unique_failures = []
    for reason in failures:
        if reason not in unique_failures:
            unique_failures.append(reason)
    evaluation['non_hold_recall_min'] = float(non_hold_min)
    evaluation['macro_pr_auc'] = float(macro_pr_auc)
    candidate['evaluation'] = evaluation
    conformal_failures = list(conformal.get('usability_failures', []) or [])
    conformal_only_failures = {
        'conformal_unusable',
        'conformal_singleton_floor',
    }
    blocked_otherwise_healthy = bool(conformal_failures) and not any(
        reason not in conformal_only_failures for reason in unique_failures
    )
    conformal['blocked_otherwise_healthy_model'] = blocked_otherwise_healthy
    conformal['conformal_block_reason_counts'] = {
        reason: int(count)
        for reason, count in Counter(conformal_failures).items()
    }
    candidate['conformal'] = conformal
    candidate['deployment_eligible'] = not unique_failures
    candidate['eligibility_failures'] = unique_failures
    return candidate['deployment_eligible'], unique_failures


def _candidate_conformal_rescue_ready(candidate: dict, cfg: dict) -> bool:
    failures = set(str(x) for x in (candidate.get('eligibility_failures', []) or []))
    if not failures:
        return False
    conformal_only_failures = {
        'conformal_unusable',
        'conformal_singleton_floor',
        'degenerate_conformal_sets',
    }
    if not failures.issubset(conformal_only_failures):
        return False
    latest = dict(candidate.get('latest_signal', {}) or {})
    confidence = float(latest.get('confidence', 0.0) or 0.0)
    margin = float(latest.get('probability_margin', 0.0) or 0.0)
    return bool(
        confidence >= float(cfg.get('conformal_rescue_min_confidence', 0.45) or 0.45)
        and margin >= float(cfg.get('conformal_rescue_min_margin', 0.10) or 0.10)
    )


def _build_backtest_audit(strategy_returns: float,
                          buyhold_returns: float,
                          n_bars: int,
                          n_trades: int,
                          active_signal,
                          conformal_mult,
                          sizing_method: str,
                          activity_summary: dict | None = None,
                          strategy_return_source: str = 'static_in_sample_backtest') -> dict:
    summary = dict(activity_summary or {})
    active_arr = np.asarray(pd.Series(active_signal).fillna(0.0), dtype=np.float64)
    conformal_arr = np.asarray(pd.Series(conformal_mult).fillna(1.0), dtype=np.float64)
    n_bars = int(max(int(n_bars or 0), 0))
    n_trades = int(max(int(summary.get('n_trade_events', n_trades) or 0), 0))
    active_bar_count = int(max(int(summary.get('active_bar_count', np.sum(np.abs(active_arr) > 1e-9)) or 0), 0))
    avg_abs_position = float(summary.get('avg_abs_position', np.mean(np.abs(active_arr)) if len(active_arr) else 0.0))
    max_abs_position = float(summary.get('max_abs_position', np.max(np.abs(active_arr)) if len(active_arr) else 0.0))
    trade_density = float(n_trades / max(n_bars, 1))
    active_bar_density = float(active_bar_count / max(n_bars, 1))
    conformal_zeroed_bar_rate = float(np.mean(conformal_arr == 0.0)) if len(conformal_arr) else 0.0
    buyhold_denominator = max(abs(float(buyhold_returns or 0.0)), 0.10)
    strat_vs_bh_ratio = float(float(strategy_returns or 0.0) / buyhold_denominator)
    sanity_flags = []
    if abs(float(strategy_returns or 0.0)) > 10.0:
        sanity_flags.append('absolute_return_gt_1000pct')
    if abs(strat_vs_bh_ratio) > 10.0:
        sanity_flags.append('strategy_vs_buyhold_ratio_gt_10x')
    if trade_density > 0.50:
        sanity_flags.append('trade_density_gt_50pct')
    return {
        'strategy_return_source': str(strategy_return_source or 'static_in_sample_backtest'),
        'sizing_method': str(sizing_method or 'N/A'),
        'n_bars': n_bars,
        'n_trades': n_trades,
        'active_bar_count': active_bar_count,
        'trade_density': trade_density,
        'active_bar_density': active_bar_density,
        'avg_abs_position': avg_abs_position,
        'max_abs_position': max_abs_position,
        'active_bar_win_rate': float(summary.get('active_bar_win_rate', 0.0) or 0.0),
        'conformal_zeroed_bar_rate': conformal_zeroed_bar_rate,
        'strategy_return_vs_buyhold_ratio': strat_vs_bh_ratio,
        'strategy_return_pct': float(float(strategy_returns or 0.0) * 100.0),
        'buyhold_return_pct': float(float(buyhold_returns or 0.0) * 100.0),
        'sanity_flags': sanity_flags,
        'sanity_status': 'warning' if sanity_flags else 'ok',
    }


def _candidate_holdout_backtest(name: str,
                                family: str,
                                pred_labels: np.ndarray,
                                returns: np.ndarray,
                                eval_index=None,
                                conformal_set_sizes=None,
                                strategy_return_source: str = 'selected_candidate_holdout_backtest') -> dict:
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    returns = np.asarray(returns, dtype=np.float64)
    n = min(len(pred_labels), len(returns))
    if n <= 0:
        return {
            'strat_return': 0.0,
            'strat_annual': 0.0,
            'bh_return': 0.0,
            'strat_sharpe': 0.0,
            'strat_sortino': 0.0,
            'strat_maxdd': 0.0,
            'calmar': 0.0,
            'win_rate': 0.0,
            'n_trades': 0,
            'active_bar_count': 0,
            'active_bar_win_rate': 0.0,
            'var95': 0.0,
            'cvar95': 0.0,
            'kelly_sharpe': 0.0,
            'binary_sharpe': 0.0,
            'sizing_method': 'DecisionPolicyHoldout',
            'backtest_mode': 'selected_candidate_holdout',
            'strategy_model': name,
            'strategy_family': family,
            'audit': {
                'strategy_return_source': strategy_return_source,
                'sizing_method': 'DecisionPolicyHoldout',
                'sanity_status': 'ok',
                'sanity_flags': [],
            },
            'strat_cum': pd.Series([1.0]),
            'bh_cum': pd.Series([1.0]),
            'strat_cum_binary': pd.Series([1.0]),
            'strat_cum_kelly': pd.Series([1.0]),
        }

    idx = _coerce_index_like(eval_index)
    if len(idx) != n:
        idx = pd.RangeIndex(n)
    ret_sr = pd.Series(returns[:n], index=idx).fillna(0.0)
    signal = pd.Series(
        np.where(pred_labels[:n] == 2, 1.0, np.where(pred_labels[:n] == 0, -1.0, 0.0)),
        index=idx,
    )
    pos_change = signal.diff().abs().fillna(signal.abs())
    tx_cost = CONFIG.get('transaction_cost_bps', 10) / 10000.0
    strat = (signal * ret_sr - pos_change * tx_cost).fillna(0.0)
    sc = (1.0 + strat).cumprod()
    bhc = (1.0 + ret_sr).cumprod()
    downside = strat[strat < 0.0]
    dd = (sc - sc.cummax()) / (sc.cummax() + 1e-10)
    sharpe = float(strat.mean() / (strat.std() + 1e-10) * np.sqrt(252)) if len(strat) else 0.0
    sortino_d = float(downside.std() * np.sqrt(252)) if len(downside) else 0.0
    sortino = float(strat.mean() * np.sqrt(252) / (sortino_d + 1e-10)) if len(strat) else 0.0
    strat_return = float(sc.iloc[-1] - 1.0) if len(sc) else 0.0
    bh_return = float(bhc.iloc[-1] - 1.0) if len(bhc) else 0.0
    strat_annual = float(sc.iloc[-1] ** (252 / max(len(sc), 1)) - 1.0) if len(sc) > 1 else strat_return
    active_returns = strat[strat != 0.0].dropna()
    var95 = float(np.percentile(active_returns, 5)) if len(active_returns) else 0.0
    cvar95 = float(active_returns[active_returns <= var95].mean()) if (len(active_returns) and (active_returns <= var95).any()) else var95
    activity = _summarize_backtest_activity(signal, strat)
    set_sizes = np.asarray(conformal_set_sizes if conformal_set_sizes is not None else np.ones(n), dtype=np.int64)
    if len(set_sizes) != n:
        set_sizes = np.ones(n, dtype=np.int64)
    conformal_mult = pd.Series(
        [1.0 if s <= 1 else 0.5 if s == 2 else 0.0 for s in set_sizes],
        index=idx,
    )
    audit = _build_backtest_audit(
        strategy_returns=strat_return,
        buyhold_returns=bh_return,
        n_bars=n,
        n_trades=int(activity.get('n_trade_events', 0)),
        active_signal=signal,
        conformal_mult=conformal_mult,
        sizing_method='DecisionPolicyHoldout',
        activity_summary=activity,
        strategy_return_source=strategy_return_source,
    )
    return {
        'strat_return': strat_return,
        'strat_annual': strat_annual,
        'bh_return': bh_return,
        'strat_sharpe': sharpe,
        'strat_sortino': sortino,
        'strat_maxdd': float(dd.min()) if len(dd) else 0.0,
        'calmar': float(strat_annual / (abs(float(dd.min()) if len(dd) else 0.0) + 1e-10)),
        'win_rate': float(activity.get('active_bar_win_rate', 0.0) or 0.0),
        'n_trades': int(activity.get('n_trade_events', 0)),
        'active_bar_count': int(activity.get('active_bar_count', 0)),
        'active_bar_win_rate': float(activity.get('active_bar_win_rate', 0.0) or 0.0),
        'var95': var95,
        'cvar95': cvar95,
        'kelly_sharpe': sharpe,
        'binary_sharpe': sharpe,
        'sizing_method': 'DecisionPolicyHoldout',
        'backtest_mode': 'selected_candidate_holdout',
        'strategy_model': name,
        'strategy_family': family,
        'audit': audit,
        'strat_cum': sc,
        'bh_cum': bhc,
        'strat_cum_binary': sc,
        'strat_cum_kelly': sc,
    }


def _augment_backtest_audit_with_oos_context(audit: dict | None,
                                             bt: dict | None,
                                             wf_bt: dict | None,
                                             cpcv_res: dict | None) -> dict:
    out = dict(audit or {})
    bt = dict(bt or {})
    wf_bt = dict(wf_bt or {})
    cpcv_res = dict(cpcv_res or {})
    static_sharpe = float(bt.get('strat_sharpe', 0.0) or 0.0)
    static_return = float(bt.get('strat_return', 0.0) or 0.0)
    wf_sharpe = float(wf_bt.get('wf_sharpe', 0.0) or 0.0) if wf_bt else None
    wf_return = float(wf_bt.get('wf_return', 0.0) or 0.0) if wf_bt else None
    cpcv_p5 = float(cpcv_res.get('sharpe_p5', 0.0) or 0.0) if cpcv_res else None
    out['static_sharpe'] = static_sharpe
    out['wf_sharpe'] = wf_sharpe
    out['wf_return_pct'] = None if wf_return is None else float(wf_return * 100.0)
    out['cpcv_sharpe_p5'] = cpcv_p5
    out['static_vs_wf_sharpe_gap'] = None if wf_sharpe is None else float(static_sharpe - wf_sharpe)
    out['static_vs_cpcv_p5_gap'] = None if cpcv_p5 is None else float(static_sharpe - cpcv_p5)
    out['static_return_vs_wf_return_gap_pct'] = None if wf_return is None else float((static_return - wf_return) * 100.0)
    extra_flags = list(out.get('sanity_flags', []) or [])
    sharpe_gap_floor = float(CONFIG.get('static_backtest_sharpe_gap_warn', 1.50) or 1.50)
    return_gap_floor = float(CONFIG.get('static_backtest_return_gap_warn', 1.00) or 1.00)
    if wf_sharpe is not None and (static_sharpe - wf_sharpe) > sharpe_gap_floor:
        extra_flags.append('static_vs_wf_sharpe_gap')
    if wf_return is not None and static_return > 0.0 and wf_return <= 0.0:
        extra_flags.append('static_positive_vs_wf_nonpositive')
    elif wf_return is not None and (static_return - wf_return) > return_gap_floor:
        extra_flags.append('static_vs_wf_return_gap')
    if cpcv_p5 is not None and static_sharpe > 0.0 and cpcv_p5 < 0.0:
        extra_flags.append('static_positive_vs_cpcv_negative_tail')
    unique_flags = []
    for flag in extra_flags:
        if flag not in unique_flags:
            unique_flags.append(flag)
    out['sanity_flags'] = unique_flags
    out['sanity_status'] = 'warning' if unique_flags else 'ok'
    return out


def _summarize_backtest_activity(active_signal,
                                 strat_returns,
                                 eps: float = 1e-9) -> dict:
    active_arr = np.asarray(pd.Series(active_signal).fillna(0.0), dtype=np.float64)
    strat_arr = np.asarray(pd.Series(strat_returns).fillna(0.0), dtype=np.float64)
    active_mask = np.abs(active_arr) > float(eps)
    prev_arr = np.concatenate([[0.0], active_arr[:-1]]) if len(active_arr) else np.array([], dtype=np.float64)
    trade_event_mask = np.abs(active_arr - prev_arr) > float(eps)
    active_bar_count = int(np.sum(active_mask))
    positive_active_bars = int(np.sum((strat_arr > 0.0) & active_mask))
    return {
        'active_bar_count': active_bar_count,
        'positive_active_bars': positive_active_bars,
        'active_bar_win_rate': float(positive_active_bars / max(active_bar_count, 1)),
        'n_trade_events': int(np.sum(trade_event_mask)),
        'avg_abs_position': float(np.mean(np.abs(active_arr))) if len(active_arr) else 0.0,
        'max_abs_position': float(np.max(np.abs(active_arr))) if len(active_arr) else 0.0,
    }


def _build_execution_state(final_signal: dict,
                           selected_candidate: dict | None,
                           router_summary: dict | None = None) -> dict:
    signal = dict(final_signal or {})
    candidate = dict(selected_candidate or {})
    router = dict(router_summary or {})
    conformal = dict(candidate.get('conformal', {}) or {})
    decision = dict(candidate.get('decision_policy', {}) or {})
    execution_gate = 'actionable'
    execution_status = 'ACTIONABLE'
    eligibility_failures = list(candidate.get('eligibility_failures', signal.get('eligibility_failures', [])) or [])
    raw_abstain_reason = str(signal.get('abstain_reason', '') or '')
    router_status = str(router.get('router_status', '') or '')
    routing_actionable = bool(router.get('routing_actionable', False))
    deployment_eligible = bool(candidate.get('deployment_eligible', signal.get('deployment_eligible', not eligibility_failures)))
    conformal_usable = bool(conformal.get('usable_for_execution', not conformal))
    canonical_abstain_reason = None

    if (
        not deployment_eligible
        or str(signal.get('selection_status', '') or '') == 'reference_only_no_deployable_candidate'
        or router_status == 'fallback' and str(router.get('fallback_reason', '') or '') == 'no_eligible_family'
    ):
        execution_status = 'ABSTAIN_MODEL_UNRELIABLE'
        execution_gate = 'model_guard'
        canonical_abstain_reason = 'model_guard'
    elif raw_abstain_reason in {'conformal_gate', 'margin_gate'} or (conformal and not conformal_usable):
        execution_status = 'ABSTAIN_UNCERTAIN'
        execution_gate = 'conformal_gate' if raw_abstain_reason == 'conformal_gate' or (conformal and not conformal_usable) else 'margin_gate'
        canonical_abstain_reason = execution_gate
    elif raw_abstain_reason == 'edge_gate':
        execution_status = 'ABSTAIN_NO_EDGE'
        execution_gate = 'edge_gate'
        canonical_abstain_reason = 'edge_gate'
    elif str(signal.get('signal', 'HOLD')).upper() == 'HOLD':
        execution_status = 'HOLD_NEUTRAL'
        execution_gate = 'neutral_hold'
        canonical_abstain_reason = 'neutral_hold'

    details = {
        'abstain_reason': canonical_abstain_reason,
        'raw_abstain_reason': raw_abstain_reason or None,
        'probability_margin': float(signal.get('probability_margin', 0.0) or 0.0),
        'margin_threshold': float(decision.get('margin_threshold', 0.0) or 0.0),
        'cost_gate': float(decision.get('cost_gate', 0.0) or 0.0),
        'edge_map': decision.get('edge_map', {}) or {},
        'edge_minus_cost': signal.get('edge_minus_cost', {}) or {},
        'conformal_usable': conformal_usable,
        'conformal_failures': list(conformal.get('usability_failures', []) or []),
        'prediction_set': signal.get('prediction_set', []) or [],
        'set_size': int(signal.get('set_size', 0) or 0),
        'router_status': router_status or None,
        'routing_actionable': routing_actionable,
        'router_fallback_reason': router.get('fallback_reason'),
    }
    return {
        'abstain_reason': canonical_abstain_reason,
        'execution_status': execution_status,
        'execution_gate': execution_gate,
        'execution_gate_details': details,
        'deployment_eligible': deployment_eligible,
        'eligibility_failures': eligibility_failures,
    }


def _block_sharpe_stats(strategy_returns: np.ndarray, block_size: int = 21) -> dict:
    arr = np.asarray(strategy_returns, dtype=np.float64)
    block_size = max(5, int(block_size or 21))
    sharpes = []
    for start in range(0, len(arr), block_size):
        block = arr[start:start + block_size]
        if len(block) < 5:
            continue
        sharpes.append(float(block.mean() / (block.std() + 1e-10) * np.sqrt(252)))
    if not sharpes:
        return {'median_wf_sharpe': 0.0, 'positive_wf_share': 0.0, 'wf_block_sharpes': []}
    sharpes = np.asarray(sharpes, dtype=np.float64)
    return {
        'median_wf_sharpe': float(np.median(sharpes)),
        'positive_wf_share': float(np.mean(sharpes > 0)),
        'wf_block_sharpes': [float(x) for x in sharpes.tolist()],
    }


def _compute_conformal_sharpness(avg_set_size: float,
                                 singleton_rate: float,
                                 n_classes: int = 3) -> float:
    denom = max(1.0, float(n_classes - 1))
    set_term = 1.0 - float(np.clip((float(avg_set_size) - 1.0) / denom, 0.0, 1.0))
    return float(0.5 * float(singleton_rate) + 0.5 * set_term)


def _compute_robust_score(components: dict) -> tuple[float, dict]:
    ece_cap = float(CONFIG.get('robust_score_ece_cap', 0.25) or 0.25)
    normalized_ece = float(np.clip(float(components.get('ece', 1.0) or 1.0) / max(ece_cap, 1e-9), 0.0, 1.0))
    turnover_penalty = float(np.clip(float(components.get('annualized_trade_rate', 0.0) or 0.0), 0.0, 1.0))
    score_components = {
        'median_wf_sharpe': float(components.get('median_wf_sharpe', 0.0) or 0.0),
        'positive_wf_share': float(components.get('positive_wf_share', 0.0) or 0.0),
        'cpcv_p5_sharpe': float(components.get('cpcv_p5_sharpe', 0.0) or 0.0),
        'normalized_ece': normalized_ece,
        'conformal_sharpness': float(components.get('conformal_sharpness', 0.0) or 0.0),
        'turnover_penalty': turnover_penalty,
    }
    robust_score = (
        0.35 * score_components['median_wf_sharpe']
        + 0.20 * score_components['positive_wf_share']
        + 0.20 * score_components['cpcv_p5_sharpe']
        + 0.10 * (1.0 - score_components['normalized_ece'])
        + 0.10 * score_components['conformal_sharpness']
        - 0.05 * score_components['turnover_penalty']
    )
    return float(robust_score), score_components


def _tune_probability_thresholds(proba: np.ndarray,
                                 y_true: np.ndarray,
                                 forward_returns: np.ndarray,
                                 tx_cost: float) -> dict:
    proba = _clip_prob_matrix(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    rets = np.asarray(forward_returns, dtype=np.float64)
    if len(y_true) < int(CONFIG.get('threshold_tuning_min_rows', 40)):
        return {
            'buy_threshold': 0.50,
            'sell_threshold': 0.50,
            'margin_threshold': float(CONFIG.get('decision_margin_threshold', 0.08)),
            'cost_gate': float(tx_cost * CONFIG.get('decision_edge_cost_multiplier', 1.0)),
            'objective': CONFIG.get('threshold_objective', 'wf_sharpe_net_cost'),
        }

    edge_map = _estimate_class_edge_map(y_true, rets)
    best = None
    for buy_thr in CONFIG.get('threshold_buy_grid', [0.5]):
        for sell_thr in CONFIG.get('threshold_sell_grid', [0.5]):
            for margin_thr in CONFIG.get('threshold_margin_grid', [0.08]):
                thresholds = {
                    'buy_threshold': float(buy_thr),
                    'sell_threshold': float(sell_thr),
                    'margin_threshold': float(margin_thr),
                    'cost_gate': float(tx_cost * CONFIG.get('decision_edge_cost_multiplier', 1.0)),
                    'objective': CONFIG.get('threshold_objective', 'wf_sharpe_net_cost'),
                }
                preds = np.array([
                    _trade_signal_from_policy(row, thresholds, edge_map, tx_cost)[0]
                    for row in proba
                ], dtype=np.int64)
                metrics = _strategy_metrics_from_predictions(preds, rets, tx_cost)
                score = float(metrics['sharpe'])
                if best is None or score > best[0]:
                    best = (score, thresholds)
    return best[1] if best is not None else {
        'buy_threshold': 0.50,
        'sell_threshold': 0.50,
        'margin_threshold': float(CONFIG.get('decision_margin_threshold', 0.08)),
        'cost_gate': float(tx_cost * CONFIG.get('decision_edge_cost_multiplier', 1.0)),
        'objective': CONFIG.get('threshold_objective', 'wf_sharpe_net_cost'),
    }


def _build_threshold_regime_diagnostics(proba: np.ndarray,
                                        y_true: np.ndarray,
                                        forward_returns: np.ndarray,
                                        thresholds: dict,
                                        edge_map: dict,
                                        tx_cost: float,
                                        index_like=None,
                                        close: pd.Series | None = None) -> dict:
    proba = _clip_prob_matrix(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    rets = np.asarray(forward_returns, dtype=np.float64)
    idx = _coerce_index_like(index_like)
    labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    diagnostics = {
        'class_probability_return_bins': _build_class_edge_bins(proba, y_true, rets),
        'class_regime_returns': {},
        'threshold_utility_by_regime': {},
    }
    if close is None or len(idx) == 0:
        return diagnostics

    close_aligned = close.reindex(idx).ffill()
    if close_aligned.isna().all():
        return diagnostics
    ret_series = close.pct_change()
    realized_vol = (ret_series.rolling(21, min_periods=10).std() * np.sqrt(252)).reindex(idx).ffill()
    valid_vol = realized_vol.dropna()
    if valid_vol.empty:
        bucket_series = pd.Series('mid_vol', index=idx, dtype=object)
        q1 = q2 = None
    else:
        q1 = float(valid_vol.quantile(0.33))
        q2 = float(valid_vol.quantile(0.66))
        bucket_series = pd.Series(index=idx, dtype=object)
        bucket_series.loc[realized_vol <= q1] = 'low_vol'
        bucket_series.loc[(realized_vol > q1) & (realized_vol <= q2)] = 'mid_vol'
        bucket_series.loc[realized_vol > q2] = 'high_vol'
        bucket_series = bucket_series.fillna('mid_vol')

    preds = np.array([
        _trade_signal_from_policy(row, thresholds, edge_map, tx_cost)[0]
        for row in proba
    ], dtype=np.int64)

    regime_buckets = {}
    utility_by_bucket = {}
    for bucket in ('low_vol', 'mid_vol', 'high_vol'):
        mask = (bucket_series.values == bucket)
        if not np.any(mask):
            continue
        bucket_pred = preds[mask]
        bucket_ret = rets[mask]
        bucket_y = y_true[mask]
        metrics = _strategy_metrics_from_predictions(bucket_pred, bucket_ret, tx_cost)
        utility_by_bucket[bucket] = {
            'n': int(np.sum(mask)),
            'sharpe': float(metrics.get('sharpe', 0.0) or 0.0),
            'n_trades': int(metrics.get('n_trades', 0) or 0),
            'annualized_trade_rate': float(metrics.get('annualized_trade_rate', 0.0) or 0.0),
            'mean_return': float(np.mean(bucket_ret)) if len(bucket_ret) else 0.0,
        }
        class_rows = {}
        for cls, cls_name in labels.items():
            class_mask = bucket_y == cls
            if not np.any(class_mask):
                continue
            class_rows[cls_name] = {
                'n': int(np.sum(class_mask)),
                'mean_forward_return': float(np.mean(bucket_ret[class_mask])),
                'median_forward_return': float(np.median(bucket_ret[class_mask])),
            }
        regime_buckets[bucket] = class_rows

    diagnostics['class_regime_returns'] = regime_buckets
    diagnostics['threshold_utility_by_regime'] = utility_by_bucket
    diagnostics['regime_bucket_thresholds'] = {
        'low_vol_max': q1,
        'mid_vol_max': q2,
    }
    return diagnostics


def _tune_conformal_policy(proba: np.ndarray,
                           y_true: np.ndarray,
                           n_classes: int = 3) -> dict:
    proba = _clip_prob_matrix(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    base_method = str(CONFIG.get('conformal_method', 'aps') or 'aps').lower()
    base_alpha = float(CONFIG.get('conformal_alpha', 0.10) or 0.10)
    base_lam = float(CONFIG.get('conformal_lambda', 0.01) or 0.01)
    base_kreg = int(CONFIG.get('conformal_kreg', 1) or 1)
    mondrian_enabled = bool(CONFIG.get('conformal_mondrian_enabled', True))
    mondrian_supported, predicted_class_counts = _conformal_mondrian_supported(proba)
    method_grid = [str(x).lower() for x in (CONFIG.get('conformal_tuning_method_grid', []) or [base_method])]
    if base_method not in method_grid:
        method_grid.insert(0, base_method)
    method_grid = [m for m in dict.fromkeys(method_grid) if m in {'aps', 'raps'}] or [base_method]

    alpha_grid = [float(x) for x in (CONFIG.get('conformal_tuning_alpha_grid', []) or [])]
    lam_grid = [float(x) for x in (CONFIG.get('conformal_tuning_lambda_grid', []) or [])]
    kreg_grid = [int(x) for x in (CONFIG.get('conformal_tuning_kreg_grid', []) or [])]
    if base_alpha not in alpha_grid:
        alpha_grid.append(base_alpha)
    if base_lam not in lam_grid:
        lam_grid.append(base_lam)
    if base_kreg not in kreg_grid:
        kreg_grid.append(base_kreg)

    candidates = []
    for method in method_grid:
        lam_candidates = sorted(set(lam_grid)) if method == 'raps' else [0.0]
        kreg_candidates = sorted(set(kreg_grid)) if method == 'raps' else [0]
        for alpha in sorted(set(alpha_grid)):
            for lam in lam_candidates:
                for kreg in kreg_candidates:
                    candidates.append({
                        'alpha': alpha,
                        'lam': lam,
                        'kreg': kreg,
                        'method': method,
                        'mondrian': False,
                    })
                    if mondrian_enabled and mondrian_supported:
                        candidates.append({
                            'alpha': alpha,
                            'lam': lam,
                            'kreg': kreg,
                            'method': method,
                            'mondrian': True,
                        })

    evaluated = []
    coverage_floor = float(CONFIG.get('conformal_min_coverage_ratio', 0.80) or 0.80)
    degeneracy_probe_min = int(CONFIG.get('conformal_degeneracy_probe_min', 8) or 8)
    degenerate_probe_count = 0
    tuning_stopped_early = False
    degeneracy_reason = None
    for params in candidates:
        stats = _cross_conformal_stats(
            proba,
            y_true,
            alpha=params['alpha'],
            lam=params['lam'],
            kreg=params['kreg'],
            method=params['method'],
            mondrian=bool(params['mondrian']),
        )
        if not stats:
            continue
        stats['degenerate_execution_conformal'] = _is_degenerate_conformal_stats(stats, n_classes=n_classes)
        stats = _assess_conformal_usability(stats, n_classes=n_classes)
        stats['selected_alpha'] = float(params['alpha'])
        stats['selected_lambda'] = float(params['lam'])
        stats['selected_kreg'] = int(params['kreg'])
        stats['mondrian_requested'] = bool(params['mondrian'])
        stats['mondrian_supported'] = bool(mondrian_supported)
        stats['predicted_class_counts'] = predicted_class_counts
        stats['coverage_floor_pass'] = bool(
            float(stats.get('coverage', 0.0) or 0.0)
            >= float(stats.get('target_coverage', 1.0 - params['alpha'])) * coverage_floor
        )
        evaluated.append((params, stats))
        if bool(stats.get('degenerate_execution_conformal', False)):
            degenerate_probe_count += 1
        if len(evaluated) >= degeneracy_probe_min and degenerate_probe_count == len(evaluated):
            tuning_stopped_early = True
            degeneracy_reason = 'all_tuned_candidates_full_set'
            break

    fallback_params = {
        'alpha': base_alpha,
        'lam': base_lam,
        'kreg': base_kreg,
        'method': base_method,
        'mondrian': False,
    }
    if not evaluated:
        return {
            'params': fallback_params,
            'stats': {},
            'evaluated': [],
            'mondrian_supported': bool(mondrian_supported),
        }

    valid = [(params, stats) for params, stats in evaluated if stats.get('coverage_floor_pass')]
    ranked = valid or evaluated
    if tuning_stopped_early and degeneracy_reason == 'all_tuned_candidates_full_set':
        best_params, best_stats = next(
            (
                (params, stats)
                for params, stats in ranked
                if float(params.get('alpha', -1.0)) == float(base_alpha)
                and float(params.get('lam', -1.0)) == float(base_lam)
                and int(params.get('kreg', -1)) == int(base_kreg)
                and str(params.get('method', '') or '') == str(base_method)
                and not bool(params.get('mondrian', False))
            ),
            ranked[0],
        )
    else:
        best_params, best_stats = max(
            ranked,
            key=lambda item: (
                float(item[1].get('singleton_rate', 0.0) or 0.0),
                -float(item[1].get('avg_set_size', float(n_classes)) or float(n_classes)),
                -float(item[1].get('full_set_rate', 1.0) or 1.0),
                float(item[1].get('mean_class_singleton_rate', 0.0) or 0.0),
                float(item[1].get('coverage', 0.0) or 0.0),
                float(item[1].get('sharpness', 0.0) or 0.0),
                int(bool(item[0].get('mondrian', False))),
                int(str(item[0].get('method', 'aps') or 'aps').lower() == 'raps'),
            ),
        )
    best_stats = dict(best_stats)
    best_stats['tuned'] = True
    best_stats['tuning_candidates'] = int(len(evaluated))
    best_stats['degenerate_execution_conformal'] = bool(best_stats.get('degenerate_execution_conformal', False))
    best_stats['degeneracy_reason'] = degeneracy_reason
    best_stats['tuning_stopped_early'] = bool(tuning_stopped_early)
    best_stats['degeneracy_probe_count'] = int(degenerate_probe_count)
    best_stats['selected_tuning_params'] = {
        'alpha': float(best_params['alpha']),
        'lam': float(best_params['lam']),
        'kreg': int(best_params['kreg']),
        'method': str(best_params['method']),
        'mondrian': bool(best_params['mondrian']),
        'base_method': base_method,
    }
    best_stats['tuning_objective'] = 'singleton_rate_then_set_size_then_coverage'
    best_stats['tuning_candidates_evaluated'] = [
        {
            'alpha': float(params['alpha']),
            'lam': float(params['lam']),
            'kreg': int(params['kreg']),
            'method': str(params['method']),
            'mondrian': bool(params['mondrian']),
            'coverage': float(stats.get('coverage', 0.0) or 0.0),
            'avg_set_size': float(stats.get('avg_set_size', 0.0) or 0.0),
            'singleton_rate': float(stats.get('singleton_rate', 0.0) or 0.0),
            'full_set_rate': float(stats.get('full_set_rate', 0.0) or 0.0),
            'mean_class_singleton_rate': float(stats.get('mean_class_singleton_rate', 0.0) or 0.0),
            'sharpness': float(stats.get('sharpness', 0.0) or 0.0),
            'coverage_floor_pass': bool(stats.get('coverage_floor_pass')),
            'degenerate_execution_conformal': bool(stats.get('degenerate_execution_conformal', False)),
            'mondrian_fallback_reason': stats.get('mondrian_fallback_reason'),
        }
        for params, stats in evaluated
    ]
    best_stats['coverage_floor_pass'] = bool(best_stats.get('coverage_floor_pass'))
    return {
        'params': best_params,
        'stats': best_stats,
        'evaluated': list(best_stats.get('tuning_candidates_evaluated', [])),
        'mondrian_supported': bool(mondrian_supported),
    }


def _dl_instability_reason(summary: dict, cfg: dict) -> 'str | None':
    dom = float(summary.get('dominant_frac', 0.0) or 0.0)
    f1 = float(summary.get('f1', 0.0) or 0.0)
    predicted_classes = int(summary.get('predicted_classes', 0) or 0)
    hard_dom = float(cfg.get('dl_collapse_hard_threshold', 0.85))
    soft_dom = float(cfg.get('dl_collapse_soft_threshold', 0.75))
    min_f1 = float(cfg.get('dl_stack_min_f1', 0.35))
    soft_f1_margin = float(cfg.get('dl_collapse_soft_f1_margin', 0.02))
    min_classes = int(cfg.get('dl_collapse_min_classes', 2))

    if predicted_classes <= 1:
        return f"predicted_classes={predicted_classes}"
    if dom >= hard_dom:
        return f"dominant_frac={dom:.3f} >= hard_threshold={hard_dom:.2f}"
    if dom >= soft_dom and (f1 < (min_f1 + soft_f1_margin) or predicted_classes < min_classes):
        return (
            f"dominant_frac={dom:.3f} with weak_f1={f1:.4f} "
            f"or predicted_classes={predicted_classes}"
        )
    return None


def _dl_trainer_usable_for_stack(trainer, cfg: dict) -> bool:
    if getattr(trainer, 'is_collapsed', False):
        return False
    if getattr(trainer, 'te_proba', None) is None:
        return False
    min_f1 = float(cfg.get('dl_stack_min_f1', 0.35))
    eval_f1 = float(getattr(trainer, 'eval_f1_', 0.0) or 0.0)
    if eval_f1 < min_f1:
        return False
    summary = {
        'dominant_frac': float(getattr(trainer, 'eval_dominant_frac_', 0.0) or 0.0),
        'f1': eval_f1,
        'predicted_classes': int(getattr(trainer, 'eval_predicted_classes_', 3) or 3),
    }
    return _dl_instability_reason(summary, cfg) is None


def _tree_grid_worker_count(requested_jobs: int, cfg: dict) -> int:
    max_workers = cfg.get('tree_grid_max_workers')
    if max_workers is None:
        return max(1, int(requested_jobs))
    try:
        max_workers = int(max_workers)
    except Exception:
        max_workers = int(requested_jobs)
    return max(1, min(int(requested_jobs), max_workers))


def _tree_grid_parallel_prefer(cfg: dict) -> str:
    backend = str(cfg.get('tree_grid_parallel_backend', 'threading')).lower()
    return 'threads' if backend in ('thread', 'threads', 'threading') else 'processes'


def _run_grid_search_jobs(name: str, grid, eval_combo, parallel_jobs: int, cfg: dict):
    if parallel_jobs <= 1:
        best_score, best_params, best_kwargs = -1, None, {}
        bar = tqdm(grid, desc=f"  {name} grid", unit="combo", ncols=90,
                   file=sys.stdout, leave=True)
        for params in bar:
            score, p_, kw_ = eval_combo(params)
            if score > best_score:
                best_score, best_params, best_kwargs = score, p_, kw_
            bar.set_postfix(best=f"{best_score:.4f}", cur=f"{score:.4f}")
        bar.close()
        return best_score, best_params, best_kwargs

    prefer = _tree_grid_parallel_prefer(cfg)
    log.info(f"  Launching {len(grid)} parallel jobs on {parallel_jobs} workers "
             f"(backend={prefer})...")
    try:
        raw = Parallel(n_jobs=parallel_jobs, prefer=prefer)(
            delayed(eval_combo)(p)
            for p in tqdm(grid, desc=f"  {name} sched", unit="combo",
                          ncols=80, file=sys.stdout))
    except Exception as e:
        if not cfg.get('tree_grid_fallback_to_sequential', True):
            raise
        log.warning(f"  Parallel grid search failed for {name}: "
                    f"{type(e).__name__}: {e}. Retrying sequentially.")
        return _run_grid_search_jobs(name, grid, eval_combo, parallel_jobs=1, cfg=cfg)
    return max(raw, key=lambda x: x[0])


def _meta_stack_is_acceptable(meta_acc: float, meta_f1: float,
                              baseline_f1: float, cfg: dict) -> bool:
    min_f1 = float(cfg.get('meta_stack_min_f1', 0.34))
    max_under = float(cfg.get('meta_stack_max_f1_underperform', 0.03))
    if meta_acc < (1.0 / 3.0):
        return False
    if meta_f1 < min_f1:
        return False
    if meta_f1 + max_under < baseline_f1:
        return False
    return True


def _compute_seed_stability_for_tree(X, y, tree) -> 'dict | None':
    if not CONFIG.get('multi_seed_enabled', True):
        return None
    try:
        best_name = max(
            {k: v for k, v in tree.results.items() if k not in ('Ensemble', 'MiniROCKET')},
            key=lambda k: tree.results[k]['f1'])
        BestModel = type(tree.models[best_name])
        best_params = tree.models[best_name].get_params()
        diag_cfg = dict(getattr(tree, 'cfg', CONFIG))
        train_end = min(max(1, len(tree.X_tr)), len(X) - 1)
        seeds = list(diag_cfg.get('multi_seed_seeds', [42, 43, 44]))
        signals, f1s = [], []
        for seed in seeds:
            refit = _diagnostic_refit_fixed_model(
                X, y, BestModel, best_params, diag_cfg, int(seed),
                train_end=train_end, test_end=len(X))
            signals.append(refit['latest_signal']['signal'])
            f1s.append(refit['f1'])
        summary = _summarize_seed_stability(signals, f1s)
        summary['model_name'] = best_name
        summary['disagreement'] = 1.0 - summary.get('agreement', 0.0)
        return summary
    except Exception as e:
        if log is not None:
            log.warning(f"Multi-seed stability failed: {e}")
        return {'stable': False, 'error': str(e)}


def _apply_confidence_adjustments(final_signal: dict,
                                  adv_report: dict | None,
                                  seed_summary: dict | None,
                                  cfg: dict) -> dict:
    raw_conf = float(final_signal.get('confidence', 0.0))
    multiplier = 1.0
    adjustments = []

    if adv_report and adv_report.get('shift_detected') and cfg.get('adv_val_confidence_penalty_enabled', True):
        penalty = float(cfg.get('adv_val_confidence_penalty', 0.15))
        multiplier *= max(0.0, 1.0 - penalty)
        adjustments.append({'source': 'adversarial_validation', 'multiplier': max(0.0, 1.0 - penalty)})

    if seed_summary and 'agreement' in seed_summary and cfg.get('multi_seed_confidence_penalty_enabled', True):
        disagreement = float(seed_summary.get('disagreement', 1.0 - seed_summary.get('agreement', 0.0)))
        penalty = float(cfg.get('multi_seed_disagreement_penalty', 0.25))
        seed_mult = max(0.0, 1.0 - penalty * disagreement)
        multiplier *= seed_mult
        adjustments.append({'source': 'seed_disagreement', 'multiplier': seed_mult,
                            'disagreement': disagreement})

    final_signal['confidence_raw'] = raw_conf
    final_signal['confidence_multiplier'] = float(multiplier)
    final_signal['confidence_adjustments'] = adjustments
    final_signal['confidence'] = float(raw_conf * multiplier)
    return final_signal


def _normalize_context_series(data, name: str) -> 'pd.Series | None':
    """
    Convert any market-context payload into a deterministic daily close series.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if 'Close' not in df.columns:
            return None
        s = df['Close'].copy()
    elif isinstance(data, pd.Series):
        s = data.copy()
    else:
        s = pd.Series(data).copy()

    if s.empty:
        return None

    idx = pd.to_datetime(s.index, errors='coerce')
    valid_idx = ~pd.isna(idx)
    s = s[valid_idx]
    if s.empty:
        return None

    idx = pd.DatetimeIndex(idx[valid_idx])
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    idx = idx.normalize()

    s.index = idx
    s = pd.to_numeric(s, errors='coerce')
    s = s.groupby(level=0).last().sort_index().dropna()
    if s.empty:
        return None
    s.name = name
    return s


def _log_context_coverage(name: str, series: 'pd.Series | None'):
    if log is None:
        return
    if series is None or series.empty:
        log.warning(f"  {name}: no usable rows")
        return
    log.info(f"  {name}: {len(series)} rows  "
             f"({series.index[0].date()} to {series.index[-1].date()})")


def _align_market_series(series: pd.Series, target_index, ffill_limit: int = 5) -> pd.Series:
    """Align a market series to the stock index with bounded forward-fill."""
    name = getattr(series, 'name', None) or 'market'
    target = pd.DatetimeIndex(pd.to_datetime(target_index)).normalize()
    if len(target) == 0:
        return pd.Series(dtype=float, index=target, name=name)

    src = _normalize_context_series(series, name)
    if src is None or src.empty:
        return pd.Series(np.nan, index=target, name=name)

    aligned = src.reindex(target)
    if ffill_limit is not None and int(ffill_limit) > 0:
        aligned = aligned.ffill(limit=int(ffill_limit))
    aligned.name = name
    return aligned


def _required_market_rows(n_rows: int, cfg: dict) -> int:
    if n_rows <= 0:
        return 0
    min_rows = int(cfg.get('market_min_aligned_rows', 63))
    ratio    = float(cfg.get('market_min_coverage_ratio', 0.50))
    return min(n_rows, max(min_rows, int(np.ceil(n_rows * ratio))))


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
def _fetch_share_history(tkr) -> 'pd.Series | None':
    try:
        start = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * 5)).date()
        shares = tkr.get_shares_full(start=start)
        if shares is None:
            return None
        shares = pd.Series(shares).dropna()
        return shares if not shares.empty else None
    except Exception:
        return None


def _augment_snapshot_from_statements(snapshot: dict,
                                      income_stmt: 'pd.DataFrame | None',
                                      cashflow: 'pd.DataFrame | None',
                                      balance_sheet: 'pd.DataFrame | None') -> dict:
    out = dict(snapshot)
    mappings = {
        'interestExpense': (income_stmt, ['Interest Expense', 'Interest And Debt Expense']),
        'stockBasedCompensation': (cashflow, ['Stock Based Compensation', 'Share Based Compensation']),
        'capitalExpenditure': (cashflow, ['Capital Expenditure', 'Capital Expenditures']),
        'operatingIncome': (income_stmt, ['Operating Income', 'EBIT']),
        'commonStockEquity': (balance_sheet, ['Common Stock Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest']),
        'cashAndEquivalents': (balance_sheet, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']),
    }
    for key, (frame, labels) in mappings.items():
        current = safe_float(out.get(key), np.nan)
        if not np.isnan(current):
            continue
        value = latest_statement_value(frame, labels)
        if not np.isnan(value):
            out[key] = float(value)
    if np.isnan(safe_float(out.get('totalCash'), np.nan)):
        stmt_cash = safe_float(out.get('cashAndEquivalents'), np.nan)
        if not np.isnan(stmt_cash):
            out['totalCash'] = stmt_cash
    return out


def _compute_fundamental_quality(snapshot: dict) -> dict:
    revenue = safe_float(snapshot.get('totalRevenue'), np.nan)
    fcf = safe_float(snapshot.get('freeCashflow'), np.nan)
    ocf = safe_float(snapshot.get('operatingCashflow'), np.nan)
    op_income = safe_float(snapshot.get('operatingIncome'), np.nan)
    total_debt = safe_float(snapshot.get('totalDebt'), 0.0)
    total_cash = safe_float(snapshot.get('totalCash'), 0.0)
    ebitda = safe_float(snapshot.get('ebitda'), np.nan)
    equity = safe_float(snapshot.get('commonStockEquity'), np.nan)

    roic = np.nan
    invested_capital = (0.0 if np.isnan(equity) else equity) + max(total_debt, 0.0) - max(total_cash, 0.0)
    if not np.isnan(op_income) and invested_capital > 0:
        roic = (op_income * (1.0 - 0.21)) / invested_capital

    reinvestment_rate = np.nan
    if not np.isnan(ocf) and not np.isnan(fcf):
        nopat_proxy = max(abs(op_income), 1e-6) if not np.isnan(op_income) else max(abs(ocf), 1e-6)
        reinvestment_rate = max(0.0, ocf - fcf) / nopat_proxy

    fcf_margin = np.nan
    if revenue > 0 and not np.isnan(fcf):
        fcf_margin = fcf / revenue

    net_debt_ebitda = np.nan
    if not np.isnan(ebitda) and abs(ebitda) > 1e-6:
        net_debt_ebitda = (total_debt - total_cash) / ebitda

    return {
        'roic': None if np.isnan(roic) else float(roic),
        'reinvestment_rate': None if np.isnan(reinvestment_rate) else float(reinvestment_rate),
        'fcf_margin': None if np.isnan(fcf_margin) else float(fcf_margin),
        'net_debt_ebitda': None if np.isnan(net_debt_ebitda) else float(net_debt_ebitda),
    }


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
        'roic':                0.0,
        'reinvestment_rate':   0.0,
        'fcf_margin':          0.0,
        'net_debt_ebitda':     0.0,
        'share_dilution_1y':   0.0,
        'share_dilution_cagr_3y': 0.0,
        'sbc_ratio':           0.0,
        'dilution_risk_flag':  0.0,
        'reverse_dcf_implied_growth': 0.0,
        'dcf_surface_pct_above_price': 50.0,
        'speculative_growth_risk': 0.0,
        'speculative_growth_score': 0.0,
        'speculative_growth_haircut': 0.0,
        'fundamental_confidence_multiplier': 1.0,
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

            income_stmt = load_statement_frame(tkr, ['income_stmt', 'financials', 'income_stmt_freq'])
            cashflow = load_statement_frame(tkr, ['cashflow', 'cash_flow'])
            balance_sheet = load_statement_frame(tkr, ['balance_sheet', 'balancesheet'])
            share_history = _fetch_share_history(tkr) if CONFIG.get('fundamentals_dilution_enabled', True) else None
            snapshot = _augment_snapshot_from_statements(dict(slow), income_stmt, cashflow, balance_sheet)
            if np.isnan(safe_float(snapshot.get('currentPrice'), np.nan)):
                fast_price = safe_float(getattr(info, 'last_price', np.nan), np.nan)
                if np.isnan(fast_price):
                    try:
                        fast_price = safe_float(info.get('lastPrice'), np.nan)
                    except Exception:
                        fast_price = np.nan
                if not np.isnan(fast_price):
                    snapshot['currentPrice'] = fast_price

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
            snapshot.update({
                'currentPrice': _g('currentPrice', snapshot.get('currentPrice', np.nan)),
                'regularMarketPrice': _g('regularMarketPrice', snapshot.get('regularMarketPrice', np.nan)),
                'sharesOutstanding': _g('sharesOutstanding', snapshot.get('sharesOutstanding', np.nan)),
                'freeCashflow': _g('freeCashflow', snapshot.get('freeCashflow', np.nan)),
                'operatingCashflow': _g('operatingCashflow', snapshot.get('operatingCashflow', np.nan)),
                'totalRevenue': _g('totalRevenue', snapshot.get('totalRevenue', np.nan)),
                'totalDebt': _g('totalDebt', snapshot.get('totalDebt', np.nan)),
                'totalCash': _g('totalCash', snapshot.get('totalCash', np.nan)),
                'beta': _g('beta', snapshot.get('beta', 1.0)),
                'marketCap': _g('marketCap', snapshot.get('marketCap', np.nan)),
                'revenueGrowth': _g('revenueGrowth', snapshot.get('revenueGrowth', 0.0)),
                'ebitda': _g('ebitda', snapshot.get('ebitda', np.nan)),
            })
            quality = _compute_fundamental_quality(snapshot)
            dilution = compute_dilution_metrics(
                current_shares=snapshot.get('sharesOutstanding'),
                share_history=share_history,
                stock_based_comp=snapshot.get('stockBasedCompensation'),
                revenue=snapshot.get('totalRevenue'),
                free_cashflow=snapshot.get('freeCashflow'),
                )
            dcf_state = build_dcf_state(
                snapshot,
                years=CONFIG.get('fundamentals_dcf_years', 5),
                terminal_growth=CONFIG.get('fundamentals_dcf_terminal_growth', 0.025),
                country_risk_premium=CONFIG.get('fundamentals_country_risk_premium', 0.0),
            )
            dcf_base = dcf_valuation(dcf_state)
            reverse_dcf = (
                reverse_dcf_analysis(dcf_state)
                if CONFIG.get('fundamentals_reverse_dcf_enabled', True)
                else {'available': False, 'note': 'disabled by config'}
            )
            dcf_surface = (
                dcf_surface_analysis(
                    dcf_state,
                    growth_grid=CONFIG.get('fundamentals_dcf_growth_grid'),
                    discount_grid=CONFIG.get('fundamentals_dcf_discount_grid'),
                )
                if CONFIG.get('fundamentals_dcf_surface_enabled', True)
                else {'available': False, 'note': 'disabled by config'}
            )
            result['quality_metrics'] = quality
            result['dilution_analysis'] = dilution
            result['dcf'] = dcf_base
            result['reverse_dcf'] = reverse_dcf
            result['dcf_surface'] = dcf_surface
            spec_growth = compute_speculative_growth_profile(
                snapshot,
                dilution=dilution,
                dcf_result=dcf_base,
                reverse_dcf=reverse_dcf,
            )
            result['speculative_growth'] = spec_growth
            result['roic'] = float(quality.get('roic') or 0.0)
            result['reinvestment_rate'] = float(quality.get('reinvestment_rate') or 0.0)
            result['fcf_margin'] = float(quality.get('fcf_margin') or 0.0)
            result['net_debt_ebitda'] = float(quality.get('net_debt_ebitda') or 0.0)
            result['share_dilution_1y'] = float((dilution.get('share_growth_1y') or 0.0) / 100.0)
            result['share_dilution_cagr_3y'] = float((dilution.get('share_cagr_3y') or 0.0) / 100.0)
            result['sbc_ratio'] = float((dilution.get('sbc_ratio') or 0.0) / 100.0)
            result['dilution_risk_flag'] = float(1.0 if dilution.get('risk_flag') else 0.0)
            result['reverse_dcf_implied_growth'] = float((reverse_dcf.get('implied_growth_5y') or 0.0) / 100.0)
            result['dcf_surface_pct_above_price'] = float((dcf_surface.get('pct_above_price') or 50.0) / 100.0)
            result['speculative_growth_risk'] = float(1.0 if spec_growth.get('speculative_growth_risk') else 0.0)
            result['speculative_growth_score'] = float(spec_growth.get('speculative_growth_score') or 0.0)
            result['speculative_growth_haircut'] = float(spec_growth.get('speculative_growth_haircut') or 0.0)
            result['fundamental_confidence_multiplier'] = float(spec_growth.get('fundamental_confidence_multiplier') or 1.0)

            for k, v in result.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    result[k] = self.DEFAULTS.get(k, 0.0)

            log.info(
                f"Fundamentals fetched: "
                f"SI={result['short_pct_float']:.1%}  vel={result['short_velocity']:+.4f}  "
                f"PCR={result['put_call_ratio']:.2f}  IV={result['iv_proxy']:.0%}  "
                f"target_up={result['target_upside']:+.1%}  "
                f"days_earn={result['days_to_earnings']:.0f}  "
                f"ROIC={result['roic']:.1%}  dilution={result['share_dilution_1y']:.1%}  "
                f"rev_dcf={result['reverse_dcf_implied_growth']:.1%}  "
                f"spec_haircut={result['speculative_growth_haircut']:.1%}"
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

    def split(self, X, y=None, groups=None, event_meta: 'pd.DataFrame | None' = None):
        n = len(X)
        if n <= 1:
            return
        embargo = max(0, int(n * self.embargo_pct))
        fold_size = max(1, n // self.n_splits)

        if event_meta is None and isinstance(groups, pd.DataFrame):
            if {'event_start', 'event_end'}.issubset(groups.columns):
                event_meta = groups

        if hasattr(X, 'index'):
            base_index = pd.Index(X.index)
        elif event_meta is not None:
            base_index = pd.Index(event_meta.index)
        else:
            base_index = pd.RangeIndex(n)

        if event_meta is None:
            event_meta = _fixed_horizon_event_frame(base_index, CONFIG.get('predict_days', 1))
        else:
            event_meta = event_meta.reindex(base_index).copy()

        if 'event_start' not in event_meta or 'event_end' not in event_meta:
            event_meta = _fixed_horizon_event_frame(base_index, CONFIG.get('predict_days', 1))

        event_start = pd.to_datetime(event_meta['event_start'], errors='coerce')
        event_end = pd.to_datetime(event_meta['event_end'], errors='coerce')
        event_start = event_start.where(event_start.notna(), pd.to_datetime(base_index, errors='coerce'))
        event_end = event_end.where(event_end.notna(), event_start)

        if event_start.isna().any() or event_end.isna().any():
            pos_index = np.arange(n)
            event_start = pd.Series(pos_index, index=base_index)
            event_end = pd.Series(np.maximum(pos_index, pos_index), index=base_index)

        for i in range(self.n_splits):
            te_start = i * fold_size
            te_end = te_start + fold_size if i < self.n_splits - 1 else n
            te_idx = np.arange(te_start, te_end)
            if len(te_idx) == 0:
                continue

            test_start = event_start.iloc[te_idx].min()
            test_end = event_end.iloc[te_idx].max()
            overlap = (event_start <= test_end) & (event_end >= test_start)

            train_mask = np.ones(n, dtype=bool)
            train_mask[te_idx] = False
            train_mask &= ~overlap.to_numpy()
            if embargo > 0:
                train_mask[te_end:min(n, te_end + embargo)] = False

            tr_idx = np.flatnonzero(train_mask)
            if len(tr_idx) >= 10 and len(te_idx) >= 1:
                yield tr_idx, te_idx


# ─────────────────────────────────────────────────────────────────────────────
# CV SCORER — now uses PurgedKFold and safe split count
# ─────────────────────────────────────────────────────────────────────────────
def cv_score(model, X, y, n_splits, gap=None, event_meta: 'pd.DataFrame | None' = None):
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
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    for tr_idx, te_idx in pkf.split(X, event_meta=event_meta):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
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
    deterministic_torch = bool(CONFIG.get('torch_deterministic', True))
    if HAS_TORCH and deterministic_torch:
        _ensure_cublas_workspace_config()
    if HAS_TORCH and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        prop = torch.cuda.get_device_properties(0)
        vram = prop.total_memory / 1e9
        log.info(f"GPU    : {name}")
        log.info(f"VRAM   : {vram:.2f} GB")
        log.info(f"SM     : {prop.multi_processor_count}  |  CUDA CC: {prop.major}.{prop.minor}")
        if deterministic_torch:
            torch.backends.cudnn.benchmark        = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32       = False
            log.info("cuDNN  : benchmark=False  TF32=False (deterministic)")
            log.info(f"cuBLAS : workspace={os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'unset')}")
        else:
            torch.backends.cudnn.benchmark        = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
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


def _normalize_ohlcv_frame(data) -> 'pd.DataFrame | None':
    """Return a canonical daily OHLCV frame or None when unusable."""
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None

    df = data.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    col_map = {str(c).strip().lower(): c for c in df.columns}
    required = ['open', 'high', 'low', 'close', 'volume']
    if any(col not in col_map for col in required):
        return None

    out = df[[col_map[col] for col in required]].copy()
    out.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    idx = pd.to_datetime(out.index, errors='coerce')
    valid_idx = ~pd.isna(idx)
    out = out.loc[valid_idx]
    if out.empty:
        return None

    idx = pd.DatetimeIndex(idx[valid_idx])
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    idx = idx.normalize()
    out.index = idx
    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')]
    out = out.apply(pd.to_numeric, errors='coerce')
    out = out.dropna(how='any')
    out.index.name = None
    return out if not out.empty else None


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
        except Exception as exc:
            log.warning(f"Cache  : read failed for {ticker.upper()} ({exc})")
            return None

    @classmethod
    def inspect(cls, ticker: str, ttl_hours: int = DATA_CACHE_TTL_HOURS) -> tuple:
        """Return (frame, cache_status, meta) for logging and scheduling."""
        con = cls._connect()
        if con is None:
            return None, "disabled", {}
        try:
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=ttl_hours)
            rows = con.execute(
                "SELECT date,open,high,low,close,volume,fetched_at "
                "FROM ohlcv WHERE ticker=? ORDER BY date",
                [ticker.upper()]
            ).fetchdf()
            if rows.empty:
                return None, "miss", {"cache_path": str(cls._path) if cls._path else None}

            latest_fetch = pd.to_datetime(rows['fetched_at']).max()
            meta = {
                "latest_fetch": latest_fetch.isoformat() if pd.notna(latest_fetch) else None,
                "cache_path": str(cls._path) if cls._path else None,
            }
            if latest_fetch < pd.Timestamp(cutoff):
                return None, "stale", meta

            rows['date'] = pd.to_datetime(rows['date'])
            df = rows.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index.name = None
            return df, "hit", meta
        except Exception as exc:
            log.warning(f"Cache  : read failed for {ticker.upper()} ({exc})")
            return None, "error", {"error": str(exc)}

    @staticmethod
    def _frame_to_rows(ticker: str, df: pd.DataFrame) -> 'pd.DataFrame | None':
        normalised = _normalize_ohlcv_frame(df)
        if normalised is None or normalised.empty:
            return None
        rows = normalised.reset_index()
        rows = rows.rename(columns={rows.columns[0]: 'date'})
        rows.columns = [str(c).lower() for c in rows.columns]
        rows['ticker'] = ticker.upper()
        rows['fetched_at'] = datetime.datetime.utcnow()
        return rows[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'fetched_at']]

    @classmethod
    def save(cls, ticker: str, df: pd.DataFrame):
        """Upsert OHLCV rows into the cache."""
        con = cls._connect()
        if con is None or df is None or df.empty:
            return
        try:
            rows = cls._frame_to_rows(ticker, df)
            if rows is None or rows.empty:
                return
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
        except Exception as exc:
            log.warning(f"Cache  : write failed for {ticker.upper()} ({exc})")


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
        self._primary_frame = None
        self._primary_info = {}

    def _download_yfinance_ohlcv(self, symbol: str) -> 'pd.DataFrame | None':
        if not HAS_YF:
            return None
        frame = None
        try:
            frame = _normalize_ohlcv_frame(
                yf.download(symbol, period=self.period, progress=False, auto_adjust=True)
            )
        except Exception:
            frame = None
        if frame is not None and not frame.empty:
            return frame
        try:
            hist = yf.Ticker(symbol).history(period=self.period, auto_adjust=True)
            return _normalize_ohlcv_frame(hist)
        except Exception:
            return None

    @staticmethod
    def _log_cache_event(symbol_label: str, info: dict, df: 'pd.DataFrame | None'):
        state = info.get("cache_state", "unknown")
        source = info.get("source", "unknown")
        rows = len(df) if isinstance(df, pd.DataFrame) else 0
        latest_fetch = info.get("latest_fetch")
        if state == "reused_primary_frame":
            log.info(f"  {symbol_label}: reused_primary_frame ({rows} rows)")
            return
        if state == "hit":
            extra = f" @ {latest_fetch}" if latest_fetch else ""
            log.info(f"  {symbol_label}: cache hit ({rows} rows{extra})")
            return
        if state == "stale":
            extra = f" @ {latest_fetch}" if latest_fetch else ""
            log.info(f"  {symbol_label}: cache stale{extra} -> refetched via {source} ({rows} rows)")
            return
        if state == "miss":
            log.info(f"  {symbol_label}: cache miss -> refetched via {source} ({rows} rows)")
            return
        if state == "disabled":
            log.info(f"  {symbol_label}: cache disabled -> fetched via {source} ({rows} rows)")
            return
        if state == "error":
            log.warning(f"  {symbol_label}: cache read error -> fetched via {source} ({rows} rows)")
            return
        log.info(f"  {symbol_label}: fetched via {source} ({rows} rows)")

    def _fetch_ohlcv(self, symbol: str, reuse_primary: bool = False) -> tuple:
        symbol = symbol.upper()
        if reuse_primary and self._primary_frame is not None and symbol == self.ticker.upper():
            return self._primary_frame.copy(), {
                "cache_state": "reused_primary_frame",
                "source": self._primary_info.get("source", "primary"),
            }

        cached_df, cache_state, meta = DataCache.inspect(symbol)
        if cached_df is not None:
            info = dict(meta)
            info.update({"cache_state": "hit", "source": "cache"})
            return cached_df, info

        download_source = "Tiingo" if (self._tiingo.available() and not symbol.startswith("^")) else "yfinance"
        df = None
        if download_source == "Tiingo":
            df = _normalize_ohlcv_frame(self._tiingo.fetch(symbol, self.period))
            if df is None or len(df) <= 50:
                df = None
                download_source = "yfinance"
        if df is None:
            df = self._download_yfinance_ohlcv(symbol)

        if df is None or df.empty:
            raise RuntimeError(f"Could not fetch data for {symbol}")

        DataCache.save(symbol, df)
        info = dict(meta)
        info.update({"cache_state": cache_state, "source": download_source})
        return df, info

    def fetch(self) -> pd.DataFrame:
        log_section("DATA DOWNLOAD")
        t0 = time.time()
        log.info(f"Ticker: {self.ticker}  Period: {self.period}")

        # ── 1. Try DuckDB cache ────────────────────────────────────────────────
        df, info = self._fetch_ohlcv(self.ticker, reuse_primary=False)
        self._primary_frame = df.copy()
        self._primary_info = dict(info)
        cache_state = info.get("cache_state", "unknown")
        if cache_state == "hit":
            log.info(f"Cache  : HIT  ({len(df)} rows from {info.get('cache_path') or 'DuckDB cache'})")
        elif cache_state == "stale":
            log.info(f"Cache  : STALE  ({info.get('latest_fetch') or 'unknown'})")
            log.info(f"Cache  : REFETCHED via {info.get('source', 'unknown')}")
        elif cache_state == "miss":
            log.info("Cache  : MISS")
            log.info(f"Cache  : REFETCHED via {info.get('source', 'unknown')}")
        elif cache_state == "disabled":
            log.info(f"Cache  : DISABLED  (fetched via {info.get('source', 'unknown')})")
        elif cache_state == "error":
            log.warning(f"Cache  : READ ERROR  ({info.get('error', 'unknown')})")
            log.info(f"Cache  : REFETCHED via {info.get('source', 'unknown')}")
        else:
            log.info(f"Cache  : {cache_state.upper()}  via {info.get('source', 'unknown')}")
        if df is None or df.empty:
            raise RuntimeError(f"Could not fetch data for {self.ticker}")

        log.info(f"Source : {info.get('source', 'unknown')}")
        log.info(f"Rows   : {len(df)}  ({df.index[0].date()} to {df.index[-1].date()})")
        log.info(f"Latest : ${df['Close'].iloc[-1]:.4f}")
        log.info(f"52w H/L: ${df['High'].tail(252).max():.2f} / ${df['Low'].tail(252).min():.2f}")
        log.info(f"Avg vol: {df['Volume'].mean():,.0f}")
        log.info(f"Time   : {elapsed(t0)}")
        return df

    def fetch_context(self) -> pd.DataFrame:
        symbols = ["SPY", "QQQ", "GLD", "TLT"]
        requested_symbols = symbols + ["VIX"]
        log.info(f"Downloading market context: SPY, QQQ, GLD, TLT, VIX...")
        if not HAS_YF and not HAS_DUCKDB and not self._tiingo.available():
            log.warning("Context disabled: no yfinance, no Tiingo, and no cache available")
            return pd.DataFrame()

        frames = {}
        symbol_map = {"SPY": "SPY", "QQQ": "QQQ", "GLD": "GLD", "TLT": "TLT", "^VIX": "VIX"}
        for raw_symbol, output_name in symbol_map.items():
            try:
                frame, info = self._fetch_ohlcv(raw_symbol, reuse_primary=True)
                self._log_cache_event(output_name, info, frame)
                close = _normalize_context_series(frame, output_name)
                if close is None or len(close) < 30:
                    log.warning(f"  {output_name}: insufficient usable rows - skipping")
                    continue
                frames[output_name] = close
                _log_context_coverage(output_name, close)
            except Exception as e:
                log.warning(f"  {output_name}: fetch failed ({e})")

        log.info(f"Context requested: {len(requested_symbols)} symbols  returned: {len(frames)}")
        if not frames:
            log.warning("Context: no usable market symbols")
            return pd.DataFrame()

        usable_frames = [series for series in frames.values()
                         if isinstance(series, pd.Series) and not series.empty]
        if not usable_frames:
            log.warning("Context: all downloaded market frames were empty after validation")
            return pd.DataFrame()

        out = pd.concat(usable_frames, axis=1, join='outer').sort_index()
        out = out.loc[:, ~out.columns.duplicated()]
        out = out.dropna(axis=1, how='all')
        out.index.name = None
        if out.empty or out.shape[1] == 0:
            log.warning("Context: assembled frame is empty after cleanup")
            return pd.DataFrame()

        log.info(f"Context: {out.shape[1]} symbols, {len(out)} union rows")
        spy = next((c for c in out.columns if 'SPY' in str(c).upper()), None)
        vix = next((c for c in out.columns if 'VIX' in str(c).upper()), None)
        if spy is None:
            log.warning("  Context overlap: SPY missing - regime model will be skipped")
            return out

        spy_rows = int(out[[spy]].dropna().shape[0])
        if spy_rows > 0:
            log.info(f"  Context overlap: SPY-only={spy_rows} rows")
        else:
            log.warning("  Context overlap: SPY-only=0 rows")

        if vix is not None:
            duo_rows = int(out[[spy, vix]].dropna().shape[0])
            min_rows = int(CONFIG.get('regime_min_rows', 252))
            if duo_rows >= min_rows:
                log.info(f"  Context overlap: SPY+VIX={duo_rows} rows")
            elif duo_rows > 0:
                log.warning(f"  Context overlap: SPY+VIX={duo_rows} rows (<{min_rows}) - "
                            f"VIX will remain optional for regime modeling")
            else:
                log.warning("  Context overlap: SPY+VIX=0 rows - VIX kept optional")
        else:
            log.warning("  Context overlap: VIX unavailable - SPY-only fallback will be used")
        return out

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
        if CONFIG.get('feature_volatility_regime_enabled', True):
            log_range = np.log((h + 1e-10) / (l + 1e-10)).replace([np.inf, -np.inf], np.nan)
            abs_r = r.abs()
            downside_r = r.clip(upper=0).abs()
            for w in [10, 21, 63]:
                total_vol = r.rolling(w).std() * np.sqrt(252)
                downside_vol = np.sqrt((downside_r.pow(2)).rolling(w).mean()) * np.sqrt(252)
                self.feat[f'downside_vol_{w}'] = downside_vol
                self.feat[f'semivol_ratio_{w}'] = downside_vol / (total_vol + 1e-10)
                self.feat[f'parkinson_vol_{w}'] = np.sqrt(
                    log_range.pow(2).rolling(w).mean() / (4.0 * np.log(2))
                ) * np.sqrt(252)
                dd = c / (c.rolling(w).max() + 1e-10) - 1.0
                self.feat[f'drawdown_{w}'] = dd
                self.feat[f'ulcer_index_{w}'] = np.sqrt((dd.clip(upper=0).pow(2)).rolling(w).mean())
                self.feat[f'days_since_high_{w}'] = c.rolling(w).apply(
                    lambda x: float(len(x) - 1 - np.where(x == np.max(x))[0][-1]) if len(x) else np.nan,
                    raw=True,
                )
            self.feat['vol_cluster_21_63'] = abs_r.rolling(21).mean() / (abs_r.rolling(63).mean() + 1e-10)
            self.feat['vol_of_vol_21'] = self.feat['rv_21_annualised'].rolling(21).std()
            self.feat['absret_autocorr_21'] = abs_r.rolling(21).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else np.nan,
                raw=False,
            )
            self.feat['drawdown_stress_63'] = (self.feat['drawdown_63'] < -0.20).astype(int)
            self.feat['vol_regime_transition'] = self.feat['vol_regime_hi'].diff().abs().fillna(0.0)

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
        if CONFIG.get('feature_liquidity_enhanced_enabled', True):
            dollar_vol = c * v
            up_vol = v.where(r > 0, 0.0)
            for w in [10, 21, 63]:
                self.feat[f'dollar_vol_{w}'] = dollar_vol.rolling(w).mean()
                self.feat[f'vol_spike_pct_{w}'] = v.rolling(w).apply(
                    lambda x: float(np.mean(x <= x[-1])) if len(x) else np.nan,
                    raw=True,
                )
                self.feat[f'up_volume_share_{w}'] = (
                    up_vol.rolling(w).sum() / (v.rolling(w).sum() + 1e-10)
                )
            shares_out = safe_float(
                self.fundamentals.get('sharesOutstanding', self.fundamentals.get('shares_outstanding'))
            ) if self.fundamentals else np.nan
            if np.isfinite(shares_out) and shares_out > 0:
                turnover = v / shares_out
                for w in [10, 21, 63]:
                    self.feat[f'turnover_ratio_{w}'] = turnover.rolling(w).mean()
        if CONFIG.get('feature_anchored_vwap_enabled', True):
            for w in [21, 63, 126]:
                anchor_vwap = (c * v).rolling(w).sum() / (v.rolling(w).sum() + 1e-10)
                self.feat[f'anchored_vwap_dev_{w}'] = (
                    (c - anchor_vwap) / (anchor_vwap.abs() + 1e-10)
                )

        if CONFIG.get('feature_trend_quality_enabled', True):
            abs_path = c.diff().abs()
            for w in [10, 21, 63]:
                self.feat[f'efficiency_ratio_{w}'] = (
                    (c - c.shift(w)).abs() / (abs_path.rolling(w).sum() + 1e-10)
                )
                prior_high = c.shift(1).rolling(w).max()
                prior_low = c.shift(1).rolling(w).min()
                prior_break_high = c.shift(2).rolling(w).max()
                prior_break_low = c.shift(2).rolling(w).min()
                yday_break_up = (c.shift(1) > prior_break_high)
                yday_break_dn = (c.shift(1) < prior_break_low)
                self.feat[f'breakout_up_{w}'] = (c > prior_high).astype(int)
                self.feat[f'breakout_down_{w}'] = (c < prior_low).astype(int)
                self.feat[f'breakout_cont_up_{w}'] = (
                    yday_break_up & (c >= prior_break_high)
                ).astype(int)
                self.feat[f'breakout_fail_up_{w}'] = (
                    yday_break_up & (c < prior_break_high)
                ).astype(int)
                self.feat[f'breakout_cont_down_{w}'] = (
                    yday_break_dn & (c <= prior_break_low)
                ).astype(int)
                self.feat[f'breakout_fail_down_{w}'] = (
                    yday_break_dn & (c > prior_break_low)
                ).astype(int)

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
            ffill_limit   = int(CONFIG.get('market_ffill_limit', 5))
            required_rows = _required_market_rows(len(c.index), CONFIG)
            for sym in self.market.columns:
                m = _align_market_series(self.market[sym], c.index,
                                         ffill_limit=ffill_limit)
                valid_rows = int(m.notna().sum())
                if valid_rows < required_rows:
                    log.info(f"  Market feature skip: {sym} coverage={valid_rows}/{len(c.index)} "
                             f"rows (<{required_rows})")
                    continue
                m_ret = m.pct_change(fill_method=None)
                for w in [5,10,21,63]:
                    self.feat[f'rs_{sym}_{w}'] = (
                        c.pct_change(w) - m.pct_change(w, fill_method=None))
                self.feat[f'beta_{sym}_30'] = (
                    r.rolling(30).cov(m_ret)/(m_ret.rolling(30).var()+1e-10))
                self.feat[f'corr_{sym}_20'] = r.rolling(20).corr(m_ret)
                self.feat[f'corr_{sym}_60'] = r.rolling(60).corr(m_ret)
                if CONFIG.get('feature_market_residual_enabled', True):
                    for w in [30, 63]:
                        beta_w = r.rolling(w).cov(m_ret) / (m_ret.rolling(w).var() + 1e-10)
                        resid = r - beta_w * m_ret
                        self.feat[f'resid_ret_{sym}_{w}'] = resid.rolling(5).sum()
                        self.feat[f'idio_vol_{sym}_{w}'] = resid.rolling(w).std() * np.sqrt(252)

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

    # Safety floor: minimum meaningful move before a label is actionable.
    # BUGFIX: old formula (atr_pct/100 * horizon * 0.5) is LINEAR in horizon.
    # Volatility scales with √time, not time itself.  For RKLB (daily ATR=6.64%,
    # horizon=15d): old formula → 6.64%×15×0.5 = 49.8%, which overrides the
    # quantile thresholds entirely.  Correct scaling:
    #   min_move = atr_pct/100 × √horizon × 0.3
    #   RKLB: 6.64% × √15 × 0.3 = 7.7%  (sensible, below the 25th-pct threshold)
    # Cap at 15% so even extreme FAST stocks get reasonable thresholds.
    min_move = min(0.15, max(0.01, atr_pct / 100 * np.sqrt(best_horizon) * 0.3))
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
        event_end = pd.Series(pd.NaT, index=close.index, dtype='datetime64[ns]')

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
            event_ts = close.index[end - 1]
            for ts, price in future.items():
                event_ts = ts
                if price >= upper:
                    lbl = 2  # BUY (take-profit)
                    break
                elif price <= lower:
                    lbl = 0  # SELL (stop-loss)
                    break
            labels.iloc[i] = lbl
            event_end.iloc[i] = event_ts

        self.event_end_ = event_end
        return labels


# ─────────────────────────────────────────────────────────────────────────────
# RISK-ADJUSTED RETURN (RAR) LABELER  — balanced labels for trending stocks
# ─────────────────────────────────────────────────────────────────────────────
class RARLabeler:
    """
    Risk-Adjusted Return labels: normalise h-day forward return by rolling
    realised volatility, then apply quantile thresholds on the normalised score.

    Why this beats Triple Barrier for trending / small-dataset stocks
    ─────────────────────────────────────────────────────────────────
    Triple Barrier with symmetric barriers on a strongly trending stock
    (e.g. RKLB: $3 → $68 over 4 years) naturally produces lopsided labels —
    far more BUY hits than SELL hits because upside moves dominate.
    RKLB run log: SELL=16%, HOLD=51%, BUY=32%.  The 2:1 BUY:SELL imbalance
    makes SELL the hardest class and pushes F1-macro down toward 0.33.

    RAR normalization removes the trend effect entirely:
        rar[t] = forward_return[t,h] / (σ_d[t] × √h)
    This is the h-day Sharpe ratio (risk-adjusted alpha per bar).  Because it
    divides out current volatility, a +10% move in a low-vol environment and a
    +10% move in a high-vol environment receive very different scores — only the
    former is "large relative to noise" and gets labelled BUY.

    Labels are then assigned by PERCENTILE RANK on a rolling 252-day window of
    RAR scores, which:
      • Guarantees ~25/50/25 SELL/HOLD/BUY split in every regime
      • Is robust to non-stationarity (bull/bear shifts don't change the scale)
      • Is fully look-ahead-free (percentile computed on past window only)

    RAR formula
    ───────────
      σ_d[t]  = std(daily_returns[t-vol_window:t])   (rolling, look-back only)
      vol_h   = σ_d[t] × √h
      rar[t]  = pct_change(close, h, shift=-h)[t] / (vol_h[t] + ε)
      pct_rank(rar, window) → {0='SELL', 1='HOLD', 2='BUY'} by quantile

    Parameters
    ──────────
    close       : pd.Series of close prices
    horizon     : forward-return horizon in days
    vol_window  : lookback window for realised vol estimation (default 21)
    rank_window : rolling window for percentile ranking (default 252)
    buy_q       : upper quantile threshold for BUY   (default 0.75)
    sell_q      : lower quantile threshold for SELL  (default 0.25)
    """

    def __init__(self, close: pd.Series, horizon: int,
                 vol_window: int = 21, rank_window: int = 252,
                 buy_q: float = 0.75, sell_q: float = 0.25):
        self.close       = close
        self.h           = horizon
        self.vol_window  = vol_window
        self.rank_window = rank_window
        self.buy_q       = buy_q
        self.sell_q      = sell_q

    def label(self) -> pd.Series:
        c   = self.close
        h   = self.h

        # ── h-day forward return (look-ahead shift, so labels are future-aligned) ──
        fwd_ret = c.pct_change(h).shift(-h)   # return over next h days

        # ── Rolling realised daily volatility (no look-ahead) ────────────────────
        daily_r = c.pct_change()
        sigma_d = daily_r.rolling(self.vol_window, min_periods=max(5, self.vol_window//2)).std()
        vol_h   = sigma_d * np.sqrt(h)

        # ── Risk-adjusted return (RAR) ────────────────────────────────────────────
        rar = fwd_ret / (vol_h + 1e-8)

        # ── Rolling percentile rank → label ──────────────────────────────────────
        # percentile_rank[t] = fraction of past `rank_window` RAR values ≤ rar[t]
        # Computed with expanding window for the first rank_window rows, then rolling.
        # All lookback-only — no future information used.
        labels = pd.Series(np.nan, index=c.index, dtype=float)
        rar_vals = rar.values
        rw = self.rank_window

        for i in range(len(c)):
            if np.isnan(rar_vals[i]):
                continue
            # Lookback window: [max(0, i-rw+1) : i]  (exclude i itself — pure past)
            start = max(0, i - rw + 1)
            window = rar_vals[start:i]
            window = window[~np.isnan(window)]
            if len(window) < max(10, rw // 10):  # need at least 10% of window
                continue
            pct = float((window < rar_vals[i]).sum()) / len(window)
            if pct >= self.buy_q:
                labels.iloc[i] = 2   # BUY
            elif pct <= self.sell_q:
                labels.iloc[i] = 0   # SELL
            else:
                labels.iloc[i] = 1   # HOLD

        return labels


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAKER  —  uses Triple Barrier (NEW 1) or per-stock quantile fallback
# ─────────────────────────────────────────────────────────────────────────────
def make_labels(df: pd.DataFrame, regime: dict) -> pd.Series:
    """
    Label dispatcher.  Controlled by CONFIG['label_method']:

      'rar'            — Risk-Adjusted Return labels (default, recommended).
                         Normalises h-day return by rolling vol → percentile rank.
                         Always produces ~25/50/25 SELL/HOLD/BUY regardless of trend.
                         Best for trending / small-data stocks (RKLB, new IPOs).

      'triple_barrier' — López de Prado Triple Barrier (path-dependent).
                         Correct for mean-reverting, large-sample stocks.
                         Can produce lopsided labels on strongly trending stocks.

      'quantile'       — Fixed quantile thresholds on raw h-day return.
                         Simple fallback; uses IC-selected horizon.

    Class imbalance is handled in the loss function (FocalLoss + class_weight
    clipping) — no barrier tightening or forced balancing is applied here.
    """
    log_section("LABEL GENERATION")
    method = CONFIG.get('label_method', 'rar')

    # ── RAR labels (default) ─────────────────────────────────────────────────
    if method == 'rar':
        try:
            h        = regime['predict_days']
            buy_q    = CONFIG.get('label_buy_quantile',  0.75)
            sell_q   = CONFIG.get('label_sell_quantile', 0.25)
            vol_win  = CONFIG.get('tb_vol_window', 21)
            rank_win = CONFIG.get('rar_rank_window', 252)
            log.info(f"Label method : Risk-Adjusted Return (RAR)  "
                     f"h={h}d  vol_win={vol_win}  rank_win={rank_win}  "
                     f"BUY>pct{buy_q*100:.0f}  SELL<pct{sell_q*100:.0f}")
            log.info("  RAR normalises return by σ×√h → percentile rank on rolling window.")
            log.info("  Guarantees balanced labels regardless of trend direction.")
            labeler = RARLabeler(df['Close'], horizon=h,
                                 vol_window=vol_win, rank_window=rank_win,
                                 buy_q=buy_q, sell_q=sell_q)
            y = labeler.label()
            n_valid = y.notna().sum()
            if n_valid < 50:
                raise ValueError(f"Only {n_valid} valid RAR labels — too few")
            vc = y.value_counts().sort_index()
            for idx, cnt in vc.items():
                log.info(f"  {['SELL','HOLD','BUY'][int(idx)]:<5}: "
                         f"{cnt:>5}  ({cnt/n_valid*100:.1f}%)")
            event_meta = _fixed_horizon_event_frame(df.index, h)
            return _attach_label_event_meta(y, event_meta, 'rar')
        except Exception as e:
            log.warning(f"RAR labeler failed ({e}) — falling back to Triple Barrier")
            method = 'triple_barrier'

    # ── Triple Barrier labels ────────────────────────────────────────────────
    use_tb      = CONFIG.get('use_triple_barrier', True)
    min_samples = CONFIG.get('tb_min_samples', 150)
    n_rows      = len(df)

    if (method == 'triple_barrier' or use_tb) and n_rows >= min_samples:
        pt_sl = list(CONFIG.get('tb_pt_sl', [1.0, 1.0]))
        vol_w = CONFIG.get('tb_vol_window', 20)
        try:
            log.info(f"Label method : Triple Barrier  "
                     f"pt={pt_sl[0]:.2f}×σ√h  sl={pt_sl[1]:.2f}×σ√h  "
                     f"h={regime['predict_days']}d  vol_win={vol_w}")
            log.info("  Class imbalance handled by FocalLoss + weight clipping — "
                     "no barrier tightening applied.")
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
            event_meta = pd.DataFrame({
                'event_start': pd.DatetimeIndex(df.index).normalize(),
                'event_end': pd.to_datetime(getattr(labeler, 'event_end_', None), errors='coerce'),
            }, index=df.index)
            return _attach_label_event_meta(y, event_meta, 'triple_barrier')
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
    event_meta = _fixed_horizon_event_frame(df.index, n)
    return _attach_label_event_meta(y, event_meta, 'quantile')


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
    def __init__(self, estimators: dict, classes: np.ndarray = None,
                 weights: dict | None = None):
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
        self.weights_    = dict(weights or {})

    # sklearn VotingClassifier API compatibility stubs ─────────────────────────
    def fit(self, X, y, sample_weight=None):
        """No-op: models are already fitted."""
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average softmax probabilities across all sub-models."""
        probas = []
        names = []
        for name, model in self.estimators_.items():
            try:
                p = model.predict_proba(X)
                if p.ndim == 2 and p.shape[1] == len(self.classes_):
                    probas.append(p)
                    names.append(name)
            except Exception:
                pass   # skip models that can't produce probabilities for this X
        if not probas:
            # Fallback: uniform distribution
            return np.full((len(X), len(self.classes_)),
                           1.0 / len(self.classes_))
        weights = np.array([float(self.weights_.get(name, 1.0)) for name in names], dtype=float)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(probas), dtype=float)
        weights = weights / weights.sum()
        out = np.zeros_like(probas[0], dtype=float)
        for w, p in zip(weights, probas):
            out += w * p
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # Allow get_params / set_params so downstream sklearn utilities work ───────
    def get_params(self, deep=True):
        return {'estimators': self.estimators_, 'classes': self.classes_,
                'weights': self.weights_}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# TREE MODEL TRAINER — FIX 4: feature name tracking + FIX 6: CatBoost
# ─────────────────────────────────────────────────────────────────────────────
class TreeTrainer:
    def __init__(self, X, y, cfg, use_gpu: bool, event_meta: 'pd.DataFrame | None' = None):
        self.X, self.y, self.cfg = X, y, cfg
        self.use_gpu = use_gpu
        self.event_meta = event_meta
        self.models, self.results = {}, {}
        self.scaler   = RobustScaler()
        self.var_sel  = None
        self.feat_sel = None
        self.feat_names_selected_ = None   # FIX 4
        self.ensemble_weights_ = {}

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
        event_tr = _slice_event_meta(self.event_meta, X_tr.index)
        event_te = _slice_event_meta(self.event_meta, X_te.index)

        m_tr = y_tr.notna(); m_te = y_te.notna()
        X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        X_te, y_te = X_te[m_te], y_te[m_te]
        event_tr = _slice_event_meta(event_tr, X_tr.index)
        event_te = _slice_event_meta(event_te, X_te.index)
        self.train_event_meta_ = event_tr
        self.test_event_meta_ = event_te

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
            sc     = cv_score(model, Xs_tr, y_tr, n_splits, event_meta=self.train_event_meta_)
            return sc['f1_mean'], params, kwargs

        best_score, best_params, best_kwargs = _run_grid_search_jobs(
            name, grid, _eval_combo, parallel_jobs, self.cfg)

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

        sw = getattr(self, 'sample_weights_', None)  # overlap-aware weights
        n_cpu = multiprocessing.cpu_count()
        grid_workers = _tree_grid_worker_count(n_cpu, self.cfg)
        grid_backend = _tree_grid_parallel_prefer(self.cfg)
        log.info(f"CPU cores available: {n_cpu}  |  grid workers: {grid_workers} "
                 f"|  backend: {grid_backend}  |  sample_weights: {'yes' if sw is not None else 'no'}")
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
        if _is_model_enabled('RandomForest', self.cfg):
            log.info(f"\n[1/6] Random Forest (CPU, outer parallel_jobs={grid_workers})...")
            t0 = time.time()
            rf, rfp, _ = self._grid_search(
                "RF", RandomForestClassifier, self.cfg['rf_grid'], Xs_tr, y_tr,
                {'class_weight': 'balanced', 'n_jobs': 1}, parallel_jobs=grid_workers)
            _fit_with_sw(rf, Xs_tr, y_tr.values)
            preds = rf.predict(Xs_te)
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro', zero_division=0)
            trained['RandomForest'] = rf
            self.results['RandomForest'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': rfp}
            log.info(f"  Test accuracy: {acc:.4f}  F1-macro: {f1:.4f}  |  {elapsed(t0)}")
            log.info(f"\n{classification_report(y_te, preds, target_names=['SELL','HOLD','BUY'], zero_division=0)}")
        else:
            log.info("\n[1/6] Random Forest skipped by active-model policy")

        # 2. HistGradientBoosting (doesn't support sample_weight in sklearn < 1.4 properly, skip)
        if _is_model_enabled('HistGradBoost', self.cfg):
            log.info(f"[2/6] HistGradientBoosting (CPU, outer parallel_jobs={grid_workers})...")
            t0 = time.time()
            gb, gbp, _ = self._grid_search(
                "HistGB", HistGradientBoostingClassifier, self.cfg['hist_gb_grid'],
                Xs_tr, y_tr,
                {'class_weight': 'balanced', 'early_stopping': False,
                 'random_state': self.cfg['random_state']}, parallel_jobs=grid_workers)
            gb.fit(Xs_tr, y_tr.values)   # HistGB uses class_weight internally
            preds = gb.predict(Xs_te)
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro', zero_division=0)
            trained['HistGradBoost'] = gb
            self.results['HistGradBoost'] = {'accuracy': acc, 'f1': f1, 'preds': preds, 'params': gbp}
            log.info(f"  Test accuracy: {acc:.4f}  F1-macro: {f1:.4f}  |  {elapsed(t0)}")
        else:
            log.info("[2/6] HistGradientBoosting skipped by active-model policy")

        # 3. XGBoost — GPU only when n_train ≥ 3k (GPU launch overhead dominates below)
        if HAS_XGB and _is_model_enabled('XGBoost', self.cfg):
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
                xgb_parallel = grid_workers
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
        elif not HAS_XGB:
            log.warning("[3/6] XGBoost not installed")
        else:
            log.info("[3/6] XGBoost skipped by active-model policy")

        # 4. LightGBM — same GPU break-even
        if HAS_LGB and _is_model_enabled('LightGBM', self.cfg):
            n_train      = len(Xs_tr)
            _GPU_BREAK_EVEN = 3000
            use_lgb_gpu  = self.use_gpu and (n_train >= _GPU_BREAK_EVEN)
            lgb_device   = 'gpu' if use_lgb_gpu else 'cpu'
            lgb_parallel = 1 if use_lgb_gpu else grid_workers
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
        elif not HAS_LGB:
            log.warning("[4/6] LightGBM not installed")
        else:
            log.info("[4/6] LightGBM skipped by active-model policy")

        # 5. CatBoost
        if HAS_CB and _is_model_enabled('CatBoost', self.cfg):
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
        elif not HAS_CB:
            log.info("[5/6] CatBoost not installed  (pip install catboost)")
        else:
            log.info("[5/6] CatBoost skipped by active-model policy")

        # ── NEW 9: MiniROCKET ─────────────────────────────────────────────────
        if _is_model_enabled('MiniROCKET', self.cfg):
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
        else:
            log.info("[6/6] MiniROCKET skipped by active-model policy")

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
            ens_weights = _derive_ensemble_weights(self.results, list(ens_models.keys()), self.cfg)
            ens = ManualSoftVoter(ens_models, weights=ens_weights)
            preds = ens.predict(Xs_te)
            acc = accuracy_score(y_te, preds)
            f1  = f1_score(y_te, preds, average='macro', zero_division=0)
            trained['Ensemble'] = ens
            self.ensemble_weights_ = ens_weights
            self.results['Ensemble'] = {
                'accuracy': acc, 'f1': f1, 'preds': preds,
                'params': {'weights': ens_weights}
            }
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


def _clone_model_for_refit(model):
    try:
        return type(model)(**model.get_params())
    except Exception:
        return clone(model)


def _fit_estimator_optional_weight(model, Xs, ys, sample_weight=None):
    if sample_weight is None:
        model.fit(Xs, ys)
        return model
    try:
        model.fit(Xs, ys, sample_weight=sample_weight)
    except TypeError:
        model.fit(Xs, ys)
    return model


def _crossfit_probabilities(model,
                            Xs: np.ndarray,
                            y: np.ndarray,
                            event_meta: 'pd.DataFrame | None' = None,
                            sample_weight: np.ndarray | None = None,
                            n_splits: int = 5) -> dict:
    y = np.asarray(y, dtype=np.int64)
    pkf = PurgedKFold(n_splits=min(int(max(2, n_splits)), max(2, len(y) // 20)),
                      embargo_pct=CONFIG.get('purged_embargo_pct', 0.01))
    oof_proba = np.full((len(y), len(np.unique(y))), np.nan, dtype=np.float64)
    oof_preds = np.full(len(y), -1, dtype=np.int64)
    fold_scores = []
    for tr_idx, te_idx in pkf.split(Xs, event_meta=event_meta):
        if len(tr_idx) < 20 or len(te_idx) < 5:
            continue
        refit = _clone_model_for_refit(model)
        sw = sample_weight[tr_idx] if sample_weight is not None else None
        _fit_estimator_optional_weight(refit, Xs[tr_idx], y[tr_idx], sw)
        try:
            proba = np.asarray(refit.predict_proba(Xs[te_idx]), dtype=np.float64)
        except Exception:
            preds = np.asarray(refit.predict(Xs[te_idx])).ravel().astype(np.int64)
            proba = np.full((len(preds), 3), 1.0 / 3.0, dtype=np.float64)
            proba[np.arange(len(preds)), preds] = 1.0
        preds = np.asarray(proba.argmax(axis=1), dtype=np.int64)
        oof_proba[te_idx] = proba
        oof_preds[te_idx] = preds
        fold_scores.append(float(f1_score(y[te_idx], preds, average='macro', zero_division=0)))
    valid = np.all(np.isfinite(oof_proba), axis=1) & (oof_preds >= 0)
    return {
        'proba': oof_proba[valid],
        'preds': oof_preds[valid],
        'y': y[valid],
        'valid_mask': valid,
        'fold_f1': fold_scores,
    }


def _forward_returns_from_close(close: pd.Series, index: pd.Index) -> pd.Series:
    aligned = close.reindex(index).ffill()
    return aligned.pct_change().shift(-1).reindex(index).fillna(0.0)


def _split_tuning_and_conformal(proba: np.ndarray,
                                y_true: np.ndarray,
                                returns: np.ndarray,
                                index_like=None) -> dict:
    proba = _clip_prob_matrix(np.asarray(proba, dtype=np.float64))
    y_true = np.asarray(y_true, dtype=np.int64)
    returns = np.asarray(returns, dtype=np.float64)
    n = min(len(proba), len(y_true), len(returns))
    proba = proba[:n]
    y_true = y_true[:n]
    returns = returns[:n]
    idx = _coerce_index_like(index_like)
    if len(idx) != n:
        idx = pd.RangeIndex(n)

    policy_min_rows = int(CONFIG.get('threshold_tuning_min_rows', 12) or 12)
    conformal_min_rows = int(CONFIG.get('conformal_min_cal_rows', 20) or 20)
    if n < max(policy_min_rows + conformal_min_rows, 18):
        return {
            'policy_proba': proba,
            'policy_y': y_true,
            'policy_returns': returns,
            'policy_index': idx,
            'conformal_proba': proba,
            'conformal_y': y_true,
            'conformal_returns': returns,
            'conformal_index': idx,
            'split_policy': 'shared',
        }

    conformal_rows = max(conformal_min_rows, int(n * 0.35))
    policy_rows = n - conformal_rows
    if policy_rows < policy_min_rows:
        policy_rows = policy_min_rows
        conformal_rows = n - policy_rows
    if policy_rows < policy_min_rows or conformal_rows < conformal_min_rows:
        return {
            'policy_proba': proba,
            'policy_y': y_true,
            'policy_returns': returns,
            'policy_index': idx,
            'conformal_proba': proba,
            'conformal_y': y_true,
            'conformal_returns': returns,
            'conformal_index': idx,
            'split_policy': 'shared',
        }
    return {
        'policy_proba': proba[:policy_rows],
        'policy_y': y_true[:policy_rows],
        'policy_returns': returns[:policy_rows],
        'policy_index': idx[:policy_rows],
        'conformal_proba': proba[policy_rows:],
        'conformal_y': y_true[policy_rows:],
        'conformal_returns': returns[policy_rows:],
        'conformal_index': idx[policy_rows:],
        'split_policy': 'disjoint',
    }


def _evaluate_candidate_from_probabilities(name: str,
                                           family: str,
                                           eval_proba: np.ndarray,
                                           eval_y: np.ndarray,
                                           eval_returns: np.ndarray,
                                           latest_proba: np.ndarray,
                                           tuning_proba: np.ndarray,
                                           tuning_y: np.ndarray,
                                           tuning_returns: np.ndarray,
                                           calibration_report: dict,
                                           cpcv_proxy: float = 0.0,
                                           eval_index=None,
                                           tuning_index=None,
                                           close: pd.Series | None = None) -> dict:
    tx_cost = CONFIG.get('transaction_cost_bps', 10) / 10000.0
    split_ctx = _split_tuning_and_conformal(tuning_proba, tuning_y, tuning_returns, tuning_index)
    policy_prob = split_ctx['policy_proba']
    policy_y = np.asarray(split_ctx['policy_y'], dtype=np.int64)
    policy_ret = np.asarray(split_ctx['policy_returns'], dtype=np.float64)
    policy_idx = _coerce_index_like(split_ctx.get('policy_index'))
    conformal_prob = split_ctx['conformal_proba']
    conformal_y = np.asarray(split_ctx['conformal_y'], dtype=np.int64)
    conformal_idx = _coerce_index_like(split_ctx.get('conformal_index'))

    thresholds = _tune_probability_thresholds(policy_prob, policy_y, policy_ret, tx_cost)
    edge_map = _estimate_class_edge_map(policy_y, policy_ret)
    class_edge_bins = _build_class_edge_bins(policy_prob, policy_y, policy_ret)
    threshold_regime_diagnostics = _build_threshold_regime_diagnostics(
        policy_prob,
        policy_y,
        policy_ret,
        thresholds,
        edge_map,
        tx_cost,
        index_like=policy_idx,
        close=close,
    )

    execution_cp = None
    conf_stats = {}
    if len(conformal_y) >= int(CONFIG.get('conformal_min_cal_rows', 20)):
        conf_method = str(CONFIG.get('conformal_method', 'aps') or 'aps').lower()
        tuned = _tune_conformal_policy(
            conformal_prob,
            conformal_y,
            n_classes=eval_proba.shape[1],
        ) if CONFIG.get('conformal_tuning_enabled', True) else {
            'params': {
                'alpha': float(CONFIG.get('conformal_alpha', 0.10) or 0.10),
                'lam': float(CONFIG.get('conformal_lambda', 0.01) or 0.01),
                'kreg': int(CONFIG.get('conformal_kreg', 1) or 1),
                'method': conf_method,
                'mondrian': bool(CONFIG.get('conformal_mondrian_enabled', True)),
            },
            'stats': {},
            'evaluated': [],
            'mondrian_supported': False,
        }
        params = dict(tuned.get('params', {}) or {})
        execution_cp = _fit_execution_conformal(
            conformal_prob,
            conformal_y,
            alpha=float(params.get('alpha', CONFIG.get('conformal_alpha', 0.10))),
            lam=float(params.get('lam', CONFIG.get('conformal_lambda', 0.01))),
            kreg=int(params.get('kreg', CONFIG.get('conformal_kreg', 1))),
            method=str(params.get('method', conf_method) or conf_method),
            mondrian=bool(params.get('mondrian', False)),
        )
        conf_stats = dict(tuned.get('stats', {}) or {})
        if execution_cp is not None:
            conf_stats.setdefault('method', execution_cp.get('method', conf_method))
            conf_stats.setdefault('selected_alpha', float(execution_cp.get('alpha', CONFIG.get('conformal_alpha', 0.10))))
            conf_stats.setdefault('selected_lambda', float(execution_cp.get('lam', CONFIG.get('conformal_lambda', 0.01))))
            conf_stats.setdefault('selected_kreg', int(execution_cp.get('kreg', CONFIG.get('conformal_kreg', 1))))
            conf_stats['mondrian'] = bool(execution_cp.get('mondrian', False))
            conf_stats['mondrian_supported'] = bool(execution_cp.get('mondrian_supported', False))
            conf_stats['mondrian_fallback_reason'] = execution_cp.get('mondrian_fallback_reason')
            conf_stats['predicted_class_counts'] = execution_cp.get('predicted_class_counts', {})
            conf_stats['nonconformity_summary'] = _conformal_nonconformity_summary(
                conformal_prob,
                conformal_y,
                method=str(execution_cp.get('method', conf_method) or conf_method),
                lam=float(execution_cp.get('lam', CONFIG.get('conformal_lambda', 0.01)) or 0.0),
                kreg=int(execution_cp.get('kreg', CONFIG.get('conformal_kreg', 1)) or 0),
            )
        conf_stats.setdefault('selected_tuning_params', dict(tuned.get('stats', {}).get('selected_tuning_params', {}) or {}))
        conf_stats.setdefault('tuning_candidates_evaluated', list(tuned.get('evaluated', []) or []))
        if not conf_stats and execution_cp is not None:
            conf_stats = execution_cp['global_predictor'].empirical_coverage(conformal_prob, conformal_y)
    qhat_value = None
    if execution_cp is not None and execution_cp.get('global_predictor') is not None:
        qhat_value = float(execution_cp['global_predictor'].qhat)
    sorted_conf = np.sort(conformal_prob, axis=1) if len(conformal_prob) else np.empty((0, 0))
    mean_top1 = float(np.mean(sorted_conf[:, -1])) if sorted_conf.size else None
    mean_margin = float(np.mean(sorted_conf[:, -1] - sorted_conf[:, -2])) if sorted_conf.shape[1] >= 2 else None
    conf_stats['threshold_tuning_rows'] = int(len(policy_y))
    conf_stats['calibration_split_rows'] = int(len(conformal_y))
    conf_stats['conformal_split_policy'] = str(split_ctx.get('split_policy', 'shared') or 'shared')
    conf_stats['qhat'] = qhat_value
    conf_stats['qhat_vs_mean_top1_gap'] = None if qhat_value is None or mean_top1 is None else float(qhat_value - mean_top1)
    conf_stats['qhat_vs_mean_margin_gap'] = None if qhat_value is None or mean_margin is None else float(qhat_value - mean_margin)
    conf_stats = _assess_conformal_usability(conf_stats, n_classes=eval_proba.shape[1])

    pred_labels = []
    abstain_reasons = {'margin_gate': 0, 'edge_gate': 0, 'conformal_gate': 0}
    eval_proba = _clip_prob_matrix(eval_proba)
    eval_conformal_set_sizes = []
    for row in eval_proba:
        conformal_ok = True
        if execution_cp is not None:
            pred_set, _ = _execution_conformal_predict_set(execution_cp, row)
            eval_conformal_set_sizes.append(int(len(pred_set)))
            conformal_ok = (len(pred_set) == 1) or (_prob_margin(row) > float(thresholds.get('margin_threshold', 0.0)))
            if not conformal_ok:
                abstain_reasons['conformal_gate'] += 1
        else:
            eval_conformal_set_sizes.append(1)
        label, reason = _trade_signal_from_policy(row, thresholds, edge_map, tx_cost, uncertainty_ok=conformal_ok)
        if reason:
            abstain_reasons[reason] = abstain_reasons.get(reason, 0) + 1
        pred_labels.append(label)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    strat_metrics = _strategy_metrics_from_predictions(pred_labels, eval_returns, tx_cost)
    wf_stats = _block_sharpe_stats(strat_metrics['strategy_returns'], block_size=CONFIG.get('router_block_size', 21))
    eval_metrics = _compute_classification_metrics(eval_y, pred_labels, n_classes=eval_proba.shape[1])
    pred_summary = _prediction_summary(eval_y, pred_labels, n_classes=eval_proba.shape[1])

    latest_row = _clip_prob_matrix(np.asarray(latest_proba, dtype=np.float64))[0]
    latest_uncertainty_ok = True
    latest_conformal = {}
    if execution_cp is not None:
        pred_set, latest_conformal = _execution_conformal_predict_set(execution_cp, latest_row)
        latest_uncertainty_ok = (len(pred_set) == 1) or (_prob_margin(latest_row) > float(thresholds.get('margin_threshold', 0.0)))
    latest_label, latest_reason = _trade_signal_from_policy(
        latest_row, thresholds, edge_map, tx_cost, uncertainty_ok=latest_uncertainty_ok)
    lmap = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    latest_buy_edge = float(latest_row[2] * float(edge_map.get('BUY', 0.0)))
    latest_sell_edge = float(latest_row[0] * float(edge_map.get('SELL', 0.0)))
    latest_margin = _prob_margin(latest_row)
    latest_signal = {
        'signal': lmap[int(latest_label)],
        'label': int(latest_label),
        'confidence': float(latest_row[int(latest_label)]),
        'probabilities': {lmap[i]: float(latest_row[i]) for i in range(len(latest_row))},
        'model_used': name,
        'probability_margin': float(latest_margin),
        'edge_minus_cost': {
            'SELL': float(latest_sell_edge - float(thresholds.get('cost_gate', tx_cost))),
            'BUY': float(latest_buy_edge - float(thresholds.get('cost_gate', tx_cost))),
        },
        'expected_edge': {
            'SELL': latest_sell_edge,
            'BUY': latest_buy_edge,
        },
    }
    latest_signal.update(latest_conformal)
    if latest_reason:
        latest_signal['abstain_reason'] = latest_reason

    sharpness = float((conf_stats or {}).get('sharpness', 0.0) or 0.0)
    robust_score, score_components = _compute_robust_score({
        'median_wf_sharpe': wf_stats.get('median_wf_sharpe', 0.0),
        'positive_wf_share': wf_stats.get('positive_wf_share', 0.0),
        'cpcv_p5_sharpe': cpcv_proxy,
        'ece': calibration_report.get('post_calibration_ece', calibration_report.get('ece', 1.0)),
        'conformal_sharpness': sharpness,
        'annualized_trade_rate': strat_metrics.get('annualized_trade_rate', 0.0),
    })
    eval_idx = _coerce_index_like(eval_index)
    holdout_backtest = _candidate_holdout_backtest(
        name=name,
        family=family,
        pred_labels=pred_labels,
        returns=eval_returns,
        eval_index=eval_idx,
        conformal_set_sizes=eval_conformal_set_sizes,
        strategy_return_source='selected_candidate_holdout_backtest',
    )
    return {
        'name': name,
        'family': family,
        'latest_signal': latest_signal,
        'calibration': calibration_report,
        'decision_policy': {
            **thresholds,
            'edge_map': {k: round(float(v), 6) for k, v in edge_map.items()},
            'class_edge_bins': class_edge_bins,
            'threshold_regime_diagnostics': threshold_regime_diagnostics,
            'abstain_reason_counts': abstain_reasons,
        },
        'conformal': conf_stats,
        'evaluation': {
            **eval_metrics,
            **pred_summary,
            'macro_pr_auc': float((calibration_report or {}).get('pr_auc_macro', 0.0) or 0.0),
        },
        'selection': {
            'robust_score': robust_score,
            'components': score_components,
            'wf_block_sharpes': wf_stats.get('wf_block_sharpes', []),
            'eval_sharpe': float(strat_metrics.get('sharpe', 0.0)),
            'annualized_trade_rate': float(strat_metrics.get('annualized_trade_rate', 0.0)),
            'deployment_rank_metric': CONFIG.get('deployment_ranking_metric', 'robust_score'),
        },
        'strategy_returns': [float(x) for x in np.asarray(strat_metrics.get('strategy_returns', []), dtype=np.float64).tolist()],
        'eval_labels': pred_labels,
        'eval_proba': eval_proba,
        'eval_returns': np.asarray(eval_returns, dtype=np.float64),
        'eval_y': np.asarray(eval_y, dtype=np.int64),
        'eval_conformal_set_sizes': [int(x) for x in eval_conformal_set_sizes],
        'holdout_backtest': holdout_backtest,
        'eval_index': list(eval_idx),
        'tuning_index': list(policy_idx),
        'conformal_index': list(conformal_idx),
    }


def _coerce_index_like(value) -> pd.Index:
    if value is None:
        return pd.Index([])
    if isinstance(value, pd.Index):
        return value
    try:
        return pd.Index(value)
    except Exception:
        return pd.Index([])


def _preserve_conformal_result(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _collect_tree_candidates(tree, close: pd.Series) -> list:
    candidates = []
    last_xs = tree._transform_latest(tree.X.iloc[[-1]])
    ret_train = _forward_returns_from_close(close, tree.X_tr.index)
    ret_eval = _forward_returns_from_close(close, tree.X_te.index)
    sw = getattr(tree, 'sample_weights_', None)
    for name, model in tree.models.items():
        if _model_family(name) != 'tree_family' or name in {'Ensemble', 'MiniROCKET'}:
            continue
        if not _is_model_enabled(name, CONFIG):
            continue
        try:
            raw_eval = np.asarray(model.predict_proba(tree.Xs_te), dtype=np.float64)
            raw_latest = np.asarray(model.predict_proba(last_xs), dtype=np.float64)
        except Exception:
            continue
        cf = _crossfit_probabilities(
            model,
            tree.Xs_tr,
            tree.y_tr.values,
            event_meta=getattr(tree, 'train_event_meta_', None),
            sample_weight=sw,
            n_splits=CONFIG.get('cv_folds', 5),
        )
        if len(cf['y']) < 20:
            cal_report = _build_calibration_report(raw_eval, raw_eval, tree.y_te.values, calibrator_type='identity')
            cal_eval = raw_eval
            cal_latest = raw_latest
            cpcv_proxy = 0.0
            tune_proba = raw_eval
            tune_y = tree.y_te.values
            tune_ret = ret_eval.values
        else:
            calibrator, _, cal_report = _fit_probability_calibrator(
                cf['proba'], cf['y'],
                method=CONFIG.get('calibration_tree_method', 'auto'),
                min_rows=CONFIG.get('global_family_calibration_min_rows', 80),
            )
            cal_eval = calibrator.transform(raw_eval) if calibrator is not None else raw_eval
            cal_latest = calibrator.transform(raw_latest) if calibrator is not None else raw_latest
            fold_returns = ret_train.values[cf['valid_mask']]
            cpcv_proxy = _block_sharpe_stats(
                _strategy_metrics_from_predictions(cf['preds'], fold_returns, CONFIG.get('transaction_cost_bps', 10) / 10000.0)['strategy_returns'],
                block_size=CONFIG.get('router_block_size', 21),
            ).get('median_wf_sharpe', 0.0)
            tune_proba = calibrator.transform(cf['proba']) if calibrator is not None else cf['proba']
            tune_y = cf['y']
            tune_ret = fold_returns
        candidate = _evaluate_candidate_from_probabilities(
            name=name,
            family='tree_family',
            eval_proba=cal_eval,
            eval_y=tree.y_te.values,
            eval_returns=ret_eval.values,
            latest_proba=cal_latest,
            tuning_proba=tune_proba,
            tuning_y=tune_y,
            tuning_returns=tune_ret,
            calibration_report=cal_report,
            cpcv_proxy=float(cpcv_proxy or 0.0),
            eval_index=tree.X_te.index,
            tuning_index=tree.X_tr.index[cf['valid_mask']] if len(cf['y']) >= 20 else tree.X_te.index,
            close=close,
        )
        candidates.append(candidate)
    return candidates


def _collect_dl_candidates(dl_trainers, close: pd.Series) -> list:
    candidates = []
    for trainer in dl_trainers:
        if not _is_model_enabled(getattr(trainer, 'name', ''), CONFIG):
            continue
        if getattr(trainer, 'te_proba', None) is None or getattr(trainer, 'y_te_vals', None) is None:
            continue
        eval_idx = _coerce_index_like(getattr(trainer, 'y_te_index', None))
        cal_idx = _coerce_index_like(getattr(trainer, 'y_cal_index', None))
        if len(eval_idx) == 0 or len(cal_idx) == 0:
            continue
        ret_eval = _forward_returns_from_close(close, eval_idx).values
        ret_cal = _forward_returns_from_close(close, cal_idx).values
        latest = trainer.predict_latest()
        candidate = _evaluate_candidate_from_probabilities(
            name=trainer.name,
            family='dl_family',
            eval_proba=np.asarray(trainer.te_proba, dtype=np.float64),
            eval_y=np.asarray(trainer.y_te_vals, dtype=np.int64),
            eval_returns=ret_eval,
            latest_proba=_signal_probability_vector(latest),
            tuning_proba=np.asarray(trainer.cal_proba if trainer.cal_proba is not None else trainer.te_proba, dtype=np.float64),
            tuning_y=np.asarray(trainer.y_cal_vals if trainer.y_cal_vals is not None else trainer.y_te_vals, dtype=np.int64),
            tuning_returns=ret_cal if len(ret_cal) else ret_eval,
            calibration_report=getattr(trainer, 'calibration_report_', {}) or _build_calibration_report(
                np.asarray(trainer.te_proba, dtype=np.float64),
                np.asarray(trainer.te_proba, dtype=np.float64),
                np.asarray(trainer.y_te_vals, dtype=np.int64),
                calibrator_type='identity',
            ),
            cpcv_proxy=0.0,
            eval_index=eval_idx,
            tuning_index=cal_idx if len(cal_idx) else eval_idx,
            close=close,
        )
        candidate['is_collapsed'] = bool(getattr(trainer, 'is_collapsed', False))
        candidates.append(candidate)
    return candidates


def _collect_stack_candidate(tree, meta_signal, close: pd.Series) -> list:
    if not meta_signal:
        return []
    ctx = getattr(tree, '_meta_calibration_context', None)
    if not ctx:
        return []
    eval_index = _coerce_index_like(ctx.get('eval_index', None))
    if len(eval_index) == 0:
        eval_index = tree.y_te.index[-len(ctx.get('y_eval', [])):]
    eval_returns = _forward_returns_from_close(close, eval_index).values
    calibration = _build_calibration_report(
        np.asarray(ctx.get('proba_eval'), dtype=np.float64),
        np.asarray(ctx.get('proba_eval'), dtype=np.float64),
        np.asarray(ctx.get('y_eval'), dtype=np.int64),
        calibrator_type='stack_precalibrated',
    )
    candidate = _evaluate_candidate_from_probabilities(
        name=str(meta_signal.get('model_used', 'Meta-Stack')),
        family='stack_family',
        eval_proba=np.asarray(ctx.get('proba_eval'), dtype=np.float64),
        eval_y=np.asarray(ctx.get('y_eval'), dtype=np.int64),
        eval_returns=eval_returns,
        latest_proba=np.asarray(ctx.get('latest_proba'), dtype=np.float64),
        tuning_proba=np.asarray(ctx.get('proba_eval'), dtype=np.float64),
        tuning_y=np.asarray(ctx.get('y_eval'), dtype=np.int64),
        tuning_returns=eval_returns,
        calibration_report=calibration,
        cpcv_proxy=0.0,
        eval_index=eval_index,
        tuning_index=eval_index,
        close=close,
    )
    return [candidate]


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    try:
        val = series.autocorr(lag=lag)
        return 0.0 if pd.isna(val) else float(val)
    except Exception:
        return 0.0


def _sector_numeric(sector: str) -> float:
    text = str(sector or '').strip().upper()
    if not text:
        return 0.0
    return float(sum(ord(ch) for ch in text) % 997) / 997.0


def _router_feature_row(df: pd.DataFrame,
                        market: pd.DataFrame,
                        fundamentals: dict,
                        as_of,
                        recent_ece: float,
                        recent_set_size: float,
                        regime_value: int = 1) -> np.ndarray:
    idx = pd.Index(df.index)
    if as_of not in idx:
        as_of = idx[idx.get_indexer([as_of], method='ffill')[0]]
    hist = df.loc[:as_of].copy()
    close = hist['Close'].astype(float)
    ret = close.pct_change().dropna()
    high = hist.get('High', close)
    low = hist.get('Low', close)
    atr = (high.sub(low).abs()).rolling(14).mean().iloc[-1] if len(hist) else 0.0
    atr_pct = float(atr / (close.iloc[-1] + 1e-10)) if len(close) else 0.0
    if len(close) >= 10:
        y = close.tail(min(21, len(close))).values
        x = np.arange(len(y), dtype=float)
        trend = float(np.polyfit(x, y, deg=1)[0] / (np.mean(y) + 1e-10))
    else:
        trend = 0.0
    vol = float(ret.tail(21).std() * np.sqrt(252)) if len(ret) else 0.0
    vol_stability = 0.0
    if 'Volume' in hist and len(hist['Volume']) >= 5:
        v = hist['Volume'].astype(float).tail(21)
        vol_stability = float(1.0 - np.clip(v.std() / (v.mean() + 1e-10), 0.0, 1.0))
    sector = (
        (fundamentals.get('fundamentals', {}) or {}).get('sector')
        or fundamentals.get('sector')
        or ''
    )
    quote_type = str((fundamentals.get('fundamentals', {}) or {}).get('quoteType', '') or fundamentals.get('quoteType', '')).upper()
    is_etf = 1.0 if 'ETF' in quote_type else 0.0
    return np.array([
        vol,
        atr_pct,
        trend,
        _safe_autocorr(ret.tail(63), 1),
        _safe_autocorr(ret.tail(63), 5),
        vol_stability,
        float(regime_value),
        _sector_numeric(sector),
        is_etf,
        float(recent_ece),
        float(recent_set_size),
    ], dtype=np.float64)


def _route_family(candidates: list,
                  df: pd.DataFrame,
                  market: pd.DataFrame,
                  fundamentals: dict,
                  regime_sig: dict | None = None) -> dict:
    def _router_result(selected: dict | None,
                       global_champion: dict,
                       eligible_global_champion: dict | None,
                       leaderboard: list,
                       selection_frequency: dict,
                       confidence: float,
                       fallback: bool,
                       enabled: bool,
                       router_status: str = 'inactive_fallback',
                       fallback_reason: str | None = None,
                       family_probabilities: dict | None = None) -> dict:
        selected_sel = (selected or {}).get('selection', {}) or {}
        global_sel = (global_champion.get('selection', {}) or {})
        selected_components = (selected_sel.get('components', {}) or {})
        global_components = (global_sel.get('components', {}) or {})
        selected_eval = None if selected is None else float(selected_sel.get('eval_sharpe', 0.0) or 0.0)
        global_eval = float(global_sel.get('eval_sharpe', 0.0) or 0.0)
        selected_cpcv = None if selected is None else float(selected_components.get('cpcv_p5_sharpe', 0.0) or 0.0)
        global_cpcv = float(global_components.get('cpcv_p5_sharpe', 0.0) or 0.0)
        selected_robust = None if selected is None else float(selected_sel.get('robust_score', 0.0) or 0.0)
        global_robust = float(global_sel.get('robust_score', 0.0) or 0.0)
        return {
            'chosen_family': None if selected is None else selected['family'],
            'chosen_model': None if selected is None else selected['name'],
            'confidence': float(confidence),
            'fallback_to_global': bool(fallback),
            'fallback_reason': fallback_reason,
            'global_champion_family': global_champion['family'],
            'global_champion_model': global_champion['name'],
            'eligible_global_champion_family': None if eligible_global_champion is None else eligible_global_champion['family'],
            'eligible_global_champion_model': None if eligible_global_champion is None else eligible_global_champion['name'],
            'family_leaderboard': leaderboard,
            'selection_frequency': selection_frequency,
            'enabled': bool(enabled),
            'router_status': str(router_status or 'inactive_fallback'),
            'routing_actionable': bool((router_status or '') == 'active' and selected is not None),
            'router_family_probabilities': family_probabilities or {},
            'selected_eval_sharpe': selected_eval,
            'global_eval_sharpe': global_eval,
            'selected_cpcv_p5_sharpe': selected_cpcv,
            'global_cpcv_p5_sharpe': global_cpcv,
            'selected_robust_score': selected_robust,
            'global_champion_robust_score': global_robust,
            'delta_eval_sharpe': None if selected_eval is None else float(selected_eval - global_eval),
            'delta_cpcv_p5_sharpe': None if selected_cpcv is None else float(selected_cpcv - global_cpcv),
            'delta_robust_score': None if selected_robust is None else float(selected_robust - global_robust),
            'routing_improved_vs_global': bool(
                selected_eval is not None and selected_cpcv is not None and (
                    (selected_eval > global_eval + 1e-9)
                    or (selected_cpcv > global_cpcv + 1e-9)
                )
            ),
        }

    all_family_best = {}
    for candidate in candidates:
        fam = candidate.get('family')
        if fam not in {'tree_family', 'dl_family', 'stack_family'}:
            continue
        if fam not in all_family_best or _candidate_sort_key(candidate) > _candidate_sort_key(all_family_best[fam]):
            all_family_best[fam] = candidate
    if not all_family_best:
        return {}
    global_champion = max(all_family_best.values(), key=_candidate_sort_key)
    family_best = {
        fam: cand for fam, cand in all_family_best.items()
        if bool(cand.get('deployment_eligible', False))
    }
    eligible_global_champion = max(family_best.values(), key=_candidate_sort_key) if family_best else None
    leaderboard = [
        {
            'family': fam,
            'model': cand['name'],
            'robust_score': float(cand['selection']['robust_score']),
            'eval_sharpe': float((cand.get('selection', {}) or {}).get('eval_sharpe', 0.0) or 0.0),
            'cpcv_p5_sharpe': float((((cand.get('selection', {}) or {}).get('components', {}) or {}).get('cpcv_p5_sharpe', 0.0) or 0.0)),
            'deployment_eligible': bool(cand.get('deployment_eligible', False)),
            'reference_champion': bool(cand.get('name') == global_champion.get('name')),
            'eligible_global_champion': bool(eligible_global_champion is not None and cand.get('name') == eligible_global_champion.get('name')),
        }
        for fam, cand in sorted(all_family_best.items(), key=lambda item: _candidate_sort_key(item[1]), reverse=True)
    ]
    if not family_best:
        return _router_result(
            selected=None,
            global_champion=global_champion,
            eligible_global_champion=eligible_global_champion,
            leaderboard=leaderboard,
            selection_frequency={global_champion['family']: 1.0},
            confidence=0.0,
            fallback=True,
            enabled=True,
            router_status='fallback',
            fallback_reason='no_eligible_family',
        )
    if not CONFIG.get('router_enabled', True) or len(family_best) < 2:
        return _router_result(
            selected=eligible_global_champion,
            global_champion=global_champion,
            eligible_global_champion=eligible_global_champion,
            leaderboard=leaderboard,
            selection_frequency={(eligible_global_champion or global_champion)['family']: 1.0},
            confidence=1.0 if len(family_best) == 1 else 0.0,
            fallback=len(family_best) < 2,
            enabled=False,
            router_status='inactive_fallback',
            fallback_reason='router_disabled' if not CONFIG.get('router_enabled', True) else 'single_eligible_family',
        )

    aligned = pd.DataFrame({
        fam: pd.Series(cand['strategy_returns'], index=pd.Index(cand['eval_index']))
        for fam, cand in family_best.items()
    }).dropna(how='any')
    block_size = int(CONFIG.get('router_block_size', 21))
    min_blocks = int(CONFIG.get('router_min_blocks', 6))
    if len(aligned) < block_size * 2:
        return _router_result(
            selected=eligible_global_champion,
            global_champion=global_champion,
            eligible_global_champion=eligible_global_champion,
            leaderboard=leaderboard,
            selection_frequency={(eligible_global_champion or global_champion)['family']: 1.0},
            confidence=0.0,
            fallback=True,
            enabled=True,
            router_status='insufficient_coverage',
            fallback_reason='insufficient_router_rows',
        )

    X_router, y_router = [], []
    families = list(aligned.columns)
    selection_history = []
    regime_value = 0 if str((regime_sig or {}).get('regime', '')).upper() == 'BEAR' else 1
    for start in range(0, len(aligned), block_size):
        block = aligned.iloc[start:start + block_size]
        if len(block) < 5:
            continue
        sharpes = {
            fam: float(block[fam].mean() / (block[fam].std() + 1e-10) * np.sqrt(252))
            for fam in families
        }
        winner = max(sharpes, key=sharpes.get)
        winner_candidate = family_best[winner]
        feat = _router_feature_row(
            df, market, fundamentals,
            block.index[0],
            recent_ece=float((winner_candidate.get('calibration', {}) or {}).get('post_calibration_ece', 0.0)),
            recent_set_size=float((winner_candidate.get('conformal', {}) or {}).get('avg_set_size', 1.0)),
            regime_value=regime_value,
        )
        X_router.append(feat)
        y_router.append(families.index(winner))
        selection_history.append(winner)

    if len(y_router) < min_blocks or len(set(y_router)) < 2:
        return _router_result(
            selected=eligible_global_champion,
            global_champion=global_champion,
            eligible_global_champion=eligible_global_champion,
            leaderboard=leaderboard,
            selection_frequency={fam: float(selection_history.count(fam) / max(1, len(selection_history))) for fam in set(selection_history)},
            confidence=0.0,
            fallback=True,
            enabled=True,
            router_status='insufficient_coverage',
            fallback_reason='router_class_coverage',
        )

    router = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=CONFIG.get('random_state', 42),
        class_weight='balanced_subsample',
    )
    X_router = np.asarray(X_router, dtype=np.float64)
    y_router = np.asarray(y_router, dtype=np.int64)
    router.fit(X_router, y_router)
    latest_feat = _router_feature_row(
        df, market, fundamentals, df.index[-1],
        recent_ece=float(((eligible_global_champion or global_champion).get('calibration', {}) or {}).get('post_calibration_ece', 0.0)),
        recent_set_size=float(((eligible_global_champion or global_champion).get('conformal', {}) or {}).get('avg_set_size', 1.0)),
        regime_value=regime_value,
    ).reshape(1, -1)
    proba = router.predict_proba(latest_feat)[0]
    pred_idx = int(np.argmax(proba))
    chosen_family = families[pred_idx]
    confidence = float(np.max(proba))
    fallback = confidence < float(CONFIG.get('router_confidence_floor', 0.55))
    selected = eligible_global_champion if fallback else family_best[chosen_family]
    return _router_result(
        selected=selected,
        global_champion=global_champion,
        eligible_global_champion=eligible_global_champion,
        leaderboard=leaderboard,
        selection_frequency={fam: float(selection_history.count(fam) / max(1, len(selection_history))) for fam in set(selection_history)},
        confidence=confidence,
        fallback=bool(fallback),
        enabled=True,
        router_status='low_confidence' if fallback else 'active',
        fallback_reason='low_confidence' if fallback else None,
        family_probabilities={fam: float(proba[i]) for i, fam in enumerate(families)},
    )


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
class PatchTSTClassifier(nn.Module):
    """Patch-based long-horizon classifier inspired by PatchTST."""
    def __init__(self, input_size, patch_len=8, stride=4, d_model=96,
                 nhead=4, n_layers=3, dropout=0.20, n_classes=3, seq_len=60):
        super().__init__()
        self.patch_len = max(2, int(patch_len))
        self.stride = max(1, int(stride))
        self.seq_len = int(seq_len)
        self.input_bn = nn.BatchNorm1d(input_size)
        self.patch_proj = nn.Linear(input_size * self.patch_len, d_model)
        max_patches = max(1, 1 + max(0, self.seq_len - self.patch_len) // self.stride)
        self.pos_embed = nn.Embedding(max_patches, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        h1 = max(n_classes * 4, min(128, d_model))
        h2 = max(n_classes * 2, h1 // 2)
        self.head = nn.Sequential(
            nn.Linear(d_model, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Linear(h2, n_classes))

    def forward(self, x):
        b, s, f = x.shape
        x = self.input_bn(x.reshape(b * s, f)).reshape(b, s, f)
        if s < self.patch_len:
            x = F.pad(x, (0, 0, self.patch_len - s, 0))
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(b, patches.shape[1], self.patch_len * f)
        pos = torch.arange(patches.shape[1], device=x.device).unsqueeze(0).expand(b, -1)
        tokens = self.patch_proj(patches) + self.pos_embed(pos)
        tokens = self.norm(self.encoder(tokens))
        return self.head(tokens.mean(dim=1))


class _TiDEResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.20):
        super().__init__()
        inner = hidden_dim * 2
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TiDEClassifier(nn.Module):
    """Lightweight TiDE-style dense temporal classifier."""
    def __init__(self, input_size, hidden=96, n_layers=3, dropout=0.20,
                 n_classes=3, seq_len=60):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_bn = nn.BatchNorm1d(input_size)
        flat_dim = int(input_size) * self.seq_len
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Linear(int(input_size), hidden)
        self.blocks = nn.ModuleList(
            [_TiDEResidualBlock(hidden, dropout=dropout) for _ in range(max(1, int(n_layers)))]
        )
        h1 = max(n_classes * 4, min(128, hidden))
        h2 = max(n_classes * 2, h1 // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Linear(h2, n_classes))

    def forward(self, x):
        b, s, f = x.shape
        x = self.input_bn(x.reshape(b * s, f)).reshape(b, s, f)
        if s < self.seq_len:
            x = F.pad(x, (0, 0, self.seq_len - s, 0))
        elif s > self.seq_len:
            x = x[:, -self.seq_len:, :]
        flat = x.reshape(b, self.seq_len * f)
        h = self.encoder(flat) + self.context_proj(x.mean(dim=1))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


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

    def __init__(self, alpha: float = 0.10, lam: float = 0.01, kreg: int = 1,
                 method: str = 'aps'):
        self.alpha = alpha
        self.lam   = lam
        self.kreg  = kreg
        self.method = str(method or 'raps').lower()
        self.qhat  = None
        self.n_cal = 0

    def _nonconformity(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Canonical non-randomized APS/RAPS nonconformity scores."""
        scores = np.zeros(len(y), dtype=np.float64)
        for i in range(len(y)):
            order  = np.argsort(proba[i])[::-1]   # descending prob
            cumsum = 0.0
            for rank, cls in enumerate(order):
                cumsum += proba[i, cls]
                if cls == int(y[i]):
                    penalty = 0.0
                    if self.method == 'raps':
                        penalty = self.lam * max(0, rank + 1 - self.kreg)
                    scores[i] = cumsum + penalty
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
        """Return canonical APS/RAPS prediction set for a single (3,) probability vector."""
        if self.qhat is None:
            raise RuntimeError("Call calibrate() first.")
        order    = np.argsort(proba)[::-1]
        pred_set = []
        cumsum   = 0.0
        for rank, cls in enumerate(order):
            cumsum += proba[cls]
            penalty = self.lam * max(0, rank + 1 - self.kreg) if self.method == 'raps' else 0.0
            score = cumsum + penalty
            if score <= self.qhat or not pred_set:
                pred_set.append(int(cls))
            else:
                break
        return sorted(set(pred_set)) if pred_set else [int(order[0])]

    def empirical_coverage(self, proba: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate empirical coverage and average set size on hold-out."""
        pred_sets = [self.predict_set(proba[i]) for i in range(len(y))]
        return _conformal_summary_from_sets(
            pred_sets,
            np.asarray(y, dtype=np.int64),
            n_classes=int(proba.shape[1]),
            alpha=float(self.alpha),
            method=self.method,
            qhat=self.qhat,
            n_cal=self.n_cal,
        )

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
            'conformal_method':         self.method,
        }


def _cross_conformal_stats(proba: np.ndarray,
                           y_true: np.ndarray,
                           alpha: float,
                           lam: float,
                           kreg: int,
                           method: str = 'aps',
                           n_splits: int = 3,
                           mondrian: bool = False) -> dict:
    proba = _clip_prob_matrix(proba)
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(y_true) < max(12, n_splits * 4):
        return {}
    fold_size = max(1, len(y_true) // int(max(2, n_splits)))
    all_sets = []
    all_y = []
    mondrian_counts = None
    for start in range(0, len(y_true), fold_size):
        stop = min(len(y_true), start + fold_size)
        test_idx = np.arange(start, stop)
        cal_idx = np.concatenate([np.arange(0, start), np.arange(stop, len(y_true))])
        if len(test_idx) < 3 or len(cal_idx) < 6:
            continue
        if mondrian:
            pred_cal = np.argmax(proba[cal_idx], axis=1)
            mondrian_counts = {
                ConformalPredictor.LABELS.get(cls, str(cls)): int(np.sum(pred_cal == cls))
                for cls in range(proba.shape[1])
            }
            cp_map = {}
            global_cp = ConformalPredictor(alpha=alpha, lam=lam, kreg=kreg, method=method)
            global_cp.calibrate(proba[cal_idx], y_true[cal_idx])
            for cls in range(proba.shape[1]):
                cls_mask = pred_cal == cls
                if int(np.sum(cls_mask)) < int(CONFIG.get('conformal_mondrian_min_class_rows', 24) or 24):
                    continue
                cp_map[cls] = ConformalPredictor(alpha=alpha, lam=lam, kreg=kreg, method=method).calibrate(
                    proba[cal_idx][cls_mask],
                    y_true[cal_idx][cls_mask],
                )
            for row in proba[test_idx]:
                cls = int(np.argmax(row))
                cp = cp_map.get(cls, global_cp)
                all_sets.append(cp.predict_set(row))
        else:
            cp = ConformalPredictor(alpha=alpha, lam=lam, kreg=kreg, method=method)
            cp.calibrate(proba[cal_idx], y_true[cal_idx])
            for row in proba[test_idx]:
                all_sets.append(cp.predict_set(row))
        all_y.extend(y_true[test_idx].tolist())
    if not all_sets:
        return {}
    out = _conformal_summary_from_sets(
        all_sets,
        np.asarray(all_y, dtype=np.int64),
        n_classes=int(proba.shape[1]),
        alpha=float(alpha),
        method=method,
        qhat=None,
        n_cal=len(y_true),
    )
    out['n_folds'] = int(max(1, len(all_y) // max(1, fold_size)))
    out['cross_conformal'] = True
    out['mondrian'] = bool(mondrian)
    sorted_proba = np.sort(proba, axis=1)
    out['mean_top1_probability'] = float(np.mean(sorted_proba[:, -1])) if len(sorted_proba) else 0.0
    out['mean_probability_margin'] = float(np.mean(sorted_proba[:, -1] - sorted_proba[:, -2])) if sorted_proba.shape[1] >= 2 else 0.0
    if mondrian_counts is not None:
        out['predicted_class_counts'] = mondrian_counts
    return out


def _signal_probability_vector(signal: dict) -> np.ndarray:
    labels = ['SELL', 'HOLD', 'BUY']
    probs = signal.get('probabilities', {}) if isinstance(signal, dict) else {}
    return np.array([float(probs.get(lbl, 1.0 / 3.0)) for lbl in labels], dtype=np.float64)


def _resolve_conformal_source(final_signal: dict, tree, dl_trainers, regime_mdl=None) -> 'dict | None':
    model_name = str(final_signal.get('model_used', ''))
    latest_proba = _signal_probability_vector(final_signal)

    meta_ctx = getattr(tree, '_meta_calibration_context', None)
    if meta_ctx and model_name == meta_ctx.get('model_name'):
        return {
            'source_type': 'meta',
            'model_name': meta_ctx['model_name'],
            'proba_eval': np.asarray(meta_ctx.get('proba_eval')),
            'y_eval': np.asarray(meta_ctx.get('y_eval')),
            'latest_proba': latest_proba,
        }

    if model_name in getattr(tree, 'models', {}):
        model = tree.models[model_name]
        try:
            proba_eval = np.asarray(model.predict_proba(tree.Xs_te), dtype=np.float64)
            y_eval = np.asarray(tree.y_te.values)
            return {
                'source_type': 'tree',
                'model_name': model_name,
                'proba_eval': proba_eval,
                'y_eval': y_eval,
                'latest_proba': latest_proba,
            }
        except Exception as e:
            log.warning(f"Conformal source '{model_name}' unavailable from tree model: {e}")
            return None

    for trainer in dl_trainers:
        if getattr(trainer, 'name', None) == model_name:
            proba_eval = getattr(trainer, 'te_proba', None)
            y_eval = getattr(trainer, 'y_te_vals', None)
            if proba_eval is None or y_eval is None:
                log.warning(f"Conformal source '{model_name}' has no evaluation probabilities")
                return None
            return {
                'source_type': 'dl',
                'model_name': model_name,
                'proba_eval': np.asarray(proba_eval, dtype=np.float64),
                'y_eval': np.asarray(y_eval),
                'latest_proba': latest_proba,
            }

    if regime_mdl is not None and model_name.startswith('Regime-'):
        ctx = getattr(regime_mdl, 'get_conformal_context', lambda _: None)(model_name)
        if ctx is None:
            log.warning(f"Conformal source '{model_name}' has no held-out regime context")
            return None
        proba_eval = ctx.get('proba_eval')
        y_eval = ctx.get('y_eval')
        if proba_eval is None or y_eval is None:
            log.warning(f"Conformal source '{model_name}' has incomplete regime probabilities")
            return None
        return {
            'source_type': 'regime',
            'model_name': ctx.get('model_name', model_name),
            'proba_eval': np.asarray(proba_eval, dtype=np.float64),
            'y_eval': np.asarray(y_eval),
            'latest_proba': latest_proba,
        }

    log.warning(f"Conformal source unavailable for final model '{model_name}'")
    return None


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


def _patchtst_param_count(input_size: int, patch_len: int, d_model: int,
                          n_layers: int) -> int:
    """Approximate trainable parameter count for PatchTSTClassifier."""
    patch_dim = input_size * patch_len
    count = patch_dim * d_model            # patch projection
    count += d_model * 32                  # position embedding budget
    count += n_layers * (12 * d_model * d_model)
    count += d_model * 128 + 128 * 64 + 64 * 3
    return count


def _tide_param_count(input_size: int, seq_len: int, hidden: int,
                      n_layers: int) -> int:
    """Approximate trainable parameter count for TiDEClassifier."""
    flat_dim = input_size * seq_len
    count = flat_dim * hidden + hidden
    count += input_size * hidden + hidden
    count += n_layers * (4 * hidden * hidden)
    count += hidden * 128 + 128 * 64 + 64 * 3
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
    model_name  : 'bilstm' | 'transformer' | 'tft' | 'tide' | 'patchtst'
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

    elif model_name == 'tide':
        base_ep, base_pat = cfg['tide_epochs'], cfg['tide_patience']
        epochs, patience = _scaled(base_ep, base_pat)
        candidates = [
            (128, 3), (96, 3), (64, 2), (48, 2), (32, 1)
        ]
        for hidden, layers in candidates:
            est = _tide_param_count(n_features, seq_len, hidden, layers)
            if est <= budget:
                log.info(
                    f"  [Adaptive TiDE] hidden={hidden}  layers={layers}  "
                    f"est_params={est:,}  budget={budget:,}  "
                    f"(n_seqs={n_seqs})  epochs={epochs}  patience={patience}")
                return dict(model_kw=dict(
                    input_size=n_features,
                    hidden=hidden,
                    n_layers=layers,
                    dropout=cfg['tide_dropout'],
                    seq_len=seq_len,
                ), epochs=epochs, patience=patience)
        log.warning("  [Adaptive TiDE] Minimum architecture (hidden=32, L=1).")
        return dict(model_kw=dict(
            input_size=n_features,
            hidden=32,
            n_layers=1,
            dropout=cfg['tide_dropout'],
            seq_len=seq_len,
        ), epochs=epochs, patience=patience)

    # Fallback - should never reach here
    elif model_name == 'patchtst':
        base_ep, base_pat = cfg['patchtst_epochs'], cfg['patchtst_patience']
        epochs, patience = _scaled(base_ep, base_pat)
        patch_len = min(cfg.get('patchtst_patch_len', 8), seq_len)
        stride = max(1, cfg.get('patchtst_stride', 4))
        candidates = [
            (96, 3, 4), (64, 2, 4), (48, 2, 4), (32, 1, 4)
        ]
        for d_model, layers, nhead in candidates:
            while nhead > 1 and d_model % nhead != 0:
                nhead //= 2
            est = _patchtst_param_count(n_features, patch_len, d_model, layers)
            if est <= budget:
                log.info(
                    f"  [Adaptive PatchTST] d_model={d_model}  layers={layers}  "
                    f"patch={patch_len}/{stride}  est_params={est:,}  budget={budget:,}  "
                    f"(n_seqs={n_seqs})  epochs={epochs}  patience={patience}")
                return dict(model_kw=dict(
                    input_size=n_features,
                    patch_len=patch_len,
                    stride=stride,
                    d_model=d_model,
                    nhead=nhead,
                    n_layers=layers,
                    dropout=cfg['patchtst_dropout'],
                    seq_len=seq_len,
                ), epochs=epochs, patience=patience)
        log.warning("  [Adaptive PatchTST] Minimum architecture (d_model=32, L=1).")
        return dict(model_kw=dict(
            input_size=n_features,
            patch_len=patch_len,
            stride=stride,
            d_model=32,
            nhead=4,
            n_layers=1,
            dropout=cfg['patchtst_dropout'],
            seq_len=seq_len,
        ), epochs=epochs, patience=patience)

    raise ValueError(f"Unknown model_name '{model_name}'. "
                     "Expected 'bilstm', 'transformer', 'tft', 'tide', or 'patchtst'.")


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
        self.y_te_index = None
        self.cal_proba = None
        self.y_cal_vals = None
        self.y_cal_index = None
        self.X_arr   = None
        self.seq_len = None
        self.temperature = 1.0
        self.calibration_report_ = {}

    def _make_seqs(self, X: np.ndarray, y: np.ndarray, seq_len: int,
                   augment: bool = False,
                   aug_factor: int = 3,
                   noise_std: float = 0.005,
                   mag_warp: bool = True,
                   rng_seed: int = 42) -> tuple:
        """
        Build sliding-window sequences. rng_seed is now a parameter so each
        DL model (BiLSTM, Transformer, TFT) gets DISTINCT augmented sequences.
        Using seed=42 for all models is a bug: identical augmentation provides
        zero diversity benefit — three models train on the exact same data.
        """
        rng  = np.random.default_rng(rng_seed)
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
        tr_df    = X_df.iloc[:sp_raw]
        te_df    = X_df.iloc[sp_raw:]
        tr_med   = tr_df.median()
        X_tr_raw = tr_df.fillna(tr_med).values.astype(np.float32)
        X_te_raw = te_df.fillna(tr_med).values.astype(np.float32)

        # B3 FIX — Double-scaling: X_dl is already RobustScaler-transformed by
        # tree.scaler (applied in main() before DL training).  Applying a second
        # RobustScaler compresses data by ~0.74× (median≈0, IQR≈1.35 of already-
        # scaled data → second scaler divides by ~1.35).  Skip rescaling entirely
        # when input is flagged as pre-scaled; use an identity transform instead.
        pre_scaled = getattr(self, '_pre_scaled', False)
        if pre_scaled:
            X_arr = np.vstack([X_tr_raw, X_te_raw])
            log.info("Scaling: skipped (input already tree.scaler-transformed)")
        else:
            self.scaler.fit(X_tr_raw)
            X_arr = np.vstack([
                self.scaler.transform(X_tr_raw),
                self.scaler.transform(X_te_raw),
            ])

        # ── Build raw sequences first, then split, then augment training only ─
        Xs_raw, ys_raw = self._make_seqs(X_arr, y_arr, seq_len, augment=False)
        seq_index = np.asarray(X_df.index[seq_len:])

        # B2 FIX — Sequence split leakage: using int(len(Xs_raw)*0.80)=849 places
        # training sequences whose context windows reach into test rows (897-908).
        # For sequence j, context = X_arr[j : j+seq_len].  Training sequences must
        # not reference any test-period rows.  The last SAFE training sequence ends
        # at row sp_raw-1, which is sequence index sp_raw - seq_len.
        #   Old sp=849 → last train seq uses rows 848:908 → 11 test rows leaked!
        #   New sp=837 → last train seq uses rows 837:897 → no test rows ✓
        sp    = sp_raw - seq_len      # guaranteed leak-free split
        train_seq_end = max(0, min(sp, len(Xs_raw)))
        cal_len = max(20, int(train_seq_end * 0.15))
        cal_len = min(cal_len, max(5, train_seq_end // 3)) if train_seq_end else 0
        cal_start = max(0, train_seq_end - cal_len)

        X_fit, y_fit = Xs_raw[:cal_start], ys_raw[:cal_start]
        idx_fit = seq_index[:cal_start]
        X_cal, y_cal = Xs_raw[cal_start:train_seq_end], ys_raw[cal_start:train_seq_end]
        idx_cal = seq_index[cal_start:train_seq_end]
        X_te  = Xs_raw[sp:]
        y_te  = ys_raw[sp:]
        idx_te = seq_index[sp:]

        if len(X_fit) < 20 or len(X_cal) < 5 or len(X_te) < 5:
            raise ValueError("insufficient chronological sequences for fit/cal/test split")

        # Augmentation: use ONLY training rows (X_arr[:sp_raw] = rows 0:sp_raw)
        # The augmentation slice X_arr[:sp+seq_len] = X_arr[:sp_raw] ✓
        do_aug = self.cfg.get('dl_augment', True) and self.device.type == 'cuda'
        # Each model gets a unique augmentation seed so BiLSTM/Transformer/TFT
        # train on DISTINCT augmented sequences (diversity benefit).
        aug_seed = (hash(self.name) + self.cfg.get('random_state', 42)) & 0xFFFFFFFF
        if do_aug:
            aug_f = self.cfg.get('dl_aug_factor', 3)
            aug_n = self.cfg.get('dl_aug_noise_std', 0.005)
            aug_m = self.cfg.get('dl_aug_mag_warp', True)
            fit_rows = len(X_fit) + seq_len
            X_tr, y_tr = self._make_seqs(X_arr[:fit_rows], y_arr[:fit_rows],
                                          seq_len, augment=True, aug_factor=aug_f,
                                          noise_std=aug_n, mag_warp=aug_m,
                                          rng_seed=aug_seed)
            log.info(f"Augmentation: {aug_f}× seed={aug_seed} "
                     f"(noise σ={aug_n}, mag_warp={aug_m}) "
                     f"→ {len(X_tr)} train seqs from {sp} originals")
        else:
            X_tr, y_tr = X_fit, y_fit
            if not do_aug:
                log.info("Augmentation: disabled (CPU mode)")

        log.info(f"Fit seqs: {len(X_fit)}  Cal seqs: {len(X_cal)}  Test seqs: {len(X_te)}  "
                 f"Features: {X_tr.shape[2]}")

        counts  = np.bincount(y_tr % 3, minlength=3).astype(float)  # mod 3 safe for aug
        # recount from originals for class weight (aug copies same label)
        counts_orig = np.bincount(y_fit, minlength=3).astype(float)
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
        cal_loader = DataLoader(TensorDataset(torch.tensor(X_cal), torch.tensor(y_cal)),
                                batch_size=512, shuffle=False,
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
        best_acc, best_f1, best_state, no_imp = 0.0, 0.0, None, 0
        selection_metric = str(self.cfg.get('dl_selection_metric', 'f1_macro')).lower()
        use_f1_selection = selection_metric in ('f1', 'f1_macro', 'macro_f1')
        # Lucky-init guard: random weights can score high if majority class
        # dominates test set. Gate best_state saves to post-warmup epochs only
        # and reject dominated (>85%) checkpoints. If nothing valid is saved,
        # best_state stays None and the final-epoch weights are used.
        min_save_epoch = max(3, int(epochs * 0.10))   # earliest epoch to save
        # Random-init baseline: a model that always predicts majority class
        # gets acc = majority_class_fraction. If we never beat this, we've collapsed.
        majority_frac  = float(np.bincount(y_cal, minlength=3).max()) / len(y_cal)
        # FIX B: plateau threshold = random-chance baseline + 5% margin.
        # Using majority_frac (~67% for HOLD-heavy labels) kills DL models that
        # predict all 3 classes at ~46% accuracy — which is genuinely good for
        # imbalanced 3-class data. A model only needs to beat random chance (33%),
        # not the "predict-HOLD-always" shortcut.
        n_classes      = len(np.unique(y_cal))
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

            val_summary = self._eval_summary(cal_loader, y_cal, amp_dtype, use_amp)
            val_acc = float(val_summary['acc'])
            val_f1 = float(val_summary['f1'])
            val_score = val_f1 if use_f1_selection else val_acc
            best_score = best_f1 if use_f1_selection else best_acc
            if (val_score > best_score + 1e-9) or (
                abs(val_score - best_score) <= 1e-9 and val_acc > best_acc
            ):
                # BUG FIX: guard against saving a lucky-init or collapsed state.
                # Skip if too early in training, OR if this epoch is dominated.
                _ep_dom = float(val_summary['dominant_frac'])
                _is_early = (epoch < min_save_epoch)
                _is_dom   = (_ep_dom > 0.85)
                if not _is_early and not _is_dom:
                    best_acc   = val_acc
                    best_f1    = val_f1
                    best_state = {k: v.cpu().clone() for k, v in model_raw.state_dict().items()}
                    no_imp     = 0
                elif _is_dom and val_score > (best_score + 0.02):
                    # If ALL states are dominated (common for extreme imbalance)
                    # still save the least-bad one rather than keeping nothing.
                    best_acc   = val_acc
                    best_f1    = val_f1
                    best_state = {k: v.cpu().clone() for k, v in model_raw.state_dict().items()}
                    no_imp     = 0
                else:
                    no_imp += 1
            else:
                no_imp += 1

            mem_str = (f"  vram={torch.cuda.memory_allocated()/1e9:.2f}GB"
                       if self.device.type == 'cuda' else "")
            epoch_bar.set_postfix(loss=f"{ep_loss/len(tr_loader):.6f}",
                                  val=f"{val_acc:.4f}", f1=f"{val_f1:.4f}",
                                  best=f"{(best_f1 if use_f1_selection else best_acc):.4f}",
                                  p=f"{no_imp}/{patience}")
            if epoch % 10 == 0 or epoch == 1:
                log.debug(f"  ep{epoch:4d}  loss={ep_loss/len(tr_loader):.6f}  "
                          f"val={val_acc:.4f}  f1={val_f1:.4f}  "
                          f"best={best_f1 if use_f1_selection else best_acc:.4f}{mem_str}")
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
            swa_acc = self._eval(cal_loader, amp_dtype, use_amp)
            if swa_acc >= best_acc:
                log.info(f"  SWA acc={swa_acc:.4f} >= best-checkpoint {best_acc:.4f} — keeping SWA")
                best_acc = swa_acc
            else:
                log.info(f"  SWA acc={swa_acc:.4f} < best-checkpoint {best_acc:.4f} — reverting")
                model_raw.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        cal_preds, cal_proba = [], []
        model_raw.eval()
        with torch.no_grad():
            for Xb, _ in cal_loader:
                with torch.amp.autocast(device_type=self.device.type,
                                        dtype=amp_dtype, enabled=use_amp):
                    logits = model_raw(Xb.to(self.device))
                proba = torch.softmax(logits.float(), dim=1).cpu().numpy()
                cal_preds.extend(proba.argmax(axis=1)); cal_proba.extend(proba)

        all_preds, all_proba = [], []
        model_raw.eval()
        with torch.no_grad():
            for Xb, _ in te_loader:
                with torch.amp.autocast(device_type=self.device.type,
                                        dtype=amp_dtype, enabled=use_amp):
                    logits = model_raw(Xb.to(self.device))
                proba = torch.softmax(logits.float(), dim=1).cpu().numpy()
                all_preds.extend(proba.argmax(axis=1)); all_proba.extend(proba)

        self.cal_proba = np.array(cal_proba)
        self.y_cal_vals = y_cal
        self.y_cal_index = pd.Index(idx_cal)
        self.te_preds   = np.array(all_preds)
        self.te_proba   = np.array(all_proba)
        self.y_te_vals  = y_te
        self.y_te_index = pd.Index(idx_te)
        self.X_arr      = X_arr
        self.seq_len    = seq_len
        self._model_raw = model_raw

        # ── NEW 4: Model Collapse Detection + Restart ─────────────────────────
        unique_preds, counts = np.unique(self.te_preds, return_counts=True)
        initial_summary = _prediction_summary(y_te, self.te_preds)
        dominant_frac = float(initial_summary['dominant_frac'])
        collapse_reason = _dl_instability_reason(initial_summary, self.cfg)
        self.is_collapsed = False
        self.collapse_reason_ = None
        if collapse_reason:
            dominant_cls = ['SELL', 'HOLD', 'BUY'][unique_preds[counts.argmax()]]
            hard_collapse = (
                int(initial_summary.get('predicted_classes', 0)) <= 1
                or dominant_frac >= float(self.cfg.get('dl_collapse_hard_threshold', 0.85))
            )
            if hard_collapse:
                # ── Hard collapse: single class predicted or extreme dominance.
                # A one-shot restart with seed+1 cannot fix this reliably and
                # wastes 80 epochs. Mark immediately and skip the retry.
                log.warning(
                    f"⚠  MODEL COLLAPSE (HARD): {self.name} predicts '{dominant_cls}' "
                    f"on {dominant_frac*100:.1f}% of test samples — skipping restart."
                )
                self.is_collapsed = True
                self.collapse_reason_ = collapse_reason
            else:
                # ── Soft instability: dominant but not single-class. One retry.
                log.warning(
                    f"DL instability detected: {self.name} is one-class dominant "
                    f"('{dominant_cls}' {dominant_frac*100:.1f}%, "
                    f"f1={initial_summary['f1']:.4f})."
                )
            if not hard_collapse:
                try:
                    # Save the best-checkpoint val F1 BEFORE the restart overwrites
                    # anything — this is the quality bar the restart must meet.
                    # Using o_f1 (collapsed test F1) as the bar was the old bug:
                    # o_f1 is often very low (e.g. 0.15) after collapse, so any
                    # restart that produces f1=0.30 looks "good" by comparison even
                    # though it is substantially worse than the pre-collapse best
                    # validation f1 (e.g. 0.41).
                    pre_collapse_best_f1 = best_f1   # validation F1 from main loop
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
                        summary_r = self._eval_model_summary(model_r, te_loader, y_te, amp_dtype, use_amp)
                        score_r = summary_r['f1'] if use_f1_selection else summary_r['acc']
                        if score_r > best_r:
                            best_r = score_r
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
                    # Evaluate restart model after the retry loop, not mid-training.
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
                    r_summary = _prediction_summary(y_te, r_preds)
                    o_summary = _prediction_summary(y_te, self.te_preds)
                    r_dom   = float(r_summary['dominant_frac'])
                    r_acc   = float(r_summary['acc'])
                    r_f1    = float(r_summary['f1'])
                    o_acc   = float(o_summary['acc'])
                    o_f1    = float(o_summary['f1'])
                    r_reason = _dl_instability_reason(r_summary, self.cfg)
                    # Accept restart if:
                    #   (a) no new instability, AND
                    #   (b) restart f1 is within 15% of the pre-collapse BEST
                    #       VALIDATION F1 (not the collapsed test f1 — using the
                    #       collapsed floor made any restart look "better").
                    # Condition (b) uses pre_collapse_best_f1 × 0.85 as the floor:
                    # a restart that regresses more than 15% below the best
                    # checkpoint is not good enough to be accepted.
                    restart_f1_floor = max(0.0, pre_collapse_best_f1 * 0.85)
                    restart_ok = (
                        r_reason is None and (
                            r_f1 >= restart_f1_floor or
                            (r_dom < dominant_frac and r_f1 >= o_f1 - 0.02)
                        )
                    )
                    if restart_ok:
                        log.info(f"  Restart SUCCESS: dom={r_dom*100:.1f}%  acc={r_acc:.4f}  "
                                 f"f1={r_f1:.4f}  (was dom={dominant_frac*100:.1f}%  "
                                 f"acc={o_acc:.4f}  f1={o_f1:.4f})")
                        self.te_preds   = r_preds
                        self.te_proba   = np.array(r_proba)
                        self._model_raw = model_r
                        model_raw       = model_r
                        dominant_frac   = r_dom
                        self.is_collapsed = False
                        self.collapse_reason_ = None
                        log.info(f"  Collapse resolved by restart.")
                    else:
                        reason_msg = r_reason or f"dom={r_dom*100:.1f}%"
                        log.warning(f"  Restart did not resolve instability "
                                    f"({reason_msg}). Marking model as collapsed.")
                        self.is_collapsed = True
                        self.collapse_reason_ = r_reason or collapse_reason
                except Exception as e_r:
                    log.warning(f"  Collapse restart failed: {e_r}")
                    self.is_collapsed = True
                    self.collapse_reason_ = collapse_reason   # keep original reason on exception

        # ── NEW 5: Temperature Scaling calibration ────────────────────────────
        # Learn scalar T on the TEST set via cross-entropy minimisation.
        # After scaling, logits are divided by T before softmax.
        self.temperature = 1.0
        self.calibration_report_ = {}
        if HAS_TORCH and self.cal_proba is not None and len(self.cal_proba) >= 20:
            try:
                pre_cal = np.asarray(self.cal_proba, dtype=np.float64)
                self.temperature = self._fit_temperature(
                    pre_cal, y_cal, device=self.device)
                post_cal = self._scale_proba(pre_cal, self.temperature)
                self.calibration_report_ = _build_calibration_report(
                    pre_cal, post_cal, y_cal,
                    calibrator_type='temperature',
                    learned_temperature=self.temperature,
                )
                if self.temperature != 1.0:
                    log.info(f"  Temperature scaling: T={self.temperature:.3f}")
                self.cal_proba = post_cal
                self.te_proba = self._scale_proba(self.te_proba, self.temperature)
                self.te_preds = self.te_proba.argmax(axis=1)
            except Exception as e:
                log.debug(f"  Temperature scaling skipped: {e}")
                self.calibration_report_ = _build_calibration_report(
                    self.cal_proba, self.cal_proba, y_cal,
                    calibrator_type='identity',
                    learned_temperature=1.0,
                )

        final_summary = _prediction_summary(y_te, self.te_preds)
        final_acc = float(final_summary['acc'])
        final_f1 = float(final_summary['f1'])
        final_reason = _dl_instability_reason(final_summary, self.cfg)
        if final_reason:
            self.is_collapsed = True
            self.collapse_reason_ = final_reason
            log.warning(f"  DL stability gate: {self.name} {final_reason} - excluded from stack/final DL use")
        self.eval_acc_ = final_acc
        self.eval_f1_ = final_f1
        self.eval_dominant_frac_ = float(final_summary['dominant_frac'])
        self.eval_pred_counts_ = dict(final_summary['pred_counts'])
        self.eval_predicted_classes_ = int(final_summary['predicted_classes'])
        if final_f1 < float(self.cfg.get('dl_stack_min_f1', 0.35)):
            log.warning(f"  DL stack gate: {self.name} f1={final_f1:.4f} "
                        f"< {self.cfg.get('dl_stack_min_f1', 0.35):.2f} - excluded from stack")
        log.info(f"\n  Final test acc : {final_acc:.4f}")
        log.info(f"  Final test f1  : {final_f1:.4f}")
        log.info(f"  Best val acc   : {best_acc:.4f}")
        log.info(f"  Best val f1    : {best_f1:.4f}")
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
        """Fit scalar temperature T by minimizing NLL on calibration data only."""
        logits = np.log(np.clip(proba, 1e-9, 1.0)).astype(np.float32)
        logits_t = torch.tensor(logits, device=device)
        labels_t = torch.tensor(y_true.astype(np.int64), device=device)
        T = torch.nn.Parameter(torch.ones(1, device=device))
        optim = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
        criterion = torch.nn.CrossEntropyLoss()

        def _closure():
            optim.zero_grad()
            loss = criterion(logits_t / T.clamp(min=0.05), labels_t)
            loss.backward()
            return loss

        try:
            optim.step(_closure)
        except Exception:
            return 1.0

        t_val = float(T.detach().cpu().item())
        return float(np.clip(t_val, 0.05, 10.0))

    def _eval(self, loader, amp_dtype=None, use_amp=False):
        return self._eval_model(self.model, loader, amp_dtype, use_amp)

    def _eval_summary(self, loader, y_true, amp_dtype=None, use_amp=False):
        return self._eval_model_summary(self.model, loader, y_true, amp_dtype, use_amp)

    @staticmethod
    def _eval_model(model, loader, amp_dtype=None, use_amp=False):
        """Evaluate any model on a loader, returning accuracy."""
        summary = TorchTrainer._eval_model_summary(model, loader, None, amp_dtype, use_amp)
        return float(summary['acc'])

    @staticmethod
    def _eval_model_summary(model, loader, y_true=None, amp_dtype=None, use_amp=False):
        if amp_dtype is None:
            amp_dtype = torch.float32
        device = next(model.parameters()).device
        preds = []
        labels = []
        model.eval()
        with torch.no_grad():
            for Xb, yb in loader:
                with torch.amp.autocast(device_type=device.type,
                                        dtype=amp_dtype, enabled=use_amp):
                    logits = model(Xb.to(device))
                preds.extend(logits.float().argmax(dim=1).cpu().numpy())
                if y_true is None:
                    labels.extend(yb.cpu().numpy())
        labels_arr = np.asarray(y_true if y_true is not None else labels, dtype=np.int64)
        preds_arr = np.asarray(preds, dtype=np.int64)
        return _prediction_summary(labels_arr, preds_arr)

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
            'collapse_reason': getattr(self, 'collapse_reason_', None),
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
        self.model_names_ = {}
        self.calibration_contexts_ = {}
        self.bear_st_ = 0
        self.cur_reg_ = 1      # default bull
        self.obs_cols_= []
        self.regime_mode_ = "SPY"

    def _find_market_column(self, token: str):
        token = token.upper()
        return next((c for c in self.market.columns
                     if token in str(c).upper()), None)

    @staticmethod
    def _regime_name(rid: int) -> str:
        return 'BEAR' if int(rid) == 0 else 'BULL'

    def _regime_model_name(self, rid: int, model=None) -> str:
        label = self._regime_name(rid)
        if model is None and rid in self.models_:
            model = self.models_[rid][0]
        suffix = 'MODEL'
        if model is not None:
            cls_name = model.__class__.__name__
            if 'XGB' in cls_name.upper():
                suffix = 'XGB'
            elif 'RANDOMFOREST' in cls_name.upper():
                suffix = 'RF'
            else:
                suffix = cls_name
        return f'Regime-{label}-{suffix}'

    def get_conformal_context(self, model_name: str):
        return self.calibration_contexts_.get(str(model_name))

    def _build_hmm_observations(self, log_details: bool = True):
        if self.market is None or self.market.empty:
            if log_details:
                log.warning("  Market context empty - skipping HMM regime model")
            return None, None

        min_rows = int(self.cfg.get('regime_min_rows', 252))
        spy = self._find_market_column('SPY')
        if spy is None:
            if log_details:
                log.warning("  SPY not in market context - skipping HMM")
            return None, None

        market = self.market.sort_index()
        spy_raw = market[[spy]].dropna().copy()
        if len(spy_raw) < min_rows:
            if log_details:
                log.warning(f"  SPY-only overlap={len(spy_raw)} rows (<{min_rows}) - "
                            f"skipping HMM")
            return None, None

        spy_raw['spy_ret'] = spy_raw[spy].pct_change().fillna(0.0)
        spy_obs = spy_raw[['spy_ret']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(spy_obs) < min_rows:
            if log_details:
                log.warning(f"  SPY-only usable rows={len(spy_obs)} (<{min_rows}) - "
                            f"skipping HMM")
            return None, None

        vix = self._find_market_column('VIX')
        if vix is not None:
            duo_raw = market[[spy, vix]].dropna().copy()
            if len(duo_raw) >= min_rows:
                duo_raw['spy_ret'] = duo_raw[spy].pct_change().fillna(0.0)
                roll = duo_raw[vix].rolling(63, min_periods=20)
                duo_raw['vix_z'] = (
                    (duo_raw[vix] - roll.mean()) / (roll.std() + 1e-10)
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                duo_obs = duo_raw[['spy_ret', 'vix_z']].dropna()
                if len(duo_obs) >= min_rows:
                    if log_details:
                        log.info(f"  HMM observations: mode=SPY+VIX  rows={len(duo_obs)}")
                    return duo_obs, 'SPY+VIX'
                if log_details:
                    log.info(f"  HMM fallback: SPY+VIX usable rows={len(duo_obs)} "
                             f"(<{min_rows}) - using SPY-only")
            elif log_details:
                log.info(f"  HMM fallback: SPY+VIX overlap={len(duo_raw)} rows "
                         f"(<{min_rows}) - using SPY-only")
        elif log_details:
            log.info("  HMM fallback: VIX unavailable - using SPY-only")

        if log_details:
            log.info(f"  HMM observations: mode=SPY-only  rows={len(spy_obs)}")
        return spy_obs, 'SPY'

    # ── HMM fitting ────────────────────────────────────────────────
    def _fit_hmm(self) -> 'pd.Series | None':
        if not HAS_HMM:
            log.warning("  hmmlearn not installed - pip install hmmlearn")
            return None
        obs_df, mode = self._build_hmm_observations(log_details=True)
        if obs_df is None:
            return None
        obs = obs_df.values
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
            self.obs_cols_ = list(obs_df.columns)
            self.regime_mode_ = mode
            regimes = pd.Series(
                np.where(states == self.bear_st_, 0, 1),
                index=obs_df.index, name='regime')
            bull_pct = (regimes == 1).mean()
            log.info(f"  HMM: mode={mode}  bull={bull_pct*100:.0f}%  "
                     f"bear={(1-bull_pct)*100:.0f}%")
            return regimes
        except Exception as e:
            log.warning(f"  HMM fit error: {e}")
            return None

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
            model_name = self._regime_model_name(rid, model_r)
            self.model_names_[rid] = model_name

            mask_te = reg_al.iloc[sp:] == rid
            n_te    = int(mask_te.sum())
            if n_te > 0:
                Xte = vt.transform(sc.transform(Xf.iloc[sp:][mask_te]))
                pte = model_r.predict(Xte)
                try:
                    proba_te = np.asarray(model_r.predict_proba(Xte), dtype=np.float64)
                except Exception:
                    proba_te = None
                self.calibration_contexts_[model_name] = {
                    'source_type': 'regime',
                    'model_name': model_name,
                    'proba_eval': proba_te,
                    'y_eval': y.iloc[sp:][mask_te].values,
                    'regime': self._regime_name(rid),
                }
            if n_te >= 10:
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
        if self.hmm_ is None:
            self.cur_reg_ = 1; return
        obs_df, mode = self._build_hmm_observations(log_details=False)
        if obs_df is None or obs_df.empty:
            self.cur_reg_ = 1; return
        cols = self.obs_cols_ if self.obs_cols_ else list(obs_df.columns)
        if not set(cols).issubset(obs_df.columns):
            log.warning("  Current regime detection skipped: observation columns changed")
            self.cur_reg_ = 1
            return
        obs = obs_df[cols].iloc[[-1]].values
        state       = int(self.hmm_.predict(obs)[0])
        self.cur_reg_ = 0 if state == self.bear_st_ else 1
        log.info(f"  Current regime: {'BEAR' if self.cur_reg_==0 else 'BULL'}  ({mode})")
        return

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
        rname = self._regime_name(rid)
        model_name = self.model_names_.get(rid, self._regime_model_name(rid, model_r))
        return {
            'signal':        lmap[pred],
            'label':         pred,
            'confidence':    float(proba[pred]),
            'probabilities': {lmap[i]: float(p) for i, p in enumerate(proba)},
            'model_used':    model_name,
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
        active_dl = []
        for t in dl_trainers:
            if _dl_trainer_usable_for_stack(t, CONFIG):
                active_dl.append(t)
            else:
                eval_f1 = float(getattr(t, 'eval_f1_', 0.0) or 0.0)
                log.info(f"  Stacking skip: {t.name} "
                         f"(collapsed={getattr(t, 'is_collapsed', False)}, "
                         f"f1={eval_f1:.4f})")
        n_dl = 0
        if active_dl:
            n_dl = min(len(t.te_proba) for t in active_dl)
        n = min(n_dl, len(Xs_te)) if n_dl > 0 else len(Xs_te)
        if n < 30:
            log.warning("Not enough test samples for stacking"); return None

        ens = tree.models.get('Ensemble') or primary
        cols = [ens.predict_proba(Xs_te[-n:])]
        stack_feature_names = [f"{'Ensemble' if 'Ensemble' in tree.models else best_name}_{lbl}"
                               for lbl in ('SELL', 'HOLD', 'BUY')]
        for t in active_dl:
            cols.append(t.te_proba[-n:])
            stack_feature_names.extend(f"{t.name}_{lbl}" for lbl in ('SELL', 'HOLD', 'BUY'))
        if len(active_dl) == 0:
            log.info("  Stacking: no usable DL models - using tree ensemble only")
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
        eval_proba_st = stack.predict_proba(meta_X[sp:])
        preds_st = eval_proba_st.argmax(axis=1)
        acc_st   = accuracy_score(y_meta[sp:], preds_st)
        f1_st    = f1_score(y_meta[sp:], preds_st, average='macro', zero_division=0)
        # ── Meta-stack sanity gate ────────────────────────────────────────────
        # If stacking accuracy < random-chance baseline (1/3 for 3 classes),
        # the meta stack is worse than useless — return None so the fallback
        # chain uses tree or regime signal directly.  RKLB had acc=16.7%
        # (below 33% chance) yet was still returned as the final signal.
        n_classes_meta = len(np.unique(y_meta))
        random_chance  = 1.0 / max(n_classes_meta, 2)
        baseline_f1 = max(
            float(tree.results.get(best_name, {}).get('f1', 0.0)),
            float(tree.results.get('Ensemble', {}).get('f1', 0.0))
        )
        if not _meta_stack_is_acceptable(acc_st, f1_st, baseline_f1, CONFIG):
            log.warning(
                f"  Meta-stack rejected: acc={acc_st:.4f}  f1={f1_st:.4f}  "
                f"baseline_f1={baseline_f1:.4f}  "
                f"(chance={random_chance:.3f}, min_f1={CONFIG.get('meta_stack_min_f1', 0.34):.2f})  "
                f"{random_chance:.3f} — discarding meta signal, "
                f"falling back to tree/regime signal")
            return None
        log.info(f"  Stacking accuracy: {acc_st:.4f}  F1-macro: {f1_st:.4f}")
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
        explain_model = stack
        try:
            explain_model = stack.calibrated_classifiers_[0].estimator
        except Exception:
            explain_model = stack
        tree._meta_calibration_context = {
            'model_name': f'Meta-Label+Stack ({len(cols)} models)',
            'proba_eval': eval_proba_st,
            'y_eval': y_meta[sp:],
            'latest_proba': proba_st,
            'eval_index': list(tree.y_te.index[-n:][sp:]),
        }
        tree._meta_explain_context = {
            'model_name': f'Meta-Label+Stack ({len(cols)} models)',
            'model': explain_model,
            'feature_names': stack_feature_names,
            'x_reference': meta_X[sp:],
            'x_latest': meta_feat_latest,
        }
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

        """

            sizing_method='Half-Kelly' if best_is_kelly else 'Binary±1',
        )

        """

        result = {
            'signal':           lmap[int(pred_st)],
            'label':            int(pred_st),
            'confidence':       float(proba_st[int(pred_st)]),
            'model_used':       f'Meta-Label+Stack ({len(cols)} models)',
            'probabilities':    {lmap[i]: float(p) for i, p in enumerate(proba_st)},
            'meta_accuracy':    float(acc_st),
            'meta_f1':          float(f1_st),
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
              'shift_detected': False, 'warning': False}
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
        use_holdout = (
            hasattr(tree, 'X_te') and hasattr(tree, 'Xs_te')
            and getattr(tree, 'X_te', None) is not None
            and getattr(tree, 'Xs_te', None) is not None
            and len(getattr(tree, 'X_te')) > 0
        )
        if use_holdout:
            xs = tree.Xs_te
            pred_index = tree.X_te.index
            preds = model.predict(xs).ravel()
            backtest_source = 'static_holdout_backtest'
        else:
            Xf = tree.X.fillna(tree.X.median())
            xs = tree.scaler.transform(Xf)
            xs = tree.var_sel.transform(xs)
            xs = tree.feat_sel.transform(xs)
            preds = model.predict(xs).ravel()   # FIX: CatBoost returns (n,1) — flatten to 1D
            pred_index = tree.X.index
            backtest_source = 'static_in_sample_backtest'
        pred_sr = pd.Series(preds, index=pred_index)

        close  = df['Close'].reindex(pred_sr.index)
        ret    = close.pct_change().shift(-1)
        signal = pred_sr.map({2: 1, 0: -1, 1: 0})

        # NEW 15: volume-adaptive realistic transaction cost
        close_vol  = df['Volume'].reindex(pred_sr.index, method='ffill').fillna(1e6)
        adv63_s    = close_vol.rolling(63).mean().fillna(close_vol)
        rv21_s     = close.pct_change().rolling(21).std().fillna(0.03)
        cost_bin   = _realistic_cost(signal, close, adv63_s, rv21_s)

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
        strat_full_1  = (signal * ret - cost_bin).fillna(0)   # binary ±1 (reference)

        # Build per-bar conformal multiplier (default = 1.0 = no change)
        conformal_mult = pd.Series(1.0, index=signal.index)
        if CONFIG.get('conformal_sizing_enabled', True):
            cp_stored = getattr(tree, '_conformal_predictor', None)
            if cp_stored is not None:
                try:
                    best_model = tree.models[best]
                    proba_all  = best_model.predict_proba(xs)  # (N, 3)
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

        cost_k = _realistic_cost(kelly_size, close, adv63_s, rv21_s)
        strat_k = (kelly_size * ret - cost_k).fillna(0)

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
            activity   = _summarize_backtest_activity(active_sig, strat_r)
            hit_rate   = float(activity.get('active_bar_win_rate', 0.0))
            n_tr       = int(activity.get('n_trade_events', 0))
            log.info(f"  [{label}] Return={float(sc.iloc[-1]-1)*100:+.2f}%  "
                     f"ann={strat_ann*100:+.2f}%  Sharpe={sharpe:.3f}  "
                     f"MaxDD={dd.min()*100:.1f}%  "
                     f"HitRate(active bars)={hit_rate*100:.1f}%  "
                     f"TradeEvents={n_tr}")
            return sc, float(sc.iloc[-1]-1), strat_ann, sharpe, \
                   sortino, \
                   float(dd.min()), hit_rate, int(n_tr), var95, cvar95

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
        active_sig_best = kelly_size if best_is_kelly else signal
        strat_best = strat_k if best_is_kelly else strat_full_1
        activity_best = _summarize_backtest_activity(active_sig_best, strat_best)
        audit = _build_backtest_audit(
            strategy_returns=sr_k if best_is_kelly else sr_bin,
            buyhold_returns=float(bhc.iloc[-1] - 1),
            n_bars=len(sc),
            n_trades=nt_k if best_is_kelly else nt_bin,
            active_signal=active_sig_best,
            conformal_mult=conformal_mult,
            sizing_method='Half-Kelly' if best_is_kelly else 'Binary±1',
            activity_summary=activity_best,
            strategy_return_source=backtest_source,
        )
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
            'active_bar_count': int(activity_best.get('active_bar_count', 0)),
            'active_bar_win_rate': float(activity_best.get('active_bar_win_rate', 0.0)),
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
            'audit': audit,
            'sizing_method': 'Half-Kelly' if best_is_kelly else 'Binary±1',
        }
        return result
    except Exception as e:
        log.error(f"Backtest failed: {e}\n{traceback.format_exc()}")
        return {'strat_return':0,'strat_annual':0,'bh_return':0,
                'strat_sharpe':0,'strat_sortino':0,'strat_maxdd':0,
                'calmar':0,'win_rate':0,'n_trades':0,'var95':0,'cvar95':0,
                'active_bar_count':0,'active_bar_win_rate':0,
                'kelly_sharpe':0,'binary_sharpe':0,'sizing_method':'N/A',
                'audit': {'strategy_return_source': 'static_in_sample_backtest', 'sanity_status': 'ok', 'sanity_flags': []},
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


def _chart_display_label(label: str, width: int = 12) -> str:
    text = str(label or '')
    aliases = {
        'BiLSTM-Attention': 'BiLSTM\nAttention',
        'Transformer-Encoder': 'Transformer\nEncoder',
    }
    if text in aliases:
        return aliases[text]
    if len(text) <= width:
        return text
    return '\n'.join(textwrap.wrap(text, width=width, break_long_words=False))


def _scatter_annotation_offsets(n: int) -> list[tuple[int, int]]:
    base = [(10, 8), (10, 20), (10, -8), (22, 8), (22, -8), (22, 20), (34, 8), (34, -8)]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        dx, dy = base[i % len(base)]
        out.append((dx + 12 * (i // len(base)), dy))
    return out


def make_dl_overview_chart(ticker, dl_diagnostics):
    if not dl_diagnostics:
        return

    apply_dark_theme()
    fig = plt.figure(figsize=(18, 9.5), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

    models = [d.get('model', 'DL') for d in dl_diagnostics]
    display_models = [_chart_display_label(name, width=12) for name in models]
    accuracy = [float(d.get('accuracy', 0.0)) * 100 for d in dl_diagnostics]
    f1_vals = [float(d.get('f1', 0.0)) * 100 for d in dl_diagnostics]
    dominant = [float(d.get('dominant_frac', 0.0)) * 100 for d in dl_diagnostics]
    collapsed = [bool(d.get('is_collapsed', False)) for d in dl_diagnostics]
    colors = [RED if is_bad else BLUE for is_bad in collapsed]

    ax1 = fig.add_subplot(gs[0, 0])
    xpos = np.arange(len(models))
    width = 0.35
    ax1.bar(xpos - width / 2, accuracy, width=width, color=colors, alpha=0.85, label='Accuracy')
    ax1.bar(xpos + width / 2, f1_vals, width=width, color=GREEN, alpha=0.75, label='Macro-F1')
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(display_models, rotation=12, ha='right')
    ax1.tick_params(axis='x', labelsize=8, pad=6)
    ax1.margins(x=0.05)
    ax1.set_ylabel('Score (%)', color=TEXT)
    ax1.set_title('DL Model Test Performance', color=TEXT, fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(xpos, dominant, color=colors, alpha=0.85)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(display_models, rotation=12, ha='right')
    ax2.tick_params(axis='x', labelsize=8, pad=6)
    ax2.margins(x=0.05)
    ax2.axhline(80, color=GOLD, lw=0.8, ls='--', alpha=0.7)
    ax2.set_ylabel('Dominant Class Share (%)', color=TEXT)
    ax2.set_title('Collapse / Overconfidence Monitor', color=TEXT, fontsize=11, fontweight='bold')
    for bar, diag in zip(bars, dl_diagnostics):
        temp = float(diag.get('temperature', 1.0))
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f"T={temp:.2f}", ha='center', va='bottom', fontsize=8, color=MUTED)

    fig.suptitle(f'{ticker} â€” DL Model Overview', fontsize=13, fontweight='bold', color=TEXT)
    if getattr(fig, '_suptitle', None) is not None:
        fig._suptitle.set_text(f'{ticker} - DL Model Overview')
    fig.text(0.5, 0.01, 'Red bars indicate collapsed or one-class-dominant behaviour.',
             ha='center', color=MUTED, fontsize=8)
    path = OUT_DIR / f"{ticker}_dl_models.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close('all')
    log.info(f"DL chart saved â†’ {path}")


def make_selection_diagnostics_chart(ticker,
                                     robust_leaderboard,
                                     calibration_leaderboard,
                                     conformal_leaderboard,
                                     router_summary,
                                     selection_payload=None):
    if not robust_leaderboard and not calibration_leaderboard and not conformal_leaderboard:
        return

    apply_dark_theme()
    fig = plt.figure(figsize=(21, 12), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    robust_top = list(robust_leaderboard or [])[:5]
    cal_top = list(calibration_leaderboard or [])[:5]
    conformal_top = list(conformal_leaderboard or [])[:5]
    router_probs = (router_summary or {}).get('router_family_probabilities', {}) or {}
    family_board = (router_summary or {}).get('family_leaderboard', []) or []
    selection_payload = dict(selection_payload or {})

    ax1 = fig.add_subplot(gs[0, 0])
    if robust_top:
        labels = [_chart_display_label(row.get('model', '?'), width=16) for row in robust_top]
        scores = [float(row.get('robust_score', 0.0) or 0.0) for row in robust_top]
        fam_colors = {
            'tree_family': BLUE,
            'dl_family': PURPLE,
            'stack_family': GREEN,
        }
        colors = [fam_colors.get(row.get('family'), GOLD) for row in robust_top]
        bars = ax1.barh(labels, scores, color=colors, alpha=0.85)
        ax1.axvline(0.0, color=BORDER, lw=0.8)
        ax1.set_title('Robust Score Leaderboard', color=TEXT, fontsize=11, fontweight='bold')
        ax1.set_xlabel('Robust score', color=TEXT)
        xmin = min([0.0] + scores)
        xmax = max([0.0] + scores)
        pad = max(0.02, (xmax - xmin) * 0.18 if xmax != xmin else 0.04)
        ax1.set_xlim(xmin - pad, xmax + pad)
        for bar, score, row in zip(bars, scores, robust_top):
            x = bar.get_width()
            ha = 'left' if x >= 0 else 'right'
            offset = 0.01 if x >= 0 else -0.01
            ax1.text(x + offset, bar.get_y() + bar.get_height() / 2,
                     f'{score:+.3f}', va='center', ha=ha, fontsize=7, color=TEXT)
            if not row.get('deployment_eligible', False):
                bar.set_hatch('//')
                bar.set_edgecolor(RED)
                bar.set_linewidth(1.0)
            champion_tags = []
            if row.get('raw_reference_champion'):
                champion_tags.append('RAW')
            if row.get('reference_champion'):
                champion_tags.append('REF')
            if row.get('deployment_champion'):
                champion_tags.append('LIVE')
            if champion_tags:
                ax1.text(
                    xmin - pad * 0.92,
                    bar.get_y() + bar.get_height() / 2,
                    '/'.join(champion_tags),
                    va='center',
                    ha='left',
                    fontsize=7,
                    color=TEXT,
                    bbox=dict(boxstyle='round,pad=0.18', fc=PANEL, ec=BORDER, alpha=0.9),
                )
    else:
        ax1.text(0.5, 0.5, 'No robust-score data', transform=ax1.transAxes,
                 ha='center', va='center', color=MUTED)

    ax2 = fig.add_subplot(gs[0, 1])
    if cal_top:
        labels = [_chart_display_label(row.get('model', '?'), width=14) for row in cal_top]
        ece = [float(row.get('post_calibration_ece', 0.0) or 0.0) for row in cal_top]
        nll = [float(row.get('post_calibration_nll', 0.0) or 0.0) for row in cal_top]
        xpos = np.arange(len(labels))
        ax2.bar(xpos, ece, color=ORANGE, alpha=0.80, width=0.55, label='Post-cal ECE')
        ax2.axhline(0.10, color=GOLD, lw=0.8, ls='--', alpha=0.7, label='ECE target')
        ax2.set_xticks(xpos)
        ax2.set_xticklabels(labels, rotation=10, ha='right')
        ax2.tick_params(axis='x', labelsize=7, pad=6)
        ax2.set_ylabel('ECE', color=TEXT)
        ax2.set_title('Calibration Quality', color=TEXT, fontsize=11, fontweight='bold')
        ax2b = ax2.twinx()
        ax2b.plot(xpos, nll, color=BLUE, marker='o', lw=1.4, label='Post-cal NLL')
        ax2b.set_ylabel('NLL', color=TEXT)
        lines, labels_a = ax2.get_legend_handles_labels()
        lines_b, labels_b = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines_b, labels_a + labels_b, fontsize=8, loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'No calibration data', transform=ax2.transAxes,
                 ha='center', va='center', color=MUTED)

    ax3 = fig.add_subplot(gs[1, 0])
    if conformal_top:
        labels = [row.get('model', '?') for row in conformal_top]
        sharpness = [float(row.get('sharpness', 0.0) or 0.0) for row in conformal_top]
        avg_size = [float(row.get('avg_set_size', 0.0) or 0.0) for row in conformal_top]
        singleton = [float(row.get('singleton_rate', 0.0) or 0.0) for row in conformal_top]
        colors = [GREEN if s >= 0.6 else GOLD if s >= 0.4 else RED for s in sharpness]
        sizes = [max(60.0, 320.0 * s) for s in singleton]
        edge_colors = [
            RED if row.get('blocked_otherwise_healthy_model', False)
            else BLUE if row.get('deployment_eligible', False)
            else BORDER
            for row in conformal_top
        ]
        ax3.scatter(avg_size, sharpness, s=sizes, c=colors, alpha=0.80, edgecolors=edge_colors, linewidths=1.0)
        rounded_points = {}
        for x, y in zip(avg_size, sharpness):
            key = (round(float(x), 2), round(float(y), 3))
            rounded_points[key] = rounded_points.get(key, 0) + 1
        point_seen = {}
        for x, y, label, row in zip(avg_size, sharpness, labels, conformal_top):
            key = (round(float(x), 2), round(float(y), 3))
            idx = point_seen.get(key, 0)
            point_seen[key] = idx + 1
            dx, dy = _scatter_annotation_offsets(rounded_points[key])[idx]
            label_text = _chart_display_label(label, width=14).replace('\n', ' ')
            if row.get('blocked_otherwise_healthy_model', False):
                label_text += ' [BLOCKED]'
            elif row.get('degenerate_execution_conformal', False):
                label_text += ' [DEGEN]'
            ax3.annotate(
                label_text,
                (x, y),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=7,
                color=TEXT,
                bbox=dict(boxstyle='round,pad=0.2', fc=PANEL, ec=BORDER, alpha=0.85),
                arrowprops=dict(arrowstyle='-', color=BORDER, lw=0.6, alpha=0.6),
            )
        ax3.axvline(1.0, color=GOLD, lw=0.8, ls='--', alpha=0.7)
        ax3.margins(x=0.12, y=0.18)
        ax3.set_xlabel('Average set size', color=TEXT)
        ax3.set_ylabel('Sharpness', color=TEXT)
        ax3.set_title('Conformal Sharpness vs Ambiguity', color=TEXT, fontsize=11, fontweight='bold')
        ax3.text(
            0.02, 0.98,
            'Edge color: red = conformal blocked otherwise-healthy model | [DEGEN] = all-class degeneracy',
            transform=ax3.transAxes,
            va='top',
            ha='left',
            fontsize=8,
            color=MUTED,
        )
    else:
        ax3.text(0.5, 0.5, 'No conformal data', transform=ax3.transAxes,
                 ha='center', va='center', color=MUTED)

    ax4 = fig.add_subplot(gs[1, 1])
    if router_probs or family_board:
        fam_labels = list(router_probs.keys()) or [row.get('family', '?') for row in family_board]
        prob_map = {row.get('family'): row.get('robust_score', 0.0) for row in family_board}
        router_vals = [float(router_probs.get(fam, 0.0) or 0.0) for fam in fam_labels]
        robust_vals = [float(prob_map.get(fam, 0.0) or 0.0) for fam in fam_labels]
        xpos = np.arange(len(fam_labels))
        ax4.bar(xpos - 0.18, router_vals, width=0.36, color=PURPLE, alpha=0.80, label='Router prob')
        ax4.bar(xpos + 0.18, robust_vals, width=0.36, color='#00B8D4', alpha=0.75, label='Family robust')
        ax4.set_xticks(xpos)
        ax4.set_xticklabels([_chart_display_label(lbl, width=14) for lbl in fam_labels], rotation=10, ha='right')
        ax4.tick_params(axis='x', labelsize=7, pad=6)
        ax4.set_title('Family Router / Champion View', color=TEXT, fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        chosen = (router_summary or {}).get('chosen_family')
        chosen_model = (router_summary or {}).get('chosen_model')
        conf = float((router_summary or {}).get('confidence', 0.0) or 0.0)
        fallback = bool((router_summary or {}).get('fallback_to_global', False))
        selection_status = selection_payload.get('selection_status', 'n/a')
        ref_model = selection_payload.get('reference_model_used') or 'N/A'
        dep_model = selection_payload.get('deployment_model_used') or 'NONE'
        if chosen or ref_model or dep_model:
            ax4.text(0.02, 0.98,
                     f"Chosen: {chosen or 'NONE'}\n"
                     f"Model: {chosen_model or 'NONE'}\n"
                     f"Ref: {ref_model}\n"
                     f"Live: {dep_model}\n"
                     f"Status: {selection_status}\n"
                     f"Conf: {conf:.2f}\n"
                     f"Fallback: {'YES' if fallback else 'NO'}",
                     transform=ax4.transAxes, va='top', ha='left', fontsize=9, color=MUTED,
                     family='monospace')
    else:
        ax4.text(0.5, 0.5, 'No router data', transform=ax4.transAxes,
                 ha='center', va='center', color=MUTED)

    fig.suptitle(f'{ticker} - Selection Diagnostics', fontsize=13, fontweight='bold', color=TEXT)
    fig.text(0.5, 0.01,
             'Robust-score deployment prioritizes walk-forward quality, CPCV tail behaviour, calibration, and conformal sharpness.',
             ha='center', color=MUTED, fontsize=8)
    path = OUT_DIR / f"{ticker}_selection_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close('all')
    if log is not None:
        log.info(f"Selection diagnostics chart saved -> {path}")


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
    backtest_mode = str(bt.get('backtest_mode', '') or '')

    ax4.plot(bhc.index, bhc.values,   color=MUTED,  lw=1.0, ls='--',
             label=f"Buy&Hold {bt['bh_return']*100:+.1f}%")
    if backtest_mode == 'selected_candidate_holdout':
        ax4.plot(sc.index, sc.values, color=GREEN, lw=1.2,
                 label=f"{bt.get('strategy_model', 'Selected')} {bt['strat_return']*100:+.1f}%")
    else:
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
    global RUN_METADATA, _EXPERIMENT_RECORDED, TELEGRAM_PROGRESS_SESSION
    run_start = time.time()
    ticker    = ask_ticker()
    init_run(ticker)
    TELEGRAM_PROGRESS_SESSION = _analyzer_progress_session(ticker)
    runtime_overrides = _apply_runtime_overrides()
    _EXPERIMENT_RECORDED = False
    RUN_METADATA = build_run_metadata(
        mode='single_stock_ml',
        seed=CONFIG.get('random_state', 42),
        config=CONFIG,
        config_version=CONFIG.get('config_version', DEFAULT_CONFIG_VERSION),
        enabled_models=_enabled_model_names(CONFIG),
        universe=[ticker],
        extra={'ticker': ticker, 'runtime_overrides': runtime_overrides},
    )
    stage_errors = []
    stage_defs = [('data_context', 'Data + context')]
    if CONFIG.get('fetch_fundamentals', True):
        stage_defs.append(('fundamentals_fetch', 'Fundamentals'))
    stage_defs.append(('regime_features_labels', 'Regime + features + labels'))
    stage_defs.append(('tree_models', 'Tree models'))
    stage_defs.append(('regime_model', 'Regime model'))
    if _has_enabled_dl_models(CONFIG):
        stage_defs.append(('dl_models', 'DL models'))
    stage_defs.append(('meta_final_signal', 'Selection + router'))
    if CONFIG.get('conformal_on_final_signal', True):
        stage_defs.append(('conformal_calibration', 'Conformal calibration'))
    stage_defs.append(('backtests', 'Backtests'))
    stage_defs.append(('charts_output', 'Charts + output'))
    progress = ProgressTracker(ticker, TELEGRAM_PROGRESS_SESSION, stage_defs)
    progress.start()
    if CONFIG.get('seed_everything', True):
        seed_everything(CONFIG.get('random_state', 42),
                        deterministic_torch=CONFIG.get('torch_deterministic', True))

    log.info("*"*60)
    log.info(f"  {ticker}  ML ANALYZER  v12.0")
    log.info(f"  Output: {OUT_DIR.resolve()}")
    if runtime_overrides.get("run_diagnostics_forced"):
        log.info(f"  Runtime override: diagnostics forced ON ({runtime_overrides.get('run_context', 'external')})")
    log.info("*"*60)

    device, gpu_name = get_device()
    use_gpu = (device.type == 'cuda')

    fetcher = DataFetcher(ticker, CONFIG['period'])
    progress.stage_started('data_context')
    try:
        df = fetcher.fetch()
    except Exception as e:
        progress.stage_failed('data_context', e)
        _record_stage_error(stage_errors, 'data_fetch', e, fatal=True)
        _write_failure_signal_json(ticker, stage_errors)
        return

    try:
        market = fetcher.fetch_context()
    except Exception as e:
        _record_stage_error(stage_errors, 'context_fetch', e, fatal=False)
        market = pd.DataFrame()
    progress.stage_done('data_context')

    # ── NEW 3: Fundamental / alternative data ────────────────────────────────
    fundamentals = {}
    if CONFIG.get('fetch_fundamentals', True):
        progress.stage_started('fundamentals_fetch')
        log_section("FUNDAMENTAL / ALTERNATIVE DATA")
        try:
            fundamentals = FundamentalFetcher(ticker).fetch()
        except Exception as e:
            progress.stage_failed('fundamentals_fetch', e)
            _record_stage_error(stage_errors, 'fundamentals', e, fatal=False)
            fundamentals = {}
        else:
            progress.stage_done('fundamentals_fetch')

    # ── Per-stock adaptive horizon + quantile thresholds ─────────────────────
    progress.stage_started('regime_features_labels')
    try:
        regime = analyze_stock_regime(df, CONFIG)
    except Exception as e:
        progress.stage_failed('regime_features_labels', e)
        _record_stage_error(stage_errors, 'regime_analysis', e, fatal=True)
        _write_failure_signal_json(ticker, stage_errors, partial={'fundamentals': fundamentals})
        return
    CONFIG['predict_days'] = regime['predict_days']

    try:
        fe   = FeatureEngineer(df, market, fundamentals=fundamentals)
        feat = fe.build()
    except Exception as e:
        progress.stage_failed('regime_features_labels', e)
        _record_stage_error(stage_errors, 'feature_generation', e, fatal=True)
        _write_failure_signal_json(ticker, stage_errors, partial={'fundamentals': fundamentals})
        return

    try:
        y = make_labels(df, regime)
    except Exception as e:
        progress.stage_failed('regime_features_labels', e)
        _record_stage_error(stage_errors, 'label_generation', e, fatal=True)
        _write_failure_signal_json(ticker, stage_errors, partial={'fundamentals': fundamentals})
        return
    progress.stage_done('regime_features_labels')

    common = feat.index.intersection(y.dropna().index)
    X, y   = feat.loc[common], y.loc[common]
    event_meta = _slice_event_meta(y.attrs.get('event_meta'), common)

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

    progress.stage_started('tree_models')
    tree = TreeTrainer(X, y, CONFIG, use_gpu=use_gpu, event_meta=event_meta)
    try:
        tree.train()
    except Exception as e:
        progress.stage_failed('tree_models', e)
        _record_stage_error(stage_errors, 'training', e, fatal=True)
        _write_failure_signal_json(
            ticker, stage_errors,
            partial={'fundamentals': fundamentals, 'regime': regime}
        )
        return
    progress.stage_done('tree_models')

    # NEW 13: Regime-conditional model
    progress.stage_started('regime_model')
    regime_mdl = RegimeModel(market, CONFIG)
    try:
        regime_ok  = regime_mdl.train(X, y)
        regime_sig = regime_mdl.predict_latest(tree.X.iloc[[-1]]) if regime_ok else None
    except Exception as e:
        progress.stage_failed('regime_model', e)
        _record_stage_error(stage_errors, 'regime_model', e, fatal=False)
        regime_ok = False
        regime_sig = None
    else:
        progress.stage_done('regime_model')

    dl_trainers = []
    def _train_dl_stage(stage_name: str, trainer, model_class, model_kwargs,
                        epochs: int, patience: int, seq_len: int, lr: float, batch: int):
        try:
            trainer.train(X_dl, y, ModelClass=model_class,
                          model_kwargs=model_kwargs,
                          epochs=epochs,
                          patience=patience,
                          seq_len=seq_len,
                          lr=lr,
                          batch=batch)
            dl_trainers.append(trainer)
            return True
        except Exception as e:
            _record_stage_error(stage_errors, stage_name, e, fatal=False)
            log.warning(f"{trainer.name} training failed ({e}) - continuing without it")
            if use_gpu:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return False

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
        if _is_model_enabled('BiLSTM-Attention', CONFIG):
            lstm_kw = get_adaptive_dl_kwargs(
                'bilstm', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
            lstm_t = TorchTrainer("BiLSTM-Attention", device, CONFIG)
            lstm_t._pre_scaled = True   # X_dl already tree.scaler-transformed
            _train_dl_stage('dl_bilstm', lstm_t, BiLSTMAttention,
                            lstm_kw['model_kw'],
                            lstm_kw['epochs'],
                            lstm_kw['patience'],
                            CONFIG['lstm_seq_len'],
                            CONFIG['lstm_lr'],
                            CONFIG['lstm_batch'])
        else:
            log.info("  Skipping BiLSTM-Attention (disabled in active_dl_models)")

        if use_gpu:
            torch.cuda.empty_cache()

        # ── Transformer (adaptive size) ───────────────────────────────────────
        if _is_model_enabled('Transformer-Encoder', CONFIG):
            tf_kw = get_adaptive_dl_kwargs(
                'transformer', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
            tf_t = TorchTrainer("Transformer-Encoder", device, CONFIG)
            tf_t._pre_scaled = True
            _train_dl_stage('dl_transformer', tf_t, TransformerEncoder,
                            tf_kw['model_kw'],
                            tf_kw['epochs'],
                            tf_kw['patience'],
                            CONFIG['lstm_seq_len'],
                            CONFIG['lstm_lr'],
                            CONFIG['lstm_batch'])
        else:
            log.info("  Skipping Transformer-Encoder (disabled in active_dl_models)")

        if use_gpu:
            torch.cuda.empty_cache()

        # ── NEW 16: Temporal Fusion Transformer (adaptive size) ───────────────
        if _is_model_enabled('TFT', CONFIG):
            tft_kw = get_adaptive_dl_kwargs(
                'tft', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
            tft_t = TorchTrainer("TFT", device, CONFIG)
            tft_t._pre_scaled  = True
            tft_t._feat_names  = list(X_dl.columns)  # for VSN importance logging
            _train_dl_stage('dl_tft', tft_t, TemporalFusionTransformer,
                            tft_kw['model_kw'],
                            tft_kw['epochs'],
                            tft_kw['patience'],
                            CONFIG['lstm_seq_len'],
                            CONFIG['tft_lr'],
                            CONFIG['lstm_batch'])
        else:
            log.info("  Skipping TFT (disabled in active_dl_models)")

        if _is_model_enabled('TiDE', CONFIG):
            if use_gpu:
                torch.cuda.empty_cache()
            tide_kw = get_adaptive_dl_kwargs(
                'tide', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
            tide_t = TorchTrainer("TiDE", device, CONFIG)
            tide_t._pre_scaled = True
            _train_dl_stage('dl_tide', tide_t, TiDEClassifier,
                            tide_kw['model_kw'],
                            tide_kw['epochs'],
                            tide_kw['patience'],
                            CONFIG['lstm_seq_len'],
                            CONFIG['tide_lr'],
                            CONFIG['lstm_batch'])

        if _is_model_enabled('PatchTST', CONFIG):
            if use_gpu:
                torch.cuda.empty_cache()
            patch_kw = get_adaptive_dl_kwargs(
                'patchtst', n_tr_rows, n_feat_dl, CONFIG, use_gpu=use_gpu)
            patch_t = TorchTrainer("PatchTST", device, CONFIG)
            patch_t._pre_scaled = True
            _train_dl_stage('dl_patchtst', patch_t, PatchTSTClassifier,
                            patch_kw['model_kw'],
                            patch_kw['epochs'],
                            patch_kw['patience'],
                            CONFIG['lstm_seq_len'],
                            CONFIG['patchtst_lr'],
                            CONFIG['lstm_batch'])

    else:
        log.warning("PyTorch not installed — skipping LSTM & Transformer")

    progress.stage_done('dl_models')
    progress.stage_started('meta_final_signal')
    meta_signal = build_meta(tree, dl_trainers) if (dl_trainers and _is_model_enabled('Meta-Stack', CONFIG)) else None
    tree_signal = tree.get_latest_signal()
    dl_signals  = [t.predict_latest() for t in dl_trainers]

    # ── FIX D: Improved signal fallback chain ─────────────────────────────────
    # When meta_signal is None (either no DL or very-low-confidence filter):
    #   1. Gather all non-collapsed candidate signals with their confidence
    #   2. Prefer majority-vote direction when signals agree (2+ of tree/DL/regime)
    #   3. Tie-break on confidence
    # This prevents regime (single HMM bar) from silently overriding a
    # high-confidence tree signal just because it happens to be first in the chain.
    tree_candidates = _collect_tree_candidates(tree, df['Close'])
    dl_candidates = _collect_dl_candidates(dl_trainers, df['Close'])
    stack_candidates = _collect_stack_candidate(tree, meta_signal, df['Close'])
    candidate_pool = tree_candidates + dl_candidates + stack_candidates
    selection_payload = {}
    decision_policy = {}
    conformal_result = {}
    calibration_diag = {}
    router_summary = {}
    robust_leaderboard = []
    calibration_leaderboard = []
    conformal_leaderboard = []
    selected_candidate = None
    reference_champion = None
    raw_reference_champion = None
    deployment_champion = None
    final_signal = tree_signal
    if candidate_pool:
        for cand in candidate_pool:
            _assess_candidate_deployment(cand)
        pool_conformal_degenerate = False
        pool_conformal_degenerate_reason = None
        conformal_rescue_candidate_count = 0

        # ── Globally-degenerate conformal bypass ──────────────────────────────
        # When every single candidate has avg_set_size ≈ n_classes (i.e. the
        # prediction sets are always the full {SELL, HOLD, BUY}), conformal is
        # providing no signal for any model — this is a property of the stock's
        # probability landscape at this horizon, not a model quality problem.
        # In this case, blocking ALL candidates on conformal_unusable /
        # conformal_singleton_floor produces a permanent ABSTAIN even when
        # models have real directional skill (e.g. PatchTST non_hold_recall=0.17,
        # macro_pr_auc=0.44 in the MSFT case).
        # Fix: detect the degenerate state pool-wide and re-run eligibility
        # without conformal gates so the directional/calibration gates alone
        # decide deployment.
        conformal_only_failures = {'conformal_unusable', 'conformal_singleton_floor',
                                   'degenerate_conformal_sets'}
        # Check ALL candidates — the old `if cand.get('conformal')` guard silently
        # excluded candidates with empty conformal dicts ({} is falsy).  When some
        # candidates have no conformal data, all() over only the non-empty subset
        # can vacuously return True and trigger the bypass incorrectly.
        # Correct logic: the pool is degenerate only if EVERY candidate (including
        # those with empty dicts) reports a degenerate or missing conformal result.
        # _is_degenerate_conformal_stats({}) → False (full_set_rate=0.0 < 0.99),
        # so a candidate with no conformal data correctly prevents the bypass.
        pool_is_conformal_degenerate = bool(candidate_pool) and all(
            _is_degenerate_conformal_stats(cand.get('conformal', {}), n_classes=3)
            for cand in candidate_pool
        )

        if pool_is_conformal_degenerate:
            pool_conformal_degenerate = True
            pool_conformal_degenerate_reason = 'globally_degenerate_pool'
            log.warning(
                "All candidates have degenerate conformal sets (avg_set_size ≈ 3, "
                "singleton_rate = 0%) — conformal is uniformly uninformative for "
                f"this stock/horizon. Re-assessing eligibility without conformal gates."
            )
            for cand in candidate_pool:
                new_failures = [
                    f for f in (cand.get('eligibility_failures') or [])
                    if f not in conformal_only_failures
                ]
                # Mark conformal bypass on the conformal sub-dict for transparency
                conf = dict(cand.get('conformal', {}) or {})
                conf['conformal_bypass_active'] = True
                conf['conformal_bypass_reason'] = 'globally_degenerate_pool'
                conf['usable_for_execution']    = True   # bypass the gate
                cand['conformal'] = conf
                cand['eligibility_failures'] = new_failures
                cand['deployment_eligible']  = not new_failures

        eligible_candidates = [cand for cand in candidate_pool if cand.get('deployment_eligible', False)]
        if not eligible_candidates and bool(CONFIG.get('conformal_rescue_enabled', True)):
            rescue_candidates = [
                cand for cand in candidate_pool
                if _candidate_conformal_rescue_ready(cand, CONFIG)
            ]
            if rescue_candidates:
                conformal_rescue_candidate_count = int(len(rescue_candidates))
                log.warning(
                    "No deployable candidates survived strict conformal gating, "
                    "but one or more candidates were blocked only by conformal despite "
                    "strong latest-signal confidence/margin. Activating conformal rescue."
                )
                for cand in rescue_candidates:
                    conf = dict(cand.get('conformal', {}) or {})
                    conf['conformal_bypass_active'] = True
                    conf['conformal_bypass_reason'] = 'confidence_margin_override'
                    conf['usable_for_execution'] = True
                    conf['rescued_from_failures'] = list(cand.get('eligibility_failures', []) or [])
                    cand['conformal'] = conf
                    cand['eligibility_failures'] = []
                    cand['deployment_eligible'] = True

        raw_reference_champion = max(candidate_pool, key=_raw_candidate_robust_score)
        reference_champion = max(candidate_pool, key=_candidate_sort_key)
        eligible_candidates = [cand for cand in candidate_pool if cand.get('deployment_eligible', False)]
        deployment_champion = max(eligible_candidates, key=_candidate_sort_key) if eligible_candidates else None
        robust_leaderboard = [
            {
                'model': cand.get('name'),
                'family': cand.get('family'),
                'robust_score': float((cand.get('selection', {}) or {}).get('robust_score', 0.0)),
                'selection_rank_score': _selection_rank_score(cand),
                'directional_penalty': _directional_selection_penalty(cand),
                'deployment_eligible': bool(cand.get('deployment_eligible', False)),
                'eligibility_failures': list(cand.get('eligibility_failures', []) or []),
                'macro_pr_auc': float((cand.get('evaluation', {}) or {}).get('macro_pr_auc', 0.0) or 0.0),
                'non_hold_recall_min': float((cand.get('evaluation', {}) or {}).get('non_hold_recall_min', 0.0) or 0.0),
                'conformal_sharpness': float((cand.get('conformal', {}) or {}).get('sharpness', 0.0) or 0.0),
                'degenerate_execution_conformal': bool((cand.get('conformal', {}) or {}).get('degenerate_execution_conformal', False)),
                'raw_reference_champion': bool(raw_reference_champion is not None and cand.get('name') == raw_reference_champion.get('name')),
                'reference_champion': bool(reference_champion is not None and cand.get('name') == reference_champion.get('name')),
                'deployment_champion': bool(deployment_champion is not None and cand.get('name') == deployment_champion.get('name')),
            }
            for cand in sorted(candidate_pool, key=_candidate_sort_key, reverse=True)
        ]
        calibration_leaderboard = [
            {
                'model': cand.get('name'),
                'family': cand.get('family'),
                'post_calibration_nll': float((cand.get('calibration', {}) or {}).get('post_calibration_nll', 0.0)),
                'post_calibration_ece': float((cand.get('calibration', {}) or {}).get('post_calibration_ece', (cand.get('calibration', {}) or {}).get('ece', 0.0))),
                'macro_pr_auc': float((cand.get('evaluation', {}) or {}).get('macro_pr_auc', 0.0) or 0.0),
                'deployment_eligible': bool(cand.get('deployment_eligible', False)),
            }
            for cand in sorted(candidate_pool, key=lambda cand: (cand.get('calibration', {}) or {}).get('post_calibration_nll', np.inf))
        ]
        conformal_leaderboard = [
            {
                'model': cand.get('name'),
                'family': cand.get('family'),
                'sharpness': float((cand.get('conformal', {}) or {}).get('sharpness', 0.0)),
                'avg_set_size': float((cand.get('conformal', {}) or {}).get('avg_set_size', 0.0)),
                'singleton_rate': float((cand.get('conformal', {}) or {}).get('singleton_rate', 0.0)),
                'usable_for_execution': bool((cand.get('conformal', {}) or {}).get('usable_for_execution', False)),
                'blocked_otherwise_healthy_model': bool((cand.get('conformal', {}) or {}).get('blocked_otherwise_healthy_model', False)),
                'degenerate_execution_conformal': bool((cand.get('conformal', {}) or {}).get('degenerate_execution_conformal', False)),
                'deployment_eligible': bool(cand.get('deployment_eligible', False)),
            }
            for cand in sorted(candidate_pool, key=lambda cand: (cand.get('conformal', {}) or {}).get('sharpness', -np.inf), reverse=True)
        ]
        router_summary = _route_family(candidate_pool, df, market, fundamentals, regime_sig=regime_sig) if candidate_pool else {
            'chosen_family': None,
            'chosen_model': None,
            'confidence': 0.0,
            'fallback_to_global': True,
            'fallback_reason': 'no_candidate_pool',
            'global_champion_family': None,
            'global_champion_model': None,
            'eligible_global_champion_family': None,
            'eligible_global_champion_model': None,
            'family_leaderboard': [],
            'selection_frequency': {},
            'enabled': True,
            'router_status': 'fallback',
            'routing_actionable': False,
            'router_family_probabilities': {},
        }
        selected_name = router_summary.get('chosen_model')
        selected_candidate = next(
            (
                cand for cand in candidate_pool
                if cand.get('name') == selected_name and cand.get('deployment_eligible', False)
            ),
            None,
        )
        if selected_candidate is None:
            selected_candidate = deployment_champion or reference_champion
        final_signal = dict(selected_candidate['latest_signal'])
        selection_payload = dict(selected_candidate.get('selection', {}))
        selection_payload['family'] = selected_candidate.get('family')
        selection_payload['deployment_eligible'] = bool(selected_candidate.get('deployment_eligible', False))
        selection_payload['eligibility_failures'] = list(selected_candidate.get('eligibility_failures', []) or [])
        selection_payload['evaluation'] = dict(selected_candidate.get('evaluation', {}) or {})
        family_ranks = {
            row.get('family'): idx + 1
            for idx, row in enumerate((router_summary.get('family_leaderboard', []) or []))
        }
        selection_payload['family_rank'] = int(family_ranks.get(selected_candidate.get('family'), 1))
        selection_payload['raw_reference_champion'] = _candidate_public_summary(raw_reference_champion)
        selection_payload['reference_champion'] = _candidate_public_summary(reference_champion)
        selection_payload['deployment_champion'] = _candidate_public_summary(deployment_champion)
        selection_payload['raw_reference_model_used'] = (raw_reference_champion or {}).get('name')
        selection_payload['raw_reference_family_used'] = (raw_reference_champion or {}).get('family')
        selection_payload['reference_model_used'] = (reference_champion or {}).get('name')
        selection_payload['deployment_model_used'] = (deployment_champion or {}).get('name')
        selection_payload['reference_family_used'] = (reference_champion or {}).get('family')
        selection_payload['deployment_family_used'] = (deployment_champion or {}).get('family')
        selection_payload['reference_matches_deployment'] = bool(
            reference_champion is not None
            and deployment_champion is not None
            and reference_champion.get('name') == deployment_champion.get('name')
        )
        selection_payload['selection_status'] = (
            'deployed' if deployment_champion is not None else 'reference_only_no_deployable_candidate'
        )
        selection_payload['candidate_counts'] = {
            'total': int(len(candidate_pool)),
            'eligible_total': int(len(eligible_candidates)),
            'by_family': _candidate_family_counts(candidate_pool),
            'eligible_by_family': _candidate_family_counts(eligible_candidates),
        }
        conformal_blocked_models = [
            cand.get('name')
            for cand in candidate_pool
            if (cand.get('conformal', {}) or {}).get('blocked_otherwise_healthy_model', False)
        ]
        selection_payload['conformal_blocked_otherwise_healthy_count'] = int(len(conformal_blocked_models))
        selection_payload['conformal_blocked_models'] = conformal_blocked_models
        selection_payload['conformal_rescue_candidate_count'] = int(conformal_rescue_candidate_count)
        selection_payload['pool_conformal_degenerate'] = bool(pool_conformal_degenerate)
        selection_payload['pool_conformal_degenerate_reason'] = pool_conformal_degenerate_reason
        selection_payload['directional_rejection_counts'] = _candidate_rejection_counts(candidate_pool, directional_only=True)
        selection_payload['all_rejection_counts'] = _candidate_rejection_counts(candidate_pool, directional_only=False)
        selection_payload['reference_robust_score'] = float(
            ((reference_champion or {}).get('selection', {}) or {}).get('robust_score', 0.0) or 0.0
        )
        selection_payload['raw_reference_robust_score'] = float(
            ((raw_reference_champion or {}).get('selection', {}) or {}).get('robust_score', 0.0) or 0.0
        )
        selection_payload['reference_rank_score'] = (
            _selection_rank_score(reference_champion) if reference_champion is not None else None
        )
        selection_payload['raw_reference_rank_score'] = (
            _selection_rank_score(raw_reference_champion) if raw_reference_champion is not None else None
        )
        selection_payload['deployment_robust_score'] = (
            float(((deployment_champion or {}).get('selection', {}) or {}).get('robust_score', 0.0) or 0.0)
            if deployment_champion is not None else None
        )
        selection_payload['deployment_rank_score'] = (
            _selection_rank_score(deployment_champion) if deployment_champion is not None else None
        )
        decision_policy = dict(selected_candidate.get('decision_policy', {}))
        conformal_result = dict(selected_candidate.get('conformal', {}))
        calibration_diag = dict(selected_candidate.get('calibration', {}))
        calibration_diag['model_name'] = selected_candidate.get('name')
        calibration_diag['source_type'] = selected_candidate.get('family')
        final_signal['raw_reference_model_used'] = (raw_reference_champion or {}).get('name')
        final_signal['reference_model_used'] = (reference_champion or {}).get('name')
        final_signal['deployment_model_used'] = (deployment_champion or {}).get('name')
        final_signal['raw_reference_family_used'] = (raw_reference_champion or {}).get('family')
        final_signal['reference_family_used'] = (reference_champion or {}).get('family')
        final_signal['deployment_family_used'] = (deployment_champion or {}).get('family')
        final_signal['selection_status'] = selection_payload.get('selection_status')
        final_signal['model_used'] = (deployment_champion or {}).get('name')
        final_signal['family_used'] = (deployment_champion or {}).get('family')
        final_signal['robust_score'] = float(selection_payload.get('deployment_robust_score') or 0.0)
        final_signal['deployment_rank_metric'] = CONFIG.get('deployment_ranking_metric', 'robust_score')
        final_signal['deployment_eligible'] = bool(deployment_champion is not None)
        final_signal['conformal_rescue_candidate_count'] = int(conformal_rescue_candidate_count)
        final_signal['pool_conformal_degenerate'] = bool(pool_conformal_degenerate)
        final_signal['pool_conformal_degenerate_reason'] = pool_conformal_degenerate_reason
        final_signal['eligibility_failures'] = (
            list((deployment_champion or {}).get('eligibility_failures', []) or [])
            if deployment_champion is not None
            else ['no_deployable_candidate']
        )
        if deployment_champion is None:
            # No deployable model — force the public signal to HOLD.
            # MUST also update label (1 = HOLD in {0:SELL, 1:HOLD, 2:BUY}) and
            # confidence to the HOLD class probability so the dashboard doesn't
            # show a contradictory "HOLD at 54% confidence" where 54% was the
            # SELL probability of the reference model.
            final_signal['signal'] = 'HOLD'
            final_signal['label']  = 1   # HOLD index
            probs = final_signal.get('probabilities', {}) or {}
            if probs:
                # Use the HOLD class probability as the reported confidence
                final_signal['confidence'] = float(probs.get('HOLD', 1.0 / 3.0))
        log.info(
            f"Robust selection → {final_signal['signal']} via {final_signal.get('model_used','?')} "
            f"[{final_signal.get('family_used','?')}] robust_score={final_signal.get('robust_score', 0.0):+.4f}"
        )
        if router_summary:
            log.info(
                f"Router → family={router_summary.get('chosen_family', '?')}  "
                f"model={router_summary.get('chosen_model', '?')}  "
                f"conf={router_summary.get('confidence', 0.0):.2f}  "
                f"fallback={'YES' if router_summary.get('fallback_to_global') else 'NO'}"
            )
    elif regime_sig and not regime_sig.get('is_collapsed', False):
        final_signal = regime_sig
        log.warning("No calibrated candidate pool available — falling back to regime signal.")
    else:
        final_signal = tree_signal
        log.warning("No calibrated candidate pool available — falling back to tree signal.")

    progress.stage_done('meta_final_signal')
    final_signal['date'] = str(df.index[-1].date())
    seed_stability = _compute_seed_stability_for_tree(X, y, tree)
    final_signal = _apply_confidence_adjustments(final_signal, adv_report, seed_stability, CONFIG)

    # ── NEW 17: Conformal Prediction — calibrate on best DL model ────────────
    progress.stage_started('conformal_calibration')
    conformal_result = _preserve_conformal_result(conformal_result)
    tree._conformal_predictor = None
    conformal_source = None
    if False and CONFIG.get('conformal_on_final_signal', True):
        conformal_source = _resolve_conformal_source(final_signal, tree, dl_trainers, regime_mdl)
    if conformal_source is not None:
        try:
            log_section("CONFORMAL PREDICTION (NEW 17)")
            alpha    = CONFIG.get('conformal_alpha', 0.10)
            cal_pct  = CONFIG.get('conformal_cal_pct', 0.25)
            lam      = CONFIG.get('conformal_lambda', 0.01)
            kreg     = CONFIG.get('conformal_kreg', 1)
            min_cal  = int(CONFIG.get('conformal_min_cal_rows', 20))
            cp = ConformalPredictor(alpha=alpha, lam=lam, kreg=kreg)

            proba_all = np.asarray(conformal_source['proba_eval'], dtype=np.float64)
            y_all     = np.asarray(conformal_source['y_eval'])
            n_cal     = max(min_cal, int(len(y_all) * cal_pct))
            if len(y_all) < (min_cal + 5) or n_cal >= len(y_all):
                raise ValueError(f"insufficient chronological rows for calibration ({len(y_all)})")
            # Use first n_cal as calibration (chronological — no look-ahead)
            proba_cal, proba_eval = proba_all[:n_cal], proba_all[n_cal:]
            y_cal,     y_eval     = y_all[:n_cal],     y_all[n_cal:]

            cp.calibrate(proba_cal, y_cal)
            cov_stats = cp.empirical_coverage(proba_eval, y_eval)
            cov_stats['calibrated_model'] = conformal_source['model_name']
            cov_stats['source_type'] = conformal_source.get('source_type', 'unknown')
            if conformal_source.get('source_type') == 'tree':
                tree._conformal_predictor = cp
            log.info(f"  Model         : {conformal_source['model_name']}")
            log.info(f"  Source type   : {conformal_source.get('source_type', 'unknown')}")
            log.info(f"  Alpha (1-cov) : {alpha:.0%}  →  target ≥{1-alpha:.0%} coverage")
            log.info(f"  Empirical cov : {cov_stats['coverage']:.3f}  "
                     f"(target {cov_stats['target_coverage']:.3f})")
            log.info(f"  Avg set size  : {cov_stats['avg_set_size']:.2f}")
            log.info(f"  Singleton rate: {cov_stats['singleton_rate']:.1%}  "
                     f"← fraction with unambiguous signal")
            log.info(f"  qhat          : {cov_stats['qhat']:.4f}")

            cp_annot = cp.annotate_signal(conformal_source['latest_proba'])
            final_signal.update(cp_annot)
            final_signal['conformal_model'] = conformal_source['model_name']
            if cp_annot.get('is_conformal_singleton'):
                log.info(f"  Final signal prediction set: "
                         f"{cp_annot['prediction_set']}  ← SINGLETON (high confidence)")
            else:
                log.info(f"  Final signal prediction set: "
                         f"{cp_annot['prediction_set']}  "
                         f"(set_size={cp_annot['set_size']} — consider reducing size)")
            conformal_result = cov_stats
        except Exception as e:
            progress.stage_failed('conformal_calibration', e)
            _record_stage_error(stage_errors, 'calibration', e, fatal=False)
    calibration_diag = calibration_diag or {}
    if conformal_source is not None:
        try:
            calibration_diag = _compute_calibration_diagnostics(
                conformal_source['proba_eval'],
                conformal_source['y_eval'],
                n_bins=CONFIG.get('calibration_bins', 5))
            calibration_diag['model_name'] = conformal_source['model_name']
            calibration_diag['source_type'] = conformal_source.get('source_type', 'unknown')
        except Exception as e:
            _record_stage_error(stage_errors, 'calibration_diagnostics', e, fatal=False)

    if selected_candidate is not None:
        selected_candidate['conformal'] = _assess_conformal_usability(conformal_result, n_classes=3)
        conformal_result = dict(selected_candidate['conformal'])
        final_signal.update(_build_execution_state(final_signal, selected_candidate, router_summary))
    else:
        final_signal.update(_build_execution_state(final_signal, None, router_summary))
    progress.stage_done('conformal_calibration')

    progress.stage_started('backtests')
    tree_bt = backtest(df, tree)
    tree_wf_bt = backtest_walkforward(df, tree)  # NEW 14
    tree_cpcv_res = run_cpcv(tree, df)  # NEW 12
    selected_family = (selected_candidate or {}).get('family')
    bt = dict((selected_candidate or {}).get('holdout_backtest', {}) or {})
    if not bt:
        bt = dict(tree_bt or {})
    if selected_family == 'tree_family':
        bt['audit'] = _augment_backtest_audit_with_oos_context(
            (bt or {}).get('audit', {}),
            bt,
            tree_wf_bt,
            tree_cpcv_res,
        )
    else:
        bt['audit'] = dict((bt or {}).get('audit', {}) or {})
    wf_bt = tree_wf_bt if selected_family == 'tree_family' else {}
    cpcv_res = tree_cpcv_res if selected_family == 'tree_family' else {}
    tree_family_diagnostics = {
        'backtest': {k: v for k, v in (tree_bt or {}).items() if not isinstance(v, pd.Series)},
        'backtest_audit': dict((tree_bt or {}).get('audit', {}) or {}),
        'walkforward_backtest': {k: v for k, v in (tree_wf_bt or {}).items() if not isinstance(v, pd.Series)},
        'cpcv': dict(tree_cpcv_res or {}),
        'scope': 'tree_family',
    }
    evidence_scope = {
        'selected_candidate_model': (selected_candidate or {}).get('name'),
        'selected_candidate_family': selected_family,
        'selected_candidate_is_tree': bool(selected_family == 'tree_family'),
        'static_backtest_scope': 'selected_candidate',
        'walkforward_scope': 'selected_candidate' if selected_family == 'tree_family' else 'tree_family_diagnostics',
        'cpcv_scope': 'selected_candidate' if selected_family == 'tree_family' else 'tree_family_diagnostics',
    }
    feat_imp = tree.get_feature_importance()
    explainability = _generate_final_explainability(final_signal, tree, dl_trainers, regime_mdl)
    progress.stage_done('backtests')
    progress.stage_started('charts_output')
    try:
        make_charts(ticker, df, tree, bt, final_signal, feat_imp)
    except Exception as e:
        _record_stage_error(stage_errors, 'chart_generation', e, fatal=False)

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
            'collapse_reason': getattr(t, 'collapse_reason_', None),
            'temperature':  getattr(t, 'temperature',  1.0),
            'calibration':  getattr(t, 'calibration_report_', {}),
            'accuracy':     float(getattr(t, 'eval_acc_', 0.0) or 0.0),
            'f1':           float(getattr(t, 'eval_f1_', 0.0) or 0.0),
            'dominant_frac': float(getattr(t, 'eval_dominant_frac_', 0.0) or 0.0),
            'predicted_classes': int(getattr(t, 'eval_predicted_classes_', 0) or 0),
            'pred_counts':  getattr(t, 'eval_pred_counts_', {}),
        })
    try:
        make_dl_overview_chart(ticker, dl_diagnostics)
    except Exception as e:
        _record_stage_error(stage_errors, 'dl_chart_generation', e, fatal=False)
    try:
        make_selection_diagnostics_chart(
            ticker,
            robust_leaderboard,
            calibration_leaderboard,
            conformal_leaderboard,
            router_summary,
            selection_payload,
        )
    except Exception as e:
        _record_stage_error(stage_errors, 'selection_chart_generation', e, fatal=False)

    completed_run_metadata = complete_run_metadata(
        RUN_METADATA,
        status='OK' if not any(err.get('fatal') for err in stage_errors) else 'FAILED',
    )

    out = {
        'ticker': ticker, 'generated': datetime.datetime.now().isoformat(),
        'schema_version': SIGNAL_SCHEMA_VERSION,
        'signal': final_signal, 'tree_signal': tree_signal,
        'dl_signals': dl_signals, 'meta_signal': meta_signal,
        'dl_diagnostics': dl_diagnostics,
        'seed_stability': seed_stability,
        'calibration_diagnostics': calibration_diag,
        'calibration': calibration_diag,
        'selection': selection_payload,
        'selection_leaderboard': robust_leaderboard,
        'decision_policy': decision_policy,
        'router': router_summary,
        'calibration_leaderboard': calibration_leaderboard,
        'conformal_leaderboard': conformal_leaderboard,
        'explainability': explainability,
        'ensemble_weights': getattr(tree, 'ensemble_weights_', {}),
        'pipeline_errors': stage_errors,
        'pipeline_status': 'OK' if not any(err.get('fatal') for err in stage_errors) else 'FAILED',
        'run_metadata': completed_run_metadata,
        'backtest': {k: v for k, v in bt.items() if not isinstance(v, pd.Series)},
        'backtest_audit': dict((bt or {}).get('audit', {}) or {}),
        'tree_family_diagnostics': tree_family_diagnostics,
        'evidence_scope': evidence_scope,
        'model_accuracy': all_acc, 'gpu_info': gpu_name,
        'regime': {
            'predict_days':   regime['predict_days'],
            'buy_thresh':     regime['buy_thresh'],
            'sell_thresh':    regime['sell_thresh'],
            'ann_vol':        regime['ann_vol'],
            'speed':          regime['speed'],
            'best_ic':        regime['best_ic'],
            'ic_table':       regime['ic_table'],
            'label_method':   y.attrs.get('label_method', CONFIG.get('label_method', 'rar')),
        },
        'fundamentals': fundamentals,
        'adversarial_validation': adv_report,
        'cpcv':          cpcv_res,
        'walkforward_backtest': {k: v for k, v in (wf_bt or {}).items()
                                 if not isinstance(v, pd.Series)},
        'regime_signal': regime_sig,
        'conformal_prediction': conformal_result,
        'conformal': conformal_result,
        'total_runtime': elapsed(run_start),
    }
    out['artifact_invariants'] = validate_signal_artifact(out)
    path = OUT_DIR / f"{ticker}_signal.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, cls=_NumpyEncoder)
    if out['artifact_invariants'].get('status') != 'OK':
        log.warning(
            "Artifact invariants flagged issues: "
            + ", ".join(out['artifact_invariants'].get('failure_names', []) or out['artifact_invariants'].get('warning_names', []))
        )
    progress.stage_done('charts_output')

    log_section("FINAL SUMMARY")
    s = final_signal
    log.info(f"TICKER   : {ticker}")
    log.info(f"SIGNAL   : {s['signal']}")
    log.info(f"CONF     : {s.get('confidence',0)*100:.2f}%")
    log.info(f"MODEL    : {s.get('deployment_model_used') or s.get('reference_model_used') or s.get('model_used','')}")
    log.info(f"SELECT   : {s.get('selection_status', 'n/a')}")
    log.info(f"REGIME   : {regime['speed']}  horizon={regime['predict_days']}d  "
             f"IC={regime['best_ic']:.4f}  vol={regime['ann_vol']*100:.1f}%")
    log.info(f"LABELS   : {y.attrs.get('label_method', CONFIG.get('label_method', 'rar'))}")
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
            log.info(f"  {d['model']:<30} {col}  "
                     f"acc={d.get('accuracy', 0.0):.4f}  "
                     f"f1={d.get('f1', 0.0):.4f}  "
                     f"T={d['temperature']:.3f}")
    if conformal_result:
        ps_str = str(final_signal.get('prediction_set', '—'))
        log.info(f"CONFORMAL: set={ps_str}  "
                 f"singleton={'YES' if final_signal.get('is_conformal_singleton') else 'NO'}  "
                 f"cov={conformal_result.get('coverage',0):.3f}  "
                 f"avg_size={conformal_result.get('avg_set_size',0):.2f}")
    if seed_stability:
        log.info(f"SEEDS    : model={seed_stability.get('model_name','?')}  "
                 f"consensus={seed_stability.get('consensus','?')}  "
                 f"agreement={seed_stability.get('agreement',0):.0%}  "
                 f"disagreement={seed_stability.get('disagreement',0):.0%}")
    if calibration_diag:
        pr_auc = calibration_diag.get('pr_auc_macro')
        pr_auc_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
        log.info(f"CAL-DIAG : model={calibration_diag.get('model_name','?')}  "
                 f"Brier={calibration_diag.get('brier_score',0):.4f}  "
                 f"ECE={calibration_diag.get('ece',0):.4f}  "
                 f"PR-AUC={pr_auc_str}")
    if getattr(tree, 'ensemble_weights_', {}):
        log.info("ENS-WTS  : " + ", ".join(
            f"{k}={v:.2f}" for k, v in tree.ensemble_weights_.items()))
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
        try:
            run_diagnostic_tests(ticker, df, X, y, tree, dl_trainers,
                                 tree_signal, OUT_DIR,
                                 adv_report=adv_report,
                                 seed_stability=seed_stability,
                                 calibration_diag=calibration_diag,
                                 run_metadata=completed_run_metadata)
        except Exception as e:
            _record_stage_error(stage_errors, 'diagnostics', e, fatal=False)

    _record_experiment(out.get('pipeline_status', 'OK'), {
        'ticker': ticker,
        'signal': final_signal.get('signal'),
        'model_used': final_signal.get('model_used'),
        'confidence': final_signal.get('confidence'),
    })


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC TEST SUITE  v1.0
# 6 modules that probe stability, reliability, and calibration.
# Enable with "run_diagnostics": True in CONFIG (adds ~3 extra tree fits).
# Output: reports/<TICKER>/<TICKER>_diagnostics.json
# ─────────────────────────────────────────────────────────────────────────────
def _diagnostic_score_fn(method: str, seed: int):
    if method == 'mi':
        return lambda X_arr, y_arr: mutual_info_classif(
            X_arr, y_arr, random_state=seed)
    return f_classif


def _diagnostic_sample_weights(n_rows: int, gap: int) -> np.ndarray:
    gap = max(1, min(int(gap), int(n_rows)))
    w = np.ones(int(n_rows), dtype=np.float32)
    for i in range(gap):
        w[n_rows - gap + i] = 0.1 + 0.9 * (i / gap)
    return w


def _fit_model_optional_sample_weight(model, Xs, ys, sample_weight):
    if sample_weight is None:
        model.fit(Xs, ys)
        return model
    try:
        model.fit(Xs, ys, sample_weight=sample_weight)
    except TypeError:
        model.fit(Xs, ys)
    return model


def _extract_model_feature_importance(model, feature_names):
    if not feature_names:
        return pd.Series(dtype=float)
    imp = None
    if hasattr(model, 'feature_importances_'):
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            imp = None
    elif hasattr(model, 'get_feature_importance'):
        try:
            imp = np.asarray(model.get_feature_importance(), dtype=float)
        except Exception:
            imp = None
    elif hasattr(model, 'coef_'):
        try:
            coef = np.asarray(model.coef_, dtype=float)
            imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        except Exception:
            imp = None
    if imp is None or len(imp) != len(feature_names):
        return pd.Series(dtype=float)
    return pd.Series(imp, index=feature_names, dtype=float)


def _build_tree_explainability_context(tree, model_name: str) -> 'dict | None':
    if model_name not in getattr(tree, 'models', {}):
        return None
    model = tree.models[model_name]
    feature_names = list(getattr(tree, 'feat_names_selected_', []) or [])
    if not feature_names:
        return None
    x_ref = getattr(tree, 'Xs_te', None)
    if x_ref is None or len(x_ref) == 0:
        x_ref = getattr(tree, 'Xs_tr', None)
    if x_ref is None or len(x_ref) == 0:
        return None
    x_latest = tree._transform_latest(tree.X.iloc[[-1]])
    return {
        'model': model,
        'feature_names': feature_names,
        'x_reference': x_ref,
        'x_latest': x_latest,
    }


def _build_regime_explainability_context(tree, regime_mdl, model_name: str) -> 'dict | None':
    if regime_mdl is None or not hasattr(regime_mdl, 'models_'):
        return None
    rid = 0 if 'BEAR' in str(model_name).upper() else 1
    if rid not in regime_mdl.models_:
        return None
    model, scaler, vt = regime_mdl.models_[rid]
    x_fill = tree.X.fillna(tree.X.median())
    x_reference = vt.transform(scaler.transform(x_fill))
    x_latest = vt.transform(scaler.transform(tree.X.iloc[[-1]].fillna(tree.X.median())))
    if hasattr(vt, 'get_support'):
        feature_names = list(np.asarray(tree.X.columns)[vt.get_support()])
    else:
        feature_names = list(tree.X.columns)
    return {
        'model': model,
        'feature_names': feature_names,
        'x_reference': x_reference,
        'x_latest': x_latest,
    }


def _build_dl_explainability_context(dl_trainers, model_name: str) -> 'dict | None':
    for trainer in dl_trainers:
        if getattr(trainer, 'name', '') != model_name:
            continue
        feature_names = list(getattr(trainer, '_feat_names', []) or [])
        return {
            'model': None,
            'feature_names': feature_names,
            'x_reference': None,
            'x_latest': None,
            'reason': 'dl_model_not_supported',
        }
    return None


def _generate_final_explainability(final_signal: dict,
                                   tree,
                                   dl_trainers,
                                   regime_mdl) -> dict:
    if not CONFIG.get('explainability_enabled', True):
        return {}
    model_name = str(final_signal.get('model_used', ''))
    top_n = int(CONFIG.get('explainability_top_n', 10))
    class_index = int(final_signal.get('label', 0) or 0)

    context = _build_tree_explainability_context(tree, model_name)
    if context is None:
        meta_ctx = getattr(tree, '_meta_explain_context', None)
        if meta_ctx and model_name == meta_ctx.get('model_name'):
            context = meta_ctx
    if context is None and model_name.startswith('Regime-'):
        context = _build_regime_explainability_context(tree, regime_mdl, model_name)
    if context is None:
        context = _build_dl_explainability_context(dl_trainers, model_name)

    if context is None or context.get('model') is None:
        payload = {
            'available': False,
            'model_name': model_name,
            'method': 'unavailable',
            'reason': context.get('reason', 'unsupported_final_model') if context else 'unsupported_final_model',
            'top_global': [],
            'top_local': [],
        }
    else:
        payload = build_explainability(
            context['model'],
            context['x_reference'],
            context['x_latest'],
            context['feature_names'],
            class_index=class_index,
            model_name=model_name,
            top_n=top_n,
        )
    if log is not None:
        if payload.get('available'):
            log.info(f"Explainability: {payload.get('method', 'unknown')} for {model_name}")
        else:
            log.info(f"Explainability: unavailable for {model_name} ({payload.get('reason', 'unknown')})")
    return payload


def _diagnostic_latest_signal(model, X, fill_values, scaler, var_sel,
                              feat_sel, cfg):
    last_x = X.iloc[[-1]].fillna(fill_values)
    last_xs = scaler.transform(last_x)
    last_xs = var_sel.transform(last_xs)
    last_xs = feat_sel.transform(last_xs)
    pred = int(np.asarray(model.predict(last_xs)).ravel()[0])
    try:
        proba = np.asarray(model.predict_proba(last_xs), dtype=float)[0]
    except Exception:
        proba = np.full(3, 1.0 / 3.0, dtype=float)
    lmap = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    conf = float(proba[pred])
    min_conf = cfg.get('min_signal_confidence', 0.38)
    signal = lmap[pred] if conf >= min_conf else 'HOLD'
    return {
        'signal': signal,
        'label': pred,
        'confidence': conf,
        'probabilities': {lmap[i]: float(p) for i, p in enumerate(proba)},
    }


def _diagnostic_refit_fixed_model(X, y, ModelClass, model_params, cfg, seed,
                                  train_end, test_end=None):
    if test_end is None:
        test_end = len(X)
    if train_end <= 0 or test_end <= train_end:
        raise ValueError("invalid diagnostic split")

    X_tr = X.iloc[:train_end].copy()
    y_tr = y.iloc[:train_end].copy()
    X_te = X.iloc[train_end:test_end].copy()
    y_te = y.iloc[train_end:test_end].copy()

    m_tr = y_tr.notna()
    m_te = y_te.notna()
    X_tr, y_tr = X_tr.loc[m_tr], y_tr.loc[m_tr]
    X_te, y_te = X_te.loc[m_te], y_te.loc[m_te]

    if len(X_tr) < 30 or len(X_te) < 20:
        raise ValueError("insufficient rows for diagnostic split")
    if y_tr.nunique() < 2:
        raise ValueError("training split lacks class diversity")

    fill_values = X_tr.median()
    X_tr = X_tr.fillna(fill_values)
    X_te = X_te.fillna(fill_values)

    scaler = RobustScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)

    var_sel = VarianceThreshold(threshold=cfg['feat_var_thresh'])
    Xs_tr = var_sel.fit_transform(Xs_tr)
    Xs_te = var_sel.transform(Xs_te)
    n_after_var = Xs_tr.shape[1]
    if n_after_var == 0:
        raise ValueError("all features removed by variance filter")

    raw_k = cfg['feat_select_k']
    k = min(raw_k, max(20, len(X_tr) // 12), n_after_var)
    score_fn = _diagnostic_score_fn(cfg.get('feat_select_method', 'mi'), seed)
    feat_sel = SelectKBest(score_fn, k=k)
    Xs_tr = feat_sel.fit_transform(Xs_tr, y_tr.values)
    Xs_te = feat_sel.transform(Xs_te)

    cols_after_var = np.array(X.columns)[var_sel.get_support()]
    feature_names = list(cols_after_var[feat_sel.get_support()])

    model_kwargs = dict(model_params)
    model_kwargs['random_state'] = seed
    model = ModelClass(**model_kwargs)

    sw = _diagnostic_sample_weights(len(X_tr), cfg.get('predict_days', 1))
    sw = sw / sw.sum() * len(sw)
    _fit_model_optional_sample_weight(model, Xs_tr, y_tr.values, sw)

    preds = np.asarray(model.predict(Xs_te)).ravel()
    try:
        proba = np.asarray(model.predict_proba(Xs_te), dtype=float)
    except Exception:
        proba = None
    f1 = float(f1_score(y_te.values, preds, average='macro',
                        zero_division=0))
    acc = float(accuracy_score(y_te.values, preds))

    return {
        'model': model,
        'seed': seed,
        'X_tr': X_tr,
        'X_te': X_te,
        'y_tr': y_tr,
        'y_te': y_te,
        'Xs_tr': Xs_tr,
        'Xs_te': Xs_te,
        'preds': preds,
        'proba': proba,
        'acc': acc,
        'f1': f1,
        'fill_values': fill_values,
        'scaler': scaler,
        'var_sel': var_sel,
        'feat_sel': feat_sel,
        'feature_names': feature_names,
        'feature_importance': _extract_model_feature_importance(model,
                                                                feature_names),
        'latest_signal': _diagnostic_latest_signal(model, X, fill_values,
                                                   scaler, var_sel, feat_sel,
                                                   cfg),
    }


def _summarize_seed_stability(seed_signals, seed_f1s):
    if not seed_signals:
        return {'stable': False, 'error': 'all seeds failed'}
    from collections import Counter as _Counter
    mode_sig, agree_count = _Counter(seed_signals).most_common(1)[0]
    agreement = agree_count / len(seed_signals)
    f1_std = float(np.std(seed_f1s)) if len(seed_f1s) > 1 else 0.0
    stable = (agree_count * 3 >= 2 * len(seed_signals)) and f1_std < 0.04
    return {
        'signals': seed_signals,
        'f1s': seed_f1s,
        'consensus': mode_sig,
        'agreement': agreement,
        'disagreement': 1.0 - agreement,
        'agree_count': agree_count,
        'f1_std': f1_std,
        'stable': stable,
    }


def _diagnostic_quarter_slices(n_rows: int, min_rows: int = 10):
    if n_rows <= 0:
        return []
    qsize = max(20, n_rows // 4)
    slices = []
    for qi in range(4):
        qs = qi * qsize
        if qs >= n_rows:
            break
        qe = n_rows if qi == 3 else min(qs + qsize, n_rows)
        if qe - qs < min_rows:
            continue
        slices.append((qi + 1, qs, qe))
    return slices


def run_diagnostic_tests(ticker, df, X, y, tree, dl_trainers,
                         tree_signal, out_dir, adv_report=None,
                         seed_stability=None, calibration_diag=None,
                         run_metadata=None):
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
        'started_at': datetime.datetime.now().isoformat(),
        'n_samples': len(X),
        'n_features': X.shape[1],
    }
    if run_metadata is not None:
        results['run_metadata'] = dict(run_metadata)
    if adv_report is not None:
        results['adversarial_validation'] = adv_report
    if seed_stability is not None:
        results['main_seed_stability'] = seed_stability
    if calibration_diag is not None:
        results['main_calibration_diagnostics'] = calibration_diag
    t0_diag = time.time()

    # ── Identify best tree model (excluding MiniROCKET / Ensemble) ────────────
    best_name = max(
        {k: v for k, v in tree.results.items()
         if k not in ('Ensemble', 'MiniROCKET')},
        key=lambda k: tree.results[k]['f1'])
    BestModel = type(tree.models[best_name])
    best_params = tree.models[best_name].get_params()
    diag_cfg = dict(getattr(tree, 'cfg', CONFIG))
    base_seed = diag_cfg.get('random_state', 42)
    main_train_end = min(max(1, len(tree.X_tr)), len(X) - 1)
    results['base_model'] = best_name
    log.info(f"  Diagnostic base model: {best_name}  "
             f"(test F1={tree.results[best_name]['f1']:.4f})")
    refit_cache = {}

    def _get_refit(seed: int, train_end: int, test_end: int):
        key = (int(seed), int(train_end), int(test_end))
        cached = refit_cache.get(key)
        if cached is not None:
            return cached
        refit = _diagnostic_refit_fixed_model(
            X, y, BestModel, best_params, diag_cfg, seed,
            train_end=train_end, test_end=test_end)
        refit_cache[key] = refit
        return refit

    # ── Helper: fast refit ────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 1: SEED STABILITY
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 1/6  — Seed Stability")
    seed_signals, seed_f1s = [], []
    for seed in [42, 43, 44]:
        try:
            refit = _get_refit(seed, main_train_end, len(X))
            sig = refit['latest_signal']
            seed_signals.append(sig['signal'])
            seed_f1s.append(refit['f1'])
            log.info(f"  seed={seed}  model={best_name}  signal={sig['signal']}  "
                     f"conf={sig['confidence']*100:.1f}%  f1={refit['f1']:.4f}")
        except Exception as e:
            log.warning(f"  seed={seed} failed: {e}")

    if seed_signals:
        seed_summary = _summarize_seed_stability(seed_signals, seed_f1s)
        mode_sig = seed_summary['consensus']
        agree_frac = seed_summary['agreement']
        f1_std_seed = seed_summary['f1_std']
        stable = seed_summary['stable']
        log.info(f"\n  Consensus signal : {mode_sig}  "
                 f"agreement={agree_frac:.0%}  F1 std={f1_std_seed:.4f}")
        log.info(f"  {'✓ STABLE' if stable else '⚠ UNSTABLE'} — "
                 f"{'≥2/3 seeds agree and F1 variation low' if stable else 'signal or F1 varies across seeds'}")
        results['seed_stability'] = seed_summary
    else:
        results['seed_stability'] = {'stable': False, 'error': 'all seeds failed'}

    # ──────────────────────────────────────────────────────────────────────────
    # MODULE 2: TEMPORAL STABILITY
    # ──────────────────────────────────────────────────────────────────────────
    log_section("DIAG 2/6  — Temporal Stability")
    n_all = len(X)
    wsize = n_all // 3
    window_rows = []
    for wi in range(1, 3):
        tr_end = wi * wsize
        te_end = n_all if wi == 2 else min((wi + 1) * wsize, n_all)
        if tr_end < 30 or (te_end - tr_end) < 20:
            continue
        try:
            refit = _get_refit(base_seed, tr_end, te_end)
            window_rows.append({'window': wi+1, 'tr_rows': len(refit['X_tr']),
                                 'te_rows': len(refit['X_te']), 'acc': refit['acc'],
                                 'f1': refit['f1']})
            te_s = tr_end
            te_e = te_end
            wa = refit['acc']
            wf = refit['f1']
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
            refit = _get_refit(seed, main_train_end, len(X))
            fi = refit['feature_importance']
            if len(fi) > 0:
                top5 = set(fi.nlargest(5).index.tolist())
                top5_sets.append({'seed': seed, 'features': sorted(top5)})
                log.info(f"  seed={seed}: {sorted(top5)}")
            else:
                log.warning(f"  seed={seed} feature importance unavailable for {best_name}")
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
    quarterly = []
    for quarter, qs, qe in _diagnostic_quarter_slices(n_te):
        corr = float((preds_te[qs:qe] == ys_te[qs:qe]).mean())
        f1_q = float(f1_score(ys_te[qs:qe], preds_te[qs:qe],
                               average='macro', zero_division=0))
        strat_r  = sig_te[qs:qe] * ret_arr[qs:qe]
        sh_q     = float(strat_r.mean() / (strat_r.std() + 1e-10) * np.sqrt(252))
        n_trades = int((sig_te[qs:qe] != 0).sum())
        qi = quarter - 1
        quarterly.append({'quarter': quarter, 'rows': qe-qs,
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
        proba_te = best_model.predict_proba(tree.Xs_te)          # (n_test, 3)
        preds_te = proba_te.argmax(axis=1)
        correct  = (preds_te == tree.y_te.values).astype(int)
        max_conf = proba_te.max(axis=1)
        cal_diag = _compute_calibration_diagnostics(
            proba_te, tree.y_te.values, n_bins=CONFIG.get('calibration_bins', 5))

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
        pr_auc = cal_diag.get('pr_auc_macro')
        pr_auc_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
        log.info(f"  Brier score={cal_diag['brier_score']:.4f}  "
                 f"ECE={cal_diag['ece']:.4f}  PR-AUC={pr_auc_str}")
        results['calibration'] = {
            'model_name': best_name,
            'table': cal_rows, 'overconfident_bins': overconf_n,
            'well_calibrated': calibrated,
            'brier_score': cal_diag['brier_score'],
            'ece': cal_diag['ece'],
            'classwise': cal_diag['classwise'],
            'pr_auc_macro': cal_diag['pr_auc_macro'],
            'reliability_bins': cal_diag['reliability_bins']}
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
    completed_at = datetime.datetime.now().isoformat()
    results['completed_at'] = completed_at
    results['generated'] = completed_at

    diag_path = out_dir / f"{ticker}_diagnostics.json"
    with open(diag_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    log.info(f"\n  Full diagnostics → {diag_path}")
    return results


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        err_path = None
        if OUT_DIR is not None and TICKER:
            try:
                err_path = OUT_DIR / f"{TICKER}_fatal_traceback.log"
                err_path.write_text(err, encoding='utf-8')
            except Exception:
                pass
        if TELEGRAM_PROGRESS_SESSION is not None and getattr(TELEGRAM_PROGRESS_SESSION, "enabled", False):
            TELEGRAM_PROGRESS_SESSION.mark_failed("Failed")
        if _telegram_notifications_enabled_here():
            notify_failure(TICKER or "analyzer", err[-3200:], log_path=str(err_path) if err_path else None)
        raise
    else:
        if TICKER:
            _notify_analyzer_outcome(TICKER)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC PANEL API
# These helpers are consumed by panel_runner.py which imports this module
# directly (not as a subprocess).  Exposing them via clearly named public
# aliases decouples panel_runner from internal naming conventions and makes
# refactoring safe.
# ─────────────────────────────────────────────────────────────────────────────

def slice_event_meta(event_meta, index):
    """Public alias for _slice_event_meta (used by panel_runner)."""
    return _slice_event_meta(event_meta, index)


def fixed_horizon_event_frame(index, predict_days: int):
    """Public alias for _fixed_horizon_event_frame (used by panel_runner)."""
    return _fixed_horizon_event_frame(index, predict_days)


def enabled_model_names(cfg: dict) -> list:
    """Public alias for _enabled_model_names (used by panel_runner)."""
    return _enabled_model_names(cfg)


def compute_calibration_diagnostics(proba, y_true, n_bins: int = 5) -> dict:
    """Public alias for _compute_calibration_diagnostics (used by panel_runner)."""
    return _compute_calibration_diagnostics(proba, y_true, n_bins=n_bins)


def compute_seed_stability_for_tree(X, y, trainer) -> dict:
    """Public alias for _compute_seed_stability_for_tree (used by panel_runner)."""
    return _compute_seed_stability_for_tree(X, y, trainer)


# clarity in external modules.
NumpyEncoder = _NumpyEncoder
