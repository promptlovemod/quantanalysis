# Stock ML Analyzer  v0.6.0
## Analyze ANY Stock — ML + BiLSTM + Transformer + Monte Carlo + Fundamentals

## Quick Start

### Step 1 — Install PyTorch with CUDA (RTX 3050 / any RTX)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Install everything else
```
pip install -r requirements.txt
```

### Step 3 (Optional) — CatBoost
```
pip install catboost
```

### Step 4
```
python run_all.py
```
Or pass ticker directly:
```
python run_all.py AAPL
python run_all.py TSLA
python run_all.py SBUX
```

---

## File Overview

| File | What it does |
|---|---|
| `run_all.py` | **Master runner** — runs all 3 modules, builds & open self-contained HTML dashboard |
| `analyzer.py` | 220+ features, 5 tree models + BiLSTM + Transformer, PurgedKFold CV, meta-stacking |
| `fundamental.py` | Fundamentals + DCF + Graham Number + Piotroski F-Score + 12-factor composite score |
| `monte_carlo.py` | 5 simulation models (GBM/Merton/Heston/Regime/Stressed), full charts, risk tables |

Each module can be run standalone:
```
python analyzer.py AAPL
python fundamental.py TSLA
python monte_carlo.py NVDA
```

---

## Output Files

All outputs go to `reports/<TICKER>/`:

| File | Contents |
|---|---|
| `<TICKER>_dashboard.html` | Self-contained HTML dashboard (images embedded as base64) |
| `<TICKER>_analysis.png` | 6-panel ML chart — price/signals/RSI/MACD/backtest/feature importance |
| `<TICKER>_montecarlo.png` | 3-panel MC chart — fan chart / distribution / VaR timeline |
| `<TICKER>_signal.json` | ML signal + model accuracies + backtest stats including CVaR |
| `<TICKER>_fundamentals.json` | Fundamentals + DCF + Graham + Piotroski + composite score |
| `<TICKER>_montecarlo.json` | MC risk summary (5 models) for all horizons |
| `<TICKER>_run.log` | Full ML training log |

---

## Models Trained in system

**Tree Models (5):**
1. Random Forest — balanced class weights, F1-macro CV scoring
2. HistGradientBoosting — sklearn, no GPU needed
3. XGBoost — GPU when ≥10k samples
4. LightGBM — GPU
5. CatBoost — GPU (optional)
6. Soft-Vote Ensemble — all tree models

**Deep Learning (2):**
7. PyTorch BiLSTM with Self-Attention — 3-layer bidirectional
8. PyTorch Transformer Encoder — 4-layer, norm-first

**Meta-Learner (1):**
9. Isotonic-calibrated Logistic Regression on all model probabilities

**Monte Carlo Models (5):**
1. GBM — baseline
2. Merton Jump-Diffusion — earnings/news gaps
3. Heston Stochastic Volatility — vol clustering
4. 2-State Markov Regime-Switching — bull/bear regimes
5. Stressed GBM (vol × 1.5) — tail risk scenario

## Disclaimer
Educational and research use only. Not financial advice.
Past performance does not guarantee future results!
