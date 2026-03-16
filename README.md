# Stock ML Analyzer  0.7.0
## Analyze ANY Stock/ETFs — ML + BiLSTM + Transformer + Monte Carlo + Fundamentals

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

### Step 4 — Run
```
python run_all.py
```
Or pass ticker directly :
```
python run_all.py AAPL
python run_all.py TSLA
python run_all.py SBUX
```

---

## File Overview

| File | What it does |
|---|---|
| `run_all.py` | **Master runner** — runs all 3 modules, builds self-contained HTML dashboard |
| `analyzer.py` | 220+ features, 5 tree models + BiLSTM + Transformer, PurgedKFold CV, meta-stacking |
| `fundamental.py` | Fundamentals + DCF + Graham Number + Piotroski F-Score + 12-factor composite score |
| `monte_carlo.py` | 5 simulation models (GBM/Merton/Heston/Regime/Stressed), full charts, risk tables |

Each module can be run standalone :
```
python analyzer.py AAPL
python fundamental.py TSLA
python monte_carlo.py NVDA
```
---

## Models Trained

**Tree Models (5):**
1. Random Forest — balanced class weights, F1-macro CV scoring
2. HistGradientBoosting — sklearn, no GPU needed
3. XGBoost — GPU when ≥10k samples
4. LightGBM — GPU
5. CatBoost — GPU (optional)
6. Soft-Vote Ensemble — all tree models

**Deep Learning (2):**
7. PyTorch BiLSTM with Self-Attention w 3-layer bidirectional
8. PyTorch Transformer Encoder w 4-layer, norm-first

**Meta-Learner (1):**
9. Isotonic-calibrated Logistic Regression on all model probabilities

**Monte Carlo Models (5):**
1. GBM as baseline
2. Merton Jump-Diffusion for earnings/news gaps
3. Heston Stochastic Volatility for vol clustering
4. 2-State Markov Regime-Switching for bull/bear regimes
5. Stressed GBM (vol × 1.5) for tail risk scenario

---

## Key Accuracy Improvements

- **PurgedKFold** prevents label leakage (20/21 overlapping days in adjacent rows)
- **Adaptive thresholds** reduce HOLD label noise for high/low-vol stocks
- **F1-macro CV scoring** forces models to learn all 3 classes, not just HOLD
- **Feature name tracking** enables meaningful importance analysis
- **Regime features** help models distinguish vol regimes
- **VIX as context** adds the market's implied-fear level as a feature
- **Isotonic calibration** gives better-calibrated probability estimates

---

## Disclaimer
Educational and research use only! 
Not financial advice.
Past performance does not guarantee future results!!!
