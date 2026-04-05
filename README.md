# Quant Analyzer Build 2026.4.5

Multi-module stock analysis system with :

- ML and deep-learning classification
- execution-aware signal gating
- conformal uncertainty control
- fundamentals and speculative-growth diagnostics
- Monte Carlo scenario analysis
- single-stock, portfolio, panel, benchmark, and repo-audit flows
- HTML dashboards & Telegram notifications

The current codebase is designed around truthfulness first :

- `*_signal.json` is the source of truth for dashboards
- deployability gates are separate from raw model ranking
- portfolio allocation respects execution status and can hold cash
- tree-only robustness diagnostics are no longer shown as final-model evidence for non-tree selections

## What The Project Runs

### Single-stock analysis
- `analyzer.py`: model training, candidate selection, conformal calibration, selected-candidate holdout backtest, charts, `*_signal.json`
- `fundamental.py`: fundamentals, DCF, reverse DCF, dilution, speculative-growth haircut
- `monte_carlo.py`: GBM, Merton, Heston, regime, stressed scenarios, MC reliability
- `run_all.py`: orchestrates modules and builds the stock dashboard

### Portfolio / watchlist analysis
- standard portfolio scan with cash sleeve support
- optional panel run
- optional benchmark quality gate
- consolidated portfolio dashboard and `portfolio_summary.json`

### Audit / debugging
- repo-wide artifact audit via `run_all.py --debug audit`
- dashboard-vs-JSON consistency reporting
- artifact invariant checks embedded into signal JSON

## Installation

### 1. Create and activate a virtual environment

Windows PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows `cmd.exe` :

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install PyTorch

CUDA 12.1 example :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

or CPU-only :

```bash
pip install torch torchvision torchaudio
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional but recommended extras

```bash
pip install catboost hmmlearn shap lightgbm
```

Why these matter :

- `arch` : full GARCH volatility model for Monte Carlo; now included in `requirements.txt`
- `duckdb` : local market-data cache; now included in `requirements.txt`
- `catboost` : one of the default active tree candidates when installed
- `hmmlearn` : regime HMM / regime-conditional model
- `shap` : richer explainability
- `lightgbm` : optional challenger / meta utility path

## Environment Variables

### Market data

- `TIINGO_API_KEY`
  - optional but recommended
  - if set, the analyzer prefers Tiingo for cleaner OHLCV and falls back to `yfinance` when needed

### Telegram notifications

- `TELEGRAM_ENABLED`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TELEGRAM_PROGRESS_ENABLED`
- `TELEGRAM_DELAY_BEFORE_RESULT_MS`

Behavior:

- one editable progress message per overall run when progress is enabled
- success sends PNG artifacts only
- failure sends log/error output only
- JSON artifacts are never sent to Telegram

Windows `cmd.exe` example :

```cmd
set TELEGRAM_ENABLED=1
set TELEGRAM_PROGRESS_ENABLED=1
set TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
set TELEGRAM_CHAT_ID=YOUR_CHAT_ID
set TELEGRAM_DELAY_BEFORE_RESULT_MS=5000
```

Windows PowerShell ex. :

```powershell
$env:TELEGRAM_ENABLED = "1"
$env:TELEGRAM_PROGRESS_ENABLED = "1"
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
$env:TELEGRAM_DELAY_BEFORE_RESULT_MS = "5000"
```

Internal variable :

- `TELEGRAM_SUPPRESS_CHILD=1`
  - used by `run_all.py` for child-process suppression
  - not normally set manually

## Main Entry Points

### Interactive runner

```bash
python run_all.py
```

### Direct single ticker

```bash
python run_all.py AAPL
```

### Portfolio / watchlist

```bash
python run_all.py --portfolio watchlist_example.txt
```

### Portfolio + panel mode

```bash
python run_all.py --portfolio watchlist_example.txt --panel
```

### Portfolio + benchmark quality gate

```bash
python run_all.py --portfolio watchlist_example.txt --benchmark --benchmark-watchlist benchmark_universe.txt
```

### Repo-wide audit

```bash
python run_all.py --debug audit
```

### Standalone modules

```bash
python analyzer.py AAPL
python fundamental.py AAPL
python monte_carlo.py AAPL
```

## Current Default Model Policy

### Active tree-family models
- `RandomForest`
- `XGBoost`
- `CatBoost` when installed

### Active deep-learning models
- `TiDE`
- `PatchTST`

### Challenger models
- disabled by default unless explicitly enabled in config
- examples: `TFT`, `BiLSTM-Attention`, `Transformer-Encoder`, `LightGBM`, `HistGradBoost`

## Output Structure

### Single ticker

Files are written to:

```text
reports/<TICKER>/
```

Common artifacts :

- `<TICKER>_signal.json`
- `<TICKER>_dashboard.html`
- `<TICKER>_analysis.png`
- `<TICKER>_selection_diagnostics.png`
- `<TICKER>_dl_models.png`
- `<TICKER>_fundamentals.json`
- `<TICKER>_montecarlo.json`
- `<TICKER>_montecarlo.png`
- `<TICKER>_dashboard_consistency.json`
- `<TICKER>_master.log`

Important signal JSON blocks :

- `signal`
- `selection`
- `conformal`
- `backtest`
- `backtest_audit`
- `walkforward_backtest`
- `cpcv`
- `tree_family_diagnostics`
- `evidence_scope`
- `artifact_invariants`

### Portfolio

Common portfolio artifacts :

- `reports/portfolio_summary.json`
- `reports/portfolio_dashboard.html`
- `reports/portfolio_optimizer.png`

Important portfolio fields :

- `quality_gate`
- `actionable_universe_size`
- `non_actionable_universe_size`
- `cash_weight`
- `allocation_status`
- `optimizer.allocatable_tickers`
- `optimizer.excluded_tickers`

### Audit

- `reports/repo_debug_audit.json`
- `reports/repo_debug_audit.md`

## Important Behavioral Changes In The Current Codebase

### Execution-aware signals

`signal` remains `BUY`, `SELL`, or `HOLD` for compatibility, but execution behavior is driven by :

- `execution_status`
- `execution_gate`
- `deployment_eligible`
- `selection_status`

Typical execution states :

- `ACTIONABLE`
- `HOLD_NEUTRAL`
- `ABSTAIN_NO_EDGE`
- `ABSTAIN_UNCERTAIN`
- `ABSTAIN_MODEL_UNRELIABLE`

### Reference vs deployment model

The code now separates:

- `reference_model_used` : best raw diagnostic candidate
- `deployment_model_used` : actually deployable candidate

If nothing is deployable, the system can keep a reference winner for diagnostics while emitting no deployment model.

### Portfolio cash sleeve

If no names are actionable, the portfolio stays in :

```text
CASH = 100%
```

If position caps prevent full deployment, the remainder also stays in cash.

### Dashboard truth policy

Dashboards should reflect JSON truth, not recomputed labels. When critical dashboard values mismatch JSON, the dashboard writes a warning banner and a consistency report.

### Conformal / backtest evidence

- conformal is used as an execution-quality gate, not just a reporting metric
- selected-candidate holdout backtest is the primary static backtest surface
- tree WF/CPCV are only final-model evidence when the selected family is tree

### Monte Carlo reliability

The MC layer now separates :

- baseline reliability
- scenario reliability
- volatility fallback / degradation

If `arch` is missing, the system falls back and reports degraded reliability instead of silently pretending GARCH was used.

## Testing

Run the main regression suites :

```bash
python -m unittest tests.test_priority4_pipeline tests.test_priority5_pipeline
```

Other useful suites :

```bash
python -m unittest tests.test_priority1_pipeline
python -m unittest tests.test_priority3_pipeline
python -m unittest tests.test_analyzer_stability
python -m unittest tests.test_analyzer_diagnostics
```

## Notes

- The codebase contains graceful fallbacks for many optional dependencies, but the recommended install is the one that gives you the intended behavior.
- Live report artifacts in `reports/` can become stale after code changes. Regenerate reports after major pipeline updates.
- The project is optimized for research correctness, not papering over uncertainty.

## Disclaimer

Educational and research use only. Not financial advice!
