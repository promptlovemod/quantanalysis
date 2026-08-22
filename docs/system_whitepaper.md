# Quant Finance Analyzer - System Whitepaper

## Purpose
This document explains the current architecture of the Quant Finance Analyzer, how data moves through the system, which models and diagnostics are active, how caching and freshness work, and how to interpret the outputs. It is written as a technical operating manual for users and developers who need a durable reference beyond the source code.

## Entry Points
- `python analyzer.py TICKER`
  Runs the technical and machine-learning analysis for one ticker and writes signal and diagnostics artifacts under `reports/<TICKER>/`.
- `python fundamental.py TICKER`
  Runs the standalone fundamental analysis and writes `fundamentals.json` style outputs for the ticker.
- `python monte_carlo.py TICKER`
  Runs the scenario engine and risk summaries for the ticker.
- `python run_all.py`
  Interactive entry point for single-ticker, watchlist, benchmark, and development workflows.
- `python panel_runner.py --watchlist FILE`
  Shared-model cross-sectional machine-learning run over a watchlist.
- `python system_whitepaper.py`
  Regenerates the PDF form of this whitepaper.

## End-to-End Data Flow
The system follows a staged pipeline:

1. Market data download or cache load
2. Market-context assembly
3. Fundamentals snapshot and statement augmentation
4. Feature engineering
5. Label generation and event metadata construction
6. Tree-model and deep-learning training
7. Meta-model, regime logic, calibration, and diagnostics
8. Monte Carlo and fundamental valuation modules
9. JSON, PNG, HTML dashboard, and benchmark outputs

The core design principle is "fail loudly, continue when safe." Each major stage logs its status and, when possible, produces partial artifacts rather than crashing the entire run.

## Market Data Caching And Freshness
The analyzer uses a workspace-local DuckDB cache at `data_cache/market_data.duckdb`.

### Why the cache exists
- Repeated watchlist runs should not re-download identical daily OHLCV.
- Benchmark mode can hit the same broad ETFs and macro context symbols many times.
- A visible local cache is easier to inspect and clean than a hidden home-directory folder.

### Cache contract
- The cache stores normalized daily OHLCV rows keyed by `(ticker, date)`.
- Each row group includes `fetched_at`.
- Freshness is controlled by `DATA_CACHE_TTL_HOURS`.
- If data is within TTL, the analyzer uses the cache.
- If data is older than TTL, the analyzer refetches and replaces it.

### Context reuse rules
- Context symbols such as `SPY`, `QQQ`, `GLD`, `TLT`, and `^VIX` are now fetched through the same reusable OHLCV path instead of a separate ad hoc download path.
- If the primary analyzed ticker is itself a context symbol, the already-fetched primary frame is reused rather than downloaded again.

### Operational interpretation
- `cache hit` means the symbol was served directly from DuckDB.
- `cache stale` means the symbol existed locally but exceeded TTL, so a network refresh was performed.
- `cache miss` means the symbol was not yet in the local cache.
- `reused_primary_frame` means the symbol was already the main ticker payload for the run.

## Feature Engineering Groups
The feature stack is intentionally broad and grouped into toggleable families so feature growth does not automatically imply feature sprawl in every run.

### Price, return, and trend families
- raw returns and log returns
- moving averages and EMA relationships
- MACD, PPO, RSI, stochastic, Williams %R, and CCI
- Bollinger, Keltner, and Donchian structures
- multi-horizon momentum and trend-quality signals
- breakout continuation and failure flags
- anchored VWAP distance proxies

### Volatility and regime families
- ATR and ATR-percent features
- realized volatility across multiple windows
- downside semivolatility and semivol ratios
- Parkinson volatility
- drawdown depth, ulcer index, and days since high
- volatility clustering and volatility-of-volatility proxies
- HMM-style market-regime context based on SPY and optional VIX

### Market exposure and residual families
- relative-strength style features versus market context
- rolling beta to context series
- residual return and idiosyncratic volatility features
- market breadth proxies through shared benchmark ETFs

### Volume and liquidity families
- rolling volume and dollar-volume measures
- turnover-like ratios
- volume spike percentiles
- volume-price confirmation features

### Fundamental and alternative overlays
- snapshot metrics from yfinance and statements
- dilution-aware fields
- reverse-DCF and DCF surface summaries
- profitability, leverage, and quality metrics

All feature generation is designed to avoid lookahead. Rolling calculations use only information available up to each timestamp.

## Label Generation And Leak Safety
The analyzer builds time-aware labels and retains event metadata for model validation.

### Safety principles
- labels are generated from future windows, but training features are aligned to the present row only
- event start and event end timestamps are carried forward into validation
- purged cross-validation removes training examples whose label windows overlap the test fold
- preprocessing is fit on training folds only, not on future rows

### Why this matters
Time-series models often look strong because of leakage rather than signal. The system explicitly tracks event windows so validation is stricter than an embargo-only split.

## Modeling Stack
The analyzer is a layered prediction system, not a single-model script.

### Tree layer
The tree layer is the most stable part of the stack and typically includes:
- XGBoost
- LightGBM
- CatBoost
- Random Forest style baselines
- Extra-trees style baselines where enabled

Tree-model diagnostics drive much of the reliability scoring because they are faster to refit and easier to inspect.

### Deep-learning layer
The deep-learning layer is optional and includes classical and newer sequence models:
- BiLSTM
- Transformer classifier
- TFT-style sequence model
- PatchTST-style classifier
- TiDE-style classifier

DL outputs are monitored for one-class collapse, dominant-class overconfidence, weak macro-F1, and calibration failure. Weak or unstable DL models can be excluded from stacking.

### Regime and meta layers
- The market-regime layer uses SPY returns and optional VIX context.
- If VIX overlap is insufficient, the system falls back to SPY-only regime estimation.
- A meta-model can combine tree and DL outputs, but it is rejected when it underperforms the stronger base layer on the metrics that matter.

### Panel mode
Panel mode builds a shared tree model across multiple tickers by stacking rows on `(ticker, date)`. It adds ticker, sector, and market-cap-derived fields while retaining chronological train/test discipline.

## Calibration, Diagnostics, And Reliability
The diagnostics path is intended to answer "should this output be trusted?" rather than merely "did the model fit?"

### Current diagnostics families
- adversarial validation for train/test shift
- calibration diagnostics: Brier score, ECE, classwise precision and recall, reliability bins
- seed stability and disagreement
- temporal stability checks
- quarterly hit-rate and out-of-sample consistency
- feature-stability checks
- reliability scoring and final labels such as `TRUSTED`, `MARGINAL`, or `UNRELIABLE`

### Quality interpretation
- strong raw accuracy with weak macro-F1 usually means the model is under-serving one or more classes
- a high adversarial-validation AUC indicates regime shift or unstable train/test segmentation
- low ECE and sane bucket summaries indicate confidence is closer to reality
- disagreement across seeds lowers trust even when one run looks attractive

## Monte Carlo Scenario Engine
Monte Carlo is treated as a scenario and risk engine, not a price target oracle.

### Volatility layer
The engine can use:
- bounded historical volatility
- GARCH
- EGARCH
- GJR-GARCH

The chosen volatility model and parameters are logged and serialized so runs are inspectable.

### Risk outputs
The engine reports:
- VaR and CVaR style tail statistics
- probability of loss
- drawdown and drawdown-threshold probabilities
- terminal distribution percentiles
- time-under-water style summaries
- calibration and dispersion checks against comparable realized historical windows

The goal is realism and inspectability, not optimistic path generation.

## Fundamental Valuation Layer
Fundamentals are built to remain informative even for unprofitable or dilution-heavy names.

### Current capabilities
- quality and balance-sheet augmentation from statements
- reverse DCF
- DCF surface analysis over growth and discount assumptions
- dilution analysis and share-count trend tracking
- quality metrics such as ROIC, reinvestment, and leverage proxies

### Important limitation
Snapshot fundamentals from public APIs are not true point-in-time historical fundamentals. They are informative for current-state interpretation but should not be over-trusted as perfectly historical training data.

## Outputs And Artifacts
The main report tree lives under `reports/`.

### Per ticker
- `<TICKER>_signal.json`
- `<TICKER>_diagnostics.json`
- `<TICKER>_analysis.png`
- `<TICKER>_dl_models.png`
- `<TICKER>_montecarlo_volatility.png`
- `<TICKER>_montecarlo_diagnostics.png`
- `<TICKER>_dashboard.html`
- `<TICKER>_run.log`
- `<TICKER>_master.log`

### Portfolio and benchmark level
- `reports/portfolio_summary.json`
- `reports/benchmark_quality.json`
- `reports/panel_summary.json`
- `reports/portfolio_dashboard.html`
- `reports/portfolio_optimizer.png`
- `reports/benchmark_quality.png`
- `reports/panel_summary.png`
- `reports/experiment_runs.jsonl`

## Benchmark Methodology
The benchmark universe is a development-quality gate, not a trading approval engine.

### Default benchmark
- uses a curated 50-60 name set
- balances broad ETFs, sector ETFs, large caps, defensives, cyclicals, and a minority of high-beta names
- is intended to estimate reliability and calibration across regimes without turning every benchmark run into a giant stress test

### Full benchmark
- `benchmark_universe_full.txt` preserves the larger 90+ name stress basket
- use it when you want breadth and failure-surface discovery, not quick validation

### Reading the benchmark
The quality gate aggregates coverage, walk-forward quality, CPCV tail behavior, seed stability, reliability score, calibration, and BUY recall. A benchmark pass is evidence that the system is behaving coherently across a diverse universe. It is not evidence of tradable alpha by itself.

## Operational Guidance
- Use single-ticker mode for detailed investigation.
- Use benchmark mode when you want to measure overall system health.
- Use the full benchmark only when runtime and API budget allow it.
- On single-GPU systems, keep benchmark GPU jobs serialized unless you have explicit evidence that concurrent runs are stable.
- Treat model collapse and confidence pathologies as risk-control signals, not mere nuisance warnings.

## Known Limitations
- Public market and fundamental APIs can change or throttle.
- DL models remain the most failure-prone components.
- Single-name time series can still have weak class separability even after code fixes.
- Portfolio optimization quality still depends on the quality of the upstream expected-return inputs.
- The system is research software. Strong diagnostics are necessary but not sufficient for real capital deployment.
