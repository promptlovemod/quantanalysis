# -*- coding: utf-8 -*-
"""
Shared-model panel ML runner for watchlist mode.

Run:
  python panel_runner.py --watchlist watchlist_example.txt
"""

import argparse
import datetime
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import analyzer as az
from utils.model_explainability import build_explainability
from utils.run_metadata import append_experiment_record, build_run_metadata, complete_run_metadata

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:  # pragma: no cover - optional
    HAS_YF = False


def load_watchlist(path: str) -> list:
    watchlist = Path(path)
    if not watchlist.exists():
        raise FileNotFoundError(f"Watchlist file not found: {path}")
    tickers = []
    for line in watchlist.read_text(encoding="utf-8").splitlines():
        ticker = line.split("#")[0].strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    if not tickers:
        raise ValueError(f"No tickers found in {path}")
    return tickers


def init_panel_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("panel_runner")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    ch = logging.StreamHandler(stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def _fetch_panel_info(ticker: str) -> dict:
    info = {"sector": "Unknown", "market_cap": np.nan}
    if not HAS_YF:
        return info
    try:
        tkr = yf.Ticker(ticker)
        raw = tkr.info or {}
        info["sector"] = raw.get("sector") or "Unknown"
        info["market_cap"] = float(raw.get("marketCap")) if raw.get("marketCap") is not None else np.nan
    except Exception:
        pass
    return info


def _slice_latest_rows(panel_x: pd.DataFrame) -> dict:
    latest = {}
    if panel_x.empty:
        return latest
    dates = panel_x.index.get_level_values("date")
    tickers = panel_x.index.get_level_values("ticker")
    for ticker in pd.Index(tickers).unique():
        mask = tickers == ticker
        last_row = panel_x.loc[mask].tail(1)
        if not last_row.empty:
            latest[str(ticker)] = last_row
    return latest


class PanelTreeTrainer(az.TreeTrainer):
    def _prep_walkforward(self):
        x_fill = self.X.fillna(self.X.median())
        y = self.y
        date_index = pd.DatetimeIndex(self.X.index.get_level_values("date"))
        unique_dates = pd.Index(sorted(date_index.unique()))
        split_pos = min(max(1, int(len(unique_dates) * self.cfg["train_split"])), len(unique_dates) - 1)
        train_dates = unique_dates[:split_pos]
        test_dates = unique_dates[split_pos:]
        train_mask = date_index.isin(train_dates)
        test_mask = date_index.isin(test_dates)

        X_tr = x_fill.loc[train_mask]
        X_te = x_fill.loc[test_mask]
        y_tr = y.loc[X_tr.index]
        y_te = y.loc[X_te.index]
        event_tr = az.slice_event_meta(self.event_meta, X_tr.index)
        event_te = az.slice_event_meta(self.event_meta, X_te.index)

        valid_tr = y_tr.notna()
        valid_te = y_te.notna()
        X_tr, y_tr = X_tr.loc[valid_tr], y_tr.loc[valid_tr]
        X_te, y_te = X_te.loc[valid_te], y_te.loc[valid_te]
        event_tr = az.slice_event_meta(event_tr, X_tr.index)
        event_te = az.slice_event_meta(event_te, X_te.index)
        self.train_event_meta_ = event_tr
        self.test_event_meta_ = event_te
        self.panel_split_date_ = str(test_dates[0].date()) if len(test_dates) else None

        gap = self.cfg["predict_days"]
        n_tr = len(X_tr)
        w = np.ones(n_tr, dtype=np.float32)
        for i in range(min(gap, n_tr)):
            w[n_tr - min(gap, n_tr) + i] = 0.1 + 0.9 * (i / max(min(gap, n_tr), 1))
        self.sample_weights_ = w

        Xs_tr = self.scaler.fit_transform(X_tr)
        Xs_te = self.scaler.transform(X_te)
        self.var_sel = az.VarianceThreshold(threshold=self.cfg["feat_var_thresh"])
        Xs_tr = self.var_sel.fit_transform(Xs_tr)
        Xs_te = self.var_sel.transform(Xs_te)
        n_after_var = Xs_tr.shape[1]
        raw_k = self.cfg["feat_select_k"]
        k = min(raw_k, max(20, len(X_tr) // 12), n_after_var)
        score_fn = az.mutual_info_classif if self.cfg.get("feat_select_method", "mi") == "mi" else az.f_classif
        self.feat_sel = az.SelectKBest(score_fn, k=k)
        Xs_tr = self.feat_sel.fit_transform(Xs_tr, y_tr.values)
        Xs_te = self.feat_sel.transform(Xs_te)
        cols_after_var = np.array(self.X.columns)[self.var_sel.get_support()]
        cols_after_kbest = cols_after_var[self.feat_sel.get_support()]
        self.feat_names_selected_ = list(cols_after_kbest)
        return X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te

    def get_signal_for_row(self, row_df: pd.DataFrame) -> dict:
        x_latest = self._transform_latest(row_df)
        skip = {"MiniROCKET", "Ensemble"}
        eligible = {k: v for k, v in self.results.items() if k not in skip}
        best = max(eligible or self.results, key=lambda key: self.results[key]["f1"])
        model = self.models[best]
        label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        pred = int(np.asarray(model.predict(x_latest)).ravel()[0])
        try:
            proba = np.asarray(model.predict_proba(x_latest), dtype=float)[0]
        except Exception:
            proba = np.full(3, 1.0 / 3.0, dtype=float)
        conf = float(proba[pred])
        min_conf = self.cfg.get("min_signal_confidence", 0.38)
        signal = label_map[pred] if conf >= min_conf else "HOLD"
        return {
            "signal": signal,
            "label": pred,
            "confidence": conf,
            "model_used": best,
            "probabilities": {label_map[i]: float(p) for i, p in enumerate(proba)},
        }


def build_panel_dataset(tickers: list, cfg: dict, logger) -> tuple:
    panel_frames = []
    label_frames = []
    event_frames = []
    info_map = {}
    skipped = {}
    horizons = []

    for ticker in tickers:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  PANEL DATASET  [{ticker}]")
        logger.info("=" * 60)
        try:
            fetcher = az.DataFetcher(ticker, cfg["period"])
            df = fetcher.fetch()
            market = fetcher.fetch_context()
            fundamentals = az.FundamentalFetcher(ticker).fetch() if cfg.get("fetch_fundamentals", True) else {}
            regime = az.analyze_stock_regime(df, dict(cfg))
            horizons.append(int(regime["predict_days"]))
            feat = az.FeatureEngineer(df, market, fundamentals=fundamentals).build()
            y = az.make_labels(df, regime)
            common = feat.index.intersection(y.dropna().index)
            if len(common) < 200:
                raise ValueError(f"insufficient labelled rows ({len(common)})")

            x_t = feat.loc[common].copy()
            info = _fetch_panel_info(ticker)
            info_map[ticker] = info
            x_t["panel_ticker"] = ticker
            if cfg.get("panel_include_sector", True):
                x_t["panel_sector"] = info.get("sector", "Unknown")
            if cfg.get("panel_include_market_cap", True):
                market_cap = info.get("market_cap", np.nan)
                x_t["panel_log_market_cap"] = np.log1p(max(float(market_cap), 0.0)) if np.isfinite(market_cap) else 0.0
                x_t["panel_market_cap_missing"] = 0.0 if np.isfinite(market_cap) else 1.0

            idx = pd.MultiIndex.from_arrays([common, [ticker] * len(common)], names=["date", "ticker"])
            x_t.index = idx
            y_t = y.loc[common].copy()
            y_t.index = idx
            event_meta = az.slice_event_meta(y.attrs.get("event_meta"), common)
            if event_meta is None:
                event_meta = az.fixed_horizon_event_frame(common, int(regime["predict_days"]))
            event_meta = event_meta.copy()
            event_meta.index = idx
            panel_frames.append(x_t)
            label_frames.append(y_t)
            event_frames.append(event_meta)
        except Exception as exc:
            skipped[ticker] = str(exc)
            logger.warning(f"  {ticker}: skipped ({exc})")

    if not panel_frames:
        raise RuntimeError("No usable tickers for panel dataset")

    panel_x = pd.concat(panel_frames, axis=0).sort_index()
    panel_x = pd.get_dummies(panel_x, columns=[c for c in panel_x.columns if panel_x[c].dtype == object], dummy_na=True, dtype=float)
    panel_y = pd.concat(label_frames, axis=0).sort_index()
    panel_event = pd.concat(event_frames, axis=0).sort_index()
    panel_predict_days = int(np.median(horizons)) if horizons else int(cfg.get("predict_days", 10))
    return panel_x, panel_y, panel_event, info_map, skipped, panel_predict_days


def run_panel(watchlist_path: str, cfg: dict | None = None) -> dict:
    cfg = dict(az.CONFIG, **(cfg or {}))
    cfg["panel_mode_enabled"] = True
    run_start = time.time()
    out_dir = Path(cfg.get("output_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "panel_runner.log"
    logger = init_panel_logger(log_path)
    az.log = logger

    tickers = load_watchlist(watchlist_path)
    run_metadata = build_run_metadata(
        mode="panel_ml",
        seed=cfg.get("random_state", 42),
        config=cfg,
        config_version=cfg.get("config_version"),
        enabled_models=az.enabled_model_names(cfg),
        watchlist_path=watchlist_path,
        universe=tickers,
        extra={"watchlist_path": watchlist_path},
    )

    logger.info("*" * 60)
    logger.info(f"  PANEL RUNNER  |  watchlist={watchlist_path}")
    logger.info(f"  tickers={len(tickers)}")
    logger.info("*" * 60)

    panel_x, panel_y, panel_event, info_map, skipped, panel_predict_days = build_panel_dataset(tickers, cfg, logger)
    cfg["predict_days"] = panel_predict_days
    device, gpu_name = az.get_device()
    trainer = PanelTreeTrainer(panel_x, panel_y, cfg, use_gpu=(device.type == "cuda"), event_meta=panel_event)
    trainer.train()

    best_name = max({k: v for k, v in trainer.results.items() if k != "MiniROCKET"},
                    key=lambda key: trainer.results[key]["f1"])
    best_model = trainer.models[best_name]
    calibration_diag = {}
    try:
        proba_eval = np.asarray(best_model.predict_proba(trainer.Xs_te), dtype=float)
        calibration_diag = az.compute_calibration_diagnostics(proba_eval, trainer.y_te.values, n_bins=cfg.get("calibration_bins", 5))
        calibration_diag["model_name"] = best_name
        calibration_diag["source_type"] = "panel_tree"
    except Exception as exc:
        logger.warning(f"Calibration diagnostics unavailable ({exc})")

    adv_report = {}
    try:
        adv_cfg = dict(cfg)
        adv_cfg["adv_val_enabled"] = True
        if not cfg.get("panel_adv_val_drop_features", False):
            adv_cfg["adv_val_drop_top_n"] = 0
        _, adv_report = az.adversarial_validation(panel_x, cfg["train_split"], adv_cfg)
    except Exception as exc:
        logger.warning(f"Panel adversarial validation unavailable ({exc})")

    seed_stability = az.compute_seed_stability_for_tree(panel_x, panel_y, trainer)
    latest_rows = _slice_latest_rows(panel_x)
    panel_context = {
        "mode": "shared_tree_panel",
        "watchlist_path": watchlist_path,
        "n_requested_tickers": len(tickers),
        "n_trained_tickers": len(info_map),
        "skipped_tickers": skipped,
        "shared_model": best_name,
        "split_date": getattr(trainer, "panel_split_date_", None),
        "predict_days": panel_predict_days,
        "adversarial_validation": adv_report,
    }

    ticker_payloads = {}
    for ticker, row_df in latest_rows.items():
        signal = trainer.get_signal_for_row(row_df)
        x_latest = trainer._transform_latest(row_df)
        explainability = build_explainability(
            best_model,
            trainer.Xs_te if len(trainer.Xs_te) else trainer.Xs_tr,
            x_latest,
            trainer.feat_names_selected_,
            class_index=int(signal.get("label", 0)),
            model_name=best_name,
            top_n=int(cfg.get("explainability_top_n", 10)),
        ) if cfg.get("explainability_enabled", True) else {}
        ticker_metadata = complete_run_metadata(dict(run_metadata), status="OK")
        ticker_metadata["ticker"] = ticker
        payload = {
            "ticker": ticker,
            "generated": ticker_metadata["completed_at"],
            "signal": {
                **signal,
                "date": str(row_df.index.get_level_values("date")[0].date()),
            },
            "tree_signal": signal,
            "dl_signals": [],
            "meta_signal": None,
            "dl_diagnostics": [],
            "seed_stability": seed_stability,
            "calibration_diagnostics": calibration_diag,
            "explainability": explainability,
            "ensemble_weights": getattr(trainer, "ensemble_weights_", {}),
            "pipeline_errors": [],
            "pipeline_status": "OK",
            "backtest": {},
            "model_accuracy": {name: result["accuracy"] for name, result in trainer.results.items()},
            "gpu_info": gpu_name,
            "regime": {
                "predict_days": panel_predict_days,
                "speed": "PANEL",
                "label_method": panel_y.attrs.get("label_method", cfg.get("label_method", "rar")),
            },
            "fundamentals": {},
            "adversarial_validation": adv_report,
            "cpcv": {},
            "walkforward_backtest": {},
            "regime_signal": None,
            "conformal_prediction": {},
            "panel_context": panel_context,
            "run_metadata": ticker_metadata,
            "total_runtime": az.elapsed(run_start),
        }
        out_path = out_dir / ticker / f"{ticker}_signal.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, cls=az.NumpyEncoder)
        ticker_payloads[ticker] = payload

    summary = {
        "generated": datetime.datetime.now().isoformat(),
        "run_metadata": complete_run_metadata(run_metadata, status="OK"),
        "panel_context": panel_context,
        "model_accuracy": {name: result["accuracy"] for name, result in trainer.results.items()},
        "shared_seed_stability": seed_stability,
        "shared_calibration_diagnostics": calibration_diag,
        "shared_adversarial_validation": adv_report,
        "tickers": {ticker: payload["signal"] for ticker, payload in ticker_payloads.items()},
    }
    summary_path = out_dir / "panel_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=az.NumpyEncoder)
    append_experiment_record(out_dir, run_metadata, status="OK", summary={
        "watchlist_path": watchlist_path,
        "n_tickers": len(ticker_payloads),
        "shared_model": best_name,
    })
    logger.info(f"Panel summary saved → {summary_path}")
    return {
        "summary_path": summary_path,
        "tickers": ticker_payloads,
        "panel_context": panel_context,
        "log_path": log_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Panel ML runner")
    parser.add_argument("--watchlist", required=True, help="Watchlist file")
    return parser.parse_args()


def main():
    args = parse_args()
    run_panel(args.watchlist)


if __name__ == "__main__":
    main()
