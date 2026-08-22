"""QUA-9 modular backtesting helpers for deterministic API execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.technical_indicators import simple_moving_average


BACKTEST_ENGINE_VERSION = "qua9-backtester/v1"
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestTrade:
    """One deterministic trade-log event emitted by the backtester."""

    timestamp: str
    action: str
    price: float
    position_after: int
    reason: str

    def to_dict(self) -> dict:
        """Return a JSON-safe trade-log row."""

        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "price": round(self.price, 6),
            "position_after": self.position_after,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Deterministic backtest output for one symbol and strategy."""

    summary: dict
    trade_log: list[BacktestTrade]

    def to_dict(self) -> dict:
        """Return a JSON-safe backtest payload."""

        return {
            "summary": dict(self.summary),
            "trade_log": [trade.to_dict() for trade in self.trade_log],
        }


def run_sma_crossover_backtest(
    frame: pd.DataFrame,
    *,
    fast_window: int,
    slow_window: int,
    initial_cash: float,
) -> BacktestResult:
    """Run a long-only SMA crossover backtest without lookahead bias.

    Args:
        frame: OHLCV data containing at least ``Open`` and ``Close`` columns.
        fast_window: Lookback window for the fast SMA.
        slow_window: Lookback window for the slow SMA. Must exceed fast_window.
        initial_cash: Initial notional cash used for the equity curve.

    Returns:
        ``BacktestResult`` with deterministic summary metrics and trade-log
        events.

    Assumptions:
        Signals are computed from current and historical closes only. Position
        changes are executed on the next bar by shifting the signal forward one
        row, which prevents same-bar lookahead leakage across the train/test
        boundary.
    """

    validated_frame = _validate_backtest_frame(frame)
    validated_fast = _validate_positive_integer("fast_window", fast_window)
    validated_slow = _validate_positive_integer("slow_window", slow_window)
    validated_initial_cash = _validate_initial_cash(initial_cash)
    if validated_fast >= validated_slow:
        raise ValueError("fast_window must be smaller than slow_window")

    close = validated_frame["Close"]
    fast_sma = simple_moving_average(close, validated_fast)
    slow_sma = simple_moving_average(close, validated_slow)
    signal = (fast_sma > slow_sma).astype(int)
    position = signal.shift(1).fillna(0).astype(int)

    close_returns = close.pct_change().fillna(0.0)
    strategy_returns = close_returns * position
    equity_curve = validated_initial_cash * (1.0 + strategy_returns).cumprod()
    trade_log = _build_trade_log(validated_frame, signal, position)

    summary = _build_summary(
        initial_cash=validated_initial_cash,
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        trade_count=len(trade_log),
        rows_tested=len(validated_frame),
    )
    return BacktestResult(summary=summary, trade_log=trade_log)


def _validate_backtest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one OHLCV frame for backtesting."""

    required_columns = {"Open", "Close"}
    if frame is None or frame.empty:
        raise ValueError("backtest frame must not be empty")
    if not required_columns.issubset(frame.columns):
        missing_columns = ", ".join(sorted(required_columns.difference(frame.columns)))
        raise ValueError(f"backtest frame is missing required columns: {missing_columns}")

    normalized = frame.sort_index().copy()
    for column in ("Open", "Close"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(float)
        finite_values = normalized[column].dropna()
        if finite_values.empty:
            raise ValueError(f"{column} column must contain numeric values")
        if not np.isfinite(finite_values).all():
            raise ValueError(f"{column} column must not contain inf or -inf")
    return normalized


def _validate_positive_integer(name: str, value: int) -> int:
    """Validate a strictly positive integer parameter."""

    validated = int(value)
    if validated < 1:
        raise ValueError(f"{name} must be greater than or equal to 1")
    return validated


def _validate_initial_cash(initial_cash: float) -> float:
    """Validate initial cash for the deterministic equity curve."""

    validated = float(initial_cash)
    if validated <= 0.0:
        raise ValueError("initial_cash must be greater than 0")
    return validated


def _build_trade_log(frame: pd.DataFrame, signal: pd.Series, position: pd.Series) -> list[BacktestTrade]:
    """Create a deterministic trade log from position transitions."""

    previous_position = position.shift(1).fillna(0).astype(int)
    trade_log: list[BacktestTrade] = []
    for timestamp, current_position in position.items():
        prior_position = int(previous_position.loc[timestamp])
        next_signal = int(signal.loc[timestamp])
        if current_position == prior_position:
            continue
        action = "BUY" if current_position > prior_position else "SELL"
        reason = "fast_sma_above_slow_sma" if next_signal == 1 else "fast_sma_below_or_equal_slow_sma"
        trade_log.append(
            BacktestTrade(
                timestamp=pd.Timestamp(timestamp).to_pydatetime().isoformat(),
                action=action,
                price=float(frame.loc[timestamp, "Open"]),
                position_after=int(current_position),
                reason=reason,
            )
        )
    return trade_log


def _build_summary(
    *,
    initial_cash: float,
    equity_curve: pd.Series,
    strategy_returns: pd.Series,
    trade_count: int,
    rows_tested: int,
) -> dict:
    """Build deterministic summary metrics for the completed backtest."""

    ending_equity = float(equity_curve.iloc[-1])
    total_return = (ending_equity / initial_cash) - 1.0

    nonzero_returns = strategy_returns.iloc[1:]
    annualized_return = 0.0
    annualized_volatility = 0.0
    sharpe_ratio = 0.0
    if len(equity_curve) > 1:
        annualized_return = float((ending_equity / initial_cash) ** (TRADING_DAYS_PER_YEAR / (len(equity_curve) - 1)) - 1.0)
    if not nonzero_returns.empty:
        annualized_volatility = float(nonzero_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if annualized_volatility > 0.0:
            sharpe_ratio = float((nonzero_returns.mean() * TRADING_DAYS_PER_YEAR) / annualized_volatility)

    drawdown = (equity_curve / equity_curve.cummax()) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    return {
        "engine_version": BACKTEST_ENGINE_VERSION,
        "strategy": "sma_crossover",
        "rows_tested": int(rows_tested),
        "trade_count": int(trade_count),
        "initial_cash": round(initial_cash, 6),
        "ending_equity": round(ending_equity, 6),
        "total_return_pct": round(total_return * 100.0, 6),
        "annualized_return_pct": round(annualized_return * 100.0, 6),
        "annualized_volatility_pct": round(annualized_volatility * 100.0, 6),
        "sharpe_ratio": round(sharpe_ratio, 6),
        "max_drawdown_pct": round(max_drawdown * 100.0, 6),
    }
