"""Reusable technical indicators for QUA-6 and downstream API consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RSI_NUMERICAL_EPSILON = 1e-10


@dataclass(frozen=True)
class MACDResult:
    """Container for MACD outputs aligned to the input close series.

    Attributes:
        macd: Difference between the fast and slow EMAs.
        signal: EMA of the MACD line.
        histogram: Difference between the MACD and signal series.
    """

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


def simple_moving_average(close: pd.Series, window: int) -> pd.Series:
    """Return a simple moving average series.

    Args:
        close: Price series indexed by timestamp/order.
        window: Rolling lookback window in observations.

    Returns:
        A ``pd.Series`` aligned to ``close`` with warmup NaNs for the first
        ``window - 1`` rows.

    Assumptions:
        Lookback window is exactly ``window`` observations, and every output
        value uses only the current row plus the prior ``window - 1`` rows.
    """

    validated_close = _validate_close_series(close)
    validated_window = _validate_positive_integer("window", window)
    result = validated_close.rolling(window=validated_window, min_periods=validated_window).mean()
    return _validate_indicator_output(result, "simple_moving_average")


def exponential_moving_average(close: pd.Series, window: int) -> pd.Series:
    """Return an exponential moving average series.

    Args:
        close: Price series indexed by timestamp/order.
        window: EMA span parameter in observations.

    Returns:
        A ``pd.Series`` aligned to ``close``.

    Assumptions:
        The recursive EMA depends on the current row and all prior rows from
        the start of the series, seeded by the first available observation.
        It never consumes future data.
    """

    validated_close = _validate_close_series(close)
    validated_window = _validate_positive_integer("window", window)
    result = validated_close.ewm(span=validated_window, adjust=False).mean()
    return _validate_indicator_output(result, "exponential_moving_average")


def relative_strength_index(close: pd.Series, period: int) -> pd.Series:
    """Return an RSI series using exponentially weighted average gains/losses.

    Args:
        close: Price series indexed by timestamp/order.
        period: RSI smoothing span in observations.

    Returns:
        A ``pd.Series`` aligned to ``close`` with the initial warmup value as
        ``NaN`` and subsequent values bounded to ``[0, 100]``.

    Assumptions:
        The first finite RSI value requires the current close plus at least one
        prior close. After that, the recursive smoothing uses only current and
        historical price changes, with smoothing span ``period``.
    """

    validated_close = _validate_close_series(close)
    validated_period = _validate_positive_integer("period", period)
    delta = validated_close.diff()
    gain = delta.clip(lower=0).ewm(span=validated_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=validated_period, adjust=False).mean()
    rs = gain / (loss + RSI_NUMERICAL_EPSILON)
    result = 100 - 100 / (1 + rs)
    return _validate_indicator_output(result.clip(lower=0, upper=100), "relative_strength_index")


def moving_average_convergence_divergence(
    close: pd.Series,
    fast_window: int,
    slow_window: int,
    signal_window: int,
) -> MACDResult:
    """Return MACD line, signal line, and histogram.

    Args:
        close: Price series indexed by timestamp/order.
        fast_window: Span for the fast EMA.
        slow_window: Span for the slow EMA. Must be greater than fast_window.
        signal_window: Span for the MACD signal EMA.

    Returns:
        ``MACDResult`` containing aligned ``pd.Series`` outputs.

    Assumptions:
        The MACD line uses only current and historical closes with effective
        recursive lookbacks defined by ``fast_window`` and ``slow_window``.
        The signal line uses only current and historical MACD values with
        smoothing span ``signal_window``.
    """

    validated_close = _validate_close_series(close)
    validated_fast = _validate_positive_integer("fast_window", fast_window)
    validated_slow = _validate_positive_integer("slow_window", slow_window)
    validated_signal = _validate_positive_integer("signal_window", signal_window)
    if validated_fast >= validated_slow:
        raise ValueError("fast_window must be smaller than slow_window")

    macd_line = exponential_moving_average(validated_close, validated_fast) - exponential_moving_average(
        validated_close, validated_slow
    )
    signal_line = macd_line.ewm(span=validated_signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return MACDResult(
        macd=_validate_indicator_output(macd_line, "macd"),
        signal=_validate_indicator_output(signal_line, "macd_signal"),
        histogram=_validate_indicator_output(histogram, "macd_histogram"),
    )


def _validate_close_series(close: pd.Series) -> pd.Series:
    """Normalize and validate a close-price series for indicator use."""

    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if close.empty:
        raise ValueError("close series must not be empty")

    normalized = pd.to_numeric(close.copy(), errors="coerce").astype(float)
    finite_values = normalized.dropna()
    if finite_values.empty:
        raise ValueError("close series must contain at least one numeric value")
    if not np.isfinite(finite_values).all():
        raise ValueError("close series must not contain inf or -inf")
    return normalized


def _validate_positive_integer(name: str, value: int) -> int:
    """Validate a positive integer indicator parameter."""

    validated = int(value)
    if validated < 1:
        raise ValueError(f"{name} must be greater than or equal to 1")
    return validated


def _validate_indicator_output(result: pd.Series, indicator_name: str) -> pd.Series:
    """Reject non-finite indicator outputs outside intentional warmup NaNs."""

    finite_values = result.dropna()
    if not np.isfinite(finite_values).all():
        raise ValueError(f"{indicator_name} produced inf or -inf output")
    return result
