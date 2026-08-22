from __future__ import annotations

import math

import pandas as pd
import pytest

from utils.technical_indicators import (
    MACDResult,
    exponential_moving_average,
    moving_average_convergence_divergence,
    relative_strength_index,
    simple_moving_average,
)


def _close_series() -> pd.Series:
    return pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0], name="Close")


def _assert_series_matches(actual: pd.Series, expected: list[float | None]) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual.tolist(), expected):
        if expected_value is None:
            assert pd.isna(actual_value)
        else:
            assert actual_value == pytest.approx(expected_value, abs=1e-6)


def test_simple_moving_average_matches_deterministic_fixture():
    result = simple_moving_average(_close_series(), window=3)

    _assert_series_matches(result, [None, None, 11.0, 11.333333, 12.0, 12.0, 13.0, 13.0])


def test_exponential_moving_average_matches_deterministic_fixture():
    result = exponential_moving_average(_close_series(), window=3)

    _assert_series_matches(result, [10.0, 10.5, 11.25, 11.125, 12.0625, 12.03125, 13.015625, 13.007812])


def test_relative_strength_index_matches_deterministic_fixture():
    result = relative_strength_index(_close_series(), period=3)

    _assert_series_matches(result, [None, 100.0, 100.0, 50.0, 83.333333, 50.0, 80.769231, 50.0])


def test_moving_average_convergence_divergence_matches_deterministic_fixture():
    result = moving_average_convergence_divergence(_close_series(), fast_window=3, slow_window=5, signal_window=2)

    assert isinstance(result, MACDResult)
    _assert_series_matches(result.macd, [0.0, 0.166667, 0.361111, 0.199074, 0.445216, 0.286394, 0.519054, 0.343432])
    _assert_series_matches(result.signal, [0.0, 0.111111, 0.277778, 0.225309, 0.371914, 0.314901, 0.451003, 0.379289])
    _assert_series_matches(result.histogram, [0.0, 0.055556, 0.083333, -0.026235, 0.073302, -0.028507, 0.068051, -0.035857])


def test_indicator_validation_rejects_non_finite_inputs():
    close = pd.Series([10.0, 11.0, math.inf], name="Close")

    with pytest.raises(ValueError, match="must not contain inf or -inf"):
        simple_moving_average(close, window=2)


def test_macd_validation_rejects_invalid_window_order():
    with pytest.raises(ValueError, match="fast_window must be smaller than slow_window"):
        moving_average_convergence_divergence(_close_series(), fast_window=5, slow_window=5, signal_window=2)
