"""Real tests for ForecastKit: models, metrics, backtesting, data."""

import numpy as np
import pytest

from forecastkit import backtest as bt
from forecastkit import data
from forecastkit import models


# ---------- data ----------

def test_dataset_is_seeded_reproducible():
    a = data.make_dataset(seed=7)
    b = data.make_dataset(seed=7)
    assert set(a) == set(b) == {"running-shoes", "coffee-mugs", "winter-jackets"}
    for product in a:
        assert np.array_equal(a[product], b[product])


def test_series_are_nonnegative_and_sized():
    ds = data.make_dataset(seed=42, days=100)
    for series in ds.values():
        assert len(series) == 100
        assert (series >= 0).all()


# ---------- models ----------

def test_moving_average_on_constant_series():
    y = np.full(50, 10.0)
    m = models.MovingAverage(window=7).fit(y)
    assert np.allclose(m.predict(5), 10.0)


def test_moving_average_uses_last_window():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = models.MovingAverage(window=3).fit(y)
    assert np.allclose(m.predict(2), [4.0, 4.0])


def test_holt_on_pure_trend_recovers_slope():
    # y_t = 2*t + 5: the fitted trend should be ~2 and the forecast the
    # straight continuation of the line.
    y = 2.0 * np.arange(100) + 5.0
    m = models.Holt().fit(y)
    forecast = m.predict(5)
    expected = 2.0 * np.arange(100, 105) + 5.0
    assert np.allclose(forecast, expected, atol=1.0)


def test_seasonal_naive_repeats_last_season():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    m = models.SeasonalNaive(season_length=7).fit(y)
    assert np.allclose(m.predict(9), [1, 2, 3, 4, 5, 6, 7, 1, 2])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        models.MovingAverage().predict(3)


def test_predict_rejects_bad_horizon():
    # Non-positive horizons should fail with a clear ValueError, not a
    # cryptic numpy error from deep inside predict().
    y = np.arange(30, dtype=float)
    for model in models.all_models():
        model.fit(y)
        for bad in (0, -5):
            with pytest.raises(ValueError, match="positive integer"):
                model.predict(bad)


def test_all_models_share_interface():
    y = np.arange(30, dtype=float)
    for model in models.all_models():
        model.fit(y)
        fc = model.predict(10)
        assert fc.shape == (10,)
        assert np.isfinite(fc).all()


# ---------- metrics ----------

def test_mae_zero_on_perfect_forecast():
    assert bt.mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0


def test_mape_bounds_and_value():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])
    result = bt.mape(actual, predicted)
    assert result == pytest.approx(10.0)
    assert 0.0 <= result <= 100.0


def test_mape_nonnegative_always():
    rng = np.random.default_rng(0)
    actual = np.abs(rng.normal(50, 10, 20)) + 1.0
    predicted = np.abs(rng.normal(50, 10, 20))
    assert bt.mape(actual, predicted) >= 0.0


# ---------- backtesting ----------

def test_backtest_runs_on_tiny_series():
    y = np.arange(40, dtype=float) + 10.0
    results = bt.rolling_backtest(y, horizon=3, n_splits=2)
    assert set(results) == {"moving_average", "holt", "seasonal_naive"}
    for scores in results.values():
        assert scores["mae"] >= 0.0
        assert scores["mape"] >= 0.0


def test_backtest_prefers_holt_on_trend():
    # On a clean linear trend, Holt should beat the flat moving average.
    y = 0.5 * np.arange(200) + 20.0
    results = bt.rolling_backtest(y, horizon=10, n_splits=3)
    assert bt.pick_best(results) == "holt"


def test_comparison_table_sorted_by_mae():
    table = bt.comparison_table({"a": {"mae": 5.0, "mape": 1.0, "n_splits": 2},
                                 "b": {"mae": 2.0, "mape": 0.5, "n_splits": 2}})
    assert [r["model"] for r in table] == ["b", "a"]
    assert "MAE" in bt.format_table(table)


def test_backtest_rejects_short_series():
    with pytest.raises(ValueError):
        bt.rolling_backtest(np.arange(10, dtype=float), horizon=30, n_splits=2)
