"""Classical forecasting models, implemented from scratch.

Every model shares a tiny ``fit`` / ``predict`` interface so backtesting,
plotting and the CLI can treat them interchangeably::

    model = Holt()
    model.fit(train_values)
    forecast = model.predict(horizon)

Models:
- :class:`MovingAverage` — mean of the last ``window`` observations.
- :class:`Holt` — Holt's linear trend method (level + trend exponential
  smoothing), from scratch.
- :class:`SeasonalNaive` — repeats the last observed season.
"""

from __future__ import annotations

import abc

import numpy as np


class ForecastModel(abc.ABC):
    """Base class for all ForecastKit models."""

    name = "base"

    @abc.abstractmethod
    def fit(self, y: np.ndarray) -> "ForecastModel":
        """Learn parameters from a 1-D training series. Returns self."""

    @abc.abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` steps as a 1-D array."""

    def _check_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError(f"{self.name} must be fitted before predicting")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"


class MovingAverage(ForecastModel):
    """Simple moving average: forecast = mean of last ``window`` points."""

    name = "moving_average"

    def __init__(self, window: int = 7):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._level = 0.0

    def fit(self, y: np.ndarray) -> "MovingAverage":
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("cannot fit on an empty series")
        self._level = float(np.mean(y[-self.window:]))
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        return np.full(horizon, self._level)


class Holt(ForecastModel):
    """Holt's linear trend method (double exponential smoothing).

    Recurrence, for ``alpha`` (level) and ``beta`` (trend) in (0, 1]::

        level_t = alpha * y_t + (1 - alpha) * (level_{t-1} + trend_{t-1})
        trend_t = beta * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}

    Forecast h steps ahead: ``level_T + h * trend_T``.
    """

    name = "holt"

    def __init__(self, alpha: float = 0.8, beta: float = 0.2):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        self.alpha = alpha
        self.beta = beta
        self._level = 0.0
        self._trend = 0.0

    def fit(self, y: np.ndarray) -> "Holt":
        y = np.asarray(y, dtype=float)
        if y.size < 2:
            raise ValueError("Holt needs at least 2 observations")
        level = float(y[0])
        trend = float(y[1] - y[0])
        for value in y[1:]:
            prev_level = level
            level = self.alpha * value + (1.0 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
        self._level = level
        self._trend = trend
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        steps = np.arange(1, horizon + 1, dtype=float)
        return np.maximum(self._level + steps * self._trend, 0.0)


class SeasonalNaive(ForecastModel):
    """Seasonal naive: the forecast repeats the last observed season."""

    name = "seasonal_naive"

    def __init__(self, season_length: int = 7):
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length
        self._season = np.array([])

    def fit(self, y: np.ndarray) -> "SeasonalNaive":
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("cannot fit on an empty series")
        self._season = y[-self.season_length:].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        idx = np.arange(horizon) % len(self._season)
        return self._season[idx].copy()


def all_models() -> list[ForecastModel]:
    """The standard model set used by backtesting and the CLI."""
    return [MovingAverage(window=7), Holt(alpha=0.8, beta=0.2), SeasonalNaive(7)]
