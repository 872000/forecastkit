"""Rolling-origin backtesting with MAE and MAPE.

Each model is evaluated on several expanding-window origins: fit on the
history available at the origin, forecast ``horizon`` steps, and score
against the held-out future. Scores are averaged across origins, ranked,
and the best model (lowest mean MAE) is picked automatically.
"""

from __future__ import annotations

import numpy as np

from .models import ForecastModel, all_models

_EPS = 1e-8


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error (percent, 0 at best)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.maximum(np.abs(actual), _EPS)
    return float(np.mean(np.abs(actual - predicted) / denom) * 100.0)


def rolling_backtest(y: np.ndarray, models: list[ForecastModel] | None = None,
                     horizon: int = 30, n_splits: int = 4,
                     step: int | None = None) -> dict[str, dict[str, float]]:
    """Run rolling-origin backtesting.

    Returns ``{model_name: {"mae": mean_mae, "mape": mean_mape}}`` averaged
    over origins. Latest origin ends ``horizon`` steps before the series end
    so every split has a full holdout window.
    """
    y = np.asarray(y, dtype=float)
    models = list(models) if models is not None else all_models()
    if step is None:
        step = max(1, horizon // 2)
    n = len(y)
    min_train = max(14, horizon + 1)
    if n < min_train + horizon:
        raise ValueError(
            f"series too short ({n}) for horizon={horizon}; need at least "
            f"{min_train + horizon} points"
        )

    origins = [n - horizon - i * step for i in range(n_splits)]
    origins = [o for o in origins if o >= min_train]
    if not origins:
        raise ValueError("no valid origins; use a shorter horizon or fewer splits")

    results: dict[str, dict[str, float]] = {}
    for model in models:
        maes, mapes = [], []
        for origin in origins:
            train, test = y[:origin], y[origin:origin + horizon]
            try:
                model.fit(train)
                forecast = model.predict(horizon)
            except Exception:
                continue  # model can't handle this slice; skip the split
            maes.append(mae(test, forecast))
            mapes.append(mape(test, forecast))
        if not maes:
            raise ValueError(f"model {model.name} failed on every split")
        results[model.name] = {"mae": float(np.mean(maes)),
                               "mape": float(np.mean(mapes)),
                               "n_splits": len(maes)}
    return results


def comparison_table(results: dict[str, dict[str, float]]) -> list[dict]:
    """Sort backtest results by MAE (best first) into a printable table."""
    rows = [
        {"model": name, "mae": r["mae"], "mape": r["mape"], "n_splits": r["n_splits"]}
        for name, r in results.items()
    ]
    return sorted(rows, key=lambda r: r["mae"])


def pick_best(results: dict[str, dict[str, float]]) -> str:
    """Return the model name with the lowest mean MAE."""
    return comparison_table(results)[0]["model"]


def format_table(table: list[dict]) -> str:
    """Render a comparison table as aligned text."""
    header = f"{'model':<16}{'MAE':>10}{'MAPE %':>10}{'splits':>8}"
    lines = [header, "-" * len(header)]
    for row in table:
        lines.append(
            f"{row['model']:<16}{row['mae']:>10.2f}{row['mape']:>10.2f}"
            f"{row['n_splits']:>8}"
        )
    return "\n".join(lines)
