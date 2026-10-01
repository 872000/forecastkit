"""Matplotlib charts: history + forecast, and per-model overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"moving_average": "#e67e22", "holt": "#2980b9", "seasonal_naive": "#27ae60"}


def plot_forecast_overlay(history: np.ndarray,
                          forecasts: dict[str, np.ndarray],
                          path: str | Path, title: str,
                          history_tail: int = 180) -> Path:
    """Plot recent history plus one forecast line per model.

    Saves a PNG to ``path`` and returns it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hist = np.asarray(history, dtype=float)[-history_tail:]
    n_hist = len(hist)
    horizon = len(next(iter(forecasts.values())))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(n_hist), hist, color="#2c3e50", linewidth=1.6, label="history")

    for name, fc in forecasts.items():
        color = COLORS.get(name, "#7f8c8d")
        x = range(n_hist - 1, n_hist + horizon)
        y = np.concatenate([[hist[-1]], np.asarray(fc, dtype=float)])
        ax.plot(x, y, color=color, linewidth=1.6, linestyle="--",
                label=name.replace("_", " ").title())

    ax.axvline(n_hist - 1, color="#95a5a6", linestyle=":", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel("units sold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_backtest_error(table: list[dict], path: str | Path,
                        title: str = "Backtest MAE by model") -> Path:
    """Bar chart comparing backtest MAE across models."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = [r["model"].replace("_", " ").title() for r in table]
    maes = [r["mae"] for r in table]
    colors = [COLORS.get(r["model"], "#7f8c8d") for r in table]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, maes, color=colors, edgecolor="#2c3e50")
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set_title(title)
    ax.set_ylabel("mean MAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
