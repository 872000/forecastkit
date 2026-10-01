"""Seeded synthetic sales-data generator.

Each product is built from interpretable components so the toolkit can be
validated end to end without any external data or API keys::

    sales(t) = base + trend*t + weekly(t) + yearly(t) + noise(t)

Weekly seasonality has period 7 (day-of-week effects), yearly seasonality
has period 365. All noise is clipped at zero so units sold stay realistic.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PRODUCTS = {
    # product: (base, trend/day, weekly_amp, yearly_amp, noise_std, phase)
    "running-shoes": (120.0, 0.08, 18.0, 12.0, 8.0, 0.4),
    "coffee-mugs": (60.0, -0.02, 10.0, 6.0, 5.0, 1.7),
    "winter-jackets": (40.0, 0.03, 6.0, 35.0, 7.0, 2.9),
}

DEFAULT_DAYS = 730
START_DATE = np.datetime64("2024-01-01")


def generate_product_series(rng: np.random.Generator, days: int, base: float,
                            trend: float, weekly_amp: float, yearly_amp: float,
                            noise_std: float, phase: float = 0.0) -> np.ndarray:
    """Generate one daily demand series from its components."""
    t = np.arange(days, dtype=float)
    weekly = weekly_amp * np.sin(2.0 * np.pi * t / 7.0 + phase)
    yearly = yearly_amp * np.sin(2.0 * np.pi * t / 365.0)
    noise = rng.normal(0.0, noise_std, days)
    y = base + trend * t + weekly + yearly + noise
    return np.maximum(y, 0.0).round(2)


def make_dataset(seed: int = 42, days: int = DEFAULT_DAYS) -> dict[str, np.ndarray]:
    """Build the full multi-product dataset. Same seed -> same numbers."""
    rng = np.random.default_rng(seed)
    dataset = {}
    for product, params in PRODUCTS.items():
        dataset[product] = generate_product_series(rng, days, *params)
    return dataset


def dataset_to_rows(dataset: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
    """Flatten a dataset to (date, product, units_sold) rows."""
    rows = []
    for product, series in dataset.items():
        dates = START_DATE + np.arange(len(series))
        for date, value in zip(dates, series):
            rows.append((str(date), product, float(value)))
    return rows


def save_demo_csv(path: str | Path, seed: int = 42, days: int = DEFAULT_DAYS) -> Path:
    """Write the bundled demo dataset to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = dataset_to_rows(make_dataset(seed=seed, days=days))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "product", "units_sold"])
        writer.writerows(rows)
    return path


def load_product(path: str | Path, product: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one product's series from a demo CSV. Returns (dates, values)."""
    dates, values = [], []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if row["product"] == product:
                dates.append(np.datetime64(row["date"]))
                values.append(float(row["units_sold"]))
    if not values:
        raise ValueError(f"product {product!r} not found in {path}")
    return np.array(dates), np.array(values)
