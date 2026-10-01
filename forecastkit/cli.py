"""Command-line interface for ForecastKit.

Commands:
    backtest --product NAME [--horizon 30] [--splits 4]
        Rolling-origin backtest; prints the MAE/MAPE comparison table.
    forecast --product NAME [--horizon 30] [--out PATH]
        Fit every model, pick the backtest winner, and save a forecast PNG.
    demo
        End-to-end run: backtests all products, saves overlay charts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import data
from .backtest import (comparison_table, format_table, pick_best,
                       rolling_backtest)
from .models import all_models
from .plotting import plot_backtest_error, plot_forecast_overlay

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_CSV = REPO_ROOT / "data" / "demo_sales.csv"
DOCS_DIR = REPO_ROOT / "docs"


def _ensure_demo_csv() -> Path:
    if not DEMO_CSV.exists():
        data.save_demo_csv(DEMO_CSV)
    return DEMO_CSV


def cmd_backtest(args: argparse.Namespace) -> int:
    csv_path = _ensure_demo_csv()
    _, series = data.load_product(csv_path, args.product)
    results = rolling_backtest(series, horizon=args.horizon, n_splits=args.splits)
    table = comparison_table(results)
    print(f"Backtest: {args.product} (horizon={args.horizon}, splits={args.splits})")
    print(format_table(table))
    print(f"\nBest model: {pick_best(results)}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    csv_path = _ensure_demo_csv()
    _, series = data.load_product(csv_path, args.product)
    horizon = args.horizon

    # Pick the winner with a quick backtest, then refit on all data.
    results = rolling_backtest(series, horizon=min(horizon, 30), n_splits=3)
    winner = pick_best(results)
    print(f"Best model for {args.product}: {winner} "
          f"(MAE={results[winner]['mae']:.2f}, MAPE={results[winner]['mape']:.2f}%)")

    forecasts = {}
    for model in all_models():
        model.fit(series)
        forecasts[model.name] = model.predict(horizon)

    out = Path(args.out) if args.out else DOCS_DIR / f"forecast_{args.product}.png"
    plot_forecast_overlay(
        series, forecasts, out,
        title=f"{args.product}: {horizon}-day forecast (all models)")
    print(f"Saved chart: {out}")
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    csv_path = _ensure_demo_csv()
    print(f"Using dataset: {csv_path}\n")
    for product in data.PRODUCTS:
        _, series = data.load_product(csv_path, product)
        results = rolling_backtest(series, horizon=30, n_splits=4)
        table = comparison_table(results)
        print(f"== {product} ==")
        print(format_table(table))
        print(f"Best model: {pick_best(results)}\n")

        forecasts = {}
        for model in all_models():
            model.fit(series)
            forecasts[model.name] = model.predict(30)
        plot_forecast_overlay(
            series, forecasts, DOCS_DIR / f"forecast_{product}.png",
            title=f"{product}: 30-day forecast (all models)")
        plot_backtest_error(table, DOCS_DIR / f"backtest_{product}.png",
                            title=f"{product}: backtest MAE by model")
    print(f"Charts saved to {DOCS_DIR}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecastkit", description="Demand forecasting toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="rolling-origin backtest for a product")
    p_bt.add_argument("--product", required=True,
                      choices=sorted(data.PRODUCTS),
                      help="product to backtest")
    p_bt.add_argument("--horizon", type=int, default=30)
    p_bt.add_argument("--splits", type=int, default=4)
    p_bt.set_defaults(func=cmd_backtest)

    p_fc = sub.add_parser("forecast", help="forecast and save a chart")
    p_fc.add_argument("--product", required=True,
                      choices=sorted(data.PRODUCTS),
                      help="product to forecast")
    p_fc.add_argument("--horizon", type=int, default=30)
    p_fc.add_argument("--out", default=None, help="output PNG path")
    p_fc.set_defaults(func=cmd_forecast)

    p_demo = sub.add_parser("demo", help="end-to-end demo on all products")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
