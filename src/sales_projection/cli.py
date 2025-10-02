from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .backtest import BacktestConfig, rolling_backtest
from .data import ensure_monthly_frequency, load_invoice_data
from .pipeline import ForecastConfig, build_forecasts


def parse_sku_filter(raw: Optional[str]) -> Optional[Iterable[str]]:
    if raw is None:
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def summarize_metrics(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "No metrics were generated."

    lines: list[str] = []
    valid = metrics[metrics["wmape"].notna()]
    if valid.empty:
        return "Metrics contain only NaN values; inspect error column."

    aggregate = (
        valid.groupby("model")[['wmape', 'mase']]
        .mean()
        .sort_values("wmape")
    )
    lines.append("Aggregate accuracy (lower is better):")
    lines.append(aggregate.to_string(float_format=lambda x: f"{x:.4f}"))

    per_model_counts = valid.groupby("model").size().sort_values(ascending=False)
    lines.append("\nFold counts per model:")
    lines.append(per_model_counts.to_string())

    if metrics["error"].str.len().gt(0).any():
        lines.append("\nWarnings:")
        for _, row in metrics[metrics["error"].str.len().gt(0)].iterrows():
            lines.append(
                f"- {row['ItemSku']} @ {row['cutoff']}: {row['model']} -> {row['error']}"
            )

    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monthly sales forecasting with ARIMA, Random Forest, and model selection.",
    )
    parser.add_argument(
        "--invoice-path",
        type=Path,
        required=True,
        help="Path to the input Invoice CSV (columns: ItemSku, Quantity, InvoiceTxnDate, ItemType).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Number of future months to forecast (default: 6).",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=18,
        help="Minimum history (months) before starting rolling backtests (default: 18).",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=12,
        help="Season length in months for MASE and naive comparisons (default: 12).",
    )
    parser.add_argument(
        "--sku-filter",
        type=str,
        help="Comma-separated list of ItemSku values to include (optional).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional path to write fold-level backtest metrics as CSV.",
    )
    parser.add_argument(
        "--forecast-output",
        type=Path,
        help="Optional path to write future forecasts as CSV.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    invoice_df = load_invoice_data(args.invoice_path)
    if args.sku_filter:
        allowed = set(parse_sku_filter(args.sku_filter))
        invoice_df = invoice_df[invoice_df["ItemSku"].isin(allowed)]
        if invoice_df.empty:
            raise ValueError("SKU filter removed all rows. Check the provided ItemSku values.")

    monthly = ensure_monthly_frequency(invoice_df)

    backtest_cfg = BacktestConfig(
        horizon=args.horizon,
        min_train=args.min_train,
        season_length=args.season_length,
    )
    backtest_result = rolling_backtest(monthly, backtest_cfg)

    print(summarize_metrics(backtest_result.metrics))

    forecast_cfg = ForecastConfig(horizon=args.horizon)
    forecast_df = build_forecasts(
        monthly_df=monthly,
        config=forecast_cfg,
        model_preferences=backtest_result.model_selection,
    )

    if forecast_df.empty:
        print("\nNo forecasts generated. Insufficient data? Check inputs.")
    else:
        print("\nGenerated forecasts (first 10 rows):")
        print(forecast_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if args.metrics_output:
        backtest_result.metrics.to_csv(args.metrics_output, index=False)
        print(f"\nSaved metrics to {args.metrics_output}")

    if args.forecast_output:
        forecast_df.to_csv(args.forecast_output, index=False)
        print(f"Saved forecasts to {args.forecast_output}")


if __name__ == "__main__":
    main()
