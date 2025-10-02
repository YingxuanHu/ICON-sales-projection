# ICON Sales Projection Toolkit

A forecasting workflow that pairs ARIMA with machine learning to compare, blend, and select the best sales projections per SKU.

## Features
- Cleans Invoice CSV exports (expects columns `ItemSku`, `Quantity`, `InvoiceTxnDate`, `ItemType`).
- Aggregates sales into monthly series for every SKU.
- Runs rolling backtests for ARIMA, Random Forest, and an ensemble blend.
- Automatically selects the best-performing model per SKU based on WMAPE.
- Generates side-by-side forecasts, the ensemble mix, and the chosen projection for each future month.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage
```bash
sales-projection \
  --invoice-path /path/to/Invoice.csv \
  --horizon 6 \
  --min-train 18 \
  --metrics-output metrics.csv \
  --forecast-output forecasts.csv
```

Optional flags:
- `--sku-filter sku1,sku2` to focus on select SKUs.
- `--season-length 12` to tune metrics for different seasonal cycles.

Inspect the console output for aggregate accuracy tables and a preview of the generated forecasts.

## Outputs
- `metrics.csv`: fold-level backtest results with WMAPE, MASE, and any model warnings.
- `forecasts.csv`: future monthly projections containing ARIMA, Random Forest, ensemble, and per-SKU selected forecasts.

Use the outputs to quantify performance improvements and feed downstream inventory or planning workflows.
