from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .metrics import mase, wmape
from .models import ArimaConfig, RandomForestSeriesForecaster, ensemble_average, forecast_with_arima


@dataclass
class BacktestConfig:
    horizon: int = 3
    min_train: int = 18
    season_length: int = 12
    arima: ArimaConfig = ArimaConfig()


@dataclass
class BacktestResult:
    metrics: pd.DataFrame
    model_selection: Dict[str, str]


def rolling_backtest(df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    records: List[dict] = []

    for sku, group in df.groupby("ItemSku"):
        sku_df = group.sort_values("ds").reset_index(drop=True)
        if len(sku_df) < config.min_train + config.horizon:
            continue

        for cutoff in range(config.min_train, len(sku_df) - config.horizon + 1):
            train = sku_df.iloc[:cutoff].copy()
            test = sku_df.iloc[cutoff : cutoff + config.horizon].copy()
            insample = train["y"].to_numpy(dtype=float)
            actual = test["y"].to_numpy(dtype=float)
            cutoff_date = train["ds"].iloc[-1]

            # ARIMA
            try:
                arima_forecast = forecast_with_arima(train, config.horizon, config.arima)
                records.append(
                    {
                        "ItemSku": sku,
                        "model": "arima",
                        "cutoff": cutoff_date,
                        "wmape": wmape(actual, arima_forecast),
                        "mase": mase(actual, arima_forecast, insample, config.season_length),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                records.append(
                    {
                        "ItemSku": sku,
                        "model": "arima",
                        "cutoff": cutoff_date,
                        "wmape": np.nan,
                        "mase": np.nan,
                        "error": str(exc),
                    }
                )
                arima_forecast = np.full(config.horizon, np.nan)

            # Random Forest
            try:
                rf_model = RandomForestSeriesForecaster()
                rf_model.fit(train)
                rf_forecast = rf_model.forecast(train, config.horizon)
                records.append(
                    {
                        "ItemSku": sku,
                        "model": "random_forest",
                        "cutoff": cutoff_date,
                        "wmape": wmape(actual, rf_forecast),
                        "mase": mase(actual, rf_forecast, insample, config.season_length),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                records.append(
                    {
                        "ItemSku": sku,
                        "model": "random_forest",
                        "cutoff": cutoff_date,
                        "wmape": np.nan,
                        "mase": np.nan,
                        "error": str(exc),
                    }
                )
                rf_forecast = np.full(config.horizon, np.nan)

            # Ensemble
            blended = ensemble_average(arima_forecast, rf_forecast)
            records.append(
                {
                    "ItemSku": sku,
                    "model": "ensemble",
                    "cutoff": cutoff_date,
                    "wmape": wmape(actual, blended),
                    "mase": mase(actual, blended, insample, config.season_length),
                    "error": "",
                }
            )

    metrics = pd.DataFrame.from_records(records)
    model_selection: Dict[str, str] = {}
    if metrics.empty:
        return BacktestResult(metrics=metrics, model_selection=model_selection)

    valid = metrics[metrics["wmape"].notna()]
    if valid.empty:
        return BacktestResult(metrics=metrics, model_selection=model_selection)

    sku_model_scores = (
        valid.groupby(["ItemSku", "model"])["wmape"].mean().reset_index()
    )
    for sku, group in sku_model_scores.groupby("ItemSku"):
        best_row = group.sort_values("wmape").iloc[0]
        model_selection[sku] = str(best_row["model"])

    return BacktestResult(metrics=metrics, model_selection=model_selection)
