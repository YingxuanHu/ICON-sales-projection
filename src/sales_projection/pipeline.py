from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .models import ArimaConfig, RandomForestSeriesForecaster, ensemble_average, forecast_with_arima


@dataclass
class ForecastConfig:
    horizon: int = 6
    arima: ArimaConfig = ArimaConfig()


def build_forecasts(
    monthly_df: pd.DataFrame,
    config: ForecastConfig,
    model_preferences: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    records: list[dict] = []
    model_preferences = model_preferences or {}

    for sku, group in monthly_df.groupby("ItemSku"):
        sku_df = group.sort_values("ds").reset_index(drop=True)
        if sku_df.empty:
            continue

        try:
            arima_pred = forecast_with_arima(sku_df, config.horizon, config.arima)
        except Exception:  # noqa: BLE001
            arima_pred = np.full(config.horizon, np.nan)

        rf_model = RandomForestSeriesForecaster()
        try:
            rf_model.fit(sku_df)
            rf_pred = rf_model.forecast(sku_df, config.horizon)
        except Exception:  # noqa: BLE001
            rf_pred = np.full(config.horizon, np.nan)

        ensemble_pred = ensemble_average(arima_pred, rf_pred)

        last_date = sku_df["ds"].iloc[-1]
        future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(config.horizon)]

        preferred = model_preferences.get(sku, "ensemble")
        selections = {
            "arima": arima_pred,
            "random_forest": rf_pred,
            "ensemble": ensemble_pred,
        }
        chosen = selections.get(preferred)
        if chosen is None or np.all(np.isnan(chosen)):
            preferred = "ensemble"
            chosen = ensemble_pred

        for idx, future_date in enumerate(future_dates):
            records.append(
                {
                    "ItemSku": sku,
                    "ds": future_date,
                    "arima": float(arima_pred[idx]) if not np.isnan(arima_pred[idx]) else np.nan,
                    "random_forest": float(rf_pred[idx]) if not np.isnan(rf_pred[idx]) else np.nan,
                    "ensemble": float(ensemble_pred[idx]) if not np.isnan(ensemble_pred[idx]) else np.nan,
                    "selected_model": preferred,
                    "selected_forecast": float(chosen[idx]) if not np.isnan(chosen[idx]) else np.nan,
                }
            )

    return pd.DataFrame.from_records(records)
