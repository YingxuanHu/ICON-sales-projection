from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class ArimaConfig:
    max_p: int = 2
    max_d: int = 1
    max_q: int = 2
    seasonal: bool = True
    max_P: int = 1
    max_D: int = 1
    max_Q: int = 1
    seasonal_period: int = 12


def select_sarima_order(series: pd.Series, config: ArimaConfig) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    best_aic = np.inf
    best_order = (1, 0, 0)
    best_seasonal = (0, 0, 0, 0)

    p_values = range(0, config.max_p + 1)
    d_values = range(0, config.max_d + 1)
    q_values = range(0, config.max_q + 1)

    seasonal_orders = [(0, 0, 0, 0)]
    if config.seasonal and config.seasonal_period > 1:
        seasonal_orders.extend(
            (
                P,
                D,
                Q,
                config.seasonal_period,
            )
            for P in range(0, config.max_P + 1)
            for D in range(0, config.max_D + 1)
            for Q in range(0, config.max_Q + 1)
        )

    for order in [(p, d, q) for p in p_values for d in d_values for q in q_values]:
        for seasonal_order in seasonal_orders:
            try:
                if order == (0, 0, 0) and seasonal_order == (0, 0, 0, 0):
                    continue
                model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization="approximate_diffuse",
                )
                fitted = model.fit(disp=False)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = order
                    best_seasonal = seasonal_order
            except (ValueError, np.linalg.LinAlgError):
                continue
    return best_order, best_seasonal


def forecast_with_arima(train_df: pd.DataFrame, horizon: int, config: Optional[ArimaConfig] = None) -> np.ndarray:
    if config is None:
        config = ArimaConfig()
    series = train_df.set_index("ds")["y"].astype(float)
    order, seasonal_order = select_sarima_order(series, config)
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        initialization="approximate_diffuse",
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=horizon)
    return forecast.predicted_mean.to_numpy()


class RandomForestSeriesForecaster:
    def __init__(
        self,
        n_estimators: int = 400,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.max_lag = 12
        self.fitted = False
        self.fallback_value = 0.0

    def _add_time_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["month"] = df["ds"].dt.month
        df["quarter"] = df["ds"].dt.quarter
        df["year"] = df["ds"].dt.year
        df["time_index"] = np.arange(len(df))
        return df

    def _add_lag_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        for lag in (1, 3, 6, 12):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        for window in (3, 6):
            df[f"rolling_mean_{window}"] = df["y"].rolling(window).mean()
        return df

    def _prepare_training_matrix(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = self._add_time_features(frame)
        df = self._add_lag_features(df)
        df = df.dropna()
        if df.empty:
            return df, pd.Series(dtype=float)
        features = df.drop(columns=["y"])
        target = df["y"].astype(float)
        return features, target

    def fit(self, train_df: pd.DataFrame) -> None:
        ordered = train_df.sort_values("ds").copy()
        ordered = ordered.reset_index(drop=True)
        self.fallback_value = float(ordered["y"].iloc[-1])

        features, target = self._prepare_training_matrix(ordered)
        if len(target) == 0:
            self.fitted = False
            return
        self.model.fit(features, target)
        self.fitted = True

    def _feature_row(self, history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        history = history.sort_values("ds")
        base = pd.DataFrame({"ds": [target_date]})
        base["month"] = target_date.month
        base["quarter"] = target_date.quarter
        base["year"] = target_date.year
        base["time_index"] = len(history)

        history_indexed = history.set_index("ds")
        for lag in (1, 3, 6, 12):
            lookup_date = target_date - DateOffset(months=lag)
            value = history_indexed["y"].get(lookup_date, np.nan)
            if pd.isna(value):
                value = history_indexed["y"].iloc[-1] if not history_indexed.empty else 0.0
            base[f"lag_{lag}"] = float(value)

        for window in (3, 6):
            window_dates = [target_date - DateOffset(months=i + 1) for i in range(window)]
            values = [history_indexed["y"].get(date, np.nan) for date in window_dates]
            valid = [float(v) for v in values if not pd.isna(v)]
            base[f"rolling_mean_{window}"] = float(np.mean(valid)) if valid else float(self.fallback_value)

        return base

    def forecast(self, train_df: pd.DataFrame, horizon: int) -> np.ndarray:
        history = train_df.sort_values("ds").copy().reset_index(drop=True)
        predictions = []
        current_history = history.copy()
        for step in range(horizon):
            if current_history.empty:
                raise ValueError("RandomForest forecaster requires non-empty history.")
            next_date = current_history["ds"].iloc[-1] + relativedelta(months=1)
            feature_row = self._feature_row(current_history, next_date)
            if self.fitted:
                yhat = float(self.model.predict(feature_row)[0])
            else:
                yhat = float(self.fallback_value)
            predictions.append(yhat)
            current_history = pd.concat(
                [current_history, pd.DataFrame({"ds": [next_date], "y": [yhat]})],
                ignore_index=True,
            )
        return np.asarray(predictions, dtype=float)


def ensemble_average(*arrays: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(arrays)
    return np.nanmean(stacked, axis=0)
