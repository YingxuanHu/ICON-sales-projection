import numpy as np
import pandas as pd


def wmape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    denom = np.abs(actual_arr).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(actual_arr - predicted_arr).sum() / denom)


def mase(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    insample: pd.Series | np.ndarray,
    season_length: int,
) -> float:
    insample_arr = np.asarray(insample, dtype=float)
    if season_length < 1:
        season_length = 1
    if insample_arr.size <= season_length:
        return np.nan
    denom = np.mean(np.abs(insample_arr[season_length:] - insample_arr[:-season_length]))
    if denom == 0:
        return np.nan
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    mae = np.mean(np.abs(actual_arr - predicted_arr))
    return float(mae / denom)
