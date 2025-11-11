import json
import numpy as np
import pandas as pd
import talib as ta
from typing import Any
from datetime import datetime, timedelta, timezone

from app.domain.operations.indicators.base_indicator import BaseIndicator

from app.domain import *

class EmaIndicator(BaseIndicator):
    """Exponential Moving Average Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = EmaModel.model_validate_json(param_str)
        prices = df["C"].to_numpy()
        signal_col_id = f"{IndicatorEnum.EMA.name.lower()}_signal"
        df[f"ema_{params.EMA_short}"] = ta.EMA(prices, timeperiod=params.EMA_short).round(3)
        df[f"ema_{params.EMA_long}"] = ta.EMA(prices, timeperiod=params.EMA_long).round(3)
        df["ema_gt_lt"] = np.where(df[f"ema_{params.EMA_short}"] > df[f"ema_{params.EMA_long}"], 1, 0)
        df[signal_col_id] = df["ema_gt_lt"].diff()
        return 
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = EmaModel.model_validate_json(param_str)
        prices = df["Close"].to_numpy()
        df[f"ema_{params.EMA_short}"] = ta.EMA(prices, timeperiod=params.EMA_short).round(3)
        df[f"ema_{params.EMA_long}"] = ta.EMA(prices, timeperiod=params.EMA_long).round(3)
        df["EMA_diff"] = (df[f"ema_{params.EMA_short}"] - df[f"ema_{params.EMA_long}"]).round(3)
        # df["ema_ratio"] = (df[f"ema_{params.EMA_short}"] / df[f"ema_{params.EMA_long}"] - 1).round(3)
        # df["ema_cross"] = np.where(df["ema_diff"] > 0, 1, -1)
        # df["ema_cross_change"] = df["ema_cross"].diff().fillna(0).round(3)
        # df["ema_diff_slope"] = df["ema_diff"].diff().round(3)
        return df


