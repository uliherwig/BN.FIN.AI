import json
import numpy as np
import pandas as pd
import talib as ta
from typing import Any
from datetime import datetime, timedelta, timezone

from app.domain.operations.indicators.base_indicator import BaseIndicator

from app.domain import *

class WmaIndicator(BaseIndicator):
    """Weighted Moving Average Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        
        params = WmaModel.model_validate_json(param_str)
        prices = df["C"].to_numpy()
        signal_col_id = f"{IndicatorEnum.WMA.name.lower()}_signal"
        df[f"wma_{params.WMA_short}"] = ta.WMA(prices, timeperiod=params.WMA_short).round(3)
        df[f"wma_{params.WMA_long}"] = ta.WMA(prices, timeperiod=params.WMA_long).round(3)
        df["wma_gt_lt"] = np.where(df[f"wma_{params.WMA_short}"] > df[f"wma_{params.WMA_long}"], 1, 0)
        df[signal_col_id] = df["wma_gt_lt"].diff()
        return df
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = WmaModel.model_validate_json(param_str)
        prices = df["Close"].to_numpy()
        df[f"wma_{params.WMA_short}"] = ta.WMA(prices, timeperiod=params.WMA_short).round(3)
        df[f"wma_{params.WMA_long}"] = ta.WMA(prices, timeperiod=params.WMA_long).round(3)
        df["WMA_diff"] = (df[f"wma_{params.WMA_short}"] - df[f"wma_{params.WMA_long}"]).round(3)
        df["WMA_cross"] = np.where(df["WMA_diff"] > 0, 1, -1)
        df["WMA_ratio"] = (df[f"wma_{params.WMA_short}"] / df[f"wma_{params.WMA_long}"] - 1).round(3)
        df["WMA_cross_change"] = df["WMA_cross"].diff().fillna(0)
        df["WMA_diff_slope"] = df["WMA_diff"].diff()
        return df

