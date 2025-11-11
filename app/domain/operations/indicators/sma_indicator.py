import json
import numpy as np
import pandas as pd
import talib as ta
from typing import Any
from datetime import datetime, timedelta, timezone

from app.domain.operations.indicators.base_indicator import BaseIndicator

from app.domain import *

class SmaIndicator(BaseIndicator):
    """Simple Moving Average Indicator"""
    
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        # Parse JSON string: "{\"SMA_short\": 10, \"SMA_long\": 10}" -> dict -> SmaModel
        params = SmaModel(**json.loads(param_str))
        prices = df["C"].to_numpy()
        signal_col_id = f"{IndicatorEnum.SMA.name.lower()}_signal"
        df[f"sma_{params.SMA_short}"] = ta.SMA(prices, timeperiod=params.SMA_short).round(3)
        df[f"sma_{params.SMA_long}"] = ta.SMA(prices, timeperiod=params.SMA_long).round(3)
        df["sma_gt_lt"] = np.where(df[f"sma_{params.SMA_short}"] > df[f"sma_{params.SMA_long}"], 1, 0)
        df[signal_col_id] = df["sma_gt_lt"].diff()
        df.drop(columns=["sma_gt_lt"], inplace=True)
        return df
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        # Parse JSON string: "{\"SMA_short\": 10, \"SMA_long\": 10}" -> dict -> SmaModel
        params = SmaModel(**json.loads(param_str))
        prices = df["Close"].to_numpy()
        df[f"sma_{params.SMA_short}"] = ta.SMA(prices, timeperiod=params.SMA_short).round(3)
        df[f"sma_{params.SMA_long}"] = ta.SMA(prices, timeperiod=params.SMA_long).round(3)
        df["SMA_diff"] = (df[f"sma_{params.SMA_short}"] - df[f"sma_{params.SMA_long}"]).round(3) 
        # df["sma_cross"] = (np.where(df["sma_diff"] > 0, 1, -1)).round(3)
        # df["sma_ratio"] = (df[f"sma_{params.SMA_short}"] / df[f"sma_{params.SMA_long}"] - 1).round(3)
        # df["sma_cross_change"] = df["sma_cross"].diff().fillna(0)
        # df["sma_diff_slope"] = df["sma_diff"].diff()
        return df


