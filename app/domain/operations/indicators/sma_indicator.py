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
        # Parse JSON string: "{\"short_ma\": 10, \"long_ma\": 10}" -> dict -> SmaModel
        params = SmaModel(**json.loads(param_str))
        prices = df["C"].to_numpy()
        signal_col_id = f"{StrategyLibEnum.SMA.name.lower()}_signal"
        df[f"sma_{params.short_ma}"] = ta.SMA(prices, timeperiod=params.short_ma).round(3)
        df[f"sma_{params.long_ma}"] = ta.SMA(prices, timeperiod=params.long_ma).round(3)
        df["sma_gt_lt"] = np.where(df[f"sma_{params.short_ma}"] > df[f"sma_{params.long_ma}"], 1, 0)
        df[signal_col_id] = df["sma_gt_lt"].diff()
        return df


