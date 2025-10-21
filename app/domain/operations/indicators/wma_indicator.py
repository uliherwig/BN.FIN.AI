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
        signal_col_id = f"{StrategyLibEnum.WMA.name.lower()}_signal"
        df[f"wma_{params.short_ma}"] = ta.WMA(prices, timeperiod=params.short_ma).round(3)
        df[f"wma_{params.long_ma}"] = ta.WMA(prices, timeperiod=params.long_ma).round(3)
        df["wma_gt_lt"] = np.where(df[f"wma_{params.short_ma}"] > df[f"wma_{params.long_ma}"], 1, 0)
        df[signal_col_id] = df["wma_gt_lt"].diff()
        return df

