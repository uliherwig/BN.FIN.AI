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
        signal_col_id = f"{StrategyLibEnum.EMA.name.lower()}_signal"
        df[f"ema_{params.short_ma}"] = ta.EMA(prices, timeperiod=params.short_ma).round(3)
        df[f"ema_{params.long_ma}"] = ta.EMA(prices, timeperiod=params.long_ma).round(3)
        df["ema_gt_lt"] = np.where(df[f"ema_{params.short_ma}"] > df[f"ema_{params.long_ma}"], 1, 0)
        df[signal_col_id] = df["ema_gt_lt"].diff()
        return df


