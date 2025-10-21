import json
import numpy as np
import pandas as pd
import talib as ta
from typing import Any
from datetime import datetime, timedelta, timezone

from app.domain.operations.indicators.base_indicator import BaseIndicator

from app.domain import *

class TemaIndicator(BaseIndicator):
    """Triple Exponential Moving Average Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = TemaModel.model_validate_json(param_str)
        prices = df["C"].to_numpy()
        signal_col_id = f"{StrategyLibEnum.TEMA.name.lower()}_signal"
        df[f"tema_{params.short_ma}"] = ta.TEMA(prices, timeperiod=params.short_ma).round(3)
        df[f"tema_{params.long_ma}"] = ta.TEMA(prices, timeperiod=params.long_ma).round(3)
        df["tema_gt_lt"] = np.where(df[f"tema_{params.short_ma}"] > df[f"tema_{params.long_ma}"], 1, 0)
        df[signal_col_id] = df["tema_gt_lt"].diff()
        return df


