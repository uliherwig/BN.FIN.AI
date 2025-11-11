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
        signal_col_id = f"{IndicatorEnum.TEMA.name.lower()}_signal"
        df[f"tema_{params.TEMA_short}"] = ta.TEMA(prices, timeperiod=params.TEMA_short).round(3)
        df[f"tema_{params.TEMA_long}"] = ta.TEMA(prices, timeperiod=params.TEMA_long).round(3)
        df["tema_gt_lt"] = np.where(df[f"tema_{params.TEMA_short}"] > df[f"tema_{params.TEMA_long}"], 1, 0)
        df[signal_col_id] = df["tema_gt_lt"].diff()
        return df
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = TemaModel.model_validate_json(param_str)
        prices = df["Close"].to_numpy()
        df[f"tema_{params.TEMA_short}"] = ta.TEMA(prices, timeperiod=params.TEMA_short).round(3)
        df[f"tema_{params.TEMA_long}"] = ta.TEMA(prices, timeperiod=params.TEMA_long).round(3)
        df["TEMA_diff"] = (df[f"tema_{params.TEMA_short}"] - df[f"tema_{params.TEMA_long}"]).round(3)
        df["TEMA_ratio"] = (df[f"tema_{params.TEMA_short}"] / df[f"tema_{params.TEMA_long}"] - 1).round(3)
        df["TEMA_slope"] = df["TEMA_diff"].diff().round(3)
        return df


