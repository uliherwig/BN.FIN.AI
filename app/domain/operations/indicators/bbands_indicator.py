import json
import numpy as np
import pandas as pd
import talib as ta
from typing import Any
from datetime import datetime, timedelta, timezone

from app.domain.operations.indicators.base_indicator import BaseIndicator
from app.domain.models.strategies.bbands_model import BbandsModel

from app.domain import *

class BbandsIndicator(BaseIndicator):
    """Bollinger Bands Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = BbandsModel(**json.loads(param_str))
        prices = df["Close"].to_numpy()
        num_std_dev = getattr(params, "num_std_dev", 2)  # fallback to 2 if not present
        upperband, middleband, lowerband = ta.BBANDS(
            prices,
            timeperiod=params.BBANDS_period
        )
        df["BB_upper"] = upperband.round(3)
        df["BB_middle"] = middleband.round(3)
        df["BB_lower"] = lowerband.round(3)
        df["BB_width"] = (upperband - lowerband).round(3)
        df["BB_pctb"] = ((prices - lowerband) / (upperband - lowerband)).round(3)
        return df

    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        # Parse JSON string: "{\"short_ma\": 10, \"long_ma\": 10}" -> dict -> SmaModel
        params = BbandsModel(**json.loads(param_str))
        prices = df["Close"].to_numpy()
        upperband, middleband, lowerband = ta.BBANDS(
            prices,
            timeperiod=params.BBANDS_period
        )
        df["BB_upper"] = upperband.round(3)
        df["BB_middle"] = middleband.round(3)
        df["BB_lower"] = lowerband.round(3)
        df["BB_width"] = (upperband - lowerband).round(3)
        df["BB_pctb"] = ((prices - lowerband) / (upperband - lowerband)).round(3)
        return df


