import json
import numpy as np
import pandas as pd
import talib as ta
from typing import Any
from datetime import datetime, timedelta, timezone

from app.domain.operations.indicators.base_indicator import BaseIndicator

from app.domain import *

class RsiIndicator(BaseIndicator):
    """Relative Strength Index Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        """
        Calculates the RSI indicator and generates buy/sell signals.

        Args:
            df: DataFrame with 'C' (Close prices).
           

        Returns:
            DataFrame with additional columns:
            - 'rsi_<period>': RSI values.
            - 'rsi_signal': 1 (Buy when oversold), -1 (Sell when overbought), 0 (Hold).
        """
        # Parse parameters (example with Pydantic model, similar to your MacdModel)
        params = RsiModel.model_validate_json(param_str)  # Assumption: RsiModel similar to MacdModel

        # Calculate RSI
        rsi_col = f"rsi_{params.RSI_period}"
        df[rsi_col] = ta.RSI(
            df["C"].to_numpy(),
            timeperiod=params.RSI_period
        )

        # Signal column name
        signal_col_id = f"{IndicatorEnum.RSI.name.lower()}_signal"

        # Generate signals:
        # 1 = Buy (RSI below 'oversold' and rising)
        # -1 = Sell (RSI above 'overbought' and falling)
        df[signal_col_id] = 0  # Default: Hold

        # Buy signal: RSI crosses from below to above the oversold area
        df.loc[
            (df[rsi_col].shift(1) < params.RSI_oversold) &
            (df[rsi_col] > params.RSI_oversold),
            signal_col_id
        ] = 1

        # Sell signal: RSI crosses from above to below the overbought area
        df.loc[
            (df[rsi_col].shift(1) > params.RSI_overbought) &
            (df[rsi_col] < params.RSI_overbought),
            signal_col_id
        ] = -1

        return df
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
      
        params = RsiModel.model_validate_json(param_str)        
        df["RSI"] = ta.RSI(
            df["Close"].to_numpy(),
            timeperiod=params.RSI_period
        )

        df["RSI_diff"] = df["RSI"].diff().round(3)
        df["RSI_slope"] = df["RSI_diff"].diff().round(3)

        return df
