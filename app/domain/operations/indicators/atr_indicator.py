import pandas as pd
import talib as ta
from app.domain import *

class AtrIndicator(BaseIndicator):
    """Average True Range (ATR) Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = AtrModel.model_validate_json(param_str)
        high = df["H"].to_numpy()
        low = df["L"].to_numpy()
        close = df["C"].to_numpy()
        signal_col_id = f"{IndicatorEnum.ATR.name.lower()}_signal"

        df['ATR'] = ta.ATR(
            high, low, close, timeperiod=params.ATR_period).round(3)

        # Generate ATR signals

        df[signal_col_id] = SignalEnum.HOLD.value  # Set to HOLD by default

        df.loc[df['ATR'] > params.ATR_threshold, signal_col_id] = SignalEnum.BUY.value

        df.loc[df['ATR'] < params.ATR_threshold, signal_col_id] = SignalEnum.SELL.value

        return df
    

    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:

        params = AtrModel.model_validate_json(param_str)
        high = df["High"].to_numpy()
        low = df["Low"].to_numpy()
        close = df["Close"].to_numpy()
        df['ATR'] = ta.ATR(high, low, close, timeperiod=params.ATR_period).round(3) 
        df["ATR_diff"] = df['ATR'].diff().round(3)  
        df["ATR_slope"] = df["ATR_diff"].diff().round(3)

        return df

