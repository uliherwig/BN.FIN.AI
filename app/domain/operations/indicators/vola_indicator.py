import pandas as pd
import talib as ta
from app.domain import *


class VolatilityIndicator(BaseIndicator):
    """Volatility Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:

        params = VolatilityModel.model_validate_json(param_str)
        prices = df["C"].to_numpy()
        signal_col_id = f"{IndicatorEnum.VOLA.name.lower()}_signal"

        df['volatility_ma'] = df['V'].rolling(window=params.VOLA_period).mean()
        df[signal_col_id] = SignalEnum.HOLD.value
        df.loc[df['V'] > df['volatility_ma'], signal_col_id] = SignalEnum.BUY.value
        df.loc[df['V'] < df['volatility_ma'], signal_col_id] = SignalEnum.SELL.value

        return df

    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:

        params = VolatilityModel.model_validate_json(param_str)
              
        df['Volatility'] = df['Close'].rolling(params.VOLA_period).std() 
   
        df['Volume_SMA'] = df['Volume'].rolling(params.VOLA_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']


        return df
