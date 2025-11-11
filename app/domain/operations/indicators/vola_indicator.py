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

        # 1. Volatility features
        df['Returns_1d'] = df['Close'].pct_change(1)
        df['Returns_2d'] = df['Close'].pct_change(2)

        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Volatility'] = df['Returns_1d'].rolling(params.VOLA_period).std()
        
        # 3. Price action
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # 4. Volume features
        df['Volume_SMA'] = df['Volume'].rolling(params.VOLA_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']


        return df
