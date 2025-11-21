import numpy as np
import pandas as pd
import talib as ta

from app.domain import *

class DonchianIndicator(BaseIndicator):
    
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = DonchianModel.model_validate_json(param_str)
 
        close = df['C'].to_numpy()
     
        df['donchian_high'] = ta.MAX(close, timeperiod=params.DONCHIAN_window)
        df['donchian_low'] = ta.MIN(close, timeperiod=params.DONCHIAN_window)
        df["donchian_width"] = (df["donchian_high"] - df["donchian_low"]).round(3)
        df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2

        df['breakout_up'] = close > df['donchian_high'].shift(1)
        df['breakout_down'] = close < df['donchian_low'].shift(1)
        
        # Stärke des Breakouts berechnen mit 3 Nachkommastellen
        df["donchian_breakout_strength"] = (df["C"] - df["donchian_mid"]) / df["donchian_width"].round(3)
        
        signal_col_id = f"{IndicatorEnum.DONCHIAN.name.lower()}_signal"

        df[signal_col_id] = SignalEnum.HOLD.value
        df.loc[df['breakout_up'], signal_col_id] = SignalEnum.BUY.value
        df.loc[df['breakout_down'], signal_col_id] = SignalEnum.SELL.value
        return df
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = DonchianModel.model_validate_json(param_str)
 
        close = df['Close'].to_numpy()
     
        df['donchian_high'] = ta.MAX(close, timeperiod=params.DONCHIAN_window)
        df['donchian_low'] = ta.MIN(close, timeperiod=params.DONCHIAN_window)
        df["donchian_width"] = (df["donchian_high"] - df["donchian_low"]).round(3)
        df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2

        # Stärke des Breakouts berechnen mit 3 Nachkommastellen
        df["DONCHIAN_breakout_strength"] = (
            df["Close"] - df["donchian_mid"]) / df["donchian_width"].round(3)
        
        return df