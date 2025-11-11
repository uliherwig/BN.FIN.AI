import numpy as np
import pandas as pd
import talib as ta

from app.domain import *
from app.domain.models.strategies.breakout_model import BreakoutModel
from app.domain.operations.indicators.base_indicator import BaseIndicator

class BreakoutIndicator(BaseIndicator):
    """
    Breakout Strategy Indicator
    
    Generates signals when:
    - BUY: Price exceeds the highest high of the last N periods
    - SELL: Price falls below the lowest low of the last N periods
    
    Uses TA-Lib for robust technical analysis
    """
    
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = BreakoutModel.model_validate_json(param_str)
        
        # Convert to numpy arrays for TA-Lib
        high = df['H'].to_numpy()
        low = df['L'].to_numpy()
        close = df['C'].to_numpy()

        # TA-Lib: Calculate highest high and lowest low of the last N periods
        df['breakout_high'] = ta.MAX(close, timeperiod=params.Breakout_period)
        df['breakout_low'] = ta.MIN(close, timeperiod=params.Breakout_period)
        
        # Shift by 1, since we only want to react to previous highs/lows
        prev_high = df['breakout_high'].shift(1)
        prev_low = df['breakout_low'].shift(1)
        
        # Breakout conditions
        bullish_breakout = close > prev_high
        bearish_breakout = close < prev_low   
        
        # Combine all filters
        df['breakout_up'] = bullish_breakout
        df['breakout_down'] = bearish_breakout

        # Create trading signals
        signal_col_id = f"{IndicatorEnum.BREAKOUT.name.lower()}_signal"
        
        df[signal_col_id] = SignalEnum.HOLD.value  # Set to HOLD by default
        
        # BUY Signal: Upward Breakout (+1)
        df.loc[df['breakout_up'], signal_col_id] = SignalEnum.BUY.value

        # SELL Signal: Downward Breakout (-1)
        df.loc[df['breakout_down'], signal_col_id] = SignalEnum.SELL.value

        return df
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = BreakoutModel.model_validate_json(param_str)
        
        # Convert to numpy arrays for TA-Lib
        high = df['High'].to_numpy()
        low = df['Low'].to_numpy()
        close = df['Close'].to_numpy()

        # TA-Lib: Calculate highest high and lowest low of the last N periods
        df['breakout_high'] = ta.MAX(close, timeperiod=params.Breakout_period)
        df['breakout_low'] = ta.MIN(close, timeperiod=params.Breakout_period)
        
        # Calculate breakout range
        df["breakout_range"] = (df['breakout_high'] - df['breakout_low']).round(3)
        df["breakout_mid"] = (df['breakout_high'] + df['breakout_low']) / 2

        # Calculate breakout strength with 3 decimal places
        df["BREAKOUT_strength"] = ((df["Close"] - df["breakout_mid"]) / df["breakout_range"]).round(3)
        
        return df