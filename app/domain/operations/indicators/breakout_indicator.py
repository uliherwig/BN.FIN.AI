import numpy as np
import pandas as pd
import talib as ta

from app.domain import *
from app.domain.models.strategies.breakout_model import BreakoutModel
from app.domain.operations.indicators.base_indicator import BaseIndicator

class BreakoutIndicator(BaseIndicator):
    """
    Breakout Strategie Indicator
    
    Generiert Signale wenn:
    - BUY: Preis überschreitet das höchste High der letzten N Perioden
    - SELL: Preis unterschreitet das niedrigste Low der letzten N Perioden
    
    Verwendet TA-Lib für robuste technische Analyse
    """
    
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = BreakoutModel.model_validate_json(param_str)
        
        # Konvertiere zu numpy arrays für TA-Lib
        high = df['H'].to_numpy()
        low = df['L'].to_numpy()
        close = df['C'].to_numpy()

        # TA-Lib: Berechne Highest High und Lowest Low der letzten N Perioden
        df['breakout_high'] = ta.MAX(close, timeperiod=params.breakout_period)
        df['breakout_low'] = ta.MIN(close, timeperiod=params.breakout_period) 
        
        # Shift um 1, da wir nur auf vorherige Highs/Lows reagieren wollen
        prev_high = df['breakout_high'].shift(1)
        prev_low = df['breakout_low'].shift(1)
        
        # Breakout-Bedingungen
        bullish_breakout = close > prev_high
        bearish_breakout = close < prev_low   
        
        # Kombiniere alle Filter
        df['breakout_up'] = bullish_breakout
        df['breakout_down'] = bearish_breakout

        # Erstelle Handelssignale
        signal_col_id = f"{StrategyLibEnum.BREAKOUT.name.lower()}_signal"
        
        df[signal_col_id] = SignalEnum.HOLD.value  # Standardmäßig auf HOLD setzen
        
        # BUY Signal: Upward Breakout (+1)
        df.loc[df['breakout_up'], signal_col_id] = SignalEnum.BUY.value

        # SELL Signal: Downward Breakout (-1)
        df.loc[df['breakout_down'], signal_col_id] = SignalEnum.SELL.value

        return df