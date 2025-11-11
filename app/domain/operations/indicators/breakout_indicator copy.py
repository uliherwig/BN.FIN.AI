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
        volume = df['V'].to_numpy()

        # TA-Lib: Berechne Highest High und Lowest Low der letzten N Perioden
        df['breakout_high'] = ta.MAX(high, timeperiod=params.lookback_period)
        df['breakout_low'] = ta.MIN(low, timeperiod=params.lookback_period)
        
        # TA-Lib: Volumen-Filter für stärkere Signale
        df['volume_avg'] = ta.SMA(volume, timeperiod=params.lookback_period)
        
        # TA-Lib: Rate of Change für Momentum-Filter
        df['price_roc'] = ta.ROC(close, timeperiod=1)  # 1-Perioden Preisänderung in %
        
        # Shift um 1, da wir nur auf vorherige Highs/Lows reagieren wollen
        prev_high = df['breakout_high'].shift(1)
        prev_low = df['breakout_low'].shift(1)
        
        # Breakout-Bedingungen
        bullish_breakout = close > prev_high
        bearish_breakout = close < prev_low
        
        # Volumen-Filter: Überdurchschnittliches Volumen
        volume_confirmation = volume > (df['volume_avg'] * params.volume_threshold)
        
        # Momentum-Filter: Mindest-Preisänderung in Prozent
        significant_move = np.abs(df['price_roc']) > (params.min_breakout_pct * 100)
        
        # Kombiniere alle Filter
        df['breakout_up'] = bullish_breakout & volume_confirmation & significant_move
        df['breakout_down'] = bearish_breakout & volume_confirmation & significant_move
        
        # Zusätzliche TA-Lib Indikatoren für bessere Signalqualität
        df['atr'] = ta.ATR(high, low, close, timeperiod=14)  # Average True Range
        df['volatility_filter'] = df['atr'] > df['atr'].rolling(20).mean()  # Hohe Volatilität
        
        # Erstelle Handelssignale
        signal_col_id = f"{IndicatorEnum.BREAKOUT.name.lower()}_signal"
        
        df[signal_col_id] = SignalEnum.HOLD.value
        
        # BUY Signal: Upward Breakout mit Confirmation
        buy_condition = (
            df['breakout_up'] & 
            df['volatility_filter']  # Zusätzlicher Volatilitäts-Filter
        )
        df.loc[buy_condition, signal_col_id] = SignalEnum.BUY.value
        
        # SELL Signal: Downward Breakout mit Confirmation
        sell_condition = (
            df['breakout_down'] & 
            df['volatility_filter']  # Zusätzlicher Volatilitäts-Filter
        )
        df.loc[sell_condition, signal_col_id] = SignalEnum.SELL.value
        
        return df