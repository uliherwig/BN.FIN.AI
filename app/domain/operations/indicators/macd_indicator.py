import pandas as pd
import talib as ta
from app.domain import *

class MACDIndicator(BaseIndicator):
    """MACD Indicator"""
    
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:

        params = MacdModel.model_validate_json(param_str)
        prices = df["C"].to_numpy()
        signal_col_id = f"{StrategyLibEnum.MACD.name.lower()}_signal"

        df['macd'], df['signal'], df['histogram'] = ta.MACD(
            prices,
            fastperiod=params.fast,
            slowperiod=params.slow,
            signalperiod=params.signal
        )
        # Signale generieren (wie oben)
        df[signal_col_id] = SignalEnum.HOLD.value
        df.loc[df['macd'] > df['signal'], signal_col_id] = SignalEnum.BUY.value
        df.loc[df['macd'] < df['signal'], signal_col_id] = SignalEnum.SELL.value

        return df