import pandas as pd
import talib as ta
from app.domain import *

class MACDIndicator(BaseIndicator):
    
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:

        params = MacdModel.model_validate_json(param_str)
        prices = df["C"].to_numpy()
        signal_col_id = f"{IndicatorEnum.MACD.name.lower()}_signal"

        df['MACD'], df['signal'], df['MACD_histogram'] = ta.MACD(
            prices,
            fastperiod=params.MACD_fast,
            slowperiod=params.MACD_slow,
            signalperiod=params.MACD_signal
        )
        # Signale generieren (wie oben)
        df[signal_col_id] = SignalEnum.HOLD.value
        df.loc[df['MACD'] > df['signal'], signal_col_id] = SignalEnum.BUY.value
        df.loc[df['MACD'] < df['signal'], signal_col_id] = SignalEnum.SELL.value

        return df
    
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:

        params = MacdModel.model_validate_json(param_str)
        prices = df["Close"].to_numpy()
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(
            prices,
            fastperiod=params.MACD_fast,
            slowperiod=params.MACD_slow,
            signalperiod=params.MACD_signal
        )

        return df