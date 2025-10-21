import pandas as pd
import talib as ta
from app.domain import *
from app.domain.models.strategies.atr_model import AtrModel

class ATRIndicator(BaseIndicator):
    """Average True Range (ATR) Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        params = AtrModel.model_validate_json(param_str)
        high = df["H"].to_numpy()
        low = df["L"].to_numpy()
        close = df["C"].to_numpy()
        signal_col_id = f"{StrategyLibEnum.ATR.name.lower()}_signal"

        df['atr'] = ta.ATR(high, low, close, timeperiod=params.period).round(3)

        # ATR-Signale generieren
        df[signal_col_id] = SignalEnum.HOLD.value  # Standardmäßig auf HOLD setzen
        df.loc[df['atr'] > params.threshold, signal_col_id] = SignalEnum.BUY.value
        df.loc[df['atr'] < params.threshold, signal_col_id] = SignalEnum.SELL.value

        return df
