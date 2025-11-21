import pandas as pd
import talib as ta
from app.domain import *


class RocIndicator(BaseIndicator):
    """Return Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:

        roc_model = RocModel.model_validate_json(param_str)
        roc1 = ta.ROC(df["Close"].to_numpy(), roc_model.ROC_period)
        df[f'ROC_{roc_model.ROC_period}'] = roc1.round(3)


        
        prices = df["C"].to_numpy()
        signal_col_id = f"{IndicatorEnum.ROC.name.lower()}_signal"
   
        df[signal_col_id] = SignalEnum.HOLD.value
        df.loc[df[col_id] > roc_model.ROC_threshold,
               signal_col_id] = SignalEnum.BUY.value
        df.loc[df[col_id] < -roc_model.ROC_threshold,
               signal_col_id] = SignalEnum.SELL.value

        return df

    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        
        roc_model = RocModel.model_validate_json(param_str)        
        roc1 = ta.ROC(df["Close"].to_numpy(),roc_model.ROC_period)

        # df[f'PCT_{roc_model.ROC_period}'] = df['Close'].pct_change(roc_model.ROC_period)
        df[f'ROC_1'] = roc1.round(3)
        return df
