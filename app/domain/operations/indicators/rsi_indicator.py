import json
import numpy as np
import pandas as pd
import talib as ta
from typing import Any
from datetime import datetime, timedelta, timezone

from app.domain.operations.indicators.base_indicator import BaseIndicator

from app.domain import *

class RsiIndicator(BaseIndicator):
    """Relative Strength Index Indicator"""

    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        """
        Berechnet den RSI-Indikator und generiert Kauf-/Verkaufssignale.

        Args:
            df: DataFrame mit 'C' (Close-Preisen).
           

        Returns:
            DataFrame mit zusätzlichen Spalten:
            - 'rsi_<period>': RSI-Werte.
            - 'rsi_signal': 1 (Kauf bei Überverkauft), -1 (Verkauf bei Überkauft), 0 (Hold).
        """
        # Parameter parsen (Beispiel mit Pydantic-Modell, analog zu deinem MacdModel)
        params = RsiModel.model_validate_json(param_str)  # Annahme: RsiModel ähnlich wie MacdModel

        # RSI berechnen
        rsi_col = f"rsi_{params.period}"
        df[rsi_col] = ta.RSI(
            df["C"].to_numpy(),
            timeperiod=params.period
        )

        # Signal-Spaltenname
        signal_col_id = f"{StrategyLibEnum.RSI.name.lower()}_signal"

        # Signale generieren:
        # 1 = Kauf (RSI unter 'oversold' und steigt)
        # -1 = Verkauf (RSI über 'overbought' und fällt)
        df[signal_col_id] = 0  # Default: Hold

        # Kaufsignal: RSI kreuzt von unten nach oben aus dem überverkauften Bereich
        df.loc[
            (df[rsi_col].shift(1) < params.oversold) &
            (df[rsi_col] > params.oversold),
            signal_col_id
        ] = 1

        # Verkaufssignal: RSI kreuzt von oben nach unten aus dem überkauften Bereich
        df.loc[
            (df[rsi_col].shift(1) > params.overbought) &
            (df[rsi_col] < params.overbought),
            signal_col_id
        ] = -1

        return df


