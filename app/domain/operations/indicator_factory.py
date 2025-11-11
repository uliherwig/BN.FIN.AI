# app/domain/indicators/factory.py
from typing import Any, Dict, Type, List
import pandas as pd
from app.domain import *
from app.domain.operations.indicators.base_indicator import BaseIndicator
from app.domain.operations.indicators.sma_indicator import SmaIndicator
from app.domain.operations.indicators.ema_indicator import EmaIndicator
from app.domain.operations.indicators.rsi_indicator import RsiIndicator
from app.domain.operations.indicators.donchian_indicator import DonchianIndicator
from app.domain.operations.indicators.breakout_indicator import BreakoutIndicator
from app.domain.operations.indicators.macd_indicator import MACDIndicator
from app.domain.operations.indicators.tema_indicator import TemaIndicator
from app.domain.operations.indicators.wma_indicator import WmaIndicator
from app.domain.operations.indicators.vola_indicator import VolatilityIndicator
from app.domain.operations.indicators.atr_indicator import AtrIndicator
from app.domain.models.strategies.indicator_model import IndicatorModel
from app.domain.operations.indicators.bbands_indicator import BbandsIndicator

class IndicatorFactory:
    def __init__(self):
        self.available_indicator_types = [
            IndicatorEnum.SMA,
            IndicatorEnum.EMA,
            IndicatorEnum.WMA,
            IndicatorEnum.TEMA,
            IndicatorEnum.MACD,
            IndicatorEnum.RSI,
            IndicatorEnum.DONCHIAN,
            IndicatorEnum.BREAKOUT,
            IndicatorEnum.VOLA,
            IndicatorEnum.ATR,
            IndicatorEnum.BBANDS,
        ]

    @staticmethod
    def create_indicator(strategy_type: IndicatorEnum) -> BaseIndicator:
        match strategy_type:
            case IndicatorEnum.SMA:
                return SmaIndicator()
            case IndicatorEnum.EMA:
                return EmaIndicator()
            case IndicatorEnum.WMA:
                return WmaIndicator()
            case IndicatorEnum.TEMA:
                return TemaIndicator()
            case IndicatorEnum.MACD:
                return MACDIndicator()
            case IndicatorEnum.RSI:
                return RsiIndicator()
            case IndicatorEnum.DONCHIAN:
                return DonchianIndicator()
            case IndicatorEnum.BREAKOUT:
                return BreakoutIndicator()
            case IndicatorEnum.VOLA:
                return VolatilityIndicator()
            case IndicatorEnum.ATR:
                return AtrIndicator()
            case IndicatorEnum.BBANDS:
                return BbandsIndicator()
            case _:
                raise ValueError(f"Unsupported strategy type: {strategy_type}")

    @staticmethod
    def extend_dataframe_with_indicators(
        df: pd.DataFrame,
        indicator_models: List[IndicatorModel]
    ) -> pd.DataFrame:
        for indicator_model in indicator_models:
            if isinstance(indicator_model, IndicatorModel):
                indicator = IndicatorFactory.create_indicator(indicator_model.strategyType)
                df = indicator.create_features(df, indicator_model.params)
        return df

    @staticmethod
    def get_indicator_list_by_params(params: Dict[str, Any]) -> List[IndicatorModel]:
        indicator_map = {
            "SMA": IndicatorEnum.SMA,
            "EMA": IndicatorEnum.EMA,
            "WMA": IndicatorEnum.WMA,
            "TEMA": IndicatorEnum.TEMA,
            "MACD": IndicatorEnum.MACD,
            "RSI": IndicatorEnum.RSI,
            "DONCHIAN": IndicatorEnum.DONCHIAN,
            "BREAKOUT": IndicatorEnum.BREAKOUT,
            "VOLA": IndicatorEnum.VOLA,
            "ATR": IndicatorEnum.ATR,
            "BBANDS": IndicatorEnum.BBANDS,
        }
        result = []
        grouped = {}
        for k, v in params.items():
            prefix = k.split('_')[0]
            grouped.setdefault(prefix, {})[k] = v
        for prefix, param_dict in grouped.items():
            enum = indicator_map.get(prefix)
            if enum:
                result.append(IndicatorModel(strategyType=enum, params=json.dumps(param_dict)))
        return result

