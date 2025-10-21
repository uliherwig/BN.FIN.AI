# app/domain/indicators/factory.py
from typing import Any, Dict, Type
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


class IndicatorFactory:
    
    @staticmethod
    def create_indicator(strategy_type: StrategyLibEnum) -> BaseIndicator:
        
        # test = StrategyLibEnum.SMA  # Removed unused or erroneous line
        
        match strategy_type:
            case StrategyLibEnum.SMA:
                return SmaIndicator()
            case StrategyLibEnum.EMA:
                return EmaIndicator()
            case StrategyLibEnum.WMA:
                return WmaIndicator()
            case StrategyLibEnum.TEMA:
                return TemaIndicator()   
            case StrategyLibEnum.MACD:
                return MACDIndicator()
            case StrategyLibEnum.RSI:
                return RsiIndicator()
            case StrategyLibEnum.DONCHIAN:
                return DonchianIndicator()
            case StrategyLibEnum.BREAKOUT:
                return BreakoutIndicator()
            case _:
                raise ValueError(f"Unsupported strategy type: {strategy_type}")
