# app/domain/indicators/__init__.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any

from app.domain import *

class BaseIndicator(ABC):
    """Abstract Base Class für alle Indikatoren"""
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        """Berechnet die Indikatoren und Signale für den DataFrame."""
        pass