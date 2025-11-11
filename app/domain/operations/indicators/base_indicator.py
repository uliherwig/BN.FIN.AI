from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseIndicator(ABC):
    """Abstract Base Class for all indicators"""
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        """Calculates the indicators and signals for the DataFrame."""
        pass
    
    @abstractmethod
    def create_features(self, df: pd.DataFrame, param_str: str) -> pd.DataFrame:
        """Creates additional features for the DataFrame based on the indicators."""
        pass