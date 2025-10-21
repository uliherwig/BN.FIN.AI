import numpy as np
import pandas as pd
import talib as ta

from app.domain.models.enums import StrategyLibEnum
from app.domain.operations.indicator_factory import IndicatorFactory

class DataUtils:
    @staticmethod
    def extract_ticker_data(data, ticker):
        """
        Extracts data for a specific ticker and converts it to the required format.

        Parameters:
            data (pd.DataFrame): The input DataFrame with columns 
                                 ['date', 'close', 'high', 'low', 'open', 'volume', 'tic', 'day'].
            ticker (str): The ticker symbol to filter.

        Returns:
            pd.DataFrame: A DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
        """
        # Filter data for the specified ticker
        ticker_data = data[data['tic'] == ticker]

        # Rename columns to match the required format
        formatted_data = ticker_data.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Select only the required columns
        formatted_data = formatted_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Sort by Date
        formatted_data = formatted_data.sort_values(by='Date')

        return formatted_data   

    
    @staticmethod
    def sharpe_ratio(returns, periods_per_year=252*6.5*60):  # placeholder for intraday: very large periods_per_year
        # uses mean/std of returns; annualizes by sqrt(periods_per_year)
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        if sigma == 0:
            return 0.0
        return (mu / sigma) * np.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(cum_returns):
        peak = cum_returns.cummax()
        dd = (cum_returns - peak) / peak
        return dd.min()
    
    @staticmethod
    def add_indicator(df: pd.DataFrame, strategy_type: StrategyLibEnum, params: str) -> pd.DataFrame:
        indicator = IndicatorFactory.create_indicator(strategy_type)
        df = indicator.calculate_signals(df, params)
        return df

    @staticmethod
    def add_donchian(df: pd.DataFrame, window=20) -> pd.DataFrame:
        df["donchian_upper"] = df["H"].rolling(window).max().round(3)
        df["donchian_lower"] = df["L"].rolling(window).min().round(3)
        df["donchian_mid"] = ((df["donchian_upper"] + df["donchian_lower"]) / 2).round(3)
        df["donchian_width"] = (df["donchian_upper"] - df["donchian_lower"]).round(3)
        df["donchian_breakout_strength"] = ((df["C"] - df["donchian_mid"]) / df["donchian_width"]).round(3)
        return df

    # ------------------------------------------------------------
    @staticmethod
    def add_macd(df: pd.DataFrame, fast=8, slow=17, signal=9) -> pd.DataFrame:
        df["ema_fast"] = df["C"].ewm(span=fast, adjust=False).mean().round(3)
        df["ema_slow"] = df["C"].ewm(span=slow, adjust=False).mean().round(3)
        df["macd"] = (df["ema_fast"] - df["ema_slow"]).round(3)
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean().round(3)
        df["macd_diff"] = (df["macd"] - df["macd_signal"]).round(3)
        df["macd_slope"] = df["macd"].diff().round(3)
        df["macd_cross"] = (np.sign(df["macd"] - df["macd_signal"])).round(3)
        df["macd_cross_change"] = (df["macd_cross"].diff().fillna(0)).round(3)
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period=14) -> pd.DataFrame:      
        
        high = df["H"].to_numpy()
        low = df["L"].to_numpy()
        close = df["C"].to_numpy()

        df["atr"] = np.round(ta.ATR(high, low, close, timeperiod=period), 3)
        return df
    
    @staticmethod
    def add_sma(df: pd.DataFrame, short = 20, long = 50) -> pd.DataFrame:
        df[f"sma_{short}"] = df["C"].rolling(short).mean().round(3)
        df[f"sma_{long}"] = df["C"].rolling(long).mean().round(3)

        df["sma_diff"] = (df[f"sma_{short}"] - df[f"sma_{long}"]).round(3) 
        df["sma_cross"] = (np.where(df["sma_diff"] > 0, 1, -1)).round(3)
        df["sma_ratio"] = (df[f"sma_{short}"] / df[f"sma_{long}"] - 1).round(3)
        df["sma_cross_change"] = df["sma_cross"].diff().fillna(0)
        df["sma_diff_slope"] = df["sma_diff"].diff()

        return df

    # ------------------------------------------------------------
    @staticmethod
    def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
        delta = pd.to_numeric(df["C"].diff(), errors='coerce')
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        rs = avg_gain / avg_loss
        df[f"rsi_{period}"] = ( 100 - (100 / (1 + rs)) ).round(3)
        df["rsi_overbought"] = (df[f"rsi_{period}"] > 70).astype(int)
        df["rsi_oversold"] = (df[f"rsi_{period}"] < 30).astype(int)
        return df
    
    @staticmethod
    def add_future_return(df: pd.DataFrame, horizon=12) -> pd.DataFrame:
        """
        Calculate future returns but prevent crossing to next trading day (intraday only).
        Sets future_return to NaN if the target timestamp is on a different day.
        """
        # Ensure DT column exists and is datetime
        if 'DT' not in df.columns:
            df['DT'] = pd.to_datetime(df['T'])
        
        # Extract date for each row
        df['_date'] = df['DT'].dt.date
        
        # Calculate future return
        df[f"future_return_{horizon}"] = df["C"].shift(-horizon) / df["C"] - 1
        
        # Create a shifted date column to compare
        df['_future_date'] = df['_date'].shift(-horizon)
        
        # Set future_return to NaN where dates don't match (crosses day boundary)
        mask = df['_date'] != df['_future_date']
        df.loc[mask, f"future_return_{horizon}"] = np.nan
        
        # Clean up temporary columns
        df.drop(['_date', '_future_date'], axis=1, inplace=True)
        
        # Round the result
        df[f"future_return_{horizon}"] = df[f"future_return_{horizon}"].round(3)
        
        return df
    



