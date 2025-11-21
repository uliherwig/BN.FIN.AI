import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from typing import Any
from datetime import datetime, timedelta, timezone
from app.domain.operations.indicators.ema_indicator import EmaIndicator
from app.domain.operations.indicators.tema_indicator import TemaIndicator
from app.domain.operations.indicators.wma_indicator import WmaIndicator
from app.infrastructure.redis_service import RedisService, get_redis_service
from app.infrastructure import alpaca_service
from sklearn.model_selection import ParameterGrid
from celery import Celery
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from app.domain import *
from app.domain.strategy_manager import StrategyManager
import json
import os


class YahooService():

    @staticmethod
    def download_stock_data(ticker: str, start_date: datetime, end_date: datetime,  interval: str = "1d") -> None:
        # Periods
        # You can specify the overall time range using the period parameter:

        # "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"

        # Intervals
        # You can specify the granularity of the data using the interval parameter:

        # "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo" 

        # Fetch the data
        stock_data: pd.DataFrame | None = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if(stock_data is None or stock_data.empty):
            print("No data fetched for the given ticker and date range.")
            return

        # Save the data to a CSV file
        # Convert datetime to string format yyyy-mm-dd
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Save the data to a CSV file
        stock_data.to_csv(f'csv/stock_data_{ticker}_{start_str}_{end_str}_{interval}.csv')


        # Display the first few rows
        print(stock_data.head())
    
    @staticmethod
    def load_yahoo_stock_data(ticker: str) -> pd.DataFrame:
        start_str = "2010-01-01"
        end_str = "2025-11-02"
        interval = "1d"
        filename = f'csv/stock_data_{ticker}_{start_str}_{end_str}_{interval}.csv'
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            data = data.rename(columns={'Price': 'Date'})
            # remove 2nd and 3rd line
            data = data.drop(index=0)
            data = data.drop(index=1)
            # rename col Price to Date
            # list columns
            # print(f"Columns in data: {data.columns.tolist()}")
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            # # Transform Date column from string to datetime
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
            data['DT'] = data['Date']
            data.set_index('Date', inplace=True)
            return data
        else:
            raise FileNotFoundError("Data file not found.")


