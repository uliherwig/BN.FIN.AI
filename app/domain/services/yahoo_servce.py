import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from typing import Any
from datetime import datetime, timedelta, timezone
from app.domain.models.strategies.breakout_model import BreakoutModel
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
    stock_data.to_csv(f'stock_data_{ticker}_{start_date}_{end_date}_{interval}.csv')

    # Display the first few rows
    print(stock_data.head())