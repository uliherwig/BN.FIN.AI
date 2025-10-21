from datetime import datetime
import pandas as pd
import json
from fastapi import APIRouter, BackgroundTasks, Depends
from app.domain.ai_trader import AITrader
from app.domain.models.strategy_settings_model import StrategySettingsModel
from app.domain.services.strategy_test_service import plot_strategy
from app.domain.services.yahoo_servce import download_stock_data
from app.domain.strategy_manager import StrategyManager
from app.infrastructure import alpaca_service
from app.infrastructure.alpaca_service import read_quotes
from app.domain.services.tf_learn_service import learn_from_alpaca_data, learn_from_yahoo_data  
from app.domain.services.tf_trading_service import start_live_trading 
from app.infrastructure.redis_service import RedisService, get_redis_service

from typing import Any

from app.domain.services.strategy_service import optimize_strategy, test_single_strategy, run_ai_strategy_analysis, train_lgb_model

router = APIRouter()

@router.get("/yahoo-data-download/{ticker}/{start_date}/{end_date}/{interval}")
async def download_yahoo_data(ticker: str, start_date: datetime, end_date: datetime, interval: str):
    download_stock_data(ticker, start_date, end_date, interval)
    return {"message": "Yahoo data download started"}

@router.get("/get/{key}")
def get_value(key: str, redis_service: RedisService = Depends(get_redis_service)):
    value = redis_service.get_value(key)
    return {"key": key, "value": value}

@router.get("/set/{key}/{value}")
def set_key_value(key: str, value: str, redis_service: RedisService = Depends(get_redis_service)):
    redis_service.set_value(key, value)
    return {"key": key, "value": value}

@router.get("/read-quotes/{asset}")
async def read_quotes_endpoint(asset: str) -> Any:
    print(f"Reading quotes for asset: {asset}")
    quotes = read_quotes(asset)
    return {"asset": asset, "quotes": quotes}

@router.post("/test-strategy")
async def test_strategy_endpoint(strategy_settings_dict: dict, background_tasks: BackgroundTasks):  
    """
    Expected JSON format:
    {
      "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "strategy_type": "BREAKOUT",
      "asset": "SPY",
      "quantity": 1,
      "take_profit_pct": 0.003,
      "stop_loss_pct": 0.002,
      "close_positions_eod": "True",
      "start_date": "2023-01-01",
      "end_date": "2023-12-31",
      "strategy_params": "{\"breakout_period\": 60}"
    }
    """
    strategy_settings = StrategySettingsModel(**strategy_settings_dict)
    background_tasks.add_task(test_single_strategy, strategy_settings)
    return {"status": "test_strategy started"}

@router.post("/ai-strategy")
async def ai_strategy_endpoint(strategy_settings_dict: dict, background_tasks: BackgroundTasks):  
    """
    Expected JSON format:
    {
      "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "strategy_type": "BREAKOUT",
      "asset": "SPY",
      "quantity": 1,
      "take_profit_pct": 0.003,
      "stop_loss_pct": 0.002,
      "close_positions_eod": "True",
      "start_date": "2023-01-01",
      "end_date": "2023-12-31",
      "strategy_params": "{\"breakout_period\": 60}"
    }
    """
    strategy_settings = StrategySettingsModel(**strategy_settings_dict)
    
    background_tasks.add_task(run_ai_strategy_analysis, strategy_settings)
    return {"status": "ai_strategy started"}

@router.post("/train-lgbm")
async def train_lgbm_endpoint(strategy_settings_dict: dict, background_tasks: BackgroundTasks):  
 
    strategy_settings = StrategySettingsModel(**strategy_settings_dict)
    
    background_tasks.add_task(train_lgb_model, strategy_settings)
    return {"status": "training started"}

@router.post("/optimize-strategy")
async def optimization_endpoint(strategy_settings_dict: dict, background_tasks: BackgroundTasks):
    """
        Expected JSON format:
        {
        "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "strategy_type": "BREAKOUT",
        "asset": "SPY",
        "quantity": 1,
        "take_profit_pct": 0.003,
        "stop_loss_pct": 0.002,
        "close_positions_eod": "True",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "strategy_params": "{\"breakout_period\": 10}"
        }
    """    
    
    strategy_settings = StrategySettingsModel(**strategy_settings_dict)
    
    background_tasks.add_task(optimize_strategy, strategy_settings)
    return {"status": "optimize_strategy started"}

@router.post("/learn-from-yahoo-data")
async def learn_from_yahoo_data_endpoint(ticker: str,  background_tasks: BackgroundTasks):
    
    aiTrader = AITrader(ticker)
    background_tasks.add_task(aiTrader.learn_from_yahoo_data)
    return {"status": "started"}

@router.post("/learn-from-alpaca-data")
async def learn_from_alpaca_data_endpoint(param: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(learn_from_alpaca_data, param)
    return {"status": "started"}

@router.post("/execute-trading-model")
async def execute_trading_model_endpoint(param: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_live_trading, param)
    return {"status": "started"}

@router.post("/test_chart")
async def test_chart_endpoint(param: str, start: str, end: str):
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    await plot_strategy(start_dt, end_dt)
    return {"status": "started"}
