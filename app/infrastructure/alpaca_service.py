import json
import numpy as np
import pandas as pd
from typing import Any
from datetime import datetime, timedelta, timezone
from app.infrastructure.redis_service import RedisService, get_redis_service

async def read_quotes(asset: str) -> Any:
    redis_service = get_redis_service()

    start_date = datetime(2024, 1, 1).date()
    end_date = datetime.now().date()

    all_quotes = []
    current_date = start_date
    while current_date < end_date:
        day = current_date.strftime('%Y-%m-%d')

        redis_key = f"quotes:{asset}:{day}"
        
        # quotes:SPY:2024-08-09
        quotes_json = await redis_service.get_value(redis_key)
        if quotes_json is not None:
            q = quotes_json.decode("utf-8")
            quotes = json.loads(q)
            all_quotes.extend(quotes)
        else:
            quotes = []

        # quotes = json.dumps(quotes_json)

        # Or you can process the date as needed
        current_date += timedelta(days=1)
        
    df = pd.DataFrame(all_quotes)
    df["Mid"] = ((df["BidPrice"] + df["AskPrice"]) / 2).round(3)
    
    print(f"DataTypes {df.dtypes}")
    print(f"Quotes for {asset}: {len(df)} entries")
    return df  

def read_bars(asset: str, start_date: datetime = datetime(2024, 1, 1), end_date: datetime = datetime.now()) -> Any:
    redis_service = get_redis_service()
 
    all_bars = []
    current_date = start_date
    while current_date < end_date:
        day = current_date.strftime('%Y-%m-%d')

        redis_key = f"bars:{asset}:{day}"

        bars_json =  redis_service.get_value(redis_key)
        if bars_json is not None:
            b = bars_json.decode("utf-8")
            bars = json.loads(b)
            all_bars.extend(bars)
        else:
            bars = []

        current_date += timedelta(days=1)

    df = pd.DataFrame(all_bars) 
    
    df["ask"] = df["C"] + 0.01
    df["bid"] = df["C"] - 0.01
    df["DT"] = pd.to_datetime(df["T"])

    return df
   
