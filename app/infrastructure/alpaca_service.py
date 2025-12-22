import json
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

def load_stock_data_from_redis(asset: str, period: str = "1h") -> pd.DataFrame:
    redis_service = get_redis_service()
    
    start_date = datetime(2024, 1, 1).date()
    end_date = datetime.now().date()

    all_bars = []
    current_date = start_date
    while current_date < end_date:
        day = current_date.strftime('%Y-%m-%d')

        redis_key = f"bars:{asset}:{day}"

        bars_json = redis_service.get_value(redis_key)
        if bars_json is not None:
            b = bars_json.decode("utf-8")
            bars = json.loads(b)
            
            if(period == "1d" and len(bars) > 0):
                daily_bars = {}
                daily_bars['Date'] = day
                daily_bars['Open'] = bars[0]['O']
                daily_bars['High'] = max(bar['H'] for bar in bars)
                daily_bars['Low'] = min(bar['L'] for bar in bars)
                daily_bars['Close'] = bars[-1]['C']
                daily_bars['Volume'] = sum(bar['V'] for bar in bars)
                all_bars.append(daily_bars)
                
            if (period == "1h" and len(bars) > 0):

                # split bars into hourly segments
                current_hour = 0
                while current_hour < 24:

                    # get bars for the current hour
                    hour_bars_list = [bar for bar in bars if datetime.fromisoformat(bar['T']).hour == current_hour]
                    if len(hour_bars_list) == 0:
                        current_hour += 1
                        continue
                
                    hour_bars = {}
                    hour_bars['Date'] = hour_bars_list[0]['T']
                    hour_bars['Open'] = hour_bars_list[0]['O']
                    hour_bars['High'] = max(bar['H'] for bar in hour_bars_list)
                    hour_bars['Low'] = min(bar['L'] for bar in hour_bars_list)
                    hour_bars['Close'] = hour_bars_list[-1]['C']
                    hour_bars['Volume'] = sum(bar['V'] for bar in hour_bars_list)
                    all_bars.append(hour_bars)
                    
                    current_hour += 1
                    
            if (period == "1m" and len(bars) > 0):
                current_hour = 0
                
                for bar in bars:
                    bar_time = datetime.fromisoformat(bar['T'])       

                    min_bars = {}
                    min_bars['Date'] = bar['T']
                    min_bars['Open'] = bar['O']
                    min_bars['High'] = bar['H']
                    min_bars['Low'] = bar['L']
                    min_bars['Close'] = bar['C']
                    min_bars['Volume'] = bar['V']
                    all_bars.append(min_bars)

                  
        else:
            bars = []

        current_date += timedelta(days=1)

    df = pd.DataFrame(all_bars)
    
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['Date'] = pd.to_datetime(df['Date'])


    
    # Replace period formatting with match block
    match period:
        case "1d":         
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df['DT'] = df['Date']
            df.set_index('Date', inplace=True)
        case "1h":
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        case "1m":
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        case _:
            pass
 
    
    print(f"DataTypes {df.dtypes}")
    
    print(f"Loaded {len(df)} rows of {period} data for {asset} from Redis")   

    return df
