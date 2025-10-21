import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Any
from datetime import datetime, timedelta, timezone
from app.infrastructure.redis_service import RedisService, get_redis_service
from app.infrastructure import alpaca_service
import time
import matplotlib.pyplot as plt
from app.models.schemas import Quote, Position  



def start_live_trading(asset: str) -> None:
    model = load_model("alpaca_spy_model.h5")
    if model is None:
        raise ValueError("Failed to load model: alpaca_spy_model.h5")
    scaler = joblib.load("scaler_rsi.pkl")   
    
    df = alpaca_service.read_quotes(asset)
    df["Close"] = ((df["bid"] + df["ask"]) / 2).round(3)
    df['Date'] = pd.to_datetime(df['stamp'])
    df = df.dropna() # remove lines with NaN values
    numberOfLines = len(df)
    i = 0 
    while i < numberOfLines:
        if i > 30:
            df['SMA'] = df['Close'].rolling(window=10).mean()
            df['RSI'] = 100 - (100 / (1 + (df['Close'].diff(1).clip(lower=0).rolling(14).mean() /
                                    df['Close'].diff(1).clip(upper=0).abs().rolling(14).mean())))
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            features = ['Close', 'SMA', 'RSI', 'MACD']
            df[features] = scaler.transform(df[features])  # gleiche Skalierung wie beim Training

            last_sequence = df[features].iloc[-30:].values  # shape = (30, 4)
            X_live = np.expand_dims(last_sequence, axis=0)  # shape = (1, 30, 4)
            
            predictions = model.predict(X_live, verbose=0 )
            predicted_signal = np.argmax(predictions, axis=1) - 1
            

            if(predicted_signal[0] == 1):
                print(f"Buy signal for {asset} at index {i}")
            if(predicted_signal == -1):
                print(f"Sell signal for {asset} at index {i}")
           
            
        
        
        
        
        
        
        i = i + 1
    
   
    print(f"Live trading test with model: alpaca_spy_model.h5 completed")


    
   
    
    
   
    
 



    


    
