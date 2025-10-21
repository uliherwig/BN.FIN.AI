import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
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
from app.domain.data_utils import DataUtils
import os

def learn_from_yahoo_data(ticker: str) -> None:
    start_time = time.time()
    print(f"Learn from daily data for {ticker}")
    
    if (os.path.exists(os.path.join('app/assets/', 'yahoo_data.csv'))):
        raw_data = pd.read_csv(os.path.join('app/assets/', 'yahoo_data.csv'))

  
    data = DataUtils.extract_ticker_data(raw_data, ticker)

    print("Extracted Data:")
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    #####################################################

    # 2. Berechnung technischer Indikatoren
    data['SMA'] = data['Close'].rolling(window=10).mean()  # Simple Moving Average


    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(14).mean() /
                                    data['Close'].diff(1).clip(upper=0).abs().rolling(14).mean())))

    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data = data.dropna()

    # 3. Labels erstellen (1 = Buy, -1 = Sell, 0 = Hold)
    data['Signal'] = 0

    data.loc[data['Close'].shift(-1) > data['Close'], 'Signal'] = 1
    data.loc[data['Close'].shift(-1) < data['Close'], 'Signal'] = -1
    print(data.head(10))

    # 4. Features und Labels vorbereiten
    features = ['Close', 'SMA', 'RSI', 'MACD']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    X = []
    y = []
    sequence_length = 30  # Länge des Eingabefensters

    for i in range(sequence_length, len(data)):
        X.append(data[features].iloc[i-sequence_length:i].values)
        y.append(data['Signal'].iloc[i])

    X = np.array(X)
    y = np.array(y)
    y = keras.utils.to_categorical((y + 1))  # Umwandlung in One-Hot-Encoding

    print(data.head(10))
    # 5. Daten aufteilen in Training und Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. LSTM-Modell erstellen
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features))),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='softmax')  # 3 Klassen: Buy, Sell, Hold
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




    # 7. Modell trainieren
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32
    )

    # 8. Ergebnisse evaluieren
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # 9. Predictions auf neuen Daten
    predictions = model.predict(X_test)
    predicted_signals = np.argmax(predictions, axis=1) - 1  # Zurück zu [-1, 0, 1]
    print(predicted_signals[:10])
    model.save('yahoo_aapl_model.h5')
    joblib.dump(scaler, "scaler_rsi.pkl")
    
    
    

def learn_from_alpaca_data(asset: str) -> None:
    start_time = time.time()    
    
    # Deine Hintergrundlogik
    print(f"learn from alpaca data for  {asset}")
    
    df = alpaca_service.read_bars(asset, datetime(2024, 1, 1),datetime(2024, 12, 31))
    
    
    print(f"DataTypes {df.dtypes}")
    print(f"Quotes for {asset}: {len(df)} entries")
    
    df["Close"] = ((df["bid"] + df["ask"]) / 2).round(3)
    df['Date'] = pd.to_datetime(df['stamp'])

    df = label_signals(df)    
    
    # df.set_index('Date', inplace=True)
    
    
    # df to csv


    
    # print number of lines of df
    print(f"Quotes for {asset}: {len(df)} entries")

    # 2. Berechnung technischer Indikatoren
    df['SMA'] = df['Close'].rolling(window=10).mean()
    
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff(1).clip(lower=0).rolling(14).mean() /
                                df['Close'].diff(1).clip(upper=0).abs().rolling(14).mean())))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df = df.dropna() # rmove lines with NaN values

    # 3. Labels erstellen (1 = Buy, -1 = Sell, 0 = Hold)
    # df['Signal'] = 0
    # df.loc[df['Close'].shift(-1) > df['Close'], 'Signal'] = 1
    # df.loc[df['Close'].shift(-1) < df['Close'], 'Signal'] = -1
    

    
   

   
    
        # 4. Features und Labels vorbereiten
    features = ['Close', 'SMA', 'RSI', 'MACD']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    
   
    
    # plt.plot(df['Close'])
    # plt.scatter(df[df['Signal']==1].index, df[df['Signal']==1]['Close'], color='green', label='Buy')
    # plt.scatter(df[df['Signal']==-1].index, df[df['Signal']==-1]['Close'], color='red', label='Sell')
    # plt.legend()
    # plt.show()
    
  
    
    X = []
    y = []
    sequence_length = 30

    for i in range(sequence_length, len(df)):
        X.append(df[features].iloc[i-sequence_length:i].values)
        y.append(df['Signal'].iloc[i])

    X = np.array(X)
    y = np.array(y)
    y = keras.utils.to_categorical((y + 1))  # [-1,0,1] → [0,1,2]
  
    # 5. Daten aufteilen in Training und Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. LSTM-Modell erstellen
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features))),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])   


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Ready for training : {duration:.2f} seconds")

    # 7. Modell trainieren
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32
    )
    
    model.save('alpaca_spy_model.h5')
    joblib.dump(scaler, "scaler_rsi.pkl")

    # 8. Ergebnisse evaluieren
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration:.2f} seconds")

    # 9. Predictions auf neuen Daten
    predictions = model.predict(X_test)
    predicted_signals = np.argmax(predictions, axis=1) - 1
    print(predicted_signals[:10])
    
    # Index-Offset wegen sequence_length
    start_idx = sequence_length

    # Indexe der Testdaten im df
    _, X_test_idx = train_test_split(np.arange(start_idx, start_idx + len(y)), test_size=0.2, random_state=42)
    df.loc[X_test_idx, 'TestSignal'] = predicted_signals
    df.to_csv(f"{asset}_predictions.csv", index=False)

    
def label_signals(df, tp=0.015, sl=0.01, window=30):
    
    print(f"Label positions for {len(df)} entries")
    df = df.copy()
    df['Signal'] = 0
    df['Position'] = 0
 
    # maximmum close of the next 30 days
    df['Future_High'] = df['Close'].shift(-window).rolling(window=window).max()
    df['Future_Low'] = df['Close'].shift(-window).rolling(window=window).min()  
    
    position: Position = Position (
        asset="SPY",
        position=0,
        open_price=0.0,
        close_price=0.0
    )
   
    
    numberPositions = 0

    print(f"High Low added")
   
    i = 0
    while i < len(df) - window :
        
        if(i < window):
            i += 1
            continue
            
      
        entry_price = df['Close'].iloc[i]
        
        if(position.position == 1):
            if(entry_price >= position.close_price):
                position.position = 0
            
        if(position.position == -1):
            if(entry_price <= position.close_price):
                position.position = 0
               
        if position.position != 0:
            df.at[i, 'Position'] = position.position
            i += 1
            continue
        
        #test = df.at[i]
        
        h = df['Future_High'].iloc[i]
        l = df['Future_Low'].iloc[i]
        
        if pd.isna(h) or pd.isna(l):
            i += 1
            continue
        

        if (h > entry_price * (1 + tp) and l > entry_price * (1 - sl)):
            df.at[i, 'Signal'] = 1
            position.position = 1
            position.open_price = entry_price
            position.close_price = entry_price * (1 + tp)
            numberPositions += 1    
  
            
        if (l < entry_price * (1 - tp) and h < entry_price * (1 + sl)):
            df.at[i, 'Signal'] = -1
            position.position = -1
            position.open_price = entry_price
            position.close_price = entry_price * (1 - tp)
            numberPositions += 1
        
        
        df.at[i, 'Position'] = position.position
        i += 1

    df.to_csv(f"test.csv", index=False)
    print(f"Number of positions: {numberPositions}")
    return df
