import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
# import pandas_ta as ta
import tensorflow as tf
import keras
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Any
from datetime import datetime, timedelta, timezone
from app.domain.models.enums import StrategyLibEnum
from app.domain.operations.indicator_factory import IndicatorFactory
from app.infrastructure.redis_service import RedisService, get_redis_service
from app.infrastructure import alpaca_service
import time
import matplotlib.pyplot as plt
from app.models.schemas import Quote, Position  
from app.domain.data_utils import DataUtils
import os

class AITrader:
    def __init__(self, ticker: str = "AAPL"):
        self.ticker = ticker
        # self.model = keras.models.load_model('ai_models/yahoo_aapl_model.h5')
        # self.scaler = joblib.load('ai_models/yahoo_aapl_scaler.pkl')
        
    def learn_from_yahoo_data(self) -> None:
        start_time = time.time()
        
        data = self.load_data(self.ticker)   
      
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        # --- 2. Feature Engineering: SMA ---
        
        # Calculate SMA (Simple Moving Average) for Close price
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        
        # Difference between short and long SMA
        data['SMA_diff'] = data['SMA_10'] - data['SMA_20']
        
        # Drop rows with NaN values (due to rolling windows)
        data = data.dropna()

        # --- 3. Target Variable: Next-Day Close Return ---
        
        data['target'] = data['Close'].pct_change().shift(-1)
        
        # Drop the last row (no target for the last day)
        data = data.iloc[:-1]

        # --- 4. Split Features and Target ---
        
        features = ['SMA_diff']
        X = data[features]
        y = data['target']
        
        # --- 5. Walk-Forward Validation ---
        def walk_forward_validation(X, y, train_years=5, test_years=1, start_year=2010):
            results = []
            for i in range(start_year + train_years, 2025):
                train_start = f'{i - train_years}-01-01'
                train_end = f'{i - 1}-12-31'
                test_start = f'{i}-01-01'
                test_end = f'{i}-12-31'

                # Split data
                X_train, y_train = X.loc[train_start:train_end], y.loc[train_start:train_end]
                X_test, y_test = X.loc[test_start:test_end], y.loc[test_start:test_end]

                # Skip if no test data
                if len(X_test) == 0:
                    continue

                # --- 6. Train LightGBM Model ---
                model = lgb.LGBMRegressor(random_state=42)
                model.fit(X_train, y_train)

                # --- 7. Evaluate on Test Set ---
                y_pred = model.predict(X_test)
                
         

                y_true = np.asarray(y_test).ravel()

                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                results.append({'year': i, 'MAE': mae, 'R2': r2})

            return pd.DataFrame(results)
        
      
        # --- 8. Run Walk-Forward Validation ---
        results_df = walk_forward_validation(X, y)

        # --- 9. Print Results ---
        print("Walk-Forward Validation Results:")
        print(results_df)
        print(f"\nAverage MAE: {results_df['MAE'].mean():.6f}")
        print(f"Average R2: {results_df['R2'].mean():.4f}")

        # --- 10. Feature Importance ---
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X, y)
        print("\nFeature Importance:")
        print(dict(zip(features, model.feature_importances_)))
        
        
        # BACKTESTING FINAL MODEL
        
        # --- 1. Load Data and Generate Predictions ---
        # Assume `data` is your DataFrame with features and `model` is your trained LightGBM model
        data['predicted_return'] = model.predict(data[features])

        # --- 2. Generate Trading Signals ---
        # Example: Go long if predicted return > 0, else stay flat
        data['signal'] = np.where(data['predicted_return'] > 0, 1, 0)

        # --- 3. Backtesting Logic ---
        def backtest(data, tp=0.005, sl=-0.004):
            # Initialize columns for tracking trades
            data['position'] = 0  # 0 = flat, 1 = long
            data['entry_price'] = np.nan
            data['exit_price'] = np.nan
            data['return'] = 0.0

            position_open = False
            entry_price = 0.0

            for i in range(len(data)):
                # Close existing position if TP/SL is hit
                if position_open:
                    current_price = data.iloc[i]['Close']
                    current_return = (current_price - entry_price) / entry_price

                    if current_return >= tp or current_return <= sl:
                        data.at[data.index[i], 'exit_price'] = current_price
                        data.at[data.index[i], 'return'] = current_return
                        position_open = False
                        entry_price = 0.0
                        continue

                # Open new position if no position is open and signal is 1
                if not position_open and data.iloc[i]['signal'] == 1:
                    data.at[data.index[i], 'position'] = 1
                    data.at[data.index[i], 'entry_price'] = data.iloc[i]['Close']
                    position_open = True
                    entry_price = data.iloc[i]['Close']

            # Close any open position at the end of the dataset
            if position_open:
                last_idx = data.index[-1]
                data.at[last_idx, 'exit_price'] = data.iloc[-1]['Close']
                data.at[last_idx, 'return'] = (data.iloc[-1]['Close'] - entry_price) / entry_price

            return data

        # --- 4. Run Backtest ---
        results = backtest(data, tp=0.005, sl=-0.004)

        # --- 5. Calculate Performance Metrics ---
        trades = results.dropna(subset=['return'])
        total_return = (1 + trades['return']).prod() - 1
        win_rate = len(trades[trades['return'] > 0]) / len(trades)
        sharpe_ratio = trades['return'].mean() / trades['return'].std() * np.sqrt(252)  # Annualized

        print(f"Total Return: {total_return:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Number of Trades: {len(trades)}")
        
        
        
    
        data.to_csv('csv/yahoo_data.csv')
        
        return
   
        
    def load_data(self, ticker: str) -> pd.DataFrame:
        if os.path.exists(os.path.join('app/assets/', 'yahoo_data.csv')):
            raw_data = pd.read_csv(os.path.join('app/assets/', 'yahoo_data.csv'))
            data = DataUtils.extract_ticker_data(raw_data, ticker)
            
            # # Transform Date column from string to datetime
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
            data['DT'] = data['Date']
            data['C'] = data['Close']
            data.set_index('Date', inplace=True)
            
            return data
        else:
            raise FileNotFoundError("Data file not found.")
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        
        short = 20
        long = 50
        data[f"sma_{short}"] = data["C"].rolling(short).mean().round(3)
        data[f"sma_{long}"] = data["C"].rolling(long).mean().round(3)

        data["sma_diff"] = (data[f"sma_{short}"] - data[f"sma_{long}"]).round(3) 
        data["sma_cross"] = (np.where(data["sma_diff"] > 0, 1, -1)).round(3)
        data["sma_ratio"] = (data[f"sma_{short}"] / data[f"sma_{long}"] - 1).round(3)
        data["sma_cross_change"] = data["sma_cross"].diff().fillna(0)
        data["sma_diff_slope"] = data["sma_diff"].diff()
        
        indicator = IndicatorFactory.create_indicator(StrategyLibEnum.RSI)
        data = indicator.calculate_signals(data, '{"period": 14, "overbought": 70, "oversold": 30}')        
        
        data = DataUtils.add_macd(data)

        data[f"future_return"] = data["C"].shift(-1) / data["C"] - 1
        data = data.dropna()
        return data 
    
    def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Signal'] = 0
        data.loc[data['Close'].shift(-1) > data['Close'], 'Signal'] = 1
        data.loc[data['Close'].shift(-1) < data['Close'], 'Signal'] = -1
        return data
    
    def train_model(self, data: pd.DataFrame) -> None:
        features = [
            "sma_diff",
            "sma_diff_slope",
            "rsi_14",
            "macd_cross",
            "macd_cross_change",
            "Volume",
        ]
        scaler = MinMaxScaler()
        data[features] = scaler.fit_transform(data[features])

        X = []
        y = []
        sequence_length = 30  # Länge des Eingabefensters

        # for i in range(sequence_length, len(data)):
        #     X.append(data[features].iloc[i-sequence_length:i].values)
        #     y.append(data['Signal'].iloc[i])
            
   

        X = np.array(X)
        y = np.array(y)
        # y = keras.utils.to_categorical((y + 1))  # Umwandlung in One-Hot-Encoding
        
        
        X = data[features]
        # Use all 4 future returns as targets
        y = data["future_return"]
        
        # Daten aufteilen in Training und Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        split_point = int(len(data) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        # Multi-output regression model
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
        model.fit(X_train, y_train)

        returnscores = model.score(X_test, y_test)
        print(f"Return Prediction R^2 Score: {returnscores}")
        
        return_predictions = model.predict(X_test)
        plt.figure(figsize=(12,6))
        plt.plot(y_test.index, y_test, label="True Returns")
        plt.plot(y_test.index, return_predictions, label="Predicted Returns")
        plt.title("Return Predictions vs True Returns")
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.legend()
        plt.show()

        return



    def save_model(self, model: keras.Model, scaler: MinMaxScaler) -> None:
        # Modell und Scaler speichern
        model.save('ai_models/yahoo_aapl_model.h5')
        joblib.dump(scaler, 'ai_models/yahoo_aapl_scaler.pkl')

    def get_trading_signal(self, recent_data: pd.DataFrame) -> str:
        """
        Gibt ein Trading-Signal basierend auf aktuellen Daten zurück
        """
        if not hasattr(self, 'model') or not hasattr(self, 'scaler'):
            return "NO_MODEL"
    
        # Indikatoren berechnen
        data = self.calculate_indicators(recent_data.copy())
        
        if len(data) < 30:  # Nicht genug Daten für Vorhersage
            return "INSUFFICIENT_DATA"
        
        # Features vorbereiten
        features = ['Close', 'SMA', 'RSI', 'MACD']
        scaled_data = self.scaler.transform(data[features])
        
        # Letzten 30 Werte für Vorhersage nehmen
        sequence = scaled_data[-30:].reshape(1, 30, len(features))
        
        # Vorhersage machen
        prediction = self.model.predict(sequence, verbose=0)
        signal_class = np.argmax(prediction[0]) - 1  # [0,1,2] → [-1,0,1]
        confidence = np.max(prediction[0])
        
        # Signal mit Konfidenz-Schwellwert
        if confidence < 0.6:  # Mindestens 60% Sicherheit
            return "HOLD"
        
        if signal_class == 1:
            return "BUY"
        elif signal_class == -1:
            return "SELL"
        else:
            return "HOLD"

    def should_buy(self, recent_data: pd.DataFrame) -> bool:
        """
        Einfache Funktion für Kaufentscheidung
        """
        signal = self.get_trading_signal(recent_data)
        return signal == "BUY"

    def get_detailed_signal(self, recent_data: pd.DataFrame) -> dict:
        """
        Detaillierte Signal-Informationen
        """
        if not hasattr(self, 'model'):
            return {"signal": "NO_MODEL", "confidence": 0.0, "details": "Model not loaded"}
        
        data = self.calculate_indicators(recent_data.copy())
        features = ['Close', 'SMA', 'RSI', 'MACD']
        scaled_data = self.scaler.transform(data[features])
        sequence = scaled_data[-30:].reshape(1, 30, len(features))
        
        prediction = self.model.predict(sequence, verbose=0)
        probabilities = prediction[0]
        signal_class = np.argmax(probabilities) - 1
        
        signals = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        
        return {
            "signal": signals[signal_class],
            "confidence": float(np.max(probabilities)),
            "probabilities": {
                "sell": float(probabilities[0]),
                "hold": float(probabilities[1]), 
                "buy": float(probabilities[2])
            },
            "current_price": float(data['Close'].iloc[-1]),
            "rsi": float(data['RSI'].iloc[-1]),
            "macd": float(data['MACD'].iloc[-1])
        }