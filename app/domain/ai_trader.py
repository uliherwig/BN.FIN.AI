from uuid import UUID
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score
import tensorflow as tf
import keras
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Any
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from app.domain.models.enums import SideEnum, IndicatorEnum
from app.domain.operations.indicator_factory import IndicatorFactory
from app.domain.position_manager import PositionManager
from app.infrastructure.redis_service import RedisService, get_redis_service
from app.infrastructure import alpaca_service
import time
import matplotlib.pyplot as plt
from app.models.schemas import Quote, Position
from app.domain.operations.data_utils import DataUtils
import os


class AITrader:
    def __init__(self, ticker: str = "AAPL"):
        self.ticker = ticker
        start_str = "2010-01-01"
        end_str = "2025-10-20"
        interval = "1d"
        self.filename = f'csv/stock_data_{ticker}_{start_str}_{end_str}_{interval}.csv'
        self.test_filename = f'csv/test_data_{ticker}_{start_str}_{end_str}_{interval}.csv'
        # self.model = keras.models.load_model('ai_models/yahoo_aapl_model.h5')
        # self.scaler = joblib.load('ai_models/yahoo_aapl_scaler.pkl')
        self.price_change_threshold = 0.002  # 0.2% threshold
        # Define confidence thresholds
        self.long_threshold = 0.6   # Need 60%+ confidence to go LONG
        self.short_threshold = 0.6  # Need 60%+ confidence to go SHORT


    def learn_from_yahoo_data(self) -> None:
        # Binary model predicts: 0 (DOWN) or 1 (UP)
        # But we generate 3 signals: -1 (SHORT), 0 (HOLD), 1 (LONG)

        # HOLD = Low confidence predictions (uncertain)
        data = self.load_yahoo_stock_data(self.ticker)

        data = data[['DT', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()  

        data = self.add_technical_indicators(data)

        # Update features list
        features = [
            'SMA_diff',           # Trend
            'RSI',                # Momentum
            'MACD',               # Momentum
            'ATR',                # Volatility
            'Volatility_20d',     # Volatility
            'Volume_Ratio',       # Volume
            'Close_Lag_1',        # Price history
            'ROC_1d',          # Recent return
            'EMA_diff',
        ]

        # Drop rows with NaN values (due to rolling windows)
        data = data.dropna()
       
        # --- 3. Target Variable: Next-Day DIRECTION (not exact return) ---
        # Use a threshold to filter out noise
        next_return = data['Close'].pct_change().shift(-1)
        
        # Create binary target: 1 = up, 0 = down
        data['target'] = np.where(
            next_return > self.price_change_threshold, 1,
            np.where(next_return < -self.price_change_threshold, 0, np.nan)
        )
        data.to_csv(self.test_filename)

        # Drop neutral days (within threshold)
        data = data.dropna(subset=['target'])

        # Drop the last row (no target for the last day)
        data = data.iloc[:-1]

        # --- 4. Split Features and Target ---
        X = data[features]
        y = data['target']
        
        # --- 8. Run Walk-Forward Validation ---
        results_df, final_model = self.walk_forward_validation(X, y)
        
        print(f"Features: {features}")

        # --- 9. Print Results ---
        print("Walk-Forward Validation Results:")
        print(results_df)
        print("\nFeature Importance:")
        importance_dict = dict(zip(features, final_model.feature_importances_))
        for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")

        # --- BACKTESTING: Generate Signals with Confidence Threshold ---
        # Get prediction probabilities
        y_pred_proba = final_model.predict_proba(data[features])

        # Extract probabilities for each class
        prob_down = y_pred_proba[:, 0]  # Probability of DOWN (class 0)
        prob_up = y_pred_proba[:, 1]    # Probability of UP (class 1)

        # Generate 3-way signals: LONG (1), SHORT (-1), HOLD (0)
        data['signal'] = 0  # Default to HOLD

        # LONG: High confidence that price will go UP
        data.loc[prob_up > self.long_threshold, 'signal'] = 1

        # SHORT: High confidence that price will go DOWN
        data.loc[prob_down > self.short_threshold, 'signal'] = -1

        # HOLD: Everything else (low confidence either way)
        # This happens automatically with default = 0

        # Store confidence for analysis
        data['confidence_up'] = prob_up
        data['confidence_down'] = prob_down
        data['max_confidence'] = np.maximum(prob_up, prob_down)

        # Print signal distribution
        print(f"\n{'='*60}")
        print(f"SIGNAL GENERATION - {self.ticker}")
        print(f"{'='*60}")
        print(f"Total bars: {len(data)}")
        print(f"\nSignal Distribution:")
        print(
            f"  LONG (1):  {(data['signal'] == 1).sum():5d} ({(data['signal'] == 1).mean():.1%}) - High confidence UP")
        print(
            f"  SHORT (-1): {(data['signal'] == -1).sum():5d} ({(data['signal'] == -1).mean():.1%}) - High confidence DOWN")
        print(
            f"  HOLD (0):  {(data['signal'] == 0).sum():5d} ({(data['signal'] == 0).mean():.1%}) - Low confidence")
        print(f"\nAverage Confidence:")
        print(
            f"  LONG signals:  {data[data['signal'] == 1]['confidence_up'].mean():.2%}")
        print(
            f"  SHORT signals: {data[data['signal'] == -1]['confidence_down'].mean():.2%}")
        print(
            f"  HOLD signals:  {data[data['signal'] == 0]['max_confidence'].mean():.2%}")
        print(f"{'='*60}\n")

        # --- 4. Run Backtest ---
        results = self.backtest(data, tp=0.01, sl=0.005)

        # --- 5. Calculate Performance Metrics ---
        trades = results.dropna(subset=['return'])
        total_return = (1 + trades['return']).prod() - 1
        test_return = data['return'].sum()
        win_rate = len(trades[trades['return'] > 0]) / len(trades)
        sharpe_ratio = trades['return'].mean() / trades['return'].std() * np.sqrt(252)  # Annualized
        print(f"Total Return: {total_return:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        data.to_csv(self.test_filename)
        
        # print features of final model
        print("Final Model Feature Importances:")
        importance_dict = dict(zip(features, final_model.feature_importances_))
        for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        self.save_lgb_model(final_model)
        
        
        self.load_lgb_model()

        print(f"Model: loaded {self.model}")
        print(f"Features: {self.features}")
        return

    def backtest(self, data, tp=0.005, sl=0.004):
        """
        Backtest with ONLY ONE position open at a time
        Supports LONG (1) and SHORT (-1) positions
        """
        pos_tp = 0
        pos_sl = 0
        pos_sig_change = 0

        # Initialize columns for tracking trades
        data['position'] = 0  # 0 = flat, 1 = long, -1 = short
        data['entry_price'] = np.nan
        data['exit_price'] = np.nan
        data['return'] = 0.0
        data['return_abs'] = 0.0
        pm = PositionManager.create_with_test_params(
            asset='SPY',
            quantity=Decimal('1'),
            strategy_type=IndicatorEnum.NONE,
            close_positions_eod=False,
            strategy_params=''
        )
        position_open = False
        entry_price = 0.0
        position_type = 0  # 1 = long, -1 = short, 0 = no position
        position_id = UUID(int=0)
        profit = 0.0
        for i in range(len(data)):
            current_price = data.iloc[i]['Close']
            current_signal = data.iloc[i]['signal']
            current_timestamp = data.iloc[i]['DT']
            
            if(current_timestamp.year < 2024):
                continue

            # --- STEP 1: Check if we need to CLOSE an existing position ---
            if position_open:
                # Calculate return based on position type
                if position_type == 1:  # LONG position
                    current_return = (current_price - entry_price) / entry_price
                else:  # SHORT position
                    current_return = (entry_price - current_price) / entry_price

                # Check if TP or SL is hit
                if current_return >= tp or current_return <= (-1) * sl:
                    if current_return >= tp:
                        # TP hit - exit at TP price
                        if position_type == 1:
                            exit_price = entry_price * (1 + tp)
                        else:
                            exit_price = entry_price * (1 - tp)
                        data.at[data.index[i], 'exit_price'] = exit_price
                        data.at[data.index[i], 'return'] = tp
                        data.at[data.index[i], 'return_abs'] = (exit_price - entry_price) * position_type
                        profit += (exit_price - entry_price) * position_type
                        # print(f"TP HIT at {exit_price} for position {position_id} on {current_timestamp}")
                        pm.close_position(position_id, Decimal(str(exit_price)), current_timestamp)
                        pos_tp += 1
                    else:
                        # SL hit - exit at SL price
                        if position_type == 1:
                            exit_price = entry_price * (1 - sl)
                        else:
                            exit_price = entry_price * (1 + sl)
                        data.at[data.index[i], 'exit_price'] = exit_price
                        data.at[data.index[i], 'return'] = sl * -1
                        data.at[data.index[i], 'return_abs'] = (exit_price - entry_price) * position_type
                        profit += (exit_price - entry_price) * position_type
                        # print(f"SL HIT at {exit_price} for position {position_id} on {current_timestamp}")
                        pm.close_position(position_id, Decimal(str(exit_price)), current_timestamp)
                        pos_sl += 1
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    continue

                # If TP/SL not hit, check for opposite signal to close & reverse
                if (position_type == 1 and current_signal == -1) or (position_type == -1 and current_signal == 1):
                    # CLOSE current position
                    data.at[data.index[i], 'exit_price'] = current_price
                    data.at[data.index[i], 'return'] = current_return
                    data.at[data.index[i], 'return_abs'] = current_return * entry_price * position_type
                    profit += (current_price - entry_price) * position_type
                    pm.close_position(position_id, Decimal(str(current_price)), current_timestamp)
                    pos_sig_change += 1
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    # Allow opening opposite position below

            # --- STEP 2: Open NEW position if no position is open ---
            if not position_open:
                if current_signal == 1:  # BUY signal → LONG
                    tp_price, sl_price = Decimal(
                        current_price + current_price * tp), Decimal(current_price - current_price * sl)
                    data.at[data.index[i], 'position'] = 1
                    data.at[data.index[i], 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = 1
                    position_id = pm.open_position(SideEnum.Buy, Decimal(
                        str(current_price)), tp_price, sl_price, current_timestamp)
                elif current_signal == -1:  # SELL signal → SHORT
                    tp_price, sl_price = Decimal(
                        current_price - current_price * tp), Decimal(current_price + current_price * sl)
                    data.at[data.index[i], 'position'] = -1
                    data.at[data.index[i], 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = -1
                    position_id = pm.open_position(SideEnum.Sell, Decimal(
                        str(current_price)), tp_price, sl_price, current_timestamp)
                # If signal == 0 (HOLD), do nothing - stay flat

        # --- STEP 3: Close any remaining open position at end of dataset ---
        if position_open:
            last_idx = data.index[-1]
            last_price = data.iloc[-1]['Close']
            current_timestamp = data.iloc[-1]['DT']
            if position_type == 1:  # Close LONG
                final_return = (last_price - entry_price) / entry_price
            else:  # Close SHORT
                final_return = (entry_price - last_price) / entry_price
            data.at[last_idx, 'exit_price'] = last_price
            data.at[last_idx, 'return'] = final_return
            profit += final_return * entry_price
            pm.close_position(position_id, Decimal(str(last_price)), current_timestamp)

        number_of_positions = sum(data['position'] != 0)
        print(f"Total Profit from Backtest: {profit:.2f}")
        print(f"Total Number of Positions Taken: {number_of_positions}")
        print(f"################################################")
        pos = pm.get_positions()
        pos_df = pd.DataFrame([
            {
                'Id': position_id,
                'side': position.side.name,
                'openPrice': round(float(position.price_open), 3) if position.price_open is not None else None,
                'closePrice': round(float(position.price_close), 3) if position.price_close is not None else None,
                'take_profit': round(float(position.take_profit), 3) if getattr(position, 'take_profit', None) is not None else None,
                'stop_loss': round(float(position.stop_loss), 3) if getattr(position, 'stop_loss', None) is not None else None,
                'opened': position.stamp_opened.strftime("%y.%m.%d") if position.stamp_opened else "",
                'closed': position.stamp_closed.strftime("%y.%m.%d") if position.stamp_closed else "",
                # 'minutes': (position.stamp_closed - position.stamp_opened).total_seconds() / 60 if position.stamp_closed and position.stamp_opened else 0,
                'profit_loss': float(position.profit_loss) if getattr(position, 'profit_loss', None) is not None else None,
            }
            for position_id, position in pos.items()
        ])
        pos_df.to_csv("csv/positions.csv")

        # Calculate total profit
        profit = pm.calculate_profit()
        number_of_trades = len(pos_df)
        number_of_long_signals = len(pos_df[pos_df['side'] == 'Buy'])
        number_of_short_signals = len(pos_df[pos_df['side'] == 'Sell'])
        number_of_lost_positions = len(pos_df[pos_df['profit_loss'] < 0])
        number_of_won_positions = len(pos_df[pos_df['profit_loss'] > 0])
        print(f"PM ## Profit: {profit:.2f}   Number of Trades: {number_of_trades}, Long Signals: {number_of_long_signals}, Short Signals: {number_of_short_signals}, Won Positions: {number_of_won_positions}, Lost Positions: {number_of_lost_positions}")
        print(f"TP Hits: {pos_tp}, SL Hits: {pos_sl}, Signal Reversals: {pos_sig_change}")
        return data
    
    def walk_forward_validation(self, X, y, train_years=5, test_years=1, start_year=2010):
        results = []
        
        print(f"Walk-Forward Validation X {X.head()}")
        print(f"Walk-Forward Validation y {y.head()}")
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
            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',  # Important for imbalanced data
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_test)
            # Probability of class 1 (UP)
            y_pred_confidence_up = y_pred_proba[:, 1]
            # Probability of class 0 (DOWN)
            y_pred_confidence_down = y_pred_proba[:, 0]
            # Only trade when confidence > threshold
            confidence_threshold = 0.6
            y_pred_class = np.where(
                y_pred_confidence_up > confidence_threshold, 1, 0)
            # Recalculate metrics
            accuracy = accuracy_score(y_test, y_pred_class)
            precision = precision_score(y_test, y_pred_class, zero_division=0)
            recall = recall_score(y_test, y_pred_class, zero_division=0)
            results.append({
                'year': i,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Trades': sum(y_pred_class),  # Number of signals
                'Avg_Confidence_UP': y_pred_confidence_up[y_pred_class == 1].mean(),
                'Avg_Confidence_DOWN': y_pred_confidence_down[y_pred_class == 0].mean()
            })
        return pd.DataFrame(results), model

    def save_lgb_model(self, model, scaler=None) -> None:
        """Save LightGBM model and optional scaler"""
        # Create directory if it doesn't exist
        os.makedirs('ai_models', exist_ok=True)
        # Save model
        model_path = f'ai_models/lgb_{self.ticker}_classifier.pkl'
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        # Save scaler if provided
        if scaler is not None:
            scaler_path = f'ai_models/lgb_{self.ticker}_scaler.pkl'
            joblib.dump(scaler, scaler_path)
            print(f"✅ Scaler saved to {scaler_path}")
        # Save feature names for later use
        features_path = f'ai_models/lgb_{self.ticker}_features.pkl'
        features = [
            'SMA_diff', 'RSI', 'MACD', 'ATR',
            'Volatility_20d', 'Volume_Ratio', 'Close_Lag_1', 'ROC_1d',
            'EMA_diff'  # <-- Add this line
        ]
        joblib.dump(features, features_path)
        print(f"✅ Features saved to {features_path}")

    def load_lgb_model(self) -> bool:
        """Load saved LightGBM model and features"""
        try:
            model_path = f'ai_models/lgb_{self.ticker}_classifier.pkl'
            features_path = f'ai_models/lgb_{self.ticker}_features.pkl'
            scaler_path = f'ai_models/lgb_{self.ticker}_scaler.pkl'
            # Load model
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
            # Load features
            self.features = joblib.load(features_path)
            print(f"✅ Features loaded: {self.features}")
            # Load scaler if exists
     
            return True
        except FileNotFoundError as e:
            print(f"❌ Model not found: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def load_yahoo_stock_data(self, ticker: str) -> pd.DataFrame:
        start_str = "2010-01-01"
        end_str = "2025-10-20"
        interval = "1d"
        filename = f'csv/stock_data_{ticker}_{start_str}_{end_str}_{interval}.csv'
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            data = data.rename(columns={'Price': 'Date'})
            # remove 2nd and 3rd line
            data = data.drop(index=0)
            data = data.drop(index=1)
            #rename col Price to Date
            # list columns
            print(f"Columns in data: {data.columns.tolist()}")
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            # # Transform Date column from string to datetime
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
            data['DT'] = data['Date']
            data.set_index('Date', inplace=True)

            print(data.head())
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
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.RSI)
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
        sequence_length = 30  # Length of input window
        # for i in range(sequence_length, len(data)):
        #     X.append(data[features].iloc[i-sequence_length:i].values)
        #     y.append(data['Signal'].iloc[i])
        X = np.array(X)
        y = np.array(y)
        # y = keras.utils.to_categorical((y + 1))  # Convert to one-hot encoding
        X = data[features]
        # Use all 4 future returns as targets
        y = data["future_return"]
        # Split data into training and test
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
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label="True Returns")
        plt.plot(y_test.index, return_predictions, label="Predicted Returns")
        plt.title("Return Predictions vs True Returns")
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.legend()
        plt.show()
        return

    def save_model(self, model: keras.Model, scaler: MinMaxScaler) -> None:
        
        # Save model and scaler
        model.save('ai_models/yahoo_aapl_model.h5')
        joblib.dump(scaler, 'ai_models/yahoo_aapl_scaler.pkl')

    def get_trading_signal(self, recent_data: pd.DataFrame) -> str:
        """
        Returns a trading signal based on current data
        """
        if not hasattr(self, 'model') or not hasattr(self, 'scaler'):
            return "NO_MODEL"

        # Calculate indicators
        data = self.calculate_indicators(recent_data.copy())
        if len(data) < 30:  # Not enough data for prediction
            return "INSUFFICIENT_DATA"

        # Prepare features
        features = ['Close', 'SMA', 'RSI', 'MACD']
        scaled_data = self.scaler.transform(data[features])

        # Take last 30 values for prediction
        sequence = scaled_data[-30:].reshape(1, 30, len(features))

        # Make prediction
        prediction = self.model.predict(sequence, verbose=0)
        signal_class = np.argmax(prediction[0]) - 1  # [0,1,2] → [-1,0,1]
        confidence = np.max(prediction[0])

        # Signal with confidence threshold
        if confidence < 0.6:  # At least 60% confidence
            return "HOLD"
        if signal_class == 1:
            return "BUY"
        elif signal_class == -1:
            return "SELL"
        else:
            return "HOLD"

    def should_buy(self, recent_data: pd.DataFrame) -> bool:
        """
        Simple function for buy decision
        """
        signal = self.get_trading_signal(recent_data)
        return signal == "BUY"

    def get_detailed_signal(self, recent_data: pd.DataFrame) -> dict:
        """
        Detailed signal information
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

    def predict_direction(self, recent_data: pd.DataFrame) -> dict:
        """
        Predict trading direction for new stock data
        Args:
            recent_data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        Returns:
            dict with prediction, confidence, and signal
        """
        # Load model if not already loaded
        if self.model is None:
            if not self.load_lgb_model():
                return {"error": "Model not loaded"}
        # Calculate features
        data = recent_data.copy()
        # Same feature engineering as training
        data = self.add_technical_indicators(data)
        data = data.dropna()
        if len(data) == 0:
            return {"error": "Insufficient data for prediction"}
        # Get features for prediction
        X = data[self.features].iloc[-1:]  # Last row
        # Predict
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return {
            "prediction": int(prediction),  # 0 = down, 1 = up
            "signal": "BUY" if prediction == 1 else "SELL",
            "confidence": float(proba[prediction]),
            "probabilities": {
                "down": float(proba[0]),
                "up": float(proba[1])
            },
            "current_price": float(data['Close'].iloc[-1]),
            "features": {feat: float(X[feat].iloc[0]) for feat in self.features}
        }

    def get_trading_signal_with_confidence(self, recent_data: pd.DataFrame,
                                           confidence_threshold: float = 0.6) -> str:
        """
        Get trading signal with confidence filtering
        Args:
            recent_data: Recent price data
            confidence_threshold: Minimum confidence to act (0.0-1.0)
        Returns:
            "BUY", "SELL", or "HOLD"
        """
        result = self.predict_direction(recent_data)
        if "error" in result:
            return "HOLD"
        # Only trade if confidence is high enough
        if result["confidence"] < confidence_threshold:
            return "HOLD"
        return result["signal"]

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.SMA)
        df = indicator.create_features(df, '{"short_ma": 25, "long_ma": 100}')
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.EMA)
        df = indicator.create_features(df, '{"short_ma": 25, "long_ma": 100}')
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.WMA)
        df = indicator.create_features(df, '{"short_ma": 25, "long_ma": 100}')
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.RSI)
        df = indicator.create_features(
            df, '{"period": 14, "overbought": 70, "oversold": 30}')
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.DONCHIAN)
        df = indicator.create_features(df, '{"window": 30}')
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.MACD)
        df = indicator.create_features(
            df, '{"fast": 12, "slow": 26, "signal": 9}')
        indicator = IndicatorFactory.create_indicator(
            IndicatorEnum.VOLA)
        df = indicator.create_features(df, '{"period": 20}')
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.ATR)
        df = indicator.create_features(df, '{"period": 14}')
        df = df.dropna()
        return df

    def display_lgb_feature_importance(self) -> None:
        """Display feature importance of the LightGBM model"""
      
        self.load_lgb_model()
        importance_dict = dict(
            zip(self.features, self.model.feature_importances_))
        print(f"\nFeature Importance for {self.ticker}:")
        for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")

        # Plot and save feature importance image
        fig, ax = plt.subplots(figsize=(12, 8))
        lgb.plot_importance(self.model, ax=ax)
        image_path = f'ai_models/lgb_{self.ticker}_feature_importance.png'
        plt.savefig(image_path)
        plt.close(fig)
        print(f"✅ Feature importance image saved to {image_path}")
        return
