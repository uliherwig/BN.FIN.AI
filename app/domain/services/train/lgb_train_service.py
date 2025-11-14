from uuid import UUID
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score, mean_squared_error
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
from app.domain.operations.lgb_model_factory import LGBModelFactory
from app.domain.position_manager import PositionManager
from app.infrastructure.redis_service import RedisService, get_redis_service
from app.infrastructure import alpaca_service, yahoo_service
from app.infrastructure.yahoo_service import YahooService
import time
import matplotlib.pyplot as plt
from app.models.schemas import Quote, Position
from app.domain.operations.data_utils import DataUtils
from app.domain.models.strategies.indicator_model import IndicatorModel
from app.domain.services.optimize.optuna_configurator import OptunaConfigurator
import os
import json


class LgbTrainService:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_yahoo_data(self, settings: dict) -> None:
        indicators = settings["indicators"]
        lgb_model = settings["lgb_model"]
        execution_params = settings["execution_params"]
        features = settings["features"]

        model_type=lgb_model["model_type"]
        asset = execution_params["asset"]
        start_date = execution_params["start_date"]
        end_date = execution_params["end_date"]
        long_threshold = execution_params["long_threshold"]
        short_threshold = execution_params["short_threshold"]
        tp = execution_params["tp"]
        sl = execution_params["sl"]

        print("Train LGB Model with settings:")
        print(settings)

        price_change_threshold = 0.0  # Set as needed
        test_filename = f'csv/alpaca_{asset}_{start_date}_{end_date}_1d.csv'

        data = YahooService.load_yahoo_stock_data(asset)
        data = data[['DT', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        indicators_list = IndicatorFactory.get_indicator_list_by_params(indicators)
        data = IndicatorFactory.extend_dataframe_with_indicators(data, indicators_list)
        data = data.dropna()
        next_return = data['Close'].pct_change().shift(-1)
        data["NEXT"] = next_return * 100
        data["NEXT_DAY"] = data['Close'].shift(-1)
        data["DIFF_NEXT_DAY"] = data['Close'].shift(-1) - data['Close']
        
        if( model_type == 'classification'):
            data['target'] = np.where(
                next_return > price_change_threshold, 1,
                np.where(next_return < -price_change_threshold, -1, 0)
            )
        elif( model_type == 'regression'):
            data['target'] = next_return * 100
            
        data = data.iloc[:-1]
        X = data[features]
        y = data['target']

        final_model = self.walk_forward_validation(lgb_model, X, y, train_years=6, test_years=1, start_year=2010)
        
        data.to_csv('csv/spy_results.csv', index=False)
       

        importance_dict = dict(zip(features, final_model.feature_importances_))
        for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")

        y_pred_proba = final_model.predict_proba(data[features])
        # prob_down = y_pred_proba[:, 0]
        # prob_up = y_pred_proba[:, 1]
        prob_hold = y_pred_proba[:, 2]
        prob_short = y_pred_proba[:, 0]
        prob_long = y_pred_proba[:, 1]

        data['signal'] = 0
        data.loc[prob_long > long_threshold, 'signal'] = 1
        data.loc[prob_short > short_threshold, 'signal'] = -1
        data['confidence_hold'] = prob_hold
        data['confidence_short'] = prob_short
        data['confidence_long'] = prob_long
        data['max_confidence'] = np.maximum(prob_hold, np.maximum(prob_short, prob_long))
        data = self.backtest(data, tp, sl)

        daily_returns = pd.Series(data['return'])
        cumulative_product = np.prod([1 + r for r in daily_returns])
        total_return = cumulative_product - 1
        total_return_percentage = total_return * 100
        data.to_csv('csv/spy_results.csv', index=False)
        final_equity = data['equity'].iloc[-1] if 'equity' in data.columns else None

        sharpe_ratio = DataUtils.sharpe_ratio(daily_returns)
        max_drawdown = DataUtils.max_drawdown(daily_returns)
        print(f"ABS Return: {data['return_abs'].sum()} ")
        print(f"Total Return: {total_return_percentage:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.5f}")
        print(f"Max Drawdown: {max_drawdown:.5f}")

        LGBModelFactory.save_lgb_model(final_model, asset, features)

        result = {
            "profit": data['return_abs'].sum(),
            "total_return": total_return,
            "final_equity": final_equity,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "feature_importances": importance_dict
        }
        return result
    
    def backtest(self, data, tp=0.005, sl=0.004):
        """
        Backtest with ONLY ONE position open at a time
        Supports LONG (1) and SHORT (-1) positions
        """
        initial_equity = 10000.0
        equity = initial_equity
        equity_curve = []

        data['position'] = 0
        data['entry_price'] = np.nan
        data['exit_price'] = np.nan
        data['return'] = 0.0
        data['return_abs'] = 0.0
        data['equity'] = np.nan
        pm = PositionManager.create_with_test_params(
            asset='SPY',
            quantity=Decimal('1'),
            strategy_type=IndicatorEnum.NONE,
            close_positions_eod=False,
            strategy_params=''
        )
        position_open = False
        entry_price = 0.0
        position_type = 0
        position_id = UUID(int=0)
        profit = 0.0
        for idx, row in data.iterrows():
            current_price = row['Close']
            current_signal = row['signal']
            current_timestamp = row['DT']
            confidence_up = row.get('confidence_up', 0)
            confidence_down = row.get('confidence_down', 0)

            if position_open:
                if position_type == 1:
                    current_return = (current_price - entry_price) / entry_price
                else:
                    current_return = (entry_price - current_price) / entry_price

                if current_return >= tp or current_return <= (-1) * sl:
                    if current_return >= tp:
                        if position_type == 1:
                            exit_price = entry_price * (1 + tp)
                        else:
                            exit_price = entry_price * (1 - tp)
                        pm.close_position(position_id, Decimal(str(exit_price)), current_timestamp, "TP")
                    else:
                        if position_type == 1:
                            exit_price = entry_price * (1 - sl)
                        else:
                            exit_price = entry_price * (1 + sl)
                        pm.close_position(position_id, Decimal(str(exit_price)), current_timestamp, "SL")
                    p = (exit_price - entry_price) * position_type
                    data.at[idx, 'exit_price'] = exit_price
                    data.at[idx, 'return'] = p / equity
                    data.at[idx, 'return_abs'] = (exit_price - entry_price) * position_type
                    profit += p
                    equity += p
                    data.at[idx, 'equity'] = equity
                    equity_curve.append(equity)
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    continue

                if (position_type == 1 and current_signal == -1) or (position_type == -1 and current_signal == 1):
                    data.at[idx, 'exit_price'] = current_price
                    p = (current_price - entry_price) * position_type
                    data.at[idx, 'return'] = p / equity
                    data.at[idx, 'return_abs'] = p
                    profit += p
                    equity += p
                    data.at[idx, 'equity'] = equity
                    pm.close_position(position_id, Decimal(str(current_price)), current_timestamp, "Reverse")
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    equity_curve.append(equity)

            if not position_open:
                if current_signal == 1:
                    tp_price, sl_price = Decimal(
                        current_price + current_price * tp), Decimal(current_price - current_price * sl)
                    data.at[idx, 'position'] = 1
                    data.at[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = 1
                    position_id = pm.open_position(SideEnum.Buy, Decimal(
                        str(current_price)), tp_price, sl_price, current_timestamp, confidence_up=confidence_up, confidence_down=confidence_down)
                elif current_signal == -1:
                    tp_price, sl_price = Decimal(
                        current_price - current_price * tp), Decimal(current_price + current_price * sl)
                    data.at[idx, 'position'] = -1
                    data.at[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = -1
                    position_id = pm.open_position(SideEnum.Sell, Decimal(
                        str(current_price)), tp_price, sl_price, current_timestamp, confidence_up=confidence_up, confidence_down=confidence_down)

        if position_open:
            last_idx = data.index[-1]
            last_price = data.iloc[-1]['Close']
            current_timestamp = data.iloc[-1]['DT']
            if position_type == 1:
                final_return = (last_price - entry_price) / entry_price
            else:
                final_return = (entry_price - last_price) / entry_price
            p = (last_price - entry_price) * position_type
            data.at[last_idx, 'exit_price'] = last_price
            data.at[last_idx, 'return'] = p / equity
            data.at[last_idx, 'return_abs'] = p
            profit += p
            pm.close_position(position_id, Decimal(str(last_price)), current_timestamp)
            equity += p
            data.at[last_idx, 'equity'] = equity
            equity_curve.append(equity)

        number_of_positions = sum(data['position'] != 0)
        print(f"Total Profit from Test: {profit:.2f}")
        print(f"Total Number of Positions Taken: {number_of_positions}")
        print("################################################")
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
                'signal': position.close_signal,
                'profit_loss': float(position.profit_loss) if getattr(position, 'profit_loss', None) is not None else None,
                'confidence_up': float(position.confidence_up) if getattr(position, 'confidence_up', None) is not None else None,
                'confidence_down': float(position.confidence_down) if getattr(position, 'confidence_down', None) is not None else None
            }
            for position_id, position in pos.items()
        ])
        pos_df.to_csv("csv/positions.csv")
        data.to_csv("csv/spy_test.csv")

        profit = pm.calculate_profit()
        number_of_trades = len(pos_df)
        number_of_long_signals = len(pos_df[pos_df['side'] == 'Buy'])
        number_of_short_signals = len(pos_df[pos_df['side'] == 'Sell'])
        number_of_lost_positions = len(pos_df[pos_df['profit_loss'] < 0])
        number_of_won_positions = len(pos_df[pos_df['profit_loss'] > 0])
        print(f"PM ## Profit: {profit:.2f}   Number of Trades: {number_of_trades}, Long Signals: {number_of_long_signals}, Short Signals: {number_of_short_signals}, Won Positions: {number_of_won_positions}, Lost Positions: {number_of_lost_positions}")
        print(f"Win Rate: {number_of_won_positions / number_of_trades:.2%}" if number_of_trades > 0 else "Win Rate: N/A")

        if 'DT' in data.columns and 'return_abs' in data.columns:
            data['year'] = pd.to_datetime(data['DT']).dt.year
            yearly_profit = data.groupby('year')['return_abs'].sum()
            yearly_trades = data.groupby('year')['position'].count()
            print("Yearly Profit:")
            for year, prof in yearly_profit.items():
                print(f"  {year}: Profit {prof:.2f} Number of Trades: {yearly_trades.get(year, 0)}")

        return data
    
    def walk_forward_validation(self, model_params, X, y, train_years=6, test_years=1, start_year=2010):
        results = []
        for i in range(start_year + train_years, 2025):
            train_start = f'{i - train_years}-01-01'
            train_end = f'{i - 1}-12-31'
            test_start = f'{i}-01-01'
            test_end = f'{i}-12-31'
            X_train, y_train = X.loc[train_start: train_end], y.loc[train_start:train_end]
            X_test, y_test = X.loc[test_start:test_end], y.loc[test_start:test_end]
           
            model = LGBModelFactory.create_lgb_model(
                    'regression', model_params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)

            # Evaluation
            rmse = mean_squared_error(y_test, y_pred)
            print(f"RMSE: {rmse}")
            
        direction_correct = (np.sign(y_pred) == np.sign(y_test)).mean()


        print(f"Richtungsgenauigkeit: {direction_correct:.4%}")


        # plt.figure(figsize=(12, 6))
        # plt.plot(y_test.index, y_test, label="Tatsächliche Returns", color="blue")
        # plt.plot(y_test.index, y_pred, label="Vorhergesagte Returns",
        #         color="red", linestyle="--")
        # plt.legend()
        # plt.title("Vorhersage vs. Realität")
        # plt.show()
        return model
    
    def walk_forward_validation_1h(self, X, y):
        results = []
        #  number of hours :    3306
        # from 2024-01-02 14:30:00 to 2025-08-25 20:00:00
        train_months = 5
        start_month = 1
        start_hour = pd.Timestamp('2024-01-02 14:30:00')
        for i in range(start_month + train_months, 12):            
            train_start = f'2024-{i - train_months}-01'
            train_end = f'2024-{i - 1}-31'
            test_start = f'2024-{i}-01'
            test_end = f'2024-{i}-31'
            # Split data
            X_train, y_train = X.loc[train_start:train_end], y.loc[train_start:train_end]
            X_test, y_test = X.loc[test_start:test_end], y.loc[test_start:test_end]
            print(f"Walk-Forward Month {i}: Train {train_start} to {train_end} ({len(X_train)} rows), Test {test_start} to {test_end} ({len(X_test)} rows)")
            # Skip if no training data
            if len(X_train) == 0:
                continue
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
            # Get prediction probabilities instead of hard predictions
            y_pred_proba = model.predict_proba(X_test)
            # Probability of class 1 (UP)
            y_pred_confidence = y_pred_proba[:, 1]
            # Only trade when confidence > threshold
            confidence_threshold = 0.6
            y_pred_class = np.where(
                y_pred_confidence > confidence_threshold, 1, 0)
            # Recalculate metrics
            accuracy = accuracy_score(y_test, y_pred_class)
            precision = precision_score(y_test, y_pred_class, zero_division=0)
            recall = recall_score(y_test, y_pred_class, zero_division=0)
            avg_conf_slice = y_pred_confidence[y_pred_class == 1]
            avg_conf = avg_conf_slice.mean() if len(avg_conf_slice) > 0 else 0.0
            results.append({
                'year': i,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Trades': sum(y_pred_class),  # Number of signals
                'Avg_Confidence': avg_conf
            })
        return pd.DataFrame(results)


    def get_train_results_by_test_settings(self, settings: dict, model_params, indicators) -> None:
        price_change_threshold = settings.get("price_change_threshold")
        long_threshold = settings.get("long_threshold")
        short_threshold = settings.get("short_threshold")
        tp= settings.get("tp")
        sl= settings.get("sl")
        asset = settings["asset"]
        start_date = settings["start_date"]
        end_date = settings["end_date"]
        test_filename = f'csv/alpaca_{asset}_{start_date}_{end_date}.csv'
        # Binary model predicts: 0 (DOWN) or 1 (UP)
        # But we generate 3 signals: -1 (SHORT), 0 (HOLD), 1 (LONG)
        # HOLD = Low confidence predictions (uncertain)
        data = YahooService.load_yahoo_stock_data(asset)
        data = data[['DT', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()    
        data = IndicatorFactory.extend_dataframe_with_indicators(
            data, indicators)
        indicator_enums = []
        for ind_enum in indicators:
            print(f"Using Indicator: {ind_enum.strategyType}")
            indicator_enums.append(ind_enum.strategyType)
        features = OptunaConfigurator.get_features_by_indicators(indicator_enums)
        # Drop rows with NaN values (due to rolling windows)
        data = data.dropna()
        # --- 3. Target Variable: Next-Day DIRECTION (not exact return) ---
        # Use a threshold to filter out noise
        next_return = data['Close'].pct_change().shift(-1)
        data['return'] = next_return
        # Create binary target: 1 = up, 0 = down
        data['target'] = np.where(
            next_return > price_change_threshold, 1,
            np.where(next_return < -price_change_threshold, -1, 0)
        )
        #  data.to_csv(test_filename)
        # Drop neutral days (within threshold)
        #  data = data.dropna(subset=['target'])
        # Drop the last row (no target for the last day)
        data = data.iloc[:-1]
        # --- 4. Split Features and Target ---
        X = data[features]
        y = data['target']
        # data.to_csv(test_filename)
        # --- 8. Run Walk-Forward Validation ---
        # yahoo data => more than 5 year needed
        final_model = self.walk_forward_validation(model_params,
           X, y, train_years=5, test_years=1, start_year=2010)
        # results_df, final_model = self.walk_forward_validation_1h(X, y)
        # --- 9. Print Results ---
        # print("Walk-Forward Validation Results:")
        # print(results_df)
        # print("\nFeature Importance:")
        importance_dict = dict(zip(features, final_model.feature_importances_))
        # for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        #     print(f"  {feat}: {imp:.4f}")
        # --- BACKTESTING: Generate Signals with Confidence Threshold ---
        # Get prediction probabilities
        
        y_pred_proba = final_model.predict_proba(data[features])
        # prob_down = y_pred_proba[:, 0]
        # prob_up = y_pred_proba[:, 1]
        prob_hold = y_pred_proba[:, 2]
        prob_short = y_pred_proba[:, 0]
        prob_long = y_pred_proba[:, 1]

        data['signal'] = 0
        data.loc[prob_long > long_threshold, 'signal'] = 1
        data.loc[prob_short > short_threshold, 'signal'] = -1
        data['confidence_hold'] = prob_hold
        data['confidence_short'] = prob_short
        data['confidence_long'] = prob_long
        data['max_confidence'] = np.maximum(
            prob_hold, np.maximum(prob_short, prob_long))
        
        # y_pred_proba = final_model.predict_proba(data[features])
        # # Extract probabilities for each class
        # prob_down = y_pred_proba[:, 0]  # Probability of DOWN (class 0)
        # prob_up = y_pred_proba[:, 1]    # Probability of UP (class 1)
        # # Generate 3-way signals: LONG (1), SHORT (-1), HOLD (0)
        # data['signal'] = 0  # Default to HOLD
        # # LONG: High confidence that price will go UP
        # data.loc[prob_up > long_threshold, 'signal'] = 1
        # # SHORT: High confidence that price will go DOWN
        # data.loc[prob_down > short_threshold, 'signal'] = -1
        # # HOLD: Everything else (low confidence either way)
        # # This happens automatically with default = 0
        # # --- 4. Run Backtest ---
        results = self.minimized_test(data, tp, sl)
        # --- 5. Calculate Performance Metrics ---
        trades = results.dropna(subset=['return'])
        total_return = (1 + trades['return']).prod() - 1
        win_rate = len(trades[trades['return'] > 0]) / len(trades)
        sharpe_ratio = DataUtils.sharpe_ratio(trades['return'])        
        max_drawdown = DataUtils.max_drawdown(trades['return'])
        # print(f"Profit: {trades['return_abs'].sum()}  ")
        # print(f"Win Rate: {win_rate:.2%}")
        # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        result = {
            "profit": trades['return_abs'].sum(),
            "total_return": total_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "feature_importances": importance_dict,
            "model": final_model,  # used for saving the model later
            "max_drawdown": max_drawdown
        }
        print(f"Result Summary: {result}")
        return result

    def minimized_test(self, data, tp, sl):
        data['position'] = 0  # 0 = flat, 1 = long, -1 = short
        data['entry_price'] = np.nan
        data['exit_price'] = np.nan
        data['return'] = 0.0
        data['return_abs'] = 0.0

        position_open = False
        entry_price = 0.0
        position_type = 0  # 1 = long, -1 = short, 0 = no position
        position_id = UUID(int=0)
        profit = 0.0
        for idx, row in data.iterrows():
            current_price = row['Close']
            current_signal = row['signal']
            current_timestamp = row['DT']

            # --- STEP 1: Check if we need to CLOSE an existing position ---
            if position_open:
                # Calculate return based on position type
                if position_type == 1:  # LONG position
                    current_return = (current_price - entry_price) / entry_price
                else:  # SHORT position
                    current_return = (entry_price - current_price) / entry_price

                # Check if TP is hit
                if current_return >= tp:
                    # TP hit - exit at TP price
                    if position_type == 1:
                        exit_price = entry_price * (1 + tp)
                    else:
                        exit_price = entry_price * (1 - tp)
                    data.at[idx, 'exit_price'] = exit_price
                    data.at[idx, 'return'] = tp
                    data.at[idx, 'return_abs'] = (exit_price - entry_price) * position_type
                    profit += (exit_price - entry_price) * position_type
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    continue

                # Check if SL is hit
                if current_return <= -sl:
                    # SL hit - exit at SL price
                    if position_type == 1:
                        exit_price = entry_price * (1 - sl)
                    else:
                        exit_price = entry_price * (1 + sl)
                    data.at[idx, 'exit_price'] = exit_price
                    data.at[idx, 'return'] = -sl
                    data.at[idx, 'return_abs'] = (exit_price - entry_price) * position_type
                    profit += (exit_price - entry_price) * position_type
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    continue

                if (position_type == 1 and current_signal == -1) or (position_type == -1 and current_signal == 1):
                    # CLOSE current position
                    data.at[idx, 'exit_price'] = current_price
                    data.at[idx, 'return'] = current_return
                    data.at[idx, 'return_abs'] = (
                        current_price - entry_price) * position_type
                    profit += (current_price - entry_price) * position_type
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    

            if not position_open:
                if current_signal == 1:  # BUY signal → LONG
                    tp_price, sl_price = Decimal(
                        current_price + current_price * tp), Decimal(current_price - current_price * sl)
                    data.at[idx, 'position'] = 1
                    data.at[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = 1
                elif current_signal == -1:  # SELL signal → SHORT
                    tp_price, sl_price = Decimal(
                        current_price - current_price * tp), Decimal(current_price + current_price * sl)
                    data.at[idx, 'position'] = -1
                    data.at[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = -1

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
            data.at[last_idx, 'return_abs'] = final_return * entry_price
            profit += final_return * entry_price

        number_of_positions = sum(data['position'] != 0)
        # print(f"Total Profit from Test: {profit:.2f} Number of Positions Taken: {number_of_positions}")

        return data



