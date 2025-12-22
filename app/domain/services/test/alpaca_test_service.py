import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score
import json
from app.infrastructure import alpaca_service
from app.domain.operations.lgb_model_factory import LGBModelFactory
from app.domain.operations.indicator_factory import IndicatorFactory
from app.domain.models.enums import IndicatorEnum
from app.domain.position_manager import PositionManager
from app.infrastructure.yahoo_service import YahooService
from app.domain.operations.data_utils import DataUtils

from decimal import Decimal
from uuid import UUID
from app.domain.models.enums import SideEnum

class AlpacaTestService:
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def test_lgb_model(self, settings: dict):
        print("Testing LGB Model with settings:")
        print(settings)
        
        df_filename = "csv/spy_alpaca_test.csv"

        indicator_models = IndicatorFactory.get_indicator_models_by_params(settings["indicators"])
   
        execution_params = settings["execution_params"]

        model_type = "regression"
        broker = execution_params["broker"]
        trading_period = execution_params["trading_period"]
        asset = execution_params["asset"]
        start_date = execution_params["start_date"]
        end_date = execution_params["end_date"]
        price_change_threshold = execution_params.get("price_change_threshold", 0.002)
        long_threshold = execution_params["long_threshold"]
        short_threshold = execution_params["short_threshold"]
        tp = execution_params["tp"]
        sl = execution_params["sl"]
        
        print("Train LGB Model with settings:")
        print(settings)
        
        match broker:
            case 'Yahoo':
                trading_period = '1d'  # Yahoo only supports daily data
                data = YahooService.load_yahoo_stock_data(asset)
            case 'Alpaca':
                data = alpaca_service.load_stock_data_from_redis(
                    asset, period=trading_period)

        data = data[['DT', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # Keep only rows where DT < '2020-01-01'
        # data = data[data['DT'] < '2026-01-01']
        
        print(f"Data loaded: {len(data)} rows")
   
        data = IndicatorFactory.extend_dataframe_with_indicators(
            data, indicator_models)
        data.to_csv(df_filename, index=False)

        data = data.dropna()
        print(f"Data loaded: {len(data)} rows")

        next_return = data['Close'].pct_change().shift(-1)
        data["NEXT"] = next_return
        data["NEXT_DAY"] = data['Close'].shift(-1)
        data["DIFF_NEXT_DAY"] = data['Close'].shift(-1) - data['Close']
        
        if (model_type == 'classification'):
            data['target'] = np.where(
                next_return > price_change_threshold, 1,
                np.where(next_return < -price_change_threshold, -1, 0)
            )
        elif (model_type == 'regression'):
            data['target'] = next_return
        # load model
        model, features = LGBModelFactory.load_lgb_model(asset, trading_period)

        print(f"Model: loaded {model}")
        print(f"Features: {features}")
        print(data.head())
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise KeyError(f"Missing features in DataFrame: {missing_features}")
        
        if (model_type == 'classification'):
            
            importance_dict = dict(zip(features, model.feature_importances_))
            for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {imp:.4f}")

            y_pred_proba = model.predict_proba(data[features])
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
        
        elif (model_type == 'regression'):
            importance_dict = dict(zip(features, model.feature_importances_))
            for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {imp:.4f}")

            data['predicted_return'] = model.predict(data[features])
            
            # print data["target"] max and min and mean values
            print(f"Max target value: {data['target'].max()}")
            print(f"Min target value: {data['target'].min()}")
            print(f"Mean target value: {data['target'].mean()}")
            
            # print data["predicted_return"] max  min and mean  values
            print(f"Max predicted_return value: {data['predicted_return'].max()}")
            print(f"Min predicted_return value: {data['predicted_return'].min()}")
            print(f"Mean predicted_return value: {data['predicted_return'].mean()}")
            
            
            
            data['signal'] = 0
            data.loc[data['predicted_return'] > long_threshold, 'signal'] = 1
            data.loc[data['predicted_return'] < -short_threshold, 'signal'] = -1
            
        data = self.backtest_fees(data, tp, sl)
        data.to_csv(df_filename, index=False)

        daily_returns = pd.Series(data['return'])
        cumulative_product = np.prod([1 + r for r in daily_returns])
        total_return = cumulative_product - 1
        total_return_percentage = total_return * 100
        final_equity = data['equity'].iloc[-1] if 'equity' in data.columns else None

        sharpe_ratio = DataUtils.calculate_sharpe(daily_returns)
        max_drawdown = DataUtils.max_drawdown(daily_returns)
        print(f"ABS Return: {data['return_abs'].sum()} ")
        print(f"Total Return: {total_return_percentage:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.5f}")
        print(f"Max Drawdown: {max_drawdown:.5f}")
        print(f"Final Equity: {final_equity:.2f}")
        if 'DT' in data.columns and 'return_abs' in data.columns:
            data['year'] = pd.to_datetime(data['DT']).dt.year
            yearly_profit = data.groupby('year')['return_abs'].sum()

            # sum all entries in position column not equal to 0
            yearly_trades = data.groupby(
                'year')['position'].apply(lambda x: (x != 0).sum())

            print("Yearly Profit:")
            for year, prof in yearly_profit.items():
                print(
                    f"  {year}: Profit {prof:.2f} Number of Trades: {yearly_trades.get(year, 0)}")

    def backtest(self, data, tp,sl):
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
                'minutes': (position.stamp_closed - position.stamp_opened).total_seconds() / 60 if position.stamp_closed and position.stamp_opened else 0,
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

    def backtest_fees(
        self,
        data,
        tp: float = 0.005,
        sl: float = 0.004,
        spread_per_trade: float = 0.005,
        overnight_fee_rate: float = 0.00005,
        asset: str = "SPY"
    ):

        equity = 10000.0

        # --- Setup ---
        data['position'] = 0
        data['entry_price'] = np.nan
        data['exit_price'] = np.nan
        data['return'] = 0.0
        data['return_abs'] = 0.0
        data['equity'] = np.nan

        pm = PositionManager.create_with_test_params(
            asset=asset,
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
        overnight_holds = 0
        entry_idx = None
        entry_date = None

        # --- Haupt-Loop ---
        for idx, row in data.iterrows():
            current_price = row['Close']
            current_signal = row['signal']
            current_timestamp = row['DT']

            if position_open:
                # --- Overnight Gebühren zählen ---
                prev_date = pd.to_datetime(
                    entry_date).date() if entry_date else None
                curr_date = pd.to_datetime(row['DT']).date()
                if prev_date and curr_date > prev_date:
                    overnight_holds += 1
                    entry_date = row['DT']

                if position_type == 1:
                    current_return = (
                        current_price - entry_price) / entry_price
                else:
                    current_return = (
                        entry_price - current_price) / entry_price

                # --- Take-Profit / Stop-Loss ---
                if current_return >= tp or current_return <= -sl:
                    exit_price = (
                        entry_price * (1 + tp) if current_return >= tp and position_type == 1 else
                        entry_price * (1 - tp) if current_return >= tp and position_type == -1 else
                        entry_price * (1 - sl) if position_type == 1 else
                        entry_price * (1 + sl)
                    )

                    reason = "TP" if current_return >= tp else "SL"
                    pm.close_position(position_id, Decimal(
                        str(exit_price)), current_timestamp, reason)

                    # --- Gebühren anwenden ---
                    total_fee = 2 * spread_per_trade
                    overnight_fee = overnight_holds * overnight_fee_rate * entry_price

                    p = (exit_price - entry_price) * \
                        position_type - total_fee - overnight_fee
                    data.loc[idx, 'exit_price'] = exit_price
                    data.loc[idx, 'return'] = p / equity
                    data.loc[idx, 'return_abs'] = p
                    profit += p
                    equity += p
                    data.loc[idx, 'equity'] = equity

                    # Reset
                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    overnight_holds = 0
                    entry_idx = None
                    entry_date = None
                    continue

                # --- Signalwechsel (Reverse) ---
                if (position_type == 1 and current_signal == -1) or (position_type == -1 and current_signal == 1):
                    total_fee = 2 * spread_per_trade
                    overnight_fee = overnight_holds * overnight_fee_rate * entry_price

                    data.at[idx, 'exit_price'] = current_price
                    p = (current_price - entry_price) * \
                        position_type - total_fee - overnight_fee
                    data.at[idx, 'return'] = p / equity
                    data.at[idx, 'return_abs'] = p
                    profit += p
                    equity += p
                    data.at[idx, 'equity'] = equity

                    pm.close_position(position_id, Decimal(
                        str(current_price)), current_timestamp, "Reverse")

                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    overnight_holds = 0
                    entry_idx = None
                    entry_date = None

            # --- Neue Position eröffnen ---
            if not position_open:
                if current_signal == 1:
                    tp_price = Decimal(current_price * (1 + tp))
                    sl_price = Decimal(current_price * (1 - sl))
                    data.loc[idx, 'position'] = 1
                    data.loc[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = 1
                    position_id = pm.open_position(
                        SideEnum.Buy,
                        Decimal(str(current_price)),
                        tp_price,
                        sl_price,
                        current_timestamp
                    )
                    overnight_holds = 0
                    entry_idx = idx
                    entry_date = row['DT']

                elif current_signal == -1:
                    tp_price = Decimal(current_price * (1 - tp))
                    sl_price = Decimal(current_price * (1 + sl))
                    data.loc[idx, 'position'] = -1
                    data.loc[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = -1
                    position_id = pm.open_position(
                        SideEnum.Sell,
                        Decimal(str(current_price)),
                        tp_price,
                        sl_price,
                        current_timestamp
                    )
                    overnight_holds = 0
                    entry_idx = idx
                    entry_date = row['DT']

        # --- Letzte offene Position schließen ---
        if position_open:
            last_idx = data.index[-1]
            last_price = data.iloc[-1]['Close']
            current_timestamp = data.iloc[-1]['DT']

            prev_date = pd.to_datetime(entry_date).date()
            curr_date = pd.to_datetime(current_timestamp).date()
            overnight_holds += (curr_date - prev_date).days

            total_fee = 2 * spread_per_trade
            overnight_fee = overnight_holds * overnight_fee_rate * entry_price
            p = (last_price - entry_price) * \
                position_type - total_fee - overnight_fee

            data.loc[last_idx, 'exit_price'] = last_price
            data.loc[last_idx, 'return'] = p / equity
            data.loc[last_idx, 'return_abs'] = p
            profit += p
            pm.close_position(position_id, Decimal(
                str(last_price)), current_timestamp)
            equity += p
            data.loc[last_idx, 'equity'] = equity

        # --- Summary ---
        print(f"Asset: {asset}")
        print(f"Total Profit: {profit:.2f}")
        print(
            f"Spread per trade: {spread_per_trade:.4f} USD, Overnight fee: {overnight_fee_rate:.6f}")
        print("################################################")
        pos = pm.get_positions()
        if (len(pos) == 0):
            return data
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
                'profit_loss': float(position.profit_loss) if getattr(position, 'profit_loss', None) is not None else None
            }
            for position_id, position in pos.items()
        ])
        pos_df.to_csv("csv/positions.csv")
        profit = pm.calculate_profit()
        number_of_trades = len(pos_df)
        number_of_long_signals = len(pos_df[pos_df['side'] == 'Buy'])
        number_of_short_signals = len(pos_df[pos_df['side'] == 'Sell'])
        number_of_lost_positions = len(pos_df[pos_df['profit_loss'] < 0])
        number_of_won_positions = len(pos_df[pos_df['profit_loss'] > 0])
        print(f"PM ## Profit: {profit:.2f}   Number of Trades: {number_of_trades}, Long Signals: {number_of_long_signals}, Short Signals: {number_of_short_signals}, Won Positions: {number_of_won_positions}, Lost Positions: {number_of_lost_positions}")
        print(
            f"Win Rate: {number_of_won_positions / number_of_trades:.2%}" if number_of_trades > 0 else "Win Rate: N/A")

        return data

    def test_lgb_with_alpaca_feed(self, settings: dict):
        print("Testing LGB Model with settings:")
        print(settings)

        df_filename = "csv/spy_df_20251117h.csv"

        indicator_models = IndicatorFactory.get_indicator_models_by_params(settings["indicators"])    
        execution_params = settings["execution_params"]    
        asset = execution_params["asset"]
        start_date = execution_params["start_date"]
        end_date = execution_params["end_date"]
        price_change_threshold = execution_params.get(
            "price_change_threshold", 0.002)
        long_threshold = execution_params["long_threshold"]
        short_threshold = execution_params["short_threshold"]
        tp = execution_params["tp"]
        sl = execution_params["sl"]
        
         # load model
        model, features = LGBModelFactory.load_lgb_model(asset, "20251119")
        
        print("features:", features )

        # load data
        data = alpaca_service.load_stock_data_from_redis(asset, period="1h")
        data = data[['DT', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        print(f"Data loaded: {len(data)} rows")

        data = IndicatorFactory.extend_dataframe_with_indicators(
            data, indicator_models)  
        data = data.dropna()
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise KeyError(
                
                f"Missing features in DataFrame: {missing_features}")
        data.to_csv(df_filename, index=False) 

        if (model_type == 'classification'):

            importance_dict = dict(zip(features, model.feature_importances_))
            for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {imp:.4f}")

            y_pred_proba = model.predict_proba(data[features])
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

        elif (model_type == 'regression'):
            importance_dict = dict(zip(features, model.feature_importances_))
            for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {imp:.4f}")

            data['predicted_return'] = model.predict(data[features])


            print(
                f"Max predicted_return value: {data['predicted_return'].max()}")
            print(
                f"Min predicted_return value: {data['predicted_return'].min()}")
            print(
                f"Mean predicted_return value: {data['predicted_return'].mean()}")

            data['signal'] = 0
            data.loc[data['predicted_return'] > long_threshold, 'signal'] = 1
            data.loc[data['predicted_return'] < -short_threshold, 'signal'] = -1

        data = self.backtest_alpaca_feed(data, tp, sl)
        data.to_csv(df_filename, index=False)

        daily_returns = pd.Series(data['return'])
        cumulative_product = np.prod([1 + r for r in daily_returns])
        total_return = cumulative_product - 1
        total_return_percentage = total_return * 100
        final_equity = data['equity'].iloc[-1] if 'equity' in data.columns else None

        sharpe_ratio = DataUtils.calculate_sharpe(daily_returns)
        max_drawdown = DataUtils.max_drawdown(daily_returns)
        print(f"ABS Return: {data['return_abs'].sum()} ")
        print(f"Total Return: {total_return_percentage:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.5f}")
        print(f"Max Drawdown: {max_drawdown:.5f}")
        print(f"Final Equity: {final_equity:.2f}" if final_equity is not None else "Final Equity: N/A")

        
    def backtest_alpaca_feed(
        self,
        data,
        tp: float = 0.005,
        sl: float = 0.004,
        spread_per_trade: float = 0.005,
        overnight_fee_rate: float = 0.00005,
        asset: str = "SPY"
    ):

        equity = 10000.0

        # --- Setup ---
        data['position'] = 0
        data['entry_price'] = np.nan
        data['exit_price'] = np.nan
        data['return'] = 0.0
        data['return_abs'] = 0.0
        data['equity'] = np.nan

        pm = PositionManager.create_with_test_params(
            asset=asset,
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
        overnight_holds = 0
        entry_idx = None
        entry_date = None
        
        df_rates = alpaca_service.load_stock_data_from_redis(asset, period="1m")
        df_rates = df_rates[['DT', 'Open', 'High',
                             'Low', 'Close', 'Volume']].dropna()

        # --- Haupt-Loop ---
        for idx, row in data.iterrows():
            current_price = row['Close']
            current_signal = row['signal']
            current_timestamp = row['DT'] 
            

                
           
            

            if position_open:
                
                # check if sl or tp hit in df_rates between current_timestamp and next timestamp
                
         
                current_pos = data.index.get_loc(idx)


                if current_pos + 1 < len(data):
                    next_idx = data.index[current_pos + 1]
                    next_timestamp = pd.to_datetime(data.loc[next_idx, 'DT'])
                else:
                    next_timestamp = None
                        
                # get df_rates between current_timestamp and next_timestamp
                if next_timestamp:
                    df_rate_slice = df_rates[(df_rates['DT'] >= current_timestamp) & (
                        df_rates['DT'] < next_timestamp)]
                
                    for _, rate_row in df_rate_slice.iterrows():
                        current_price = rate_row['Close']
                        current_timestamp = rate_row['DT']
                        
                        # --- Overnight Gebühren zählen ---
                        prev_date = pd.to_datetime(
                            entry_date).date() if entry_date else None
                        curr_date = pd.to_datetime(rate_row['DT']).date()
                        if prev_date and curr_date > prev_date:
                            overnight_holds += 1
                            entry_date = rate_row['DT']

                        if position_type == 1:
                            current_return = (
                                current_price - entry_price) / entry_price
                        else:
                            current_return = (
                                entry_price - current_price) / entry_price

                        # --- Take-Profit / Stop-Loss ---
                        if current_return >= tp or current_return <= -sl:
                            exit_price = (
                                entry_price * (1 + tp) if current_return >= tp and position_type == 1 else
                                entry_price * (1 - tp) if current_return >= tp and position_type == -1 else
                                entry_price * (1 - sl) if position_type == 1 else
                                entry_price * (1 + sl)
                            )

                            reason = "TP" if current_return >= tp else "SL"
                            pm.close_position(position_id, Decimal(
                                str(exit_price)), current_timestamp, reason)

                            # --- Gebühren anwenden ---
                            total_fee = 2 * spread_per_trade
                            overnight_fee = overnight_holds * overnight_fee_rate * entry_price

                            p = (exit_price - entry_price) * \
                                position_type - total_fee - overnight_fee
                            data.loc[idx, 'exit_price'] = exit_price
                            data.loc[idx, 'return'] = p / equity
                            data.loc[idx, 'return_abs'] = p
                            profit += p
                            equity += p
                            data.loc[idx, 'equity'] = equity

                            # Reset
                            position_open = False
                            entry_price = 0.0
                            position_type = 0
                            overnight_holds = 0
                            entry_idx = None
                            entry_date = None
                            break  # exit the for loop over rate_row

                # --- Signalwechsel (Reverse) ---
                if (position_type == 1 and current_signal == -1) or (position_type == -1 and current_signal == 1):
                    total_fee = 2 * spread_per_trade
                    overnight_fee = overnight_holds * overnight_fee_rate * entry_price

                    data.at[idx, 'exit_price'] = current_price
                    p = (current_price - entry_price) * \
                        position_type - total_fee - overnight_fee
                    data.at[idx, 'return'] = p / equity
                    data.at[idx, 'return_abs'] = p
                    profit += p
                    equity += p
                    data.at[idx, 'equity'] = equity

                    pm.close_position(position_id, Decimal(
                        str(current_price)), current_timestamp, "Reverse")

                    position_open = False
                    entry_price = 0.0
                    position_type = 0
                    overnight_holds = 0
                    entry_idx = None
                    entry_date = None

            # --- Neue Position eröffnen ---
            if not position_open:
                if current_signal == 1:
                    tp_price = Decimal(current_price * (1 + tp))
                    sl_price = Decimal(current_price * (1 - sl))
                    data.loc[idx, 'position'] = 1
                    data.loc[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = 1
                    position_id = pm.open_position(
                        SideEnum.Buy,
                        Decimal(str(current_price)),
                        tp_price,
                        sl_price,
                        current_timestamp
                    )
                    overnight_holds = 0
                    entry_idx = idx
                    entry_date = row['DT']

                elif current_signal == -1:
                    tp_price = Decimal(current_price * (1 - tp))
                    sl_price = Decimal(current_price * (1 + sl))
                    data.loc[idx, 'position'] = -1
                    data.loc[idx, 'entry_price'] = current_price
                    position_open = True
                    entry_price = current_price
                    position_type = -1
                    position_id = pm.open_position(
                        SideEnum.Sell,
                        Decimal(str(current_price)),
                        tp_price,
                        sl_price,
                        current_timestamp
                    )
                    overnight_holds = 0
                    entry_idx = idx
                    entry_date = row['DT']

        # --- Letzte offene Position schließen ---
        if position_open:
            last_idx = data.index[-1]
            last_price = data.iloc[-1]['Close']
            current_timestamp = data.iloc[-1]['DT']

            prev_date = pd.to_datetime(entry_date).date()
            curr_date = pd.to_datetime(current_timestamp).date()
            overnight_holds += (curr_date - prev_date).days

            total_fee = 2 * spread_per_trade
            overnight_fee = overnight_holds * overnight_fee_rate * entry_price
            p = (last_price - entry_price) * \
                position_type - total_fee - overnight_fee

            data.loc[last_idx, 'exit_price'] = last_price
            data.loc[last_idx, 'return'] = p / equity
            data.loc[last_idx, 'return_abs'] = p
            profit += p
            pm.close_position(position_id, Decimal(
                str(last_price)), current_timestamp)
            equity += p
            data.loc[last_idx, 'equity'] = equity

        # --- Summary ---
        print(f"Asset: {asset}")
        print(f"Total Profit: {profit:.2f}")
        print(
            f"Spread per trade: {spread_per_trade:.4f} USD, Overnight fee: {overnight_fee_rate:.6f}")
        print("################################################")
        pos = pm.get_positions()
        if (len(pos) == 0):
            return data
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
                'profit_loss': float(position.profit_loss) if getattr(position, 'profit_loss', None) is not None else None
            }
            for position_id, position in pos.items()
        ])
        pos_df.to_csv("csv/positions.csv")
        profit = pm.calculate_profit()
        number_of_trades = len(pos_df)
        number_of_long_signals = len(pos_df[pos_df['side'] == 'Buy'])
        number_of_short_signals = len(pos_df[pos_df['side'] == 'Sell'])
        number_of_lost_positions = len(pos_df[pos_df['profit_loss'] < 0])
        number_of_won_positions = len(pos_df[pos_df['profit_loss'] > 0])
        print(f"PM ## Profit: {profit:.2f}   Number of Trades: {number_of_trades}, Long Signals: {number_of_long_signals}, Short Signals: {number_of_short_signals}, Won Positions: {number_of_won_positions}, Lost Positions: {number_of_lost_positions}")
        print(
            f"Win Rate: {number_of_won_positions / number_of_trades:.2%}" if number_of_trades > 0 else "Win Rate: N/A")

        return data
