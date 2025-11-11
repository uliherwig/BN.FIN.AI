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

from decimal import Decimal
from uuid import UUID
from app.domain.models.enums import SideEnum

class AlpacaTestService:
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def test_lgb_model(self, settings: dict):
        print("Testing LGB Model with settings:")
        print(settings)

        indicators = settings["indicators"]
        lgb_model = settings["lgb_model"]
        execution_params = settings["execution_params"]

        asset = execution_params.get("asset", "SPY")
        start_date = execution_params.get("start_date", "2010-01-01")
        end_date = execution_params.get("end_date", "2025-10-30")
        long_threshold = execution_params.get("long_threshold", 0.6)
        short_threshold = execution_params.get("short_threshold", 0.6)
        tp = execution_params.get("tp", 0.01)
        sl = execution_params.get("sl", 0.005)
        
        
        data = YahooService.load_yahoo_stock_data(asset)
        data = data[['DT', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        # Keep only rows where DT < '2020-01-01'
        data = data[data['DT'] < '2026-01-01']

        # # load data from redis
        # data = alpaca_service.load_stock_data_from_redis(asset, period="1d")
        # data = data[['DT', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        # add indicators using provided parameters
        data = self.add_technical_indicators(data, indicators)
        data = data.dropna()

        # load model
        model, features = LGBModelFactory.load_lgb_model(asset)

        print(f"Model: loaded {model}")
        print(f"Features: {features}")
        print(data.head())
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise KeyError(f"Missing features in DataFrame: {missing_features}")

        y_pred_proba = model.predict_proba(data[features])
        prob_down = y_pred_proba[:, 0]
        prob_up = y_pred_proba[:, 1]

        data['signal'] = 0
        data.loc[prob_up > long_threshold, 'signal'] = 1
        data.loc[prob_down > short_threshold, 'signal'] = -1

        result_df = self.backtest(data, tp=tp, sl=sl)
        
    def add_technical_indicators(self, df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
        # Use provided indicator parameters
        df = IndicatorFactory.create_indicator(IndicatorEnum.SMA).create_features(
            df, json.dumps({"short_ma": indicators["SMA_short"], "long_ma": indicators["SMA_long"]}))
        df = IndicatorFactory.create_indicator(IndicatorEnum.EMA).create_features(
            df, json.dumps({"short_ma": indicators["EMA_short"], "long_ma": indicators["EMA_long"]}))
        df = IndicatorFactory.create_indicator(IndicatorEnum.MACD).create_features(
            df, json.dumps({"fast": indicators["MACD_fast"], "slow": indicators["MACD_slow"], "signal": indicators["MACD_signal"]}))
        df = IndicatorFactory.create_indicator(IndicatorEnum.RSI).create_features(
            df, json.dumps({"period": indicators["RSI_period"]}))
        df = IndicatorFactory.create_indicator(IndicatorEnum.ATR).create_features(
            df, json.dumps({"period": indicators["ATR_period"]}))
        df = IndicatorFactory.create_indicator(IndicatorEnum.VOLA).create_features(
            df, json.dumps({"period": indicators["VOL_period"]}))
        df = df.dropna()
        return df

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



