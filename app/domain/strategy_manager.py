
from uuid import UUID
import numpy as np
import pandas as pd
import talib as ta
from typing import Any, Optional
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
from scipy.stats import skew

from app.domain import *
from app.domain.operations.indicator_factory import IndicatorFactory
from app.domain.position_manager import PositionManager

class StrategyManager:
    def __init__(self, strategy_model: StrategySettingsModel, df: pd.DataFrame):       
        self.strategy_model : StrategySettingsModel = strategy_model
        self.df: pd.DataFrame = df
        
        
    def test_single_strategy(self, store_result=True) -> Any:

        self.create_indicator()  
        
        if(store_result):   
            self.df.to_csv(f"csv/ind_{self.strategy_model.strategy_type.name.lower()}.csv", index=False)
         
        result =  self.run_strategy_test(store_result)
        print(f"Total Profit: {result.profit:.2f} USD")
      
        return result
    
    def future_return_strategy(self) -> None:   
        
        df = self.df.copy()
        df['T'] = pd.to_datetime(self.df['T']) 
        df = self.create_indicators(df)  
        # --------------------------
        # 1) Target erstellen
        # --------------------------
        # Wähle Horizon: z.B. 60 Minuten (anpassen je nach Data-Freq)
        HORIZON_MIN = 60
        df = DataUtils.make_future_return(df.copy(), minutes=HORIZON_MIN)
        
        

        # Drop rows ohne Target
        target_col = f'future_return_{HORIZON_MIN}m'
        df = df.dropna(subset=[target_col])
        df.to_csv(f"csv/ai_{self.strategy_model.asset}.csv", index=False)
        print(f"csv/ai_{self.strategy_model.asset}.csv created")
       
        # --------------------------
        # 2) Features auswählen
        # --------------------------
        # alle _signal Spalten als Basisfeatures
        feature_cols = [c for c in df.columns if c.endswith('_signal')]
        if len(feature_cols) == 0:
            raise ValueError("Keine '*_signal' Spalten im df gefunden.")

        X = df[feature_cols].astype(float)
        y = df[target_col].astype(float)

        # Optionale Zusatzfeatures (Volatilität, momentum, z-score)
        # Rolling Volatility (z.B. 60 Perioden)
        
        # logret: Prozentuale Preisänderung (logarithmisch, von Periode zu Periode)
        # vol_60: Schwankungsbreite der letzten 60 Perioden (Volatilität)
        # momentum_60: Kursveränderung über die letzten 60 Perioden (Momentum)
        roll = 60
        df['logret'] = np.log(df['C']).diff()
        df['vol_60'] = df['logret'].rolling(roll).std()
        df['momentum_60'] = df['C'] / df['C'].shift(roll) - 1
        # füge hinzu (wenn verfügbar)
        for extra in ['vol_60', 'momentum_60']:
            if extra in df.columns:
                X[extra] = df[extra]


        # dropna in X
        mask = X.isnull().any(axis=1)
        X = X[~mask]
        y = y.loc[X.index]

        # --------------------------
        # 3) Chronologisches Train/Test (Walk-Forward)
        # --------------------------
        # Parameter Walk-forward
        n_splits = 5  # je nach Datenmenge anpassen
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = []
        feature_importance = pd.DataFrame(index=feature_cols + [c for c in X.columns if c not in feature_cols])

        split_idx = 0
        for train_idx, test_idx in tscv.split(X):
            split_idx += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Standardize features (fit nur auf Train)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train) # nur hier fitten
            X_test_s = scaler.transform(X_test) # hier nur transformieren

            # LightGBM Regressor (profi-default)
            lgb_train = lgb.Dataset(X_train_s, label=y_train) # 
            lgb_eval = lgb.Dataset(X_test_s, label=y_test, reference=lgb_train)

            params = {
                'objective': 'regression',  # Regressionsproblem
                'metric': 'rmse', # Root Mean Squared Error
                'learning_rate': 0.05, # kleiner Wert für stabileres Lernen
                'num_leaves': 31, # Komplexität des Baums           
                'min_data_in_leaf': 50, # Vermeidung von Overfitting
                'feature_fraction': 0.8, # zufällige Auswahl von Features
                'bagging_fraction': 0.8, # zufällige Auswahl von Daten
                'bagging_freq': 1, # Häufigkeit des Bagging 
                'verbose': -1, # Keine Ausgabe während Training
                'seed': 42 # Zufallszahl-Generator für reproduzierbare Ergebnisse
            }

            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=['train', 'eval'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)  # 0 = no output
                ]
            )

            # Vorhersage
            pred = gbm.predict(X_test_s, num_iteration=gbm.best_iteration)
            # Baseline Metrik
            r2 = r2_score(y_test, pred)
            # Wir wollen ein Trading-Signal erzeugen: thresholding
            # z.B. nur handeln, wenn predicted_return > pct_threshold, oder < -pct_threshold
            pct_threshold = X['momentum_60'].std() * 0.5  # heuristisch; später optimieren
            pred_arr = np.array(pred).flatten() if not isinstance(pred, np.ndarray) else pred
            positions = np.zeros_like(pred_arr)
            positions[pred_arr > pct_threshold] = 1.0    # long
            positions[pred_arr < -pct_threshold] = -1.0  # short

            # PnL (vereinfachte: return * position)
            pnl = pd.Series(positions * y_test.values)
            cumulative = (1 + pnl).cumprod()

            # Metriken
            ann_sharpe = DataUtils.sharpe_ratio(pnl)  # ann. Sharpe (approx)
            mdd = DataUtils.max_drawdown(cumulative)
            hit_rate = ( (positions == 1) & (y_test.values > 0) ).sum() / max(1, (positions == 1).sum()) if (positions == 1).sum() > 0 else np.nan

            results.append({
                'split': split_idx,
                'r2': r2,
                'sharpe': ann_sharpe,
                'max_dd': mdd,
                'hit_rate_long': hit_rate,
                'n_trades': int((positions != 0).sum())
            })

            # Feature Importance sammeln
            imp = pd.Series(gbm.feature_importance(importance_type='gain'), index=X.columns)
            feature_importance[f'split_{split_idx}'] = imp

        # --------------------------
        # 4) Ergebnisse ausgeben
        # --------------------------
        res_df = pd.DataFrame(results)
        print("=== Walk-forward Ergebnisse ===")
        print(res_df)
        print("\nDurchschnittswerte:")
        print(res_df.mean(numeric_only=True))

        # durchschnittliche Feature-Importance
        fi_mean = feature_importance.mean(axis=1).sort_values(ascending=False)
        print("\nTop Features (durchschnittliche Importance):")
        print(fi_mean.head(20))
        
        res_df.to_csv(f"csv/result.csv", index=False)
    
    def create_ai_strategy(self) -> Any:
        
        # Annahme: df hat eine Spalte 'timestamp' mit datetime-Objekten
        self.df['T'] = pd.to_datetime(self.df['T'])
        self.df['date'] = self.df['T'].dt.date  # Extrahiere das Datum
        unique_days = self.df['date'].unique()  # Liste aller Handelstage
        
        for day in unique_days:
            day_data = self.df[self.df['date'] == day].copy()
            
            day_data = self.create_indicators(day_data)   
            cols = day_data.columns
            
            # Überschreibe die ursprünglichen Daten
            self.df.loc[self.df['date'] == day, cols] = day_data[cols]

        # self.calculate_trades()

        # self.create_indicators()
        self.df.to_csv(f"csv/ind_ai_{self.strategy_model.asset}.csv", index=False)

        result =  self.run_backtest()
        print(f"Total Profit: {result.profit:.2f} USD")
        print(f"Number of Trades: {result.number_of_positions}")


        print("ready")
      
        return
   
    def calculate_trades(self,df: pd.DataFrame) -> pd.DataFrame:
        
        tp_percent = 0.003
        sl_percent = 0.002
        
        # Calculate TP/SL levels for each row
        df['tp'] = df['C'] * (1 + tp_percent)  # Take Profit Level
        df['sl'] = df['C'] * (1 - sl_percent)  # Stop Loss Level
        
        # Initialize hit columns
        df['hit_long'] = SignalEnum.HOLD.value
        df['hit_short'] = SignalEnum.HOLD.value        
        
        offset = 600        
        
        df['max'] = df['C'].shift(-offset).rolling(window=offset).max()
        df['min'] = df['C'].shift(-offset).rolling(window=offset).min()
        
        df['hit_long'] = df['max'] > df['tp']
        df['hit_short'] = df['min'] < df['sl']
           
        return df
  
    def create_indicator(self) -> None:

        indicator = IndicatorFactory.create_indicator(self.strategy_model.strategy_type)
        self.df = indicator.calculate_signals(self.df, self.strategy_model.strategy_params)
        return

    def create_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = self.calculate_trades(df)

        indicator = IndicatorFactory.create_indicator(IndicatorEnum.SMA)
        df = indicator.calculate_signals(df, '{"short_ma": 20, "long_ma": 70}')

        indicator = IndicatorFactory.create_indicator(IndicatorEnum.EMA)
        df = indicator.calculate_signals(df, '{"short_ma": 20, "long_ma": 70}')

        indicator = IndicatorFactory.create_indicator(IndicatorEnum.WMA)
        df = indicator.calculate_signals(df, '{"short_ma": 20, "long_ma": 70}')

        indicator = IndicatorFactory.create_indicator(IndicatorEnum.TEMA)
        df = indicator.calculate_signals(df, '{"short_ma": 20, "long_ma": 70}')

        indicator = IndicatorFactory.create_indicator(IndicatorEnum.RSI)
        df = indicator.calculate_signals(df, '{"period": 14, "overbought": 70, "oversold": 30}')
        
        indicator = IndicatorFactory.create_indicator(IndicatorEnum.DONCHIAN)
        df = indicator.calculate_signals(df, '{"window": 60}')

        # df['signal_sum'] = df[[col for col in df.columns if col.endswith('_signal')]].sum(axis=1)
        # df['signal'] = np.where(df['signal_sum'] > 0, SignalEnum.BUY.value,
        #                       np.where(df['signal_sum'] < 0, SignalEnum.SELL.value, SignalEnum.HOLD.value))

        # df['match_with_hit_long'] = np.where((df['signal'] == SignalEnum.BUY.value) & (df['hit_long'] == 1), 1, 0)
        # df['match_with_hit_short'] = np.where((df['signal'] == SignalEnum.SELL.value) & (df['hit_short'] == 1), 1, 0)

        return df
    
    def run_backtest(self) -> Any:
        self.df['T'] = pd.to_datetime(self.df['T']) 
        quantity = self.strategy_model.quantity      
        take_profit_pct = self.strategy_model.take_profit_pct
        stop_loss_pct = self.strategy_model.stop_loss_pct

        pm = PositionManager.create_with_test_params(
            asset=self.strategy_model.asset,
            quantity=quantity,
            strategy_type=self.strategy_model.strategy_type,
            close_positions_eod=self.strategy_model.close_positions_eod,
            strategy_params=self.strategy_model.strategy_params
        )
        initial_cash: Decimal =  Decimal('100000.0')
    
        # Backtest-Variablen
        position_id = UUID(int=0)
        initial_cash: Decimal =  Decimal('100000.0')  
        cash = initial_cash
        
        signal_col_id = f"{self.strategy_model.strategy_type.name.lower()}_signal"
        
     

        
        number_of_trades = 0
        number_of_short_signals = 0
        number_of_long_signals = 0
        
        signal = SignalEnum.HOLD.value

        for index, row in self.df.iterrows():
            
            # detect new day
            current_day = row['T'].date()
            ask = row["ask"]
            bid = row["bid"]
        
            stamp: datetime = row["T"]
            
            # if new day, reset position if close_positions_eod is True
            if pm.close_positions_eod and position_id != UUID(int=0) and current_day != last_day:
                position = pm.get_position(position_id)  
                
                if position is not None:
                    if(position.side == SideEnum.Buy):
                        pm.close_position(position_id, last_bid, last_stamp)
                    if(position.side == SideEnum.Sell):
                        pm.close_position(position_id, last_ask, last_stamp)

                position_id = UUID(int=0)
            
            signal = row['signal']
            
            # Close position on StopLoss/TakeProfit
            if(position_id != UUID(int=0)):
                position_id = pm.execute_sl_tp(position_id, bid, ask, stamp)
            else:
                # Long position
                if signal == SignalEnum.BUY.value:
                    position_id = pm.open_position(SideEnum.Buy, price=ask, tp=ask * (1 + take_profit_pct), sl=ask * (1 - stop_loss_pct), stamp=stamp)              

                # Short position
                if signal == SignalEnum.SELL.value:
                    position_id = pm.open_position(SideEnum.Sell, price=bid, tp=bid * (1 - take_profit_pct), sl=bid * (1 + stop_loss_pct), stamp=stamp)
              
                    
            last_day = row['T'].date()
            last_ask = row["ask"]
            last_bid = row["bid"] 
            last_stamp: datetime = row["T"]

        pos = pm.get_positions()
        
        pos_df = pd.DataFrame([
            {
                'Id': position_id,
                'side': position.side.name,
                'openPrice': position.price_open,   
                'closePrice': position.price_close,
                'take_profit': position.take_profit,
                'stop_loss': position.stop_loss,
                'opened': position.stamp_opened.strftime("%m.%d %H:%M") if position.stamp_opened else "",
                'closed': position.stamp_closed.strftime("%m.%d %H:%M") if position.stamp_closed else "",
                'minutes': (position.stamp_closed - position.stamp_opened).total_seconds() / 60 if position.stamp_closed and position.stamp_opened else 0,
                'profit_loss': position.profit_loss,
            }
            for position_id, position in pos.items()
        ])

        # Save to CSV
        pos_df.to_csv("csv/positions.csv", index=False)

        # Calculate total profit
        profit = pm.calculate_profit()     

        result = StrategyResultModel(
            strategy_id=self.strategy_model.id,
            strategy_type=self.strategy_model.strategy_type,
            asset=self.strategy_model.asset,
            quantity=self.strategy_model.quantity,
            take_profit_pct=self.strategy_model.take_profit_pct,
            stop_loss_pct=self.strategy_model.stop_loss_pct,
            strategy_params=self.strategy_model.strategy_params,
            number_of_positions=len(pos),
            profit=profit
        ) 
        
        print(f"Number of Trades: {number_of_trades}, Long Signals: {number_of_long_signals}, Short Signals: {number_of_short_signals}")

        return result
    
      
    def run_strategy_test(self, store_result=True) -> Any:
        self.df['T'] = pd.to_datetime(self.df['T']) 
        quantity = self.strategy_model.quantity      
        take_profit_pct = self.strategy_model.take_profit_pct
        stop_loss_pct = self.strategy_model.stop_loss_pct

        pm = PositionManager.create_with_test_params(
            asset=self.strategy_model.asset,
            quantity=quantity,
            strategy_type=self.strategy_model.strategy_type,
            close_positions_eod=self.strategy_model.close_positions_eod,
            strategy_params=self.strategy_model.strategy_params
        )
     
    
        # Backtest-Variablen
        position_id = UUID(int=0)

        signal_col_id = f"{self.strategy_model.strategy_type.name.lower()}_signal"

        
        signal = SignalEnum.HOLD.value

        for index, row in self.df.iterrows():
            
            # detect new day
            new_day = row['T'].date()
                  # order = row["Order"]
            ask = row["ask"]
            bid = row["bid"]
        
            stamp: datetime = row["T"]
            
            # if new day, reset position if close_positions_eod is True
            if pm.close_positions_eod and position_id != UUID(int=0) and new_day != current_day:
                position = pm.get_position(position_id)  
                if position is not None:
                    if(position.side == SideEnum.Buy):
                        pm.close_position(position_id, last_bid, last_stamp)
                    if(position.side == SideEnum.Sell):
                        pm.close_position(position_id, last_ask, last_stamp)

                position_id = UUID(int=0)

            signal = row[signal_col_id]

            # Close position on StopLoss/TakeProfit
            if(position_id != UUID(int=0)):
                position_id = pm.execute_sl_tp(position_id, bid, ask, stamp)
            else:
                # Long position
                if signal == SignalEnum.BUY.value and stamp.hour >= 9 and stamp.hour < 19:
                    position_id = pm.open_position(SideEnum.Buy, price=ask, tp=ask * (1 + take_profit_pct), sl=ask * (1 - stop_loss_pct), stamp=stamp)
             

                # Short position
                if signal == SignalEnum.SELL.value and stamp.hour >= 9 and stamp.hour < 19:
                    position_id = pm.open_position(SideEnum.Sell, price=bid, tp=bid * (1 - take_profit_pct), sl=bid * (1 + stop_loss_pct), stamp=stamp)
                
                    
            current_day = row['T'].date()
            last_ask = row["ask"]
            last_bid = row["bid"]

            last_stamp: datetime = row["T"]

        pos = pm.get_positions()
        
        pos_df = pd.DataFrame([
            {
                'Id': position_id,
                'side': position.side.name,
                'openPrice': position.price_open,   
                'closePrice': position.price_close,
                'take_profit': position.take_profit,
                'stop_loss': position.stop_loss,
                'opened': position.stamp_opened.strftime("%d.%m.%y %H:%M") if position.stamp_opened else "",
                'closed': position.stamp_closed.strftime("%d.%m.%y %H:%M") if position.stamp_closed else "",
                'minutes': (position.stamp_closed - position.stamp_opened).total_seconds() / 60 if position.stamp_closed and position.stamp_opened else 0,
                'profit_loss': position.profit_loss,
            }
            for position_id, position in pos.items()
        ])

        # Save to CSV
        if(store_result):
            pos_df.to_csv(f"csv/positions_{self.strategy_model.strategy_type.name.lower()}.csv", index=False)

        # Calculate total profit
        profit = pm.calculate_profit()     

        result = StrategyResultModel(
            strategy_id=self.strategy_model.id,
            strategy_type=self.strategy_model.strategy_type,
            asset=self.strategy_model.asset,
            quantity=self.strategy_model.quantity,
            take_profit_pct=self.strategy_model.take_profit_pct,
            stop_loss_pct=self.strategy_model.stop_loss_pct,
            strategy_params=self.strategy_model.strategy_params,
            number_of_positions=len(pos),
            profit=profit
        ) 
        
        print(f"Profit: {profit}, Number of Trades: {result.number_of_positions}")

        return result
    