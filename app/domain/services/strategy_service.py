import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Any
from datetime import datetime, timedelta, timezone
from app.domain.models.strategies.breakout_model import BreakoutModel
from app.domain.operations.indicators.ema_indicator import EmaIndicator
from app.domain.operations.indicators.tema_indicator import TemaIndicator
from app.domain.operations.indicators.wma_indicator import WmaIndicator
from app.infrastructure.redis_service import RedisService, get_redis_service
from app.infrastructure import alpaca_service
from sklearn.model_selection import ParameterGrid
from celery import Celery
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from app.domain import *
from app.domain.strategy_manager import StrategyManager
import json



# Celery App Configuration
celery_app = Celery('strategy_tasks')
celery_app.config_from_object({
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/0',
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
})

@celery_app.task
def test_single_strategy(strategy_settings: StrategySettingsModel) -> None:   
    
    try:
        # Synchroner Aufruf - kein asyncio needed!
        df = alpaca_service.read_bars(strategy_settings.asset, strategy_settings.start_date, strategy_settings.end_date)
        
        sm = StrategyManager(strategy_settings, df)
        sm.test_single_strategy()
        
        # Return als dict
        return
        
    except Exception as e:
        print(f"Error in strategy test: {e}")
        raise   


@celery_app.task
def optimize_strategy(strategy_settings: StrategySettingsModel) -> Any:    

    param_grid = {}
    
    match strategy_settings.strategy_type:
        case StrategyLibEnum.SMA:
            params = SmaModel.model_validate_json(strategy_settings.strategy_params)
             
            param_grid = {
                'short_ma': range(params.short_ma, params.short_ma + 100, 10),
                'long_ma': range(params.long_ma, params.long_ma + 100, 10)
            }
        case StrategyLibEnum.EMA:
            params = EmaModel.model_validate_json(strategy_settings.strategy_params)
             
            param_grid = {
                'short_ma': range(params.short_ma, params.short_ma + 100, 10),
                'long_ma': range(params.long_ma, params.long_ma + 100, 10)
            }
        case StrategyLibEnum.WMA:
            params = WmaModel.model_validate_json(strategy_settings.strategy_params)

            param_grid = {
                'short_ma': range(params.short_ma, params.short_ma + 100, 10),
                'long_ma': range(params.long_ma, params.long_ma + 100, 10)
            }
        case StrategyLibEnum.TEMA:
            params = TemaModel.model_validate_json(strategy_settings.strategy_params)

            param_grid = {
                'short_ma': range(params.short_ma, params.short_ma + 100, 10),
                'long_ma': range(params.long_ma, params.long_ma + 100, 10)
            }
        case StrategyLibEnum.DONCHIAN:
            params = DonchianModel.model_validate_json(strategy_settings.strategy_params)

            param_grid = {
                'window': range(params.window, params.window + 100, 10)
            }
        case StrategyLibEnum.MACD:
            params = MacdModel.model_validate_json(strategy_settings.strategy_params)
             
            param_grid = {
                'fast': range(params.fast, params.fast + 50, 5),
                'slow': range(params.slow, params.slow + 100, 10),
                'signal': range(params.signal, params.signal + 20, 5)
            }
     
        case StrategyLibEnum.RSI:
            params = RsiModel.model_validate_json(strategy_settings.strategy_params)
            
            param_grid = {
                'period': range(params.period, params.period + 100, 10),
                'overbought': range(params.overbought, params.overbought + 20, 10),
                'oversold': range(params.oversold, params.oversold + 10, 10)
            }
        #    case StrategyLibEnum.DONCHIAN:
        #         return DonchianIndicator()
        case StrategyLibEnum.BREAKOUT:
            params = BreakoutModel.model_validate_json(strategy_settings.strategy_params)
             
            param_grid = {
                'breakout_period': range(params.breakout_period, params.breakout_period + 100, 10)
            }
        case _:
            raise ValueError(f"Unsupported strategy type: {strategy_settings.strategy_type}")


        
    best_profit = float('-inf')
    best_params = None
    iteration = 0
    
    print(f"Starte Optimierung für Strategie {strategy_settings.strategy_type} mit Parametern: {param_grid} : {len(list(ParameterGrid(param_grid)))} Kombinationen")
    
    # Iteriere über alle Kombinationen
    for params in ParameterGrid(param_grid):
        
        
        match strategy_settings.strategy_type:
            case StrategyLibEnum.SMA | StrategyLibEnum.EMA | StrategyLibEnum.WMA | StrategyLibEnum.TEMA:
                if params['short_ma'] >= params['long_ma']:
                    continue  # short_ma muss kleiner als long_ma sein
            case StrategyLibEnum.RSI:
                if params['oversold'] >= params['overbought']:
                    continue  # oversold muss kleiner als overbought sein

        strategy_settings.strategy_params = json.dumps(params)
        
        df = alpaca_service.read_bars(strategy_settings.asset, strategy_settings.start_date, strategy_settings.end_date)
        sm = StrategyManager(strategy_settings, df)
        result = sm.test_single_strategy() 

    
        iteration += 1
        print(f"No {iteration}  Tested params: {params}, Profit: {result.profit}, Num Trades: {result.number_of_positions}")
        if result.profit > best_profit:
            best_profit = result.profit
            best_params = params

    print(f"Beste Parameter: {best_params}, Profit: {best_profit}")
    return    




@celery_app.task
def run_ai_strategy_test(strategy_settings: StrategySettingsModel) -> None:
    
    try:
        # Synchroner Aufruf - kein asyncio needed!
        df = alpaca_service.read_bars(strategy_settings.asset, datetime(2024, 1, 1),datetime(2024, 12, 31))
        df = df.drop(['Symbol', 'T', 'Id'], axis=1)

        
        df = DataUtils.add_sma(df)
        df = DataUtils.add_rsi(df)
        df = DataUtils.add_donchian(df)
        df = DataUtils.add_macd(df)
        df = DataUtils.add_atr(df)
        # df = DataUtils.add_future_return(df,horizon=15)
        df = DataUtils.add_future_return(df,horizon=30)
        # df = DataUtils.add_future_return(df,horizon=60)        
        # df = DataUtils.add_future_return(df,horizon=120)
        
        df.dropna(inplace=True)
   

        features = [
            "sma_diff",
            "sma_diff_slope",
            "rsi_overbought",
            "rsi_oversold",
            "macd_cross",
            "macd_cross_change", 
            "donchian_breakout_strength",
            "atr",
            "V",
        ]
        X = df[features]
        # Use all 4 future returns as targets
        y = df[["future_return_15", "future_return_30", "future_return_60", "future_return_120"]]
        y = df["future_return_60"]

        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "n_estimators": [100, 300],
            "q_low": [0.2, 0.25, 0.3],
            "q_high": [0.7, 0.75, 0.8],
            "fee_per_trade": [0.0002, 0.0005]
        }
        
        # Only select numeric columns for scaling
        # Für LightGBM ist Skalierung nicht zwingend nötig, für neuronale Netze dagegen schon.
        # numeric_features = X.select_dtypes(include=[np.number]).columns
        # X_numeric = X[numeric_features]
        
        # for tensorflow / keras
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X_numeric)
        
        
        # y_class = np.where(df["future_return"] > 0.001, 1, np.where(df["future_return"] < -0.001, -1, 0))
        
        # print(df[["C", "future_return", "rsi_14", "macd_diff", "donchian_breakout_strength"]].tail())


       
        # model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
        # model.fit(X, y)
        # df["predicted_return"] = model.predict(X)
        
        # q_low, q_high = df["predicted_return"].quantile([0.2, 0.8])
        # df["position"] = np.where(df["predicted_return"] > q_high, 1,
        #            np.where(df["predicted_return"] < q_low, -1, 0))
        
        
        split_point = int(len(df) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        # Multi-output regression model
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
        model.fit(X_train, y_train)
        

   
        # Example: Use mean of predicted returns for position logic
        test_df = df.iloc[split_point:].copy()
        test_df["predicted_return_mean"] = test_df[["predicted_return_15", "predicted_return_30", "predicted_return_60", "predicted_return_120"]].mean(axis=1)

        q_low, q_high = test_df["predicted_return_mean"].quantile([0.2, 0.8])
        test_df["position"] = np.where(test_df["predicted_return_mean"] > q_high, 1,
                           np.where(test_df["predicted_return_mean"] < q_low, -1, 0))
        
        

        fee_per_trade = 0.0005

        # Use mean of actual future returns for strategy evaluation
        df["future_return_mean"] = df[["future_return_15", "future_return_30", "future_return_60", "future_return_120"]].mean(axis=1)
        df["position"] = 0
        df.loc[split_point:, "position"] = test_df["position"]

        df["strategy_return"] = df["position"] * df["future_return_mean"]

        # Transaktionskosten, wenn sich Position ändert
        df["trade"] = df["position"].diff().abs()
        df["cost"] = df["trade"] * fee_per_trade
        df["strategy_return_after_cost"] = df["strategy_return"] - df["cost"]
        df["equity_curve"] = (1 + df["strategy_return_after_cost"]).cumprod()

        # DEBUG: Prüfe die Werte
        print(f"Strategy returns - Min: {df['strategy_return_after_cost'].min():.4f}")
        print(f"Strategy returns - Max: {df['strategy_return_after_cost'].max():.4f}")
        print(f"Strategy returns - Mean: {df['strategy_return_after_cost'].mean():.4f}")
        print(f"Equity curve - Start: {df['equity_curve'].iloc[0]:.2f}")
        print(f"Equity curve - End: {df['equity_curve'].iloc[-1]:.2f}")
        print(f"Anzahl Positionen: {(df['position'] != 0).sum()}")
        
        # Prüfe auf extreme Werte
        extreme_returns = df[abs(df['strategy_return_after_cost']) > 0.1]
        if len(extreme_returns) > 0:
            print(f"WARNUNG: {len(extreme_returns)} extreme Renditen gefunden!")
            print(extreme_returns[['DT', 'strategy_return_after_cost', 'future_return', 'position']])

        

        returns = df["strategy_return_after_cost"].dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        print(f"Sharpe Ratio: {sharpe:.2f}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(df["DT"], df["equity_curve"], label="Equity Curve (ML Strategy)")
        plt.title("ML Strategy Performance")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        import matplotlib.pyplot as plt

        lgb.plot_importance(model, figsize=(8,6))
        plt.show()

        
        df.to_csv(f"csv/ai_strategy_{strategy_settings.asset}.csv", index=False)
        
        print(f"AI Strategy Analysis completed")
        return
        
    except Exception as e:
        print(f"Error in strategy test: {e}")
        raise
    
    
@celery_app.task
def run_ai_strategy_analysis(strategy_settings: StrategySettingsModel) -> None:
    
    try:
        df = alpaca_service.read_bars(strategy_settings.asset, datetime(2024, 1, 1),datetime(2024, 12, 31))
        df = df.drop(['Symbol', 'T', 'Id'], axis=1)

        df = DataUtils.add_sma(df)
        df = DataUtils.add_rsi(df)
        df = DataUtils.add_donchian(df)
        df = DataUtils.add_macd(df)
        df = DataUtils.add_atr(df)
        df = DataUtils.add_future_return(df,horizon=30)        
        
        df.dropna(inplace=True)
   
        features = [
            "sma_diff",
            "sma_diff_slope",
            "rsi_overbought",
            "rsi_oversold",
            "macd_cross",
            "macd_cross_change", 
            "donchian_breakout_strength",
            "atr",
            "V",
        ]
        X = df[features]
        y = df["future_return_120"]

        # Train/Test split
        split_point = int(len(df) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        # Train model
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_gain_to_split=0.01,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate R² scores (coefficient of determination)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        # Correlation between predictions and actual
        train_corr = np.corrcoef(train_predictions, y_train)[0, 1] if len(set(train_predictions)) > 1 else np.nan
        test_corr = np.corrcoef(test_predictions, y_test)[0, 1] if len(set(test_predictions)) > 1 else np.nan
        
        # DIAGNOSTIC: Check prediction variance
        print("\n" + "=" * 60)
        print("DIAGNOSTIC INFORMATION")
        print("=" * 60)
        print(f"\nTarget (future_return_120) statistics:")
        print(f"  Mean:     {y_train.mean():.6f}")
        print(f"  Std Dev:  {y_train.std():.6f}")
        print(f"  Min:      {y_train.min():.6f}")
        print(f"  Max:      {y_train.max():.6f}")
        
        print(f"\nPrediction statistics (test set):")
        print(f"  Mean:     {test_predictions.mean():.6f}")
        print(f"  Std Dev:  {test_predictions.std():.6f}")
        print(f"  Min:      {test_predictions.min():.6f}")
        print(f"  Max:      {test_predictions.max():.6f}")
        print(f"  Unique values: {len(set(test_predictions))}")
        
        print(f"\nFeature statistics:")
        print(X_train.describe())
        
        print("=" * 60)
        
        print("=" * 60)
        print("MODEL FIT QUALITY ANALYSIS")
        print("=" * 60)
        print(f"\nR² Score (1.0 = perfect, 0.0 = random):")
        print(f"  Training:   {train_r2:.4f}")
        print(f"  Testing:    {test_r2:.4f}")
        print(f"  Difference: {abs(train_r2 - test_r2):.4f} {'(overfitting!)' if abs(train_r2 - test_r2) > 0.1 else ''}")
        
        print(f"\nMean Squared Error (lower is better):")
        print(f"  Training:   {train_mse:.6f}")
        print(f"  Testing:    {test_mse:.6f}")
        
        print(f"\nMean Absolute Error (lower is better):")
        print(f"  Training:   {train_mae:.6f}")
        print(f"  Testing:    {test_mae:.6f}")
        
        print(f"\nCorrelation (1.0 = perfect positive, -1.0 = perfect negative):")
        print(f"  Training:   {train_corr:.4f}")
        print(f"  Testing:    {test_corr:.4f}")
        
        # Feature correlations with target
        print(f"\nFeature correlations with target:")
        correlations = df[features + ['future_return_120']].corr()['future_return_120'].drop('future_return_120').sort_values(ascending=False)
        for feat, corr in correlations.items():
            print(f"  {feat:30s}: {corr:7.4f}")
        
        print("=" * 60)
        
        # Interpretation guide
        print("\nINTERPRETATION GUIDE:")
        print("  R² Score:")
        print("    > 0.3: Good predictive power")
        print("    0.1-0.3: Moderate predictive power")
        print("    < 0.1: Weak/no predictive power")
        print("  Train/Test R² difference > 0.1: Likely overfitting")
        
        # Plot feature importance
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Feature importance
        lgb.plot_importance(model, ax=axes[0], max_num_features=len(features))
        axes[0].set_title('Feature Importance')
        
        # Prediction vs Actual scatter plot
        axes[1].scatter(y_test, test_predictions, alpha=0.5, s=1)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Returns')
        axes[1].set_ylabel('Predicted Returns')
        axes[1].set_title(f'Predictions vs Actual (Test Set)\nR² = {test_r2:.4f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save model metrics
        results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_correlation': train_corr,
            'test_correlation': test_corr,
            'feature_correlations': correlations.to_dict()
        }
        
        import json
        with open(f"csv/model_fit_{strategy_settings.asset}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nModel fit analysis completed and saved to csv/model_fit_{strategy_settings.asset}.json")
        return
        
    except Exception as e:
        print(f"Error in model training: {e}")
        raise


    
@celery_app.task
def train_lgb_model(strategy_settings: StrategySettingsModel) -> None:
    
    try:
        df = alpaca_service.read_bars(strategy_settings.asset, datetime(2024, 1, 1),datetime(2024, 12, 31))
        df = df.drop(['Symbol', 'T', 'Id'], axis=1)

        df = DataUtils.add_sma(df)
        df = DataUtils.add_rsi(df)
        df = DataUtils.add_donchian(df)
        df = DataUtils.add_macd(df)
        df = DataUtils.add_atr(df)
        df = DataUtils.add_future_return(df,horizon=60)        
        
        df.dropna(inplace=True)
   
        features = [
            "sma_diff",
            "sma_diff_slope",
            "rsi_overbought",
            "rsi_oversold",
            "macd_cross",
            "macd_cross_change", 
            "donchian_breakout_strength",
            "atr",
            "V",
        ]
        X = df[features]
        y = df["future_return_60"]

        # PROPER TRAIN/TEST SPLIT - avoid look-ahead bias
        split_point = int(len(df) * 0.7)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        # Train only on training data
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
        model.fit(X_train, y_train)
        
        # Predict only on test data
        test_predictions = model.predict(X_test)
        
        # Initialize columns
        df["predicted_return"] = 0.0
        df["position"] = 0
        
        # Only use predictions on test set
        df.loc[df.index[split_point:], "predicted_return"] = test_predictions
        
        # Calculate quantiles only on test predictions
        q_low, q_high = pd.Series(test_predictions).quantile([0.15, 0.85])  # More extreme thresholds
        
        # Set positions only on test data with stronger signals
        test_positions = np.where(test_predictions > q_high, 1,
                                  np.where(test_predictions < q_low, -1, 0))
        df.loc[df.index[split_point:], "position"] = test_positions
        
        # ADD: Take-profit and Stop-loss logic with end-of-day closing
        take_profit = 0.004  # 0.4%
        stop_loss = 0.003    # 0.3%

        # Track entry price and apply TP/SL
        df["entry_price"] = np.nan
        df["current_pnl"] = 0.0
        df["position_managed"] = 0
        
        entry_price = None
        entry_position = 0
        
        for i in range(split_point, len(df)):
            current_pos = df["position"].iloc[i]
            current_time = df["DT"].iloc[i]
            
            # Close position at end of day (assuming market closes at or near end of trading day)
            # Check if next bar is a different day
            is_end_of_day = False
            if i < len(df) - 1:
                next_time = df["DT"].iloc[i + 5]
                if current_time.date() != next_time.date():
                    is_end_of_day = True
            
            # If we have an open position
            if entry_position != 0:
                current_price = df["C"].iloc[i]
                pnl = (current_price - entry_price) / entry_price * entry_position
                df.loc[df.index[i], "current_pnl"] = pnl
                df.loc[df.index[i], "entry_price"] = entry_price
                
                # Check stop-loss
                if pnl <= -stop_loss:
                    df.loc[df.index[i], "position_managed"] = 0  # Exit position
                    entry_price = None
                    entry_position = 0
                    continue
                
                # Check take-profit
                if pnl >= take_profit:
                    df.loc[df.index[i], "position_managed"] = 0  # Exit position
                    entry_price = None
                    entry_position = 0
                    continue
                
                # Close position at end of day
                if is_end_of_day:
                    df.loc[df.index[i], "position_managed"] = 0  # Exit position
                    entry_price = None
                    entry_position = 0
                    continue
                
                # Otherwise maintain position
                df.loc[df.index[i], "position_managed"] = entry_position
            
            # Check if we're entering a new position (only if no position is open)
            elif current_pos != 0 and entry_position == 0:
                entry_price = df["C"].iloc[i]
                entry_position = current_pos
                df.loc[df.index[i], "entry_price"] = entry_price
                df.loc[df.index[i], "position_managed"] = entry_position
            else:
                df.loc[df.index[i], "position_managed"] = 0
        
        # Use managed positions instead of raw positions
        df["position"] = df["position_managed"]
        
        fee_per_trade = 0.0005

        # Calculate returns - already using shift(1)
        df["strategy_return"] = df["position"].shift(1) * df["future_return_60"]

        df["trade"] = df["position"].diff().abs()
        df["cost"] = df["trade"] * fee_per_trade
        df["strategy_return_after_cost"] = df["strategy_return"] - df["cost"]
        
        # Start equity curve from split point
        df["equity_curve"] = np.nan
        df.loc[split_point:, "equity_curve"] = (1 + df.loc[split_point:, "strategy_return_after_cost"]).cumprod()

        # Calculate metrics only on test period
        test_df = df.iloc[split_point:]
        
        print(f"Strategy returns - Min: {test_df['strategy_return_after_cost'].min():.4f}")
        print(f"Strategy returns - Max: {test_df['strategy_return_after_cost'].max():.4f}")
        print(f"Strategy returns - Mean: {test_df['strategy_return_after_cost'].mean():.4f}")
        print(f"Equity curve - Start: {test_df['equity_curve'].iloc[0]:.2f}")
        print(f"Equity curve - End: {test_df['equity_curve'].iloc[-1]:.2f}")
        print(f"Anzahl Positionen: {(test_df['position'] != 0).sum()}")
        print(f"Total trades: {test_df['trade'].sum():.0f}")
        
        # Calculate TP/SL statistics
        tp_exits = test_df[test_df['current_pnl'] >= take_profit].shape[0]
        sl_exits = test_df[test_df['current_pnl'] <= -stop_loss].shape[0]
        eod_exits = test_df[(test_df['position'].shift(1) != 0) & (test_df['position'] == 0) & 
                            (test_df['current_pnl'] > -stop_loss) & (test_df['current_pnl'] < take_profit)].shape[0]
        print(f"Take-profit exits: {tp_exits}")
        print(f"Stop-loss exits: {sl_exits}")
        print(f"End-of-day exits: {eod_exits}")
        
        print(f"Win rate: {(test_df['strategy_return'] > 0).sum() / len(test_df[test_df['position'] != 0]):.2%}")
        print(f"Avg return per trade: {test_df.loc[test_df['trade'] > 0, 'strategy_return'].mean():.4f}")
        
        extreme_returns = test_df[abs(test_df['strategy_return_after_cost']) > 0.1]
        if len(extreme_returns) > 0:
            print(f"WARNUNG: {len(extreme_returns)} extreme Renditen gefunden!")

        returns = test_df["strategy_return_after_cost"].dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        print(f"Sharpe Ratio: {sharpe:.2f}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(test_df["DT"], test_df["equity_curve"], label="Equity Curve (ML Strategy)")
        plt.title("ML Strategy Performance (Test Period)")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        lgb.plot_importance(model, figsize=(8,6))
        plt.show()

        df.to_csv(f"csv/ai_strategy_{strategy_settings.asset}.csv", index=False)
        
        print(f"AI Strategy Analysis completed")
        return
        
    except Exception as e:
        print(f"Error in strategy test: {e}")
        raise

