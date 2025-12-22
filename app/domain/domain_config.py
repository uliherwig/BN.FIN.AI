from app.domain.models.enums import IndicatorEnum

DOMAIN_CONFIG = {
    
    "INDICATOR_PARAM_RANGES" : {
        "SMA": {"SMA_short": (10, 60), "SMA_long": (80, 250)},
        "EMA": {"EMA_short": (10, 60), "EMA_long": (80, 250)},
        "WMA": {"WMA_short": (10, 60), "WMA_long": (80, 250)},
        "TEMA": {"TEMA_short": (10, 60), "TEMA_long": (80, 250)},
        "ATR": {"ATR_period": (8, 30)},
        "MACD": {"MACD_fast": (8, 20), "MACD_slow": (21, 50), "MACD_signal": (5, 20)},
        "DONCHIAN": {"DONCHIAN_window": (8, 30)},
        "BREAKOUT": {"BREAKOUT_period": (8, 30)},
        "RSI": {"RSI_period": (8, 30)},
        "VOLA": {"VOLA_period": (8, 30)},
        "BBANDS": {"BBANDS_period": (8, 30)},
        "ROC": {"ROC_period": (1, 2)},
    },
    
    "INDICATOR_PARAM_DEFAULTS" : {
        "SMA": {"SMA_short": 25, "SMA_long": 200},
        "EMA": {"EMA_short": 25, "EMA_long": 200},
        "WMA": {"WMA_short": 25, "WMA_long": 200},
        "TEMA": {"TEMA_short": 25, "TEMA_long": 200},
        "ATR": {"ATR_period": 30},
        "MACD": {"MACD_fast": 10, "MACD_slow": 30, "MACD_signal": 10},
        "DONCHIAN": {"DONCHIAN_window": 30},
        "BREAKOUT": {"BREAKOUT_period": 30},
        "RSI": {"RSI_period": 10},
        "VOLA": {"VOLA_period": 10},
        "BBANDS": {"BBANDS_period": 30},
        "ROC": {"ROC_period": 1},
    },
    
    
    "INDICATOR_FEATURES" : {
        "SMA": "SMA_diff",
        "EMA": "EMA_diff",
        "WMA": "WMA_diff",
        "TEMA": "TEMA_diff",
        "ATR": "ATR",
        "MACD": "MACD",
        "DONCHIAN": "DONCHIAN_breakout_strength",
        "BREAKOUT": "BREAKOUT_strength",
        "RSI": "RSI",
        "VOLA": "Volume_Ratio",
        "BBANDS": "BB_pctb",
        "ROC": "ROC_1",
    },


    "INDICATOR_MAP" : {
        "SMA": IndicatorEnum.SMA,
        "EMA": IndicatorEnum.EMA,
        "WMA": IndicatorEnum.WMA,
        "TEMA": IndicatorEnum.TEMA,
        "MACD": IndicatorEnum.MACD,
        "RSI": IndicatorEnum.RSI,
        "DONCHIAN": IndicatorEnum.DONCHIAN,
        "BREAKOUT": IndicatorEnum.BREAKOUT,
        "VOLA": IndicatorEnum.VOLA,
        "ATR": IndicatorEnum.ATR,
        "BBANDS": IndicatorEnum.BBANDS,
        "ROC": IndicatorEnum.ROC,
    },
    
    "DEFAULT_EXEC_SETTINGS" : {
        "broker": "Yahoo",
       	"trading_period": "1d",
        "asset": "SPY",
        "start_date": "2010-01-01",
        "end_date": "2024-12-31",

        # fixed thresholds, TP/SL
        "price_change_threshold": 0.0016,
        "long_threshold": 0.00025,
        "short_threshold": 0.00041,
        "tp": 0.043,
        "sl": 0.005,
    },

    # fixed model params (safe defaults)
    "DEFAULT_MODEL_PARAMS" : {
        "model_type": "regression",
        "learning_rate": 0.05,
        "num_leaves": 100,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.58,
        "n_estimators": 500,
    }
}
