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
        "SMA": {"SMA_short": 50, "SMA_long": 200},
        "EMA": {"EMA_short": 50, "EMA_long": 200},
        "WMA": {"WMA_short": 50, "WMA_long": 200},
        "TEMA": {"TEMA_short": 50, "TEMA_long": 200},
        "ATR": {"ATR_period": 30},
        "MACD": {"MACD_fast": 12, "MACD_slow": 26, "MACD_signal": 9},
        "DONCHIAN": {"DONCHIAN_window": 30},
        "BREAKOUT": {"BREAKOUT_period": 30},
        "RSI": {"RSI_period": 30},
        "VOLA": {"VOLA_period": 30},
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
    }
}
