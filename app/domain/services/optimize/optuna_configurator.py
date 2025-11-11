from app.domain.models.enums import IndicatorEnum
from app.domain.operations.indicator_factory import IndicatorFactory
from app.domain import IndicatorModel


class OptunaConfigurator:
    
    @staticmethod
    def get_optuna_indicators() -> any:
        return [
            ("SMA", IndicatorEnum.SMA, {
             "SMA_short": (10, 60), "SMA_long": (80, 250)}),
            ("EMA", IndicatorEnum.EMA, {
             "EMA_short": (10, 60), "EMA_long": (80, 250)}),
            ("WMA", IndicatorEnum.WMA, {
             "WMA_short": (10, 60), "WMA_long": (80, 250)}),
            ("TEMA", IndicatorEnum.TEMA, {
             "TEMA_short": (10, 60), "TEMA_long": (80, 250)}),
            ("ATR", IndicatorEnum.ATR, {"ATR_period": (8, 30)}),
            ("MACD", IndicatorEnum.MACD, {"MACD_fast": (
                8, 20), "MACD_slow": (21, 50), "MACD_signal": (5, 20)}),
            ("DONCHIAN", IndicatorEnum.DONCHIAN, {"DONCHIAN_window": (8, 30)}),
            ("BREAKOUT", IndicatorEnum.BREAKOUT, {"BREAKOUT_period": (8, 30)}),
            ("RSI", IndicatorEnum.RSI, {"RSI_period": (8, 30)}),
            ("VOLA", IndicatorEnum.VOLA, {"VOLA_period": (8, 30)}),
            ("BBANDS", IndicatorEnum.BBANDS, {"BBANDS_period": (8, 30)}),
        ]
        
    @staticmethod
    def get_all_features() -> any:
            return [
                ("SMA", ["SMA_diff"]),
                ("EMA", ["EMA_diff"]),
                ("WMA", ["WMA_diff"]),
                ("TEMA", ["TEMA_diff"]),
                ("ATR", ["ATR"]),
                ("MACD", ["MACD"]),
                ("DONCHIAN", ["DONCHIAN_breakout_strength"]),
                ("BREAKOUT", ["BREAKOUT_strength"]),
                ("RSI", ["RSI"]),
                ("VOLA", ["Returns_1d", "Volume_Ratio"]),
                ("BBANDS", ["BB_pctb"]),
            ]
    @staticmethod
    def get_indicator_config(selected_indicators: list[IndicatorEnum]) -> any:

        all_indicators = OptunaConfigurator.get_optuna_indicators() 
        indicators = []
        
        for enum in selected_indicators:
            i = all_indicators[[x[1] for x in all_indicators].index(enum)]
            indicators.append(i)
            
        print("Selected Indicators for Optimization:", indicators)

        return indicators
    
    @staticmethod
    def get_features_by_indicators(selected_indicators: list[IndicatorEnum]) -> any:
        features = []
        
        all = OptunaConfigurator.get_all_features()
        for enum in selected_indicators:
            f = all[[x[0] for x in all].index(enum.name)]
            for y in f[1]:
                features.append(y)
        return features
