from enum import Enum

# class StrategyEnum(Enum):
#     NONE = 0
#     BREAKOUT = 1
#     MEAN_REVERSION = 2
#     MOMENTUM = 3
#     REVERSAL = 4
#     TREND_FOLLOWING = 5
#     SMA = 6
#     EMA = 7
#     WMA = 8
#     TEMA = 9
#     MACD = 10
#     DONCHIAN = 11
#     RSI = 12
    

class BrokerEnum(Enum):
    Yahoo = 0
    Alpaca = 1

class SideEnum(Enum):
    Buy = 0
    Sell = 1

class TimeFrameEnum(Enum):
    Minute = 0
    TenMinutes = 1
    ThirtyMinutes = 2
    Hour = 3
    Day = 4
    
class SignalEnum(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0
    

class IndicatorEnum(Enum):
    NONE = "NONE"
    SMA = 'SMA'  # Simple Moving Average
    EMA = 'EMA'  # Exponential Moving Average
    WMA = 'WMA'  # Weighted Moving Average
    TEMA = 'TEMA'  # Triple Exponential Moving Average
    MACD = 'MACD'  # Moving Average Convergence/Divergence
    RSI = 'RSI'  # Relative Strength Index
    DONCHIAN = 'DONCHIAN'  # Donchian Channel
    BREAKOUT = 'BREAKOUT'  # Breakout Strategy
    VOLA = 'VOLA'  # Volatility Strategy
    ATR = 'ATR'  # Average True Range
    BBANDS = 'BBANDS'  # Bollinger Bands
    ROC = 'ROC'  # Rate of Change Indicator
    

class TaLibEnum(Enum):
    NONE ="NONE"
    
    # custom Strategies
    VOLUME = 'VOLUME'  # Volume Strategy
    VOLA = 'VOLA'  # Volatility Strategy
    BREAKOUT = 'BREAKOUT'  # Breakout Strategy
    DONCHIAN = 'DONCHIAN'  # Donchian Channel
    # Volume Indicators
    AD = 'AD'  # Chaikin A/D Line
    ADOSC = 'ADOSC'  # Chaikin A/D Oscillator
    OBV = 'OBV'  # On Balance Volume
    MFI = 'MFI'  # Money Flow Index
    
    # Trend Indicators
    ADX = 'ADX'  # Average Directional Movement Index
    ADXR = 'ADXR'  # Average Directional Movement Index Rating
    AROON = 'AROON'  # Aroon
    AROONOSC = 'AROONOSC'  # Aroon Oscillator
    DX = 'DX'  # Directional Movement Index
    MINUS_DI = 'MINUS_DI'  # Minus Directional Indicator
    MINUS_DM = 'MINUS_DM'  # Minus Directional Movement
    PLUS_DI = 'PLUS_DI'  # Plus Directional Indicator
    PLUS_DM = 'PLUS_DM'  # Plus Directional Movement
    
    # Moving Averages
    SMA = 'SMA'  # Simple Moving Average
    EMA = 'EMA'  # Exponential Moving Average
    WMA = 'WMA'  # Weighted Moving Average
    DEMA = 'DEMA'  # Double Exponential Moving Average
    TEMA = 'TEMA'  # Triple Exponential Moving Average
    TRIMA = 'TRIMA'  # Triangular Moving Average
    KAMA = 'KAMA'  # Kaufman Adaptive Moving Average
    MAMA = 'MAMA'  # MESA Adaptive Moving Average
    T3 = 'T3'  # Triple Exponential Moving Average (T3)
    MA = 'MA'  # All Moving Average
    
    # Momentum Indicators
    RSI = 'RSI'  # Relative Strength Index
    MACD = 'MACD'  # Moving Average Convergence/Divergence
    MACDEXT = 'MACDEXT'  # MACD with controllable MA type
    MACDFIX = 'MACDFIX'  # Moving Average Convergence/Divergence Fix 12/26
    STOCH = 'STOCH'  # Stochastic
    STOCHF = 'STOCHF'  # Stochastic Fast
    STOCHRSI = 'STOCHRSI'  # Stochastic Relative Strength Index
    WILLR = 'WILLR'  # Williams' %R
    CMO = 'CMO'  # Chande Momentum Oscillator
    MOM = 'MOM'  # Momentum
    APO = 'APO'  # Absolute Price Oscillator
    PPO = 'PPO'  # Percentage Price Oscillator
    ROC = 'ROC'  # Rate of change : ((price/prevPrice)-1)*100
    ROCP = 'ROCP'  # Rate of change Percentage: (price-prevPrice)/prevPrice
    ROCR = 'ROCR'  # Rate of change ratio: (price/prevPrice)
    ROCR100 = 'ROCR100'  # Rate of change ratio 100 scale: (price/prevPrice)*100
    TRIX = 'TRIX'  # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    ULTOSC = 'ULTOSC'  # Ultimate Oscillator
    
    # Volatility Indicators
    ATR = 'ATR'  # Average True Range
    NATR = 'NATR'  # Normalized Average True Range
    TRANGE = 'TRANGE'  # True Range
    BBANDS = 'BBANDS'  # Bollinger Bands
    STDDEV = 'STDDEV'  # Standard Deviation
    VAR = 'VAR'  # Variance
    
    # Price Transform
    AVGPRICE = 'AVGPRICE'  # Average Price
    MEDPRICE = 'MEDPRICE'  # Median Price
    TYPPRICE = 'TYPPRICE'  # Typical Price
    WCLPRICE = 'WCLPRICE'  # Weighted Close Price
    
    # Cycle Indicators
    HT_DCPERIOD = 'HT_DCPERIOD'  # Hilbert Transform - Dominant Cycle Period
    HT_DCPHASE = 'HT_DCPHASE'  # Hilbert Transform - Dominant Cycle Phase
    HT_PHASOR = 'HT_PHASOR'  # Hilbert Transform - Phasor Components
    HT_SINE = 'HT_SINE'  # Hilbert Transform - SineWave
    HT_TRENDLINE = 'HT_TRENDLINE'  # Hilbert Transform - Instantaneous Trendline
    HT_TRENDMODE = 'HT_TRENDMODE'  # Hilbert Transform - Trend vs Cycle Mode
    
    # Statistic Functions
    BETA = 'BETA'  # Beta
    CORREL = 'CORREL'  # Pearson's Correlation Coefficient
    LINEARREG = 'LINEARREG'  # Linear Regression
    LINEARREG_ANGLE = 'LINEARREG_ANGLE'  # Linear Regression Angle
    LINEARREG_INTERCEPT = 'LINEARREG_INTERCEPT'  # Linear Regression Intercept
    LINEARREG_SLOPE = 'LINEARREG_SLOPE'  # Linear Regression Slope
    TSF = 'TSF'  # Time Series Forecast
    
    # Math Transform
    MAX = 'MAX'  # Highest value over a specified period
    MAXINDEX = 'MAXINDEX'  # Index of highest value over a specified period
    MIN = 'MIN'  # Lowest value over a specified period
    MININDEX = 'MININDEX'  # Index of lowest value over a specified period
    MINMAX = 'MINMAX'  # Lowest and highest values over a specified period
    MINMAXINDEX = 'MINMAXINDEX'  # Indexes of lowest and highest values over a specified period
    MIDPOINT = 'MIDPOINT'  # MidPoint over period
    MIDPRICE = 'MIDPRICE'  # Midpoint Price over period
    SUM = 'SUM'  # Summation
    
    # Overlap Studies
    SAR = 'SAR'  # Parabolic SAR
    SAREXT = 'SAREXT'  # Parabolic SAR - Extended
    
    # Other Indicators
    CCI = 'CCI'  # Commodity Channel Index
    BOP = 'BOP'  # Balance Of Power
    
    # Candlestick Pattern Recognition
    CDL2CROWS = 'CDL2CROWS'  # Two Crows
    CDL3BLACKCROWS = 'CDL3BLACKCROWS'  # Three Black Crows
    CDL3INSIDE = 'CDL3INSIDE'  # Three Inside Up/Down
    CDL3LINESTRIKE = 'CDL3LINESTRIKE'  # Three Outside Up/Down
    CDL3STARSINSOUTH = 'CDL3STARSINSOUTH'  # Three Stars In The South
    CDL3WHITESOLDIERS = 'CDL3WHITESOLDIERS'  # Three Advancing White Soldiers
    CDLABANDONEDBABY = 'CDLABANDONEDBABY'  # Abandoned Baby
    CDLADVANCEBLOCK = 'CDLADVANCEBLOCK'  # Advance Block
    CDLBELTHOLD = 'CDLBELTHOLD'  # Belt-hold
    CDLBREAKAWAY = 'CDLBREAKAWAY'  # Breakaway
    CDLCLOSINGMARUBOZU = 'CDLCLOSINGMARUBOZU'  # Closing Marubozu
    CDLCONCEALBABYSWALL = 'CDLCONCEALBABYSWALL'  # Concealing Baby Swallow
    CDLCOUNTERATTACK = 'CDLCOUNTERATTACK'  # Counterattack
    CDLDARKCLOUDCOVER = 'CDLDARKCLOUDCOVER'  # Dark Cloud Cover
    CDLDOJI = 'CDLDOJI'  # Doji
    CDLDOJISTAR = 'CDLDOJISTAR'  # Doji Star
    CDLDRAGONFLYDOJI = 'CDLDRAGONFLYDOJI'  # Dragonfly Doji
    CDLENGULFING = 'CDLENGULFING'  # Engulfing Pattern
    CDLEVENINGDOJISTAR = 'CDLEVENINGDOJISTAR'  # Evening Doji Star
    CDLEVENINGSTAR = 'CDLEVENINGSTAR'  # Evening Star
    CDLGAPSIDESIDEWHITE = 'CDLGAPSIDESIDEWHITE'  # Up/Down-gap side-by-side white lines
    CDLGRAVESTONEDOJI = 'CDLGRAVESTONEDOJI'  # Gravestone Doji
    CDLHAMMER = 'CDLHAMMER'  # Hammer
    CDLHANGINGMAN = 'CDLHANGINGMAN'  # Hanging Man
    CDLHARAMI = 'CDLHARAMI'  # Harami Pattern
    CDLHARAMICROSS = 'CDLHARAMICROSS'  # Harami Cross Pattern
    CDLHIGHWAVE = 'CDLHIGHWAVE'  # High-Wave Candle
    CDLHIKKAKE = 'CDLHIKKAKE'  # Hikkake Pattern
    CDLHIKKAKEMOD = 'CDLHIKKAKEMOD'  # Modified Hikkake Pattern
    CDLHOMINGPIGEON = 'CDLHOMINGPIGEON'  # Homing Pigeon
    CDLIDENTICAL3CROWS = 'CDLIDENTICAL3CROWS'  # Identical Three Crows
    CDLINNECK = 'CDLINNECK'  # In-Neck Pattern
    CDLINVERTEDHAMMER = 'CDLINVERTEDHAMMER'  # Inverted Hammer
    CDLKICKING = 'CDLKICKING'  # Kicking
    CDLKICKINGBYLENGTH = 'CDLKICKINGBYLENGTH'  # Kicking - bull/bear determined by the longer marubozu
    CDLLADDERBOTTOM = 'CDLLADDERBOTTOM'  # Ladder Bottom
    CDLLONGLEGGEDDOJI = 'CDLLONGLEGGEDDOJI'  # Long Legged Doji
    CDLLONGLINE = 'CDLLONGLINE'  # Long Line Candle
    CDLMARUBOZU = 'CDLMARUBOZU'  # Marubozu
    CDLMATCHINGLOW = 'CDLMATCHINGLOW'  # Matching Low
    CDLMATHOLD = 'CDLMATHOLD'  # Mat Hold
    CDLMORNINGDOJISTAR = 'CDLMORNINGDOJISTAR'  # Morning Doji Star
    CDLMORNINGSTAR = 'CDLMORNINGSTAR'  # Morning Star
    CDLONNECK = 'CDLONNECK'  # On-Neck Pattern
    CDLPIERCING = 'CDLPIERCING'  # Piercing Pattern
    CDLRICKSHAWMAN = 'CDLRICKSHAWMAN'  # Rickshaw Man
    CDLRISEFALL3METHODS = 'CDLRISEFALL3METHODS'  # Rising/Falling Three Methods
    CDLSEPARATINGLINES = 'CDLSEPARATINGLINES'  # Separating Lines
    CDLSHOOTINGSTAR = 'CDLSHOOTINGSTAR'  # Shooting Star
    CDLSHORTLINE = 'CDLSHORTLINE'  # Short Line Candle
    CDLSPINNINGTOP = 'CDLSPINNINGTOP'  # Spinning Top
    CDLSTALLEDPATTERN = 'CDLSTALLEDPATTERN'  # Stalled Pattern
    CDLSTICKSANDWICH = 'CDLSTICKSANDWICH'  # Stick Sandwich
    CDLTAKURI = 'CDLTAKURI'  # Takuri (Dragonfly Doji with very long lower shadow)
    CDLTASUKIGAP = 'CDLTASUKIGAP'  # Tasuki Gap
    CDLTHRUSTING = 'CDLTHRUSTING'  # Thrusting Pattern
    CDLTRISTAR = 'CDLTRISTAR'  # Tristar Pattern
    CDLUNIQUE3RIVER = 'CDLUNIQUE3RIVER'  # Unique 3 River
    CDLUPSIDEGAP2CROWS = 'CDLUPSIDEGAP2CROWS'  # Upside Gap Two Crows
    CDLXSIDEGAP3METHODS = 'CDLXSIDEGAP3METHODS'  # Upside/Downside Gap Three Methods
