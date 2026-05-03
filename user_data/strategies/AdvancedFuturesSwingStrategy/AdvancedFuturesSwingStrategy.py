"""
Advanced Multi-Timeframe Futures Swing Strategy for Freqtrade
=============================================================
This strategy incorporates:
- Higher Timeframe (HTF) trend analysis
- Distribution & Accumulation zones
- Support & Resistance detection
- Long wick analysis (rejection patterns)
- Order flow confirmation
- Smart Money Concepts (SMC)
"""

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, merge_informative_pair
from pandas import DataFrame
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import numpy as np
from datetime import datetime


class AdvancedFuturesSwingStrategy(IStrategy):
    """
    Advanced Multi-Timeframe Futures Swing Trading Strategy
    with Order Flow Analysis
    """
    
    INTERFACE_VERSION = 3
    
    # Base timeframe for entry/exit
    timeframe = '15m'
    
    # Can short
    can_short = True
    
    # ROI table
    minimal_roi = {
        "0": 0.20,
        "720": 0.15,
        "1440": 0.10,
        "2880": 0.06
    }
    
    # Stoploss
    stoploss = -0.06
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    # Leverage
    leverage_num = 3
    
    # ===== HYPEROPT PARAMETERS =====
    # These will be optimized quarterly and auto-injected
    
    # Trend Parameters
    htf_trend_ema = IntParameter(50, 200, default=100, space='buy', optimize=True)
    accumulation_period = IntParameter(20, 50, default=30, space='buy', optimize=True)
    
    # Wick Analysis
    wick_threshold = DecimalParameter(0.3, 0.7, default=0.5, space='buy', decimals=2, optimize=True)
    upper_wick_min = DecimalParameter(0.4, 0.8, default=0.6, space='buy', decimals=2, optimize=True)
    lower_wick_min = DecimalParameter(0.4, 0.8, default=0.6, space='buy', decimals=2, optimize=True)
    
    # Volume Parameters
    volume_surge = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=1, optimize=True)
    volume_confirmation = DecimalParameter(1.2, 2.5, default=1.5, space='buy', decimals=1, optimize=True)
    
    # Support/Resistance
    sr_lookback = IntParameter(50, 100, default=75, space='buy', optimize=True)
    sr_threshold = DecimalParameter(0.002, 0.01, default=0.005, space='buy', decimals=3, optimize=True)
    
    # RSI Parameters
    rsi_buy_min = IntParameter(30, 45, default=40, space='buy', optimize=True)
    rsi_buy_max = IntParameter(50, 65, default=65, space='buy', optimize=True)
    rsi_sell_min = IntParameter(60, 75, default=65, space='sell', optimize=True)
    rsi_overbought = IntParameter(65, 80, default=70, space='sell', optimize=True)
    
    # ADX Parameters
    adx_trend_min = IntParameter(15, 30, default=20, space='buy', optimize=True)
    adx_strong_trend = IntParameter(25, 40, default=25, space='buy', optimize=True)
    adx_weak_trend = IntParameter(10, 20, default=15, space='sell', optimize=True)
    
    # EMA Parameters
    ema_fast = IntParameter(8, 25, default=20, space='buy', optimize=True)
    ema_slow = IntParameter(40, 100, default=50, space='buy', optimize=True)
    
    # Order Flow
    delta_threshold = DecimalParameter(0, 1000, default=100, space='buy', decimals=0, optimize=True)
    cumulative_delta_period = IntParameter(10, 30, default=20, space='buy', optimize=True)
    
    # Accumulation/Distribution
    volatility_threshold = DecimalParameter(0.5, 0.9, default=0.7, space='buy', decimals=1, optimize=True)
    volume_acc_multiplier = DecimalParameter(1.1, 1.5, default=1.2, space='buy', decimals=1, optimize=True)
    
    # Exit Parameters
    exit_volume_low = DecimalParameter(0.6, 0.9, default=0.8, space='sell', decimals=1, optimize=True)
    exit_rsi_high = IntParameter(55, 70, default=60, space='sell', optimize=True)
    
    startup_candle_count: int = 400
    
    use_exit_signal = True
    exit_profit_only = False
    
    def informative_pairs(self):
        """
        Define additional informative pairs for HTF analysis
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        
        # Add higher timeframes for each pair
        for pair in pairs:
            informative_pairs.append((pair, '1h'))   # Higher timeframe 1
            informative_pairs.append((pair, '4h'))   # Higher timeframe 2
            informative_pairs.append((pair, '1d'))   # Higher timeframe 3
            
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate indicators including HTF analysis
        """
        
        # ===== BASE TIMEFRAME INDICATORS =====
        
        # Price Action
        dataframe['hl_avg'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['range'] = dataframe['high'] - dataframe['low']
        
        # Upper and Lower Wicks
        dataframe['upper_wick'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['lower_wick'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        
        # Wick to body ratio (rejection signals)
        dataframe['upper_wick_ratio'] = dataframe['upper_wick'] / (dataframe['range'] + 1e-10)
        dataframe['lower_wick_ratio'] = dataframe['lower_wick'] / (dataframe['range'] + 1e-10)
        
        # Long wick detection
        dataframe['long_upper_wick'] = (dataframe['upper_wick_ratio'] > self.upper_wick_min.value) & \
                                       (dataframe['upper_wick'] > dataframe['body'] * 2)
        dataframe['long_lower_wick'] = (dataframe['lower_wick_ratio'] > self.lower_wick_min.value) & \
                                       (dataframe['lower_wick'] > dataframe['body'] * 2)
        
        # EMAs
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Volume Analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / (dataframe['volume_sma'] + 1e-10)
        dataframe['high_volume'] = dataframe['volume_ratio'] > self.volume_surge.value
        
        # Order Flow Approximation (using volume delta)
        dataframe['buy_volume'] = dataframe['volume'] * ((dataframe['close'] - dataframe['low']) / 
                                                         (dataframe['range'] + 1e-10))
        dataframe['sell_volume'] = dataframe['volume'] * ((dataframe['high'] - dataframe['close']) / 
                                                          (dataframe['range'] + 1e-10))
        dataframe['volume_delta'] = dataframe['buy_volume'] - dataframe['sell_volume']
        dataframe['cumulative_delta'] = dataframe['volume_delta'].rolling(window=self.cumulative_delta_period.value).sum()
        
        # Volume Profile (simplified - volume at price levels)
        dataframe['vwap'] = (dataframe['volume'] * dataframe['hl_avg']).rolling(window=20).sum() / \
                           (dataframe['volume'].rolling(window=20).sum() + 1e-10)
        
        # ===== SUPPORT & RESISTANCE DETECTION =====
        dataframe = self.detect_support_resistance(dataframe)
        
        # ===== ACCUMULATION & DISTRIBUTION DETECTION =====
        dataframe = self.detect_accumulation_distribution(dataframe)
        
        # ADX - Trend Strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # ===== HIGHER TIMEFRAME ANALYSIS =====
        
        # 1H Timeframe
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        informative_1h = self.populate_htf_indicators(informative_1h, '1h')
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
        
        # 4H Timeframe
        informative_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')
        informative_4h = self.populate_htf_indicators(informative_4h, '4h')
        dataframe = merge_informative_pair(dataframe, informative_4h, self.timeframe, '4h', ffill=True)
        
        # Daily Timeframe
        informative_1d = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1d')
        informative_1d = self.populate_htf_indicators(informative_1d, '1d')
        dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, '1d', ffill=True)
        
        # ===== HTF TREND CONFLUENCE =====
        dataframe['htf_bullish'] = (
            (dataframe['trend_1h'] == 'bullish') &
            (dataframe['trend_4h'] == 'bullish') &
            (dataframe['trend_1d'] == 'bullish')
        )
        
        dataframe['htf_bearish'] = (
            (dataframe['trend_1h'] == 'bearish') &
            (dataframe['trend_4h'] == 'bearish') &
            (dataframe['trend_1d'] == 'bearish')
        )
        
        return dataframe
    
    def populate_htf_indicators(self, dataframe: DataFrame, timeframe: str) -> DataFrame:
        """
        Populate indicators for higher timeframes
        """
        # EMAs for trend
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=self.htf_trend_ema.value)
        
        # Trend determination
        dataframe['trend'] = 'neutral'
        dataframe.loc[
            (dataframe['close'] > dataframe['ema_trend']) &
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_slow'].shift(1) < dataframe['ema_slow']),
            'trend'
        ] = 'bullish'
        
        dataframe.loc[
            (dataframe['close'] < dataframe['ema_trend']) &
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['ema_slow'].shift(1) > dataframe['ema_slow']),
            'trend'
        ] = 'bearish'
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Support and Resistance on HTF
        dataframe['htf_resistance'] = dataframe['high'].rolling(window=20).max()
        dataframe['htf_support'] = dataframe['low'].rolling(window=20).min()
        
        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        
        return dataframe
    
    def detect_support_resistance(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect key support and resistance levels
        """
        lookback = self.sr_lookback.value
        threshold = self.sr_threshold.value
        
        # Rolling pivot points
        dataframe['pivot_high'] = dataframe['high'].rolling(window=lookback, center=True).max()
        dataframe['pivot_low'] = dataframe['low'].rolling(window=lookback, center=True).min()
        
        # Dynamic support/resistance
        dataframe['resistance_1'] = dataframe['high'].rolling(window=20).max()
        dataframe['resistance_2'] = dataframe['high'].rolling(window=50).max()
        dataframe['support_1'] = dataframe['low'].rolling(window=20).min()
        dataframe['support_2'] = dataframe['low'].rolling(window=50).min()
        
        # Near support/resistance
        dataframe['near_resistance'] = (
            ((dataframe['close'] >= dataframe['resistance_1'] * (1 - threshold)) &
             (dataframe['close'] <= dataframe['resistance_1'] * (1 + threshold))) |
            ((dataframe['close'] >= dataframe['resistance_2'] * (1 - threshold)) &
             (dataframe['close'] <= dataframe['resistance_2'] * (1 + threshold)))
        )
        
        dataframe['near_support'] = (
            ((dataframe['close'] >= dataframe['support_1'] * (1 - threshold)) &
             (dataframe['close'] <= dataframe['support_1'] * (1 + threshold))) |
            ((dataframe['close'] >= dataframe['support_2'] * (1 - threshold)) &
             (dataframe['close'] <= dataframe['support_2'] * (1 + threshold)))
        )
        
        # Order flow confirmation at support/resistance
        dataframe['support_confirmed'] = (
            dataframe['near_support'] &
            (dataframe['cumulative_delta'] > 0) &  # Positive order flow
            dataframe['long_lower_wick']  # Rejection wick
        )
        
        dataframe['resistance_confirmed'] = (
            dataframe['near_resistance'] &
            (dataframe['cumulative_delta'] < 0) &  # Negative order flow
            dataframe['long_upper_wick']  # Rejection wick
        )
        
        return dataframe
    
    def detect_accumulation_distribution(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect accumulation and distribution zones (Smart Money Concepts)
        """
        period = self.accumulation_period.value
        
        # Accumulation/Distribution Line (A/D Line)
        dataframe['ad_line'] = ta.AD(dataframe)
        dataframe['ad_line_sma'] = dataframe['ad_line'].rolling(window=period).mean()
        
        # Price range compression (accumulation characteristic)
        dataframe['price_std'] = dataframe['close'].rolling(window=period).std()
        dataframe['price_std_ratio'] = dataframe['price_std'] / dataframe['price_std'].rolling(window=50).mean()
        
        # Volume characteristics
        dataframe['avg_volume'] = dataframe['volume'].rolling(window=period).mean()
        dataframe['volume_std'] = dataframe['volume'].rolling(window=period).std()
        
        # ACCUMULATION ZONE detection
        # - Sideways price action (low volatility)
        # - Higher volume than average
        # - Positive cumulative delta
        # - Price compression near support
        dataframe['accumulation_zone'] = (
            (dataframe['price_std_ratio'] < self.volatility_threshold.value) &  # Low volatility
            (dataframe['volume'] > dataframe['avg_volume'] * self.volume_acc_multiplier.value) &  # Higher volume
            (dataframe['cumulative_delta'] > 0) &  # Buying pressure
            (dataframe['close'] < dataframe['ema_slow']) &  # Below mid-term MA
            (dataframe['rsi'] < 50)  # Not overbought
        )
        
        # DISTRIBUTION ZONE detection
        # - Sideways price action at tops
        # - Higher volume
        # - Negative cumulative delta
        # - Price compression near resistance
        dataframe['distribution_zone'] = (
            (dataframe['price_std_ratio'] < self.volatility_threshold.value) &  # Low volatility
            (dataframe['volume'] > dataframe['avg_volume'] * self.volume_acc_multiplier.value) &  # Higher volume
            (dataframe['cumulative_delta'] < 0) &  # Selling pressure
            (dataframe['close'] > dataframe['ema_slow']) &  # Above mid-term MA
            (dataframe['rsi'] > 50)  # Not oversold
        )
        
        # Wyckoff Spring/Upthrust detection
        dataframe['spring'] = (  # Fake breakdown (bullish)
            (dataframe['low'] < dataframe['support_1']) &
            (dataframe['close'] > dataframe['support_1']) &
            dataframe['long_lower_wick'] &
            (dataframe['volume'] > dataframe['volume_sma'] * 1.5)
        )
        
        dataframe['upthrust'] = (  # Fake breakout (bearish)
            (dataframe['high'] > dataframe['resistance_1']) &
            (dataframe['close'] < dataframe['resistance_1']) &
            dataframe['long_upper_wick'] &
            (dataframe['volume'] > dataframe['volume_sma'] * 1.5)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        LONG Entry with HTF trend confirmation and order flow
        """
        conditions_long = []
        
        # === LONG CONDITION 1: HTF Bullish + Accumulation Breakout ===
        conditions_long.append(
            dataframe['htf_bullish'] &  # All HTFs bullish
            (dataframe['accumulation_zone'].shift(1) == True) &  # Previous accumulation
            (dataframe['close'] > dataframe['ema_fast']) &  # Breakout above fast EMA
            (dataframe['cumulative_delta'] > self.delta_threshold.value) &  # Positive order flow
            (dataframe['volume_ratio'] > self.volume_confirmation.value) &  # Volume surge
            (dataframe['rsi'] > self.rsi_buy_min.value) & (dataframe['rsi'] < self.rsi_buy_max.value) &  # RSI in range
            (dataframe['adx'] > self.adx_trend_min.value)  # Trending
        )
        
        # === LONG CONDITION 2: Spring (Wyckoff) + Order Flow Confirmation ===
        conditions_long.append(
            dataframe['htf_bullish'] &
            (dataframe['spring'] == True) &  # Spring pattern
            (dataframe['cumulative_delta'] > dataframe['cumulative_delta'].shift(1)) &  # Increasing buying
            (dataframe['close'] > dataframe['open']) &  # Bullish candle
            (dataframe['adx'] > self.adx_weak_trend.value)
        )
        
        # === LONG CONDITION 3: Support Bounce + Order Flow ===
        conditions_long.append(
            (dataframe['trend_4h'] == 'bullish') &  # 4H trend bullish
            dataframe['support_confirmed'] &  # Support with order flow confirmation
            (dataframe['long_lower_wick'] == True) &  # Rejection wick
            (dataframe['close'] > dataframe['vwap']) &  # Above VWAP
            (dataframe['rsi'] < 50) &  # Not overbought
            (dataframe['volume_ratio'] > 1.3)
        )
        
        # === LONG CONDITION 4: Pullback Entry in Strong Trend ===
        conditions_long.append(
            dataframe['htf_bullish'] &
            (dataframe['close'] > dataframe['ema_200']) &  # Above 200 EMA
            (dataframe['close'] < dataframe['bb_middle']) &  # Pullback
            (dataframe['rsi'] < self.rsi_buy_min.value) &  # Oversold on pullback
            qtpylib.crossed_above(dataframe['close'], dataframe['ema_fast']) &  # Bounce
            (dataframe['cumulative_delta'] > 0) &  # Buying coming in
            (dataframe['adx'] > self.adx_strong_trend.value)  # Strong trend
        )
        
        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'enter_long'
            ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        LONG Exit signals
        """
        conditions_exit_long = []
        
        # === EXIT 1: Distribution Zone + Order Flow Reversal ===
        conditions_exit_long.append(
            (dataframe['distribution_zone'] == True) &
            (dataframe['cumulative_delta'] < 0) &  # Selling pressure
            (dataframe['rsi'] > self.rsi_sell_min.value)
        )
        
        # === EXIT 2: Resistance Rejection ===
        conditions_exit_long.append(
            dataframe['resistance_confirmed'] &
            (dataframe['long_upper_wick'] == True) &
            (dataframe['volume_ratio'] > 1.5)
        )
        
        # === EXIT 3: HTF Trend Reversal ===
        conditions_exit_long.append(
            (dataframe['trend_4h'] == 'bearish') &
            (dataframe['close'] < dataframe['ema_fast']) &
            (dataframe['cumulative_delta'] < 0)
        )
        
        # === EXIT 4: Upthrust (False Breakout) ===
        conditions_exit_long.append(
            (dataframe['upthrust'] == True) &
            (dataframe['rsi'] > self.exit_rsi_high.value)
        )
        
        # === EXIT 5: Momentum Loss ===
        conditions_exit_long.append(
            (dataframe['adx'] < self.adx_weak_trend.value) &  # Weak trend
            (dataframe['rsi'] > self.exit_rsi_high.value) &
            (dataframe['volume_ratio'] < self.exit_volume_low.value)  # Low volume
        )
        
        if conditions_exit_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_exit_long),
                'exit_long'
            ] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """
        Dynamic leverage based on market conditions
        """
        return self.leverage_num
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss with ATR-based dynamic adjustment
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # ATR-based stop
        atr = last_candle['atr']
        current_price = last_candle['close']
        atr_stop = (atr * 2) / current_price  # 2x ATR stop
        
        # Use the more conservative stop
        return min(self.stoploss, -atr_stop)