import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime

import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy
import talib.abstract as ta
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, BooleanParameter

# Optional for additional indicators
import pandas_ta as pta


class CompleteIndicatorStrategy2(IStrategy):
    """
    A comprehensive strategy using TA-Lib candlestick patterns and multiple indicators,
    adjusted to avoid lookahead bias (e.g., Ichimoku shift).
    Uses .value for IntParameter/DecimalParameter, preventing “object not callable” errors.
    Explicitly supports shorting (can_short = True), but also requires:
      - `enable_short: true` in your config
      - An exchange that supports margin/futures
      - A negative stoploss
    """

    # =====================================================================
    # Strategy Core Settings
    # =====================================================================

    # Allow short trades
    can_short = True

    # A negative stoploss is required to handle potential short trade losses
    stoploss = -0.20  # Example: 20% stop

    # =====================================================================
    # Strategy Hyperopt Parameters
    # =====================================================================
    # Leverage
    leverage_level = IntParameter(low=1, high=5, default=2, space="buy")

    # Score Thresholds (when to go long/short)
    long_threshold = IntParameter(low=1, high=30, default=11, space="buy")
    short_threshold = IntParameter(low=-30, high=-1, default=-4, space="sell")

    # Trailing Stop
    trailing_stop = True
    trailing_stop_positive = DecimalParameter(0.01, 0.10, default=0.03, space="sell")

    # Bullish Candlestick Pattern Weights
    hammer_weight = IntParameter(low=1, high=5, default=3, space="buy")
    morning_star_weight = IntParameter(low=1, high=5, default=3, space="buy")
    three_white_soldiers_weight = IntParameter(low=1, high=5, default=3, space="buy")
    piercing_weight = IntParameter(low=1, high=3, default=1, space="buy")

    # Bearish Candlestick Pattern Weights
    shooting_star_weight = IntParameter(low=-5, high=-1, default=-3, space="sell")
    evening_star_weight = IntParameter(low=-5, high=-1, default=-3, space="sell")
    three_black_crows_weight = IntParameter(low=-5, high=-1, default=-3, space="sell")
    dark_cloud_cover_weight = IntParameter(low=-3, high=-1, default=-1, space="sell")

    # Engulfing & Harami & 3 Inside & Rising/Falling 3 Methods
    bullish_engulfing_weight = IntParameter(low=1, high=5, default=3, space="buy")
    bearish_engulfing_weight = IntParameter(low=-5, high=-1, default=-3, space="sell")

    harami_bullish_weight = IntParameter(low=1, high=3, default=1, space="buy")
    harami_bearish_weight = IntParameter(low=-3, high=-1, default=-1, space="sell")

    three_inside_up_weight = IntParameter(low=1, high=3, default=1, space="buy")
    three_inside_down_weight = IntParameter(low=-3, high=-1, default=-1, space="sell")

    rising_three_methods_weight = IntParameter(low=1, high=3, default=1, space="buy")
    falling_three_methods_weight = IntParameter(low=-3, high=-1, default=-1, space="sell")

    # Indicator Weights
    macd_positive_weight = IntParameter(low=1, high=5, default=2, space="buy")
    macd_negative_weight = IntParameter(low=-5, high=-1, default=-2, space="sell")

    rsi_oversold_weight = IntParameter(low=1, high=5, default=3, space="buy")
    rsi_overbought_weight = IntParameter(low=-5, high=-1, default=-3, space="sell")

    stoch_oversold_weight = IntParameter(low=1, high=5, default=3, space="buy")
    stoch_overbought_weight = IntParameter(low=-5, high=-1, default=-3, space="sell")

    adx_strong_trend_weight = IntParameter(low=1, high=5, default=2, space="buy")
    adx_weak_trend_weight = IntParameter(low=-5, high=-1, default=-2, space="sell")

    sma_gap_positive_weight = IntParameter(low=1, high=5, default=2, space="buy")
    sma_gap_negative_weight = IntParameter(low=-5, high=-1, default=-2, space="sell")

    ichimoku_bullish_weight = IntParameter(low=1, high=5, default=2, space="buy")
    ichimoku_bearish_weight = IntParameter(low=-5, high=-1, default=-2, space="sell")

    # =====================================================================
    # Timeframe & Startup
    # =====================================================================
    timeframe = '1h'
    startup_candle_count = 52  # For Ichimoku + other extended indicators

    # =====================================================================
    # POPULATE INDICATORS
    # =====================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators and candlestick patterns to the dataframe.
        Convert to numeric, handle missing data, shift Ichimoku to avoid lookahead.
        """

        # Candlestick patterns (mostly 0 or +100)
        dataframe['hammer'] = pd.to_numeric(ta.CDLHAMMER(dataframe), errors='coerce').fillna(0)
        dataframe['shooting_star'] = pd.to_numeric(ta.CDLSHOOTINGSTAR(dataframe), errors='coerce').fillna(0)
        dataframe['morning_star'] = pd.to_numeric(ta.CDLMORNINGSTAR(dataframe), errors='coerce').fillna(0)
        dataframe['evening_star'] = pd.to_numeric(ta.CDLEVENINGSTAR(dataframe), errors='coerce').fillna(0)

        dataframe['three_white_soldiers'] = pd.to_numeric(ta.CDL3WHITESOLDIERS(dataframe), errors='coerce').fillna(0)
        dataframe['three_black_crows'] = pd.to_numeric(ta.CDL3BLACKCROWS(dataframe), errors='coerce').fillna(0)
        dataframe['piercing'] = pd.to_numeric(ta.CDLPIERCING(dataframe), errors='coerce').fillna(0)
        dataframe['dark_cloud_cover'] = pd.to_numeric(ta.CDLDARKCLOUDCOVER(dataframe), errors='coerce').fillna(0)

        # Candlestick patterns (± sign possible)
        dataframe['engulfing'] = pd.to_numeric(ta.CDLENGULFING(dataframe), errors='coerce').fillna(0)
        dataframe['harami'] = pd.to_numeric(ta.CDLHARAMI(dataframe), errors='coerce').fillna(0)
        dataframe['three_inside'] = pd.to_numeric(ta.CDL3INSIDE(dataframe), errors='coerce').fillna(0)
        dataframe['rise_fall_3_methods'] = pd.to_numeric(ta.CDLRISEFALL3METHODS(dataframe), errors='coerce').fillna(0)

        # SMAs
        dataframe['sma_50'] = pd.to_numeric(ta.SMA(dataframe, timeperiod=50), errors='coerce').fillna(0)
        dataframe['sma_200'] = pd.to_numeric(ta.SMA(dataframe, timeperiod=200), errors='coerce').fillna(0)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = pd.to_numeric(macd['macd'], errors='coerce').fillna(0)
        dataframe['macd_signal'] = pd.to_numeric(macd['macdsignal'], errors='coerce').fillna(0)

        # RSI
        dataframe['rsi'] = pd.to_numeric(ta.RSI(dataframe), errors='coerce').fillna(0)

        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = pd.to_numeric(stoch['slowk'], errors='coerce').fillna(0)
        dataframe['stoch_d'] = pd.to_numeric(stoch['slowd'], errors='coerce').fillna(0)

        # ADX, +DI, -DI
        dataframe['adx'] = pd.to_numeric(ta.ADX(dataframe), errors='coerce').fillna(0)
        dataframe['plus_di'] = pd.to_numeric(ta.PLUS_DI(dataframe), errors='coerce').fillna(0)
        dataframe['minus_di'] = pd.to_numeric(ta.MINUS_DI(dataframe), errors='coerce').fillna(0)

        # Bollinger Bands
        bb = ta.BBANDS(dataframe)
        dataframe['bb_upper'] = pd.to_numeric(bb['upperband'], errors='coerce').fillna(0)
        dataframe['bb_middle'] = pd.to_numeric(bb['middleband'], errors='coerce').fillna(0)
        dataframe['bb_lower'] = pd.to_numeric(bb['lowerband'], errors='coerce').fillna(0)

        # Parabolic SAR
        dataframe['sar'] = pd.to_numeric(ta.SAR(dataframe), errors='coerce').fillna(0)

        # Ichimoku (shift leading spans 26 to avoid lookahead)
        high9 = dataframe['high'].rolling(window=9).max()
        low9 = dataframe['low'].rolling(window=9).min()
        conversion_line = (high9 + low9) / 2

        high26 = dataframe['high'].rolling(window=26).max()
        low26 = dataframe['low'].rolling(window=26).min()
        base_line = (high26 + low26) / 2

        high52 = dataframe['high'].rolling(window=52).max()
        low52 = dataframe['low'].rolling(window=52).min()
        leading_span_a = ((conversion_line + base_line) / 2).shift(26)
        leading_span_b = ((high52 + low52) / 2).shift(26)

        dataframe['conversion_line'] = pd.to_numeric(conversion_line, errors='coerce').fillna(0)
        dataframe['base_line'] = pd.to_numeric(base_line, errors='coerce').fillna(0)
        dataframe['ichimoku_span_a'] = pd.to_numeric(leading_span_a, errors='coerce').fillna(0)
        dataframe['ichimoku_span_b'] = pd.to_numeric(leading_span_b, errors='coerce').fillna(0)

        # Ensure volume is numeric
        dataframe['volume'] = pd.to_numeric(dataframe['volume'], errors='coerce').fillna(0)

        return dataframe

    # =====================================================================
    # CALCULATE SCORE
    # =====================================================================
    def calculate_score(self, row: pd.Series) -> float:
        """
        Combine candlestick patterns and indicator signals into a single numeric score.
        +ve => bullish, -ve => bearish. Access IntParameter/DecimalParameter with .value.
        """

        score = 0.0

        # ---------- One-sided candlestick patterns ----------
        if row['hammer'] > 0:
            score += self.hammer_weight.value
        if row['shooting_star'] > 0:
            score += self.shooting_star_weight.value

        if row['morning_star'] > 0:
            score += self.morning_star_weight.value
        if row['evening_star'] > 0:
            score += self.evening_star_weight.value

        if row['three_white_soldiers'] > 0:
            score += self.three_white_soldiers_weight.value
        if row['three_black_crows'] > 0:
            score += self.three_black_crows_weight.value

        if row['piercing'] > 0:
            score += self.piercing_weight.value
        if row['dark_cloud_cover'] > 0:
            score += self.dark_cloud_cover_weight.value

        # ---------- Two-sided candlestick patterns (check sign) ----------
        if row['engulfing'] > 0:
            score += self.bullish_engulfing_weight.value
        elif row['engulfing'] < 0:
            score += self.bearish_engulfing_weight.value

        if row['harami'] > 0:
            score += self.harami_bullish_weight.value
        elif row['harami'] < 0:
            score += self.harami_bearish_weight.value

        if row['three_inside'] > 0:
            score += self.three_inside_up_weight.value
        elif row['three_inside'] < 0:
            score += self.three_inside_down_weight.value

        if row['rise_fall_3_methods'] > 0:
            score += self.rising_three_methods_weight.value
        elif row['rise_fall_3_methods'] < 0:
            score += self.falling_three_methods_weight.value

        # ---------- MACD ----------
        if row['macd'] > row['macd_signal']:
            score += self.macd_positive_weight.value
        elif row['macd'] < row['macd_signal']:
            score += self.macd_negative_weight.value

        # ---------- RSI ----------
        if row['rsi'] < 30:
            score += self.rsi_oversold_weight.value
        elif row['rsi'] > 70:
            score += self.rsi_overbought_weight.value

        # ---------- Stochastic ----------
        if row['stoch_k'] < 20 and row['stoch_k'] > row['stoch_d']:
            score += self.stoch_oversold_weight.value
        elif row['stoch_k'] > 80 and row['stoch_k'] < row['stoch_d']:
            score += self.stoch_overbought_weight.value

        # ---------- ADX ----------
        if row['adx'] > 25:
            if row['plus_di'] > row['minus_di']:
                score += self.adx_strong_trend_weight.value
            elif row['minus_di'] > row['plus_di']:
                score += self.adx_weak_trend_weight.value

        # ---------- SMA gap ----------
        if row['sma_50'] != 0:
            gap_50 = (row['close'] - row['sma_50']) / row['sma_50']
            if gap_50 > 0:
                score += self.sma_gap_positive_weight.value
            elif gap_50 < 0:
                score += self.sma_gap_negative_weight.value

        # ---------- Ichimoku ----------
        if row['conversion_line'] > row['base_line'] and row['close'] > row['conversion_line']:
            score += self.ichimoku_bullish_weight.value
        elif row['conversion_line'] < row['base_line'] and row['close'] < row['conversion_line']:
            score += self.ichimoku_bearish_weight.value

        return score

    # =====================================================================
    # POPULATE ENTRY TREND
    # =====================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Create 'enter_long' and 'enter_short' signals based on the 'score' thresholds.
        Remember, you need `enable_short=true` in your config and an exchange that supports margin/futures.
        """
        # Calculate the row-wise score
        dataframe['score'] = dataframe.apply(self.calculate_score, axis=1)

        # Long if score >= long_threshold
        dataframe['enter_long'] = 0
        dataframe.loc[
            dataframe['score'] >= self.long_threshold.value,
            'enter_long'
        ] = 1

        # Short if score <= short_threshold
        dataframe['enter_short'] = 0
        dataframe.loc[
            dataframe['score'] <= self.short_threshold.value,
            'enter_short'
        ] = 1

        return dataframe

    # =====================================================================
    # POPULATE EXIT TREND
    # =====================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Basic exit logic: exit positions if 'score' crosses the opposite side of zero.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # Exit long if the current score is negative
        dataframe.loc[dataframe['score'] < 0, 'exit_long'] = 1

        # Exit short if the current score is positive
        dataframe.loc[dataframe['score'] > 0, 'exit_short'] = 1

        return dataframe

    # =====================================================================
    # LEVERAGE (REQUIRED FOR FUTURES)
    # =====================================================================
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs
    ) -> float:
        """
        Optional leverage management for futures trading.
        Return a float, not an IntParameter object.
        """
        # Make sure to return the .value so freqtrade receives a numeric value.
        return float(self.leverage_level.value)
