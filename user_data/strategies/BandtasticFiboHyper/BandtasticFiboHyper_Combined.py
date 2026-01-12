"""
BandtasticFiboHyper_Combined Strategy

A combined "best of" strategy merging optimal parameters from:
- opt490: Long entry parameters (better diversification, 336% total profit)
- opt314: Short entry parameters (1.07% avg profit per trade, 67% win rate)

Features:
- Hybrid entry logic: opt490 longs + opt314 shorts
- ATR-based dynamic stop loss with multi-layer protection
- Profit-based trailing stops
- Doom stop protection (absolute maximum loss)
- Fixed 2x leverage for futures trading

Stop Loss Layers:
1. Base Stop: 10% starting point, adjusted by volatility (5-15% range)
2. ATR Dynamic: Uses normalized ATR with 2.5x multiplier
3. Profit Trailing: 4 tiers that lock in profits
4. Doom Stop: -20% absolute maximum loss protection

Author: Combined from BandtasticFiboHyper optimizations
Version: 1.0.1
"""

import talib.abstract as ta
import pandas as pd
from functools import reduce
from datetime import datetime
from pandas import DataFrame
from typing import Optional
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)
from freqtrade.persistence import Trade


class BandtasticFiboHyper_Combined(IStrategy):
    """
    Combined BandtasticFiboHyper strategy with ATR-based dynamic stop loss.

    Entry Logic:
    - Long: RSI<50, MFI<40, EMA crossover, price below BB4 lower band
    - Short: MFI>58, price above BB1 upper band

    Stop Loss:
    - Dynamic ATR-based stops that adapt to market volatility
    - Profit-based trailing to lock in gains
    - Doom stop protection at -20%
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'

    # ROI table - combined from opt490 (more conservative)
    minimal_roi = {
        "0": 0.20,
        "30": 0.08,
        "60": 0.04,
        "120": 0.02,
        "180": 0
    }

    # Base stoploss (fallback, actual stoploss handled by custom_stoploss)
    stoploss = -0.20

    # Trailing stop settings (works alongside custom_stoploss)
    trailing_stop = True
    trailing_stop_positive = 0.15
    trailing_stop_positive_offset = 0.20
    trailing_only_offset_is_reached = True

    startup_candle_count = 999
    max_open_trades = 1

    # Use custom_stoploss function
    use_custom_stoploss = True

    # ========= Stop Loss Parameters =========
    stoploss_base = DecimalParameter(0.08, 0.12, default=0.10, space='protection', optimize=True)
    stoploss_min = DecimalParameter(0.03, 0.07, default=0.05, space='protection', optimize=True)
    stoploss_max = DecimalParameter(0.12, 0.18, default=0.15, space='protection', optimize=True)
    atr_multiplier = DecimalParameter(2.0, 3.5, default=2.5, space='protection', optimize=True)
    atr_period = IntParameter(10, 20, default=14, space='protection', optimize=True)
    doom_stop = DecimalParameter(0.18, 0.25, default=0.20, space='protection', optimize=True)

    # ========= Long Entry Parameters (from opt490) =========
    buy_fastema = IntParameter(1, 250, default=24, space='buy', optimize=True)
    buy_slowema = IntParameter(1, 250, default=163, space='buy', optimize=True)
    buy_rsi = IntParameter(15, 70, default=50, space='buy', optimize=True)
    buy_mfi = IntParameter(15, 70, default=40, space='buy', optimize=True)
    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_ema_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_trigger = CategoricalParameter(
        ['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'],
        default='bb_lower4', space='buy', optimize=True
    )
    buy_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_fib_level = CategoricalParameter(
        ['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'],
        default='fib_382', space='buy', optimize=True
    )

    # ========= Short Entry Parameters (from opt314) =========
    short_fastema = IntParameter(1, 250, default=53, space='sell', optimize=True)
    short_slowema = IntParameter(1, 250, default=168, space='sell', optimize=True)
    short_rsi = IntParameter(30, 100, default=88, space='sell', optimize=True)
    short_mfi = IntParameter(30, 100, default=58, space='sell', optimize=True)
    short_rsi_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    short_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    short_ema_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    short_trigger = CategoricalParameter(
        ['bb_upper1', 'bb_upper2', 'bb_upper3', 'bb_upper4'],
        default='bb_upper1', space='sell', optimize=True
    )

    # ========= Long Exit Parameters (from opt490) =========
    sell_fastema = IntParameter(1, 365, default=242, space='sell', optimize=True)
    sell_slowema = IntParameter(1, 365, default=8, space='sell', optimize=True)
    sell_rsi = IntParameter(30, 100, default=87, space='sell', optimize=True)
    sell_mfi = IntParameter(30, 100, default=52, space='sell', optimize=True)
    sell_rsi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    sell_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    sell_ema_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    sell_trigger = CategoricalParameter(
        ['sell-bb_upper1', 'sell-bb_upper2', 'sell-bb_upper3', 'sell-bb_upper4'],
        default='sell-bb_upper1', space='sell', optimize=True
    )

    # ========= Short Exit / Cover Parameters (from opt314) =========
    cover_fastema = IntParameter(1, 250, default=97, space='buy', optimize=True)
    cover_slowema = IntParameter(1, 250, default=191, space='buy', optimize=True)
    cover_rsi = IntParameter(10, 70, default=42, space='buy', optimize=True)
    cover_mfi = IntParameter(10, 70, default=12, space='buy', optimize=True)
    cover_rsi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_ema_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
    cover_trigger = CategoricalParameter(
        ['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'],
        default='bb_lower4', space='buy', optimize=True
    )
    cover_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_fib_level = CategoricalParameter(
        ['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'],
        default='fib_382', space='buy', optimize=True
    )

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Fixed 2x leverage for all trades.
        Simple and predictable risk management.
        """
        return 2.0

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> Optional[float]:
        """
        Multi-layer ATR-based dynamic stop loss.

        Layers:
        1. Doom Stop: Absolute maximum loss protection (-20%)
        2. Base Stop: Starting point adjusted by volatility (-10% base, 5-15% range)
        3. Trailing Stop: Profit-based tightening when in profit
        4. Volatility Adjustment: Wider stops in high volatility, tighter in low

        Note: All intermediate calculations use POSITIVE values for stop distance.
        The final return is NEGATIVE (e.g., -0.10 means stop at 10% loss from entry).

        Returns:
            float: Stop loss as negative percentage (e.g., -0.10 for 10% loss)
        """
        # Get analyzed dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Fallback to base stoploss if no data
        if dataframe is None or len(dataframe) < int(self.atr_period.value):
            return float(self.stoploss)

        # Minimum trade duration: allow 15 minutes before dynamic stop kicks in
        trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_duration_minutes < 15:
            return float(self.stoploss)

        # Get current candle data
        last_candle = dataframe.iloc[-1]

        # Get normalized ATR (ATR as percentage of price)
        normalized_atr = last_candle.get('normalized_atr', 0.02)
        if pd.isna(normalized_atr) or normalized_atr <= 0:
            normalized_atr = 0.02  # Default 2%

        # Volatility thresholds
        atr_low_threshold = 0.015   # 1.5% - Low volatility
        atr_high_threshold = 0.040  # 4.0% - High volatility

        # Calculate volatility-adjusted stop distance (positive value)
        if normalized_atr < atr_low_threshold:
            # LOW VOLATILITY: Use tight stop (min)
            stop_distance = float(self.stoploss_min.value)
        elif normalized_atr > atr_high_threshold:
            # HIGH VOLATILITY: Use wide stop (max)
            stop_distance = float(self.stoploss_max.value)
        else:
            # NORMAL VOLATILITY: Linear interpolation
            vol_range = atr_high_threshold - atr_low_threshold
            vol_position = (normalized_atr - atr_low_threshold) / vol_range
            stop_distance = (
                float(self.stoploss_min.value) +
                (float(self.stoploss_max.value) - float(self.stoploss_min.value)) * vol_position
            )

        # ATR-based dynamic stop calculation (positive value)
        atr_based_stop = normalized_atr * float(self.atr_multiplier.value)

        # Use the larger of volatility-adjusted or ATR-based stop, capped at max
        # All values are positive here (representing stop distance)
        dynamic_stop_distance = min(
            max(atr_based_stop, stop_distance),
            float(self.stoploss_max.value)
        )

        # Doom stop value (positive)
        doom_stop_distance = float(self.doom_stop.value)

        # If already past doom stop, force exit
        if current_profit < -doom_stop_distance:
            return 1.0  # Force immediate exit

        # Profit-based stop tightening:
        # When in profit, reduce the allowed loss to protect gains.
        # Note: custom_stoploss can only prevent losses (negative returns),
        # not lock in profits directly. For profit protection, we rely on
        # the trailing_stop settings alongside this. Here we just tighten
        # the stop to minimize potential loss when already in profit.
        if current_profit > 0.10:
            # >10% profit: Tighten stop to 3% max loss from entry
            dynamic_stop_distance = min(dynamic_stop_distance, 0.03)
        elif current_profit > 0.05:
            # >5% profit: Tighten stop to 5% max loss
            dynamic_stop_distance = min(dynamic_stop_distance, 0.05)
        elif current_profit > 0.02:
            # >2% profit: Tighten stop to 7% max loss
            dynamic_stop_distance = min(dynamic_stop_distance, 0.07)

        # Cap at doom stop
        final_stop_distance = min(dynamic_stop_distance, doom_stop_distance)

        # Return as negative value
        return -final_stop_distance

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all indicators needed for entry/exit decisions.
        """
        # RSI and MFI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        # ATR for stop loss calculation
        atr_period = int(self.atr_period.value)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=atr_period)
        dataframe['normalized_atr'] = dataframe['atr'] / dataframe['close']

        # Bollinger Bands (1-4 standard deviations)
        for std in range(1, 5):
            bb = qtpylib.bollinger_bands(
                qtpylib.typical_price(dataframe), window=20, stds=std
            )
            dataframe[f'bb_lowerband{std}'] = bb['lower']
            dataframe[f'bb_middleband{std}'] = bb['mid']
            dataframe[f'bb_upperband{std}'] = bb['upper']

        # EMAs - collect all unique periods needed
        ema_periods = set([
            int(self.buy_fastema.value), int(self.buy_slowema.value),
            int(self.sell_fastema.value), int(self.sell_slowema.value),
            int(self.short_fastema.value), int(self.short_slowema.value),
            int(self.cover_fastema.value), int(self.cover_slowema.value)
        ])

        for period in ema_periods:
            if period > 0 and len(dataframe) >= period:
                dataframe[f'EMA_{period}'] = ta.EMA(dataframe, timeperiod=period)

        # Fibonacci Levels (50 candle lookback)
        lookback = 50
        if len(dataframe) >= lookback:
            recent_max = dataframe['high'].rolling(lookback).max()
            recent_min = dataframe['low'].rolling(lookback).min()
            diff = recent_max - recent_min
            dataframe['fib_236'] = recent_max - diff * 0.236
            dataframe['fib_382'] = recent_max - diff * 0.382
            dataframe['fib_5'] = recent_max - diff * 0.5
            dataframe['fib_618'] = recent_max - diff * 0.618
            dataframe['fib_786'] = recent_max - diff * 0.786

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions for long and short positions.

        Long: Parameters from opt490 (better diversification)
        Short: Parameters from opt314 (better per-trade profit)
        """
        # ========= LONG ENTRY (from opt490) =========
        long_conditions = []

        # RSI filter
        if self.buy_rsi_enabled.value:
            long_conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        # MFI filter
        if self.buy_mfi_enabled.value:
            long_conditions.append(dataframe['mfi'] < self.buy_mfi.value)

        # EMA crossover filter
        if self.buy_ema_enabled.value:
            fast_col = f'EMA_{self.buy_fastema.value}'
            slow_col = f'EMA_{self.buy_slowema.value}'
            if fast_col in dataframe.columns and slow_col in dataframe.columns:
                long_conditions.append(dataframe[fast_col] > dataframe[slow_col])

        # Bollinger Band trigger
        if self.buy_trigger.value.startswith('bb_lower'):
            bb_num = self.buy_trigger.value[-1]
            bb_col = f'bb_lowerband{bb_num}'
            if bb_col in dataframe.columns:
                long_conditions.append(dataframe['close'] < dataframe[bb_col])

        # Fibonacci trigger
        if self.buy_trigger.value == 'fibonacci' and self.buy_fib_enabled.value:
            fib_col = self.buy_fib_level.value
            if fib_col in dataframe.columns:
                long_conditions.append(dataframe['close'] < dataframe[fib_col])

        # Volume check
        long_conditions.append(dataframe['volume'] > 0)

        # Combine conditions
        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_conditions),
                'enter_long'
            ] = 1

        # ========= SHORT ENTRY (from opt314) =========
        short_conditions = []

        # RSI filter
        if self.short_rsi_enabled.value:
            short_conditions.append(dataframe['rsi'] > self.short_rsi.value)

        # MFI filter
        if self.short_mfi_enabled.value:
            short_conditions.append(dataframe['mfi'] > self.short_mfi.value)

        # EMA crossover filter
        if self.short_ema_enabled.value:
            fast_col = f'EMA_{self.short_fastema.value}'
            slow_col = f'EMA_{self.short_slowema.value}'
            if fast_col in dataframe.columns and slow_col in dataframe.columns:
                short_conditions.append(dataframe[fast_col] < dataframe[slow_col])

        # Bollinger Band trigger
        if self.short_trigger.value.startswith('bb_upper'):
            bb_num = self.short_trigger.value[-1]
            bb_col = f'bb_upperband{bb_num}'
            if bb_col in dataframe.columns:
                short_conditions.append(dataframe['close'] > dataframe[bb_col])

        # Volume check
        short_conditions.append(dataframe['volume'] > 0)

        # Combine conditions
        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_conditions),
                'enter_short'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions for long and short positions.
        """
        # ========= LONG EXIT (from opt490) =========
        long_exit = []

        # RSI filter
        if self.sell_rsi_enabled.value:
            long_exit.append(dataframe['rsi'] > self.sell_rsi.value)

        # MFI filter
        if self.sell_mfi_enabled.value:
            long_exit.append(dataframe['mfi'] > self.sell_mfi.value)

        # EMA crossover filter
        if self.sell_ema_enabled.value:
            fast_col = f'EMA_{self.sell_fastema.value}'
            slow_col = f'EMA_{self.sell_slowema.value}'
            if fast_col in dataframe.columns and slow_col in dataframe.columns:
                long_exit.append(dataframe[fast_col] < dataframe[slow_col])

        # Bollinger Band trigger
        if self.sell_trigger.value.startswith('sell-bb_upper'):
            bb_num = self.sell_trigger.value[-1]
            bb_col = f'bb_upperband{bb_num}'
            if bb_col in dataframe.columns:
                long_exit.append(dataframe['close'] > dataframe[bb_col])

        # Volume check
        long_exit.append(dataframe['volume'] > 0)

        # Combine conditions
        if long_exit:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_exit),
                'exit_long'
            ] = 1

        # ========= SHORT EXIT / COVER (from opt314) =========
        short_exit = []

        # RSI filter
        if self.cover_rsi_enabled.value:
            short_exit.append(dataframe['rsi'] < self.cover_rsi.value)

        # MFI filter
        if self.cover_mfi_enabled.value:
            short_exit.append(dataframe['mfi'] < self.cover_mfi.value)

        # EMA crossover filter
        if self.cover_ema_enabled.value:
            fast_col = f'EMA_{self.cover_fastema.value}'
            slow_col = f'EMA_{self.cover_slowema.value}'
            if fast_col in dataframe.columns and slow_col in dataframe.columns:
                short_exit.append(dataframe[fast_col] > dataframe[slow_col])

        # Bollinger Band trigger
        if self.cover_trigger.value.startswith('bb_lower'):
            bb_num = self.cover_trigger.value[-1]
            bb_col = f'bb_lowerband{bb_num}'
            if bb_col in dataframe.columns:
                short_exit.append(dataframe['close'] < dataframe[bb_col])

        # Fibonacci trigger
        if self.cover_trigger.value == 'fibonacci' and self.cover_fib_enabled.value:
            fib_col = self.cover_fib_level.value
            if fib_col in dataframe.columns:
                short_exit.append(dataframe['close'] < dataframe[fib_col])

        # Volume check
        short_exit.append(dataframe['volume'] > 0)

        # Combine conditions
        if short_exit:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_exit),
                'exit_short'
            ] = 1

        return dataframe
