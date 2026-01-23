# NASOSv4_Shorts - Shorts-only strategy with 3x leverage
# Converted from NASOSv4 long strategy
# for live trailing_stop = False and use_custom_stoploss = True
# for backtest trailing_stop = True and use_custom_stoploss = False
# --- Do not remove these libs ---
from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import logging

logger = logging.getLogger(__name__)

# @Rallipanos
# @pluxury
# Converted to shorts by Vasko - January 2026

# Sell (short entry) hyperspace params - inverted from buy_params
sell_params = {
    'base_nb_candles_sell': 8,  # Same as buy
    'ewo_low': -2.403,  # Inverted from ewo_high: 2.403
    'ewo_low_2': 5.585,  # Inverted from ewo_high_2: -5.585
    'ewo_high': 14.378,  # Inverted from ewo_low: -14.378
    'lookback_candles': 3,
    'high_offset': 1.016,  # Inverted from low_offset: 0.984 using 2-x
    'high_offset_2': 1.058,  # Inverted from low_offset_2: 0.942 using 2-x
    'profit_threshold': 1.008,  # Same logic but for downside
    'rsi_sell': 28  # Inverted from rsi_buy: 72 using 100-x
}

# Cover (short exit) hyperspace params - inverted from sell_params
cover_params = {
    'base_nb_candles_cover': 16,  # Same as sell
    'low_offset': 0.916,  # Inverted from high_offset: 1.084 using 2-x
    'low_offset_2': 0.599,  # Inverted from high_offset_2: 1.401 using 2-x
    # Stop loss params scaled for 3x leverage
    'pHSL': -0.45,  # 3x from -0.15 (base hard stop)
    'pPF_1': 0.048,  # 3x from 0.016 (first profit threshold)
    'pPF_2': 0.072,  # 3x from 0.024 (second profit threshold)
    'pSL_1': 0.042,  # 3x from 0.014 (first stop level)
    'pSL_2': 0.066  # 3x from 0.022 (second stop level)
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    """Elliott Wave Oscillator"""
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class NASOSv4_Shorts(IStrategy):
    """
    NASOSv4_Shorts Strategy

    A shorts-only variant of NASOSv4 with 3x leverage, designed to profit
    in bear markets and during overbought conditions.

    Original NASOSv4 Concept:
    - EWO-based entries with RSI confirmation
    - Dynamic EMA offset entries
    - Custom trailing stoploss with profit thresholds

    Shorts Adaptation:
    - Inverted entry conditions (sells high instead of buys low)
    - 3x leverage with scaled stop losses
    - All indicators inverted for bearish signals
    - Tighter risk management for short positions

    Leverage & Risk Management:
    - Fixed 3x leverage on all positions
    - Base stoploss: -45% (3x of original -15%)
    - Custom stoploss scales all values by 3x to maintain same price movement tolerance
    - Example: -0.042 stoploss = 4.2% P&L = 1.4% price movement with 3x leverage

    Entry Conditions (Short Signals):
    1. ewo1_short: Price above EMA, high EWO, overbought RSI
    2. ewo2_short: Price above EMA, moderate EWO, very overbought RSI (>75)
    3. ewohigh_short: Price above EMA, very high EWO (extreme overbought)

    Exit Conditions:
    - Signal exit: Price crosses below SMAs/EMAs with bearish RSI
    - ROI: 1000% (effectively disabled)
    - Custom trailing stops with profit-based thresholds

    Key Differences from Long Strategy:
    - Base stoploss: -45% vs -15% (3x leverage scaling)
    - Stop thresholds: All 3x larger (4.8% vs 1.6%, 7.2% vs 2.4%, etc.)
    - Entry logic: Overbought conditions instead of oversold
    - Max 4 short positions enforced

    Author: Derived from NASOSv4 by Rallipanos & pluxury
    Version: 1.0.0
    Date: January 2026
    """

    INTERFACE_VERSION = 3
    can_short = True

    # ROI table - conservative for shorts
    minimal_roi = {'0': 10}

    # Base Stoploss - scaled for 3x leverage
    # Original: -0.15 → 3x: -0.45 (allows 15% price movement before liquidation ~-30%)
    stoploss = -0.45

    # SMAOffset parameters - inverted for shorts
    base_nb_candles_sell = IntParameter(
        2, 20, default=sell_params['base_nb_candles_sell'],
        space='sell', optimize=True
    )
    base_nb_candles_cover = IntParameter(
        2, 25, default=cover_params['base_nb_candles_cover'],
        space='buy', optimize=True
    )

    # Entry offsets - inverted (price should be ABOVE EMAs for shorts)
    high_offset = DecimalParameter(
        1.01, 1.1, default=sell_params['high_offset'],
        space='sell', optimize=False
    )
    high_offset_2 = DecimalParameter(
        1.01, 1.1, default=sell_params['high_offset_2'],
        space='sell', optimize=False
    )

    # Exit offsets - inverted (price should be BELOW EMAs to cover)
    low_offset = DecimalParameter(
        0.9, 0.99, default=cover_params['low_offset'],
        space='buy', optimize=True
    )
    low_offset_2 = DecimalParameter(
        0.5, 0.99, default=cover_params['low_offset_2'],
        space='buy', optimize=True
    )

    # Protection - same periods as longs
    fast_ewo = 50
    slow_ewo = 200
    lookback_candles = IntParameter(
        1, 24, default=sell_params['lookback_candles'],
        space='sell', optimize=True
    )
    profit_threshold = DecimalParameter(
        1.0, 1.03, default=sell_params['profit_threshold'],
        space='sell', optimize=True
    )

    # EWO thresholds - inverted for shorts (looking for high/positive EWO = overbought)
    ewo_high = DecimalParameter(
        8.0, 20.0, default=sell_params['ewo_high'],
        space='sell', optimize=False
    )
    ewo_low = DecimalParameter(
        -12.0, -2.0, default=sell_params['ewo_low'],
        space='sell', optimize=False
    )
    ewo_low_2 = DecimalParameter(
        -12.0, 6.0, default=sell_params['ewo_low_2'],
        space='sell', optimize=False
    )

    # RSI threshold - inverted for shorts (looking for overbought = low value for shorting)
    rsi_sell = IntParameter(
        0, 50, default=sell_params['rsi_sell'],
        space='sell', optimize=False
    )

    # Trailing stoploss hyperopt parameters - scaled for 3x leverage
    # All values multiplied by 3 to maintain same price movement tolerance

    # Hard stoploss profit (3x from -0.15)
    pHSL = DecimalParameter(
        -0.6, -0.12, default=cover_params['pHSL'],
        decimals=3, space='buy', optimize=False, load=True
    )

    # Profit threshold 1, trigger point (3x from 0.016)
    pPF_1 = DecimalParameter(
        0.024, 0.06, default=cover_params['pPF_1'],
        decimals=3, space='buy', optimize=False, load=True
    )
    pSL_1 = DecimalParameter(
        0.024, 0.06, default=cover_params['pSL_1'],
        decimals=3, space='buy', optimize=False, load=True
    )

    # Profit threshold 2 (3x from 0.024/0.022)
    pPF_2 = DecimalParameter(
        0.12, 0.3, default=cover_params['pPF_2'],
        decimals=3, space='buy', optimize=False, load=True
    )
    pSL_2 = DecimalParameter(
        0.06, 0.21, default=cover_params['pSL_2'],
        decimals=3, space='buy', optimize=False, load=True
    )

    # Trailing stop configuration
    trailing_stop = True
    trailing_stop_positive = 0.003  # 3x from 0.001
    trailing_stop_positive_offset = 0.048  # 3x from 0.016
    trailing_only_offset_is_reached = True

    # Exit signal configuration
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.03  # 3x from 0.01
    ignore_roi_if_entry_signal = False

    # Optional order time in force
    order_time_in_force = {'entry': 'gtc', 'exit': 'ioc'}

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'
    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = False

    # Position limits for shorts
    max_open_trades = 4
    max_entry_position_adjustment = 0

    plot_config = {
        'main_plot': {
            'ma_sell': {'color': 'orange'},
            'ma_cover': {'color': 'blue'}
        }
    }

    slippage_protection = {'retries': 3, 'max_slippage': -0.02}

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Fixed 3x leverage for all short trades.

        This ensures consistent risk/reward across all positions and allows
        the custom stoploss logic to work predictably.

        With 3x leverage and -45% stoploss:
        - Allows up to 15% adverse price movement before stop
        - Exchange liquidation occurs around -30% (-10% price movement)
        - Provides buffer for volatility while protecting capital

        Returns:
            float: Leverage multiplier (3.0 = 3x)
        """
        return 3.0

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom Trailing Stoploss scaled for 3x leverage.

        Based on Perkmeister's trailing stoploss with profit thresholds,
        adapted for short positions with 3x leverage scaling.

        All stoploss values are 3x the original to maintain the same price
        movement tolerance. For example:
        - SL_1 = 0.042 means 4.2% P&L stop = 1.4% price movement with 3x
        - Original SL_1 = 0.014 meant 1.4% P&L stop = 1.4% price movement with 1x

        Logic:
        - Below PF_1 profit: Use hard stoploss (HSL = -45%)
        - Between PF_1 and PF_2: Linear interpolation between SL_1 and SL_2
        - Above PF_2: Trail at SL_2 + profit distance

        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current datetime
            current_rate: Current price
            current_profit: Current profit/loss ratio

        Returns:
            float: Stoploss value from open price
        """
        # Get parameters
        HSL = self.pHSL.value  # Hard stop: -0.45 (3x from -0.15)
        PF_1 = self.pPF_1.value  # First profit threshold: 0.048 (3x from 0.016)
        SL_1 = self.pSL_1.value  # First stop level: 0.042 (3x from 0.014)
        PF_2 = self.pPF_2.value  # Second profit threshold: 0.072 (3x from 0.024)
        SL_2 = self.pSL_2.value  # Second stop level: 0.066 (3x from 0.022)

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PF_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + (current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1)
        else:
            sl_profit = HSL

        return stoploss_from_open(sl_profit, current_profit)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        """
        Confirm trade entry - only allow shorts.

        Enforces:
        - Only short positions (no longs)
        - Maximum 4 short positions

        Args:
            pair: Trading pair
            order_type: Order type
            amount: Order amount
            rate: Order rate
            time_in_force: Time in force
            current_time: Current datetime
            entry_tag: Entry signal tag
            side: 'long' or 'short'

        Returns:
            bool: True to allow trade, False to reject
        """
        # Only allow shorts
        if side == "long":
            return False

        # Count open short positions
        short_count = sum(1 for t in Trade.get_trades_proxy(is_open=True) if t.is_short)

        # Enforce max shorts limit
        if short_count >= self.max_open_trades:
            return False

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit with shorts-specific logic.

        Block exit if:
        - Exit signal triggered but HMA shows strong downtrend still active
          (inverted from longs: HMA < EMA and price > EMA = don't exit yet)
        - Slippage protection: retry if slippage too high

        Args:
            pair: Trading pair
            trade: Trade object
            order_type: Order type
            amount: Order amount
            rate: Exit rate
            time_in_force: Time in force
            exit_reason: Reason for exit
            current_time: Current datetime

        Returns:
            bool: True to allow exit, False to block
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if last_candle is not None:
            if exit_reason in ['exit_signal']:
                # Inverted from longs: Block exit if still in strong downtrend
                # Original: hma_50 * 1.149 > ema_100 AND close < ema_100 * 0.951 → don't sell
                # Shorts: hma_50 * 0.851 < ema_100 AND close > ema_100 * 1.049 → don't cover
                if (last_candle['hma_50'] * 0.851 < last_candle['ema_100'] and
                    last_candle['close'] > last_candle['ema_100'] * 1.049):
                    return False

        # Slippage protection
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()
        slippage = rate / candle['close'] - 1

        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0
        return True

    def informative_pairs(self):
        """
        Define informative pairs for strategy.

        Returns:
            List of tuples: (pair, timeframe)
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for 1h timeframe.

        Args:
            dataframe: OHLCV dataframe
            metadata: Pair metadata

        Returns:
            DataFrame: Dataframe with indicators
        """
        assert self.dp, 'DataProvider is required for multiple timeframes.'

        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        # Add indicators if needed (currently commented out in original)
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for normal timeframe (5m).

        Args:
            dataframe: OHLCV dataframe
            metadata: Pair metadata

        Returns:
            DataFrame: Dataframe with indicators
        """
        # Calculate all ma_sell values (for short entry)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_cover values (for short exit)
        for val in self.base_nb_candles_cover.range:
            dataframe[f'ma_cover_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Additional indicators
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        # Elliott Wave Oscillator
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators.

        Args:
            dataframe: OHLCV dataframe
            metadata: Pair metadata

        Returns:
            DataFrame: Dataframe with all indicators
        """
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True
        )

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate short entry signals (inverted from long buy signals).

        Original Strategy Entry Conditions (Longs):
        1. ewo1: RSI_fast < 35, close < EMA * 0.984, EWO > 2.403, RSI < 72
        2. ewo2: Same + close < EMA * 0.942, EWO > -5.585, RSI < 25
        3. ewolow: RSI_fast < 35, close < EMA * 0.984, EWO < -14.378

        Shorts Inversion:
        1. ewo1_short: RSI_fast > 65, close > EMA * 1.016, EWO < -2.403, RSI > 28
        2. ewo2_short: Same + close > EMA * 1.058, EWO < 5.585, RSI > 75
        3. ewohigh_short: RSI_fast > 65, close > EMA * 1.016, EWO > 14.378

        Don't Sell Condition:
        - Don't short if price can't drop 3% from recent low
          (inverted from: don't buy if price can't rise 3% from recent high)

        Args:
            dataframe: OHLCV dataframe with indicators
            metadata: Pair metadata

        Returns:
            DataFrame: Dataframe with entry signals
        """
        dont_sell_conditions = []

        # Don't short if there isn't 3% downside potential
        # Inverted from: don't buy if close_1h.max() < close * profit_threshold
        # Shorts: don't sell if close_1h.min() > close / profit_threshold
        dont_sell_conditions.append(
            dataframe['close_1h'].rolling(self.lookback_candles.value).min() >
            dataframe['close'] / self.profit_threshold.value
        )

        # Short Entry Signal 1: ewo1_short
        # Original: rsi_fast < 35, close < ma_buy * 0.984, EWO > 2.403, rsi < 72
        # Shorts: rsi_fast > 65, close > ma_sell * 1.016, EWO < -2.403, rsi > 28
        dataframe.loc[
            (dataframe['rsi_fast'] > 65) &  # Inverted: 100 - 35 = 65
            (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value) &
            (dataframe['EWO'] < self.ewo_low.value) &  # Inverted sign
            (dataframe['rsi'] > self.rsi_sell.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] > dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.low_offset.value),
            ['enter_short', 'enter_tag']
        ] = (1, 'ewo1_short')

        # Short Entry Signal 2: ewo2_short (more aggressive, very overbought)
        # Original: + close < ma * 0.942, EWO > -5.585, rsi < 25
        # Shorts: + close > ma * 1.058, EWO < 5.585, rsi > 75
        dataframe.loc[
            (dataframe['rsi_fast'] > 65) &
            (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value) &
            (dataframe['EWO'] < self.ewo_low_2.value) &
            (dataframe['rsi'] > self.rsi_sell.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] > dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.low_offset.value) &
            (dataframe['rsi'] > 75),  # Inverted: 100 - 25 = 75
            ['enter_short', 'enter_tag']
        ] = (1, 'ewo2_short')

        # Short Entry Signal 3: ewohigh_short (extreme high EWO = very overbought)
        # Original: ewolow with EWO < -14.378 (extreme low = oversold)
        # Shorts: ewohigh with EWO > 14.378 (extreme high = overbought)
        dataframe.loc[
            (dataframe['rsi_fast'] > 65) &
            (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value) &
            (dataframe['EWO'] > self.ewo_high.value) &  # High EWO = overbought
            (dataframe['volume'] > 0) &
            (dataframe['close'] > dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.low_offset.value),
            ['enter_short', 'enter_tag']
        ] = (1, 'ewohigh_short')

        # Apply don't sell conditions
        if dont_sell_conditions:
            for condition in dont_sell_conditions:
                dataframe.loc[condition, 'enter_short'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate short exit signals (inverted from long sell signals).

        Original Strategy Exit Conditions (Longs):
        - close > sma_9 AND close > ma_sell * 1.401 AND rsi > 50 AND rsi_fast > rsi_slow
        - OR close < hma_50 AND close > ma_sell * 1.084 AND rsi_fast > rsi_slow

        Shorts Inversion:
        - close < sma_9 AND close < ma_cover * 0.599 AND rsi < 50 AND rsi_fast < rsi_slow
        - OR close > hma_50 AND close < ma_cover * 0.916 AND rsi_fast < rsi_slow

        Args:
            dataframe: OHLCV dataframe with indicators
            metadata: Pair metadata

        Returns:
            DataFrame: Dataframe with exit signals
        """
        conditions = []

        # Exit condition: Price drops below SMAs with bearish RSI momentum
        # Original: (close > sma_9) & (close > ma_sell * 1.401) & (rsi > 50) & (rsi_fast > rsi_slow)
        # Shorts: (close < sma_9) & (close < ma_cover * 0.599) & (rsi < 50) & (rsi_fast < rsi_slow)
        conditions.append(
            (dataframe['close'] < dataframe['sma_9']) &
            (dataframe['close'] < dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.low_offset_2.value) &
            (dataframe['rsi'] < 50) &
            (dataframe['volume'] > 0) &
            (dataframe['rsi_fast'] < dataframe['rsi_slow']) |
            # OR condition: Price rises above HMA with bearish momentum
            # Original: (close < hma_50) & (close > ma_sell * 1.084) & (rsi_fast > rsi_slow)
            # Shorts: (close > hma_50) & (close < ma_cover * 0.916) & (rsi_fast < rsi_slow)
            (dataframe['close'] > dataframe['hma_50']) &
            (dataframe['close'] < dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.low_offset.value) &
            (dataframe['volume'] > 0) &
            (dataframe['rsi_fast'] < dataframe['rsi_slow'])
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'exit_short'] = 1

        return dataframe
