"""
ElliotV5_SMA_Shorts Strategy

A shorts-only variant of the ElliotV5_SMA strategy, designed to profit in bear markets
and during overbought conditions. This strategy mirrors the successful long-only approach
but inverts the logic for short positions.

Original ElliotV5_SMA Performance (Longs):
- Win Rate: 73.7%
- Avg Profit: 0.287%
- Total Trades: 1447
- Profit Factor: 1.11

Strategy Concept:
Uses Elliott Wave Oscillator (EWO) to identify momentum divergence and mean reversion
opportunities for short entries. EWO measures the percentage difference between fast
and slow SMAs.

Entry Conditions (OR logic - either triggers entry):
1. Bear Market Rally Short: Price above EMA during downtrend (EWO < -2.5), RSI > 50
2. Parabolic Pump Short: Extreme positive momentum (EWO > 12.0), RSI > 70

Exit Conditions:
- Signal: Price falls below EMA discount
- ROI: Time-decay targets (15% → 10% → 6% → 2%)
- Stop Loss: -12% (tighter than longs due to short squeeze risk)
- Trailing Stop: +2% activation, 0.5% trail

Key Differences from Long Strategy:
- Tighter stop loss: -12% vs -18.9% (shorts are riskier)
- Lower ROI targets: 15% max vs 21.5% (faster profit-taking)
- Asymmetric EWO thresholds: Adjusted for crypto bear market dynamics
- Always uses RSI filters: Extra confirmation to avoid short squeezes
- Max 4 short positions: Position limit enforcement via confirm_trade_entry

Author: Derived from ElliotV5_SMA
Version: 1.0.0
"""

from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional
from functools import reduce
from pandas import DataFrame
from datetime import datetime

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter


# Optimized parameters for short entries
short_params = {
    "base_nb_candles_short_entry": 17,
    "ewo_high": 12.0,
    "ewo_low": -2.5,
    "high_offset": 1.022,
    "rsi_short": 50,
    "rsi_overbought": 70
}

# Optimized parameters for short exits (covers)
cover_params = {
    "base_nb_candles_short_exit": 49,
    "low_offset": 0.981
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    """
    Elliott Wave Oscillator (EWO)

    Calculates the percentage difference between fast and slow SMAs.
    Despite the name, this uses Simple Moving Averages (SMA), not Exponential (EMA).

    Formula: ((SMA_fast - SMA_slow) / Close) * 100

    Interpretation:
    - Positive EWO: Fast SMA > Slow SMA (upward momentum)
    - Negative EWO: Fast SMA < Slow SMA (downward momentum)
    - High magnitude: Strong momentum divergence

    Args:
        dataframe: Price data
        ema_length: Fast SMA period (default 5, strategy uses 50)
        ema2_length: Slow SMA period (default 35, strategy uses 200)

    Returns:
        Series: EWO values as percentage
    """
    df = dataframe.copy()
    ema1 = ta.SMA(df, timeperiod=ema_length)
    ema2 = ta.SMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class ElliotV5_SMA_Shorts(IStrategy):
    """
    Shorts-only Elliott Wave Oscillator strategy for bear markets.

    This strategy is designed to run in PARALLEL with ElliotV5_SMA (longs)
    in separate containers for 30+ days to evaluate short performance independently.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # More conservative ROI for shorts (crypto pumps are violent)
    minimal_roi = {
        "0": 0.15,      # 15% profit target (vs 21.5% for longs)
        "40": 0.10,     # 10% after 40 minutes (vs 13.2%)
        "87": 0.06,     # 6% after 87 minutes (vs 8.6%)
        "201": 0.02     # 2% after 201 minutes (vs 3%)
    }

    # Stop loss for shorts (matches long strategy for consistent risk management)
    stoploss = -0.189  # Matches long strategy - allows 6.3% price movement with 3x leverage

    # Hyperopt parameters for short entries (use 'buy' space per Freqtrade convention)
    base_nb_candles_short_entry = IntParameter(
        5, 80, default=short_params['base_nb_candles_short_entry'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        1.01, 1.1, default=short_params['high_offset'], space='buy', optimize=True)

    # Hyperopt parameters for short exits/covers (use 'sell' space per Freqtrade convention)
    base_nb_candles_short_exit = IntParameter(
        5, 80, default=cover_params['base_nb_candles_short_exit'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=cover_params['low_offset'], space='sell', optimize=True)

    # EWO thresholds (asymmetric - adjusted for crypto dynamics)
    fast_ewo = 50    # Fast SMA period for EWO
    slow_ewo = 200   # Slow SMA period for EWO
    ewo_low = DecimalParameter(
        -8.0, -1.0, default=short_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        8.0, 20.0, default=short_params['ewo_high'], space='buy', optimize=True)

    # RSI filters (always enabled for shorts to reduce risk)
    rsi_short = IntParameter(
        40, 70, default=short_params['rsi_short'], space='buy', optimize=True)
    rsi_overbought = IntParameter(
        60, 80, default=short_params['rsi_overbought'], space='buy', optimize=True)

    # Trailing stop (tighter activation for shorts)
    trailing_stop = True
    trailing_stop_positive = 0.005      # 0.5% trail distance (same as longs)
    trailing_stop_positive_offset = 0.02  # +2% activation (vs +3% for longs)
    trailing_only_offset_is_reached = True

    # Exit signal configuration
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True

    # Timeframe and data settings
    timeframe = '5m'
    informative_timeframe = '1h'
    process_only_new_candles = True
    startup_candle_count = 2000

    # Position limits (enforced via confirm_trade_entry)
    max_open_trades = 4  # Max 4 short positions
    max_short_trades = 4  # Explicit short limit

    # Custom stoploss disabled (using fixed stop)
    use_custom_stoploss = False

    # Plot configuration
    plot_config = {
        'main_plot': {
            'ma_short_entry': {'color': 'red'},
            'ma_short_exit': {'color': 'green'},
        },
    }

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Enforce maximum short position limit.

        This callback is executed before every entry to ensure we don't exceed
        max_short_trades positions. Essential for risk management in crypto shorts.

        Args:
            side: Trade direction ('long' or 'short')

        Returns:
            bool: True to confirm entry, False to reject
        """
        # Only allow shorts in this strategy
        if side == "long":
            return False  # Reject any long signals

        # Count current open short positions
        short_count = 0
        trades = Trade.get_trades_proxy(is_open=True)
        for trade in trades:
            if trade.is_short:
                short_count += 1

        # Check if we can open another short
        if short_count >= self.max_short_trades:
            return False  # Already at max short positions

        return True  # Confirm entry

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Fixed 3x leverage for all short trades.

        Increased from 2x to 3x for higher profit potential while maintaining
        risk control through -18.9% stop loss (allows 6.3% price movement).

        Returns:
            float: Leverage multiplier (3.0 = 3x)
        """
        return 3.0

    def informative_pairs(self):
        """
        Define additional timeframes to download.

        Currently requests 1h data but doesn't use it. Could be used for
        trend filtering in future versions.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        """
        Retrieve informative timeframe data.

        Currently unused but available for future multi-timeframe analysis.
        """
        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators needed for entry/exit decisions.

        Indicators:
        - EMAs: Multiple periods for entry/exit price reference
        - EWO: Elliott Wave Oscillator (momentum indicator)
        - RSI: Relative Strength Index (overbought/oversold filter)
        """
        # Calculate EMAs for entry signals (short at premium above EMA)
        for val in self.base_nb_candles_short_entry.range:
            dataframe[f'ma_short_entry_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate EMAs for exit signals (cover at discount below EMA)
        for val in self.base_nb_candles_short_exit.range:
            dataframe[f'ma_short_exit_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate Elliott Wave Oscillator
        # EWO = ((SMA_50 - SMA_200) / Close) * 100
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # Calculate RSI (14-period)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define SHORT entry conditions.

        Two conditions (OR logic - either can trigger entry):

        Condition 1: Bear Market Rally Short
        - Price pumps above EMA during downtrend (EWO < -2.5)
        - RSI > 50 confirms upward momentum
        - Short the temporary rally in bearish trend

        Condition 2: Parabolic Pump Short
        - Extreme positive momentum (EWO > 12.0)
        - RSI > 70 confirms overbought
        - Short the parabolic blow-off top

        Both conditions require:
        - Price trading at premium above EMA (high_offset * EMA)
        - Volume confirmation
        """
        conditions = []

        # Condition 1: Bear Market Rally Short (Negative EWO)
        # Short temporary pumps during downtrends
        conditions.append(
            (
                # Price above EMA discount threshold (trading at premium)
                (dataframe['close'] > (
                    dataframe[f'ma_short_entry_{self.base_nb_candles_short_entry.value}'] *
                    self.high_offset.value
                )) &
                # Weak/negative momentum (downtrend with rally)
                (dataframe['EWO'] < self.ewo_low.value) &
                # RSI confirms upward movement (rally to short)
                (dataframe['rsi'] > self.rsi_short.value) &
                # Volume confirmation
                (dataframe['volume'] > 0)
            )
        )

        # Condition 2: Parabolic Pump Short (High Positive EWO)
        # Short extreme overbought conditions
        conditions.append(
            (
                # Price above EMA discount threshold (trading at premium)
                (dataframe['close'] > (
                    dataframe[f'ma_short_entry_{self.base_nb_candles_short_entry.value}'] *
                    self.high_offset.value
                )) &
                # Strong positive momentum (parabolic/overbought)
                (dataframe['EWO'] > self.ewo_high.value) &
                # RSI confirms overbought
                (dataframe['rsi'] > self.rsi_overbought.value) &
                # Volume confirmation
                (dataframe['volume'] > 0)
            )
        )

        # Apply conditions with OR logic (either condition triggers)
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_short'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define SHORT exit (cover) conditions.

        Exit when:
        - Price falls below EMA threshold (low_offset * EMA)
        - This indicates the short has achieved profit target
        - Cover the short position to lock in gains

        Additional exits handled by:
        - ROI table (time-decay profit targets)
        - Stop loss (-12%)
        - Trailing stop (+2% activation)
        """
        conditions = []

        # Cover short when price falls below EMA threshold
        conditions.append(
            (
                # Price below EMA threshold (profit target reached)
                (dataframe['close'] < (
                    dataframe[f'ma_short_exit_{self.base_nb_candles_short_exit.value}'] *
                    self.low_offset.value
                )) &
                # Volume confirmation
                (dataframe['volume'] > 0)
            )
        )

        # Apply exit conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_short'
            ] = 1

        return dataframe
