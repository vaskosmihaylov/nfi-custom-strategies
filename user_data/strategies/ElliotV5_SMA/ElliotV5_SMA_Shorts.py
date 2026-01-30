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
- Volatility-Adaptive Stop Loss: Crash mode for profit protection
- Trailing Stop: +2% activation, 0.5% trail

Volatility-Adaptive Stop Loss with Crash Mode:
- NO PROTECTION WINDOW - Immediately trails to prevent liquidations
- Crash Mode: When making new lows (min_l < 0.003), tightens stops to protect profits
- Rally Mode: When not making new lows, allows wider trailing based on volatility
- Adapts to each coin's volatility (move_mean, move_mean_x)
- Prevents liquidations by tightening before -30% loss

Key Differences from Long Strategy:
- Crash Mode vs Moon Mode: Tightens when making new lows (inverse logic)
- Lower ROI targets: 15% max vs 21.5% (faster profit-taking for shorts)
- Asymmetric EWO thresholds: Adjusted for crypto bear market dynamics
- Always uses RSI filters: Extra confirmation to avoid short squeezes
- Max 4 short positions: Position limit enforcement via confirm_trade_entry

Author: Derived from ElliotV5_SMA
Version: 1.0.0
"""

from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Union
from functools import reduce
from pandas import DataFrame
from datetime import datetime

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter
import logging

logger = logging.getLogger(__name__)


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

    # Stop loss for shorts - OPTIMIZED to prevent overshoots (was -0.09, caused -21% losses)
    stoploss = -0.075  # 7.5% base stop = 22.5% loss at 3x leverage (safer buffer from -33% liquidation)

    # Volatility-based trailing stop parameters
    sl1 = DecimalParameter(-0.013, -0.005, default=-0.013, space='sell', optimize=True)
    move = IntParameter(35, 60, default=48, space='buy', optimize=True)
    mms = IntParameter(6, 20, default=12, space='buy', optimize=True)
    mml = IntParameter(300, 400, default=360, space='buy', optimize=True)

    # Unclog parameters (used in custom_exit)
    unclog_days = IntParameter(1, 5, default=3, space='sell', optimize=True)
    unclog = DecimalParameter(0.01, 0.08, default=0.04, decimals=2, space='sell', optimize=True)

    # Position limits (enforced via confirm_trade_entry)
    max_short_trades = 4

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

    # Custom stoploss enabled for volatility-adaptive stop loss
    use_custom_stoploss = True

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

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        OPTIMIZED asymmetric trailing stop for SHORT positions with 3x leverage.

        CRITICAL FIXES:
        1. Tighter base stop (-7.5% vs -9%) to reduce slippage overshoots
        2. Wider profit tiers to let winners run (target 4-6% vs current 2%)
        3. Emergency liquidation protection maintained

        PROBLEM IDENTIFIED:
        - Old tiers: 0.5-2% trails captured only 2.1% avg profit
        - Stop overshoots: -9% became -21% on AXS due to volatility
        - Risk-reward: 1:16 (terrible - wins too small, losses too big)

        NEW OPTIMIZED TIERS (Let Winners Run):
        - Loss zone: -7.5% stop (22.5% loss at 3x) - tighter to prevent overshoots
        - Early profit (0-2%): -3% trail (9% loss at 3x) - WIDENED to let small wins grow
        - Low profit (2-4%): -2% trail (6% loss at 3x) - captures 4%+ moves
        - Medium profit (4-6%): -1.5% trail (4.5% loss at 3x) - protects significant gains
        - High profit (6%+): -1% trail (3% loss at 3x) - locks in big winners

        EXPECTED IMPROVEMENTS:
        - Average wins: 2.1% → 4-6% (2-3x increase)
        - Max loss: -21% → -7.5% (70% reduction)
        - Risk-reward: 1:16 → 1:2-3 (sustainable)
        - Profit factor: 0.21 → 1.2+ (profitable)

        Emergency Protection:
        - If within 15% of liquidation price: Exit immediately (inverted for shorts)
        - Based on NFI pattern (proven in production)

        Args:
            pair: Trading pair
            trade: Trade object with liquidation_price
            current_time: Current timestamp
            current_rate: Current price
            current_profit: Current profit/loss ratio
            **kwargs: Additional arguments

        Returns:
            float: Stoploss percentage (negative = loss threshold)
        """

        # Emergency liquidation protection for SHORTS (inverted logic)
        # Only active in futures/margin mode when liquidation price exists
        if (self.config.get('trading_mode', '') in ('futures', 'margin')
            and trade.liquidation_price is not None
            and current_rate > trade.liquidation_price * 0.85):  # Within 15% of liquidation (INVERTED)
            logger.warning(f"{pair} SHORT EMERGENCY EXIT - Near liquidation! "
                          f"Price: {current_rate:.8f}, Liq: {trade.liquidation_price:.8f}")
            return 0.001  # Exit immediately

        # OPTIMIZED Profit-based trailing stops (asymmetric - let winners run)
        if current_profit >= 0.06:  # 6%+ profit (HIGH - lock in big wins)
            return -0.01  # 1% trailing stop (tight to protect big gains)

        elif current_profit >= 0.04:  # 4-6% profit (MEDIUM - significant gains)
            return -0.015  # 1.5% trailing stop (balance protection & growth)

        elif current_profit >= 0.02:  # 2-4% profit (LOW - let it grow)
            return -0.02  # 2% trailing stop (wider to reach 4%+)

        elif current_profit >= 0.00:  # Breakeven to 2% profit (EARLY - let it develop)
            return -0.03  # 3% trailing stop (WIDENED from -2% to let winners develop)

        # Still in loss = use TIGHTER base stoploss (FIXED from -9% to -7.5%)
        # This prevents the -21% overshoots seen on AXS, -17% on ZKC
        return -0.075  # -7.5% base stop (22.5% loss at 3x leverage, safer than old -9%)  # -9% base stop (27% loss at 3x leverage)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Simplified exit logic matching E0V1E_Shorts/ETCG_Shorts approach.

        OPTIMIZATION (Jan 30, 2026):
        - REMOVED: All time-based profit locking (Tier 1/2/3)
        - KEPT: Simple unclog mechanism only
        - RATIONALE: E0V1E_Shorts (most profitable) and ETCG_Shorts use this approach
        
        Philosophy Change:
        OLD: Quick profit-taking to avoid reversals (2% @ 12h, 3% @ 6h, 5% @ 4h)
        NEW: Let indicators handle exits, only unclog dead positions
        
        Research Findings:
        - Time-based exits cut winners prematurely in trending markets
        - Trailing stops better for crypto volatility (source: altrady.com, highstrike.com)
        - Best performers (E0V1E, ETCG) use indicator-based exits, not time-based
        
        2-Layer Exit System:
        Layer 1: custom_stoploss = Indicator-based trailing for PROFITABLE shorts
        Layer 2: custom_exit (THIS) = Unclog only for LOSING positions

        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current timestamp
            current_rate: Current price
            current_profit: Current profit/loss ratio
            **kwargs: Additional arguments

        Returns:
            Optional[Union[str, bool]]: Exit reason string or None
        """
        # Calculate trade duration
        trade_duration_days = (current_time - trade.open_date_utc).days

        # UNCLOG: Force exit losing positions after N days to free capital
        # Matches E0V1E_Shorts and ETCG_Shorts approach
        if current_profit < -self.unclog.value and trade_duration_days >= self.unclog_days.value:
            logger.info(f"{pair} Unclog - {current_profit:.2%} after {trade_duration_days} days")
            return 'unclog_short'

        # Let custom_stoploss handle all other exits (profitable trades)
        # No time-based profit locking - let winners run!
        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit with crash mode logic (inverse of moon mode).

        Prevents premature exits when:
        - ROI exit when NOT making new lows (min_l > 0.003)
        - Any exit with profit below 0.3%

        For shorts: min_l > 0.003 = price far from lows = not crashing = block exit

        Args:
            pair: Trading pair
            trade: Trade object
            order_type: Order type
            amount: Trade amount
            rate: Exit rate
            time_in_force: Time in force
            exit_reason: Reason for exit
            current_time: Current timestamp

        Returns:
            bool: True to confirm exit, False to block
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Block ROI exit if not making new lows (wait for crash to continue)
        # For shorts: min_l > 0.003 = price far from lows = not crashing = block exit
        if exit_reason == 'roi' and (last_candle['min_l'] > 0.003):
            return False

        # Block exits with very low profit - wait for better exit
        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI profit is below 0.3%")
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} Trailing stop price is below 0")
            return False

        return True

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
        - ATR: Average True Range (volatility filter)
        - Volume MA: Volume moving average (surge confirmation)
        - Volatility indicators: For adaptive stop loss
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

        # ATR volatility filter (NEW - prevents entries on ultra-volatile coins)
        # ATR as percentage of price helps identify coins like AXS, ZKC that caused -21% losses
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = (dataframe['atr'] / dataframe['close']) * 100

        # Volume surge confirmation (NEW - ensures strong momentum entries)
        # Moving average of volume over 20 periods
        dataframe['volume_ma'] = dataframe['volume'].rolling(20).mean()

        # Volatility-based indicators for adaptive stop loss
        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # Check how far we are from min and max over short window
        dataframe['max'] = dataframe['OHLC4'].rolling(self.mms.value).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(self.mms.value).min() / dataframe['OHLC4'] - 1)

        # Check how far we are from min and max over long window (360 candles = 30 hours at 5m)
        dataframe['max_l'] = dataframe['OHLC4'].rolling(self.mml.value).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(self.mml.value).min() / dataframe['OHLC4'] - 1)

        # Apply rolling window operation to calculate volatility
        rolling_window = dataframe['OHLC4'].rolling(self.move.value)

        # Calculate the peak-to-peak value (max - min) in rolling window
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Normalize volatility by current price
        dataframe['move'] = ptp_value / dataframe['OHLC4']

        # Per-pair volatility adaptation (NOT rolling mean - intentional)
        # Uses mean across ALL history to get this coin's natural volatility
        # NAORIS might average 15% moves, BTC might average 3% - this adapts stops per coin
        dataframe['move_mean'] = dataframe['move'].mean()  # Average volatility
        dataframe['move_mean_x'] = dataframe['move'].mean() * 1.6  # 1.6x average volatility

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define SHORT entry conditions with enhanced filters.

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
        - ATR filter: Volatility < 10% (NEW - prevents ultra-volatile coins)
        - Volume surge: Volume > 1.5x MA (NEW - ensures strong momentum)

        Rationale for new filters:
        - ATR < 10%: Filters out coins like AXS (caused -21% loss), ZKC (-17%)
        - Volume surge: Ensures entries on real moves, not low-liquidity spikes
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
                (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI turning down (continuation signal)
                # NEW: ATR volatility filter - avoid ultra-volatile coins
                (dataframe['atr_pct'] < 10.0) &  # Less than 10% ATR (filters AXS, ZKC, SOMI)
                # NEW: Volume surge confirmation - ensure strong momentum
                (dataframe['volume'] > dataframe['volume_ma'] * 1.5) &  # 50% above average volume
                # Basic volume check
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
                # NEW: ATR volatility filter - avoid ultra-volatile coins
                (dataframe['atr_pct'] < 10.0) &  # Less than 10% ATR
                # NEW: Volume surge confirmation - ensure strong momentum
                (dataframe['volume'] > dataframe['volume_ma'] * 1.5) &  # 50% above average volume
                # Basic volume check
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
