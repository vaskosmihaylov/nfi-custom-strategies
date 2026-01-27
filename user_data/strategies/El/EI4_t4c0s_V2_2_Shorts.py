"""
EI4_t4c0s_V2_2_Shorts Strategy

A shorts-only variant of the EI4_t4c0s_V2_2 strategy, designed to profit in bear markets
and during overbought conditions. This strategy mirrors the successful long-only approach
but inverts the logic for short positions.

Original EI4_t4c0s_V2_2 Strategy (Longs):
- Uses volatility-based dynamic stoploss with "moon mode"
- Multiple entry signals: lambo2, buy1ewo, buy2ewo, cofi
- EWO (Elliott Wave Oscillator) for momentum identification
- Unclog mechanism: -4% after 4 days

Strategy Concept:
Uses Elliott Wave Oscillator (EWO) and volatility indicators to identify overbought
conditions and mean reversion opportunities for short entries. Implements "crash mode"
stoploss that tightens during market crashes (inverse of moon mode).

Entry Conditions (OR logic - any can trigger entry):
1. Lambo Short: Parabolic pump (price > EMA, high RSI)
2. Sell1EWO Short: Rally during downtrend (EWO < MEAN_DN)
3. Sell2EWO Short: Extreme overbought (EWO > UP_FIB)
4. COFI Short: Stochastic overbought crossdown

Exit Conditions:
- Signal: Price falls below dynamic thresholds
- ROI: 0.99 (effectively disabled, let signals/stops control)
- Stop Loss: -0.99 base (custom stoploss manages risk)
- Crash Mode Stoploss: Dynamic trailing based on volatility

Key Differences from Long Strategy:
- Inverted entry/exit signals (overbought vs oversold)
- Crash mode stoploss: Tightens during crashes vs trailing during moons
- Shorter unclog window: 3 days vs 4 days (shorts are riskier)
- Uses min_l (distance from lows) vs max_l (distance from highs)
- Max 4 short positions via confirm_trade_entry()

Author: Derived from EI4_t4c0s_V2_2
Version: 1.0.0
"""

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from typing import Optional
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
import math
import logging

logger = logging.getLogger(__name__)


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class EI4_t4c0s_V2_2_Shorts(IStrategy):
    """
    Shorts-only Elliott Wave Oscillator strategy with crash mode stoploss.

    This strategy is designed to run in PARALLEL with EI4_t4c0s_V2_2 (longs)
    in separate containers to evaluate short performance independently.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # Hyperparameter configurations from long strategy
    # Will be optimized for shorts later
    buy_params = {
        "base_nb_candles_buy": 12,
        "rsi_buy": 58,
        "ewo_high": 3.001,
        "ewo_low": -10.289,
        "low_offset": 0.987,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179
    }

    sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    # ROI table (effectively disabled, let signals control)
    minimal_roi = {
        "0": 0.99,
    }

    # Stoploss (custom_stoploss manages risk)
    stoploss = -0.99
    sl1 = DecimalParameter(-0.013, -0.005, default=-0.013, space='sell', optimize=True)

    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 30, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 30, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.975, 0.995, decimals=3, default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(30, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(30, 55, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    rsi_buy = IntParameter(35, 60, default=buy_params['rsi_buy'], space='buy', optimize=True)
    move = IntParameter(35, 60, default=48, space='buy', optimize=True)
    mms = IntParameter(6, 20, default=12, space='buy', optimize=True)
    mml = IntParameter(300, 400, default=360, space='buy', optimize=True)

    # cofi
    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize=is_optimize_cofi)

    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4, space='buy', optimize=True, load=True)
    use_custom_stoploss = True
    process_only_new_candles = True

    # Custom Entry
    last_entry_price = None

    # Unclog (3 days for shorts vs 4 days for longs)
    unclog_days = IntParameter(1, 5, default=3, space='sell', optimize=True)
    unclog = DecimalParameter(0.01, 0.08, default=0.04, decimals=2, space='sell', optimize=True)

    # Phase 1: ATR-Based Dynamic Stop Loss
    atr_stop_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='sell', optimize=True)
    max_atr_stop = DecimalParameter(-0.20, -0.10, default=-0.15, space='sell', optimize=False)
    min_atr_stop = DecimalParameter(-0.05, -0.02, default=-0.03, space='sell', optimize=False)

    # Phase 2: Reversal Detection (for shorts, use higher RSI threshold)
    reversal_rsi_threshold = IntParameter(55, 70, default=60, space='sell', optimize=True)
    reversal_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.02, space='sell', optimize=True)

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

    ### Trailing Stop with Crash Mode ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss with 3-layer protection for shorts:
        
        Layer 0 (NEW): ATR-based dynamic stop for LOSING trades (prevents liquidations)
        Layer 1: 48-hour protection window (no tightening)
        Layer 2: Crash mode trailing for PROFITABLE shorts
        
        Logic:
        - LOSING trades: Use ATR-based stop immediately (NEW)
        - Hours 0-48 (profitable): Don't tighten (return 1.0)
        - Hours 48+ (profitable): Apply crash mode
          - Crash mode (min_l < 0.003): Making new lows = tighten stops
          - Rally mode (min_l >= 0.003): Not making new lows = allow trailing
        
        For shorts:
        - Crash mode (min_l < 0.003): Making new lows = profitable short → TIGHTEN stops
        - Rally mode (min_l >= 0.003): Not making new lows → Allow trailing
        
        This is the inverse of the long strategy's moon mode which loosens stops
        when making new highs. For shorts, we want to protect profit when the
        market crashes (our shorts are very profitable).
        
        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current timestamp
            current_rate: Current price
            current_profit: Current profit ratio
            **kwargs: Additional arguments
            
        Returns:
            float: Stoploss percentage (must be MORE NEGATIVE than base to tighten)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        # PHASE 1 (NEW): ATR-Based Dynamic Stop Loss (for losing trades)
        # Prevents liquidations by using volatility-adjusted stops
        if current_profit < 0:
            # Calculate ATR-based stop: 2x ATR as percentage
            atr_stop = -1 * self.atr_stop_multiplier.value * current_candle['atr_pcnt']
            
            # Clamp between bounds to prevent too tight or too loose stops
            # max_atr_stop = -0.15 (maximum loss allowed)
            # min_atr_stop = -0.03 (minimum protection)
            atr_stop = max(atr_stop, self.max_atr_stop.value)
            atr_stop = min(atr_stop, self.min_atr_stop.value)
            
            # Return ATR-based stop for losing trades
            # This prevents trades from reaching -50%+ losses
            return atr_stop
        
        # Calculate trade duration in hours
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # Phase 2: First 48 hours - PROTECTION WINDOW (for profitable trades)
        # Return 1.0 to keep base stoploss (-0.99) without tightening
        if trade_duration < 48:
            return 1.0
        
        # Phase 3: After 48 hours - Apply crash mode ONLY for profitable shorts
        SLT1 = current_candle['move_mean']
        SL1 = self.sl1.value
        SLT2 = current_candle['move_mean_x']
        SL2 = current_candle['move_mean_x'] - current_candle['move_mean']
        display_profit = current_profit * 100
        slt1 = SLT1 * 100
        sl1 = SL1 * 100
        slt2 = SLT2 * 100
        sl2 = SL2 * 100
        
        # Crash mode: When making new lows (min_l < 0.003), tighten stops
        # Small min_l = price at/near lows = SHORT IS PROFITABLE
        if current_candle['min_l'] < .003:
            # Short is very profitable (market crashing to new lows), use conservative stop
            if SLT1 is not None and current_profit > SL1:
                self.dp.send_msg(f'*** {pair} *** SHORT Profit {display_profit:.2f}% - Crash Mode {slt1:.2f}/{sl1:.2f} activated')
                logger.info(f'*** {pair} *** SHORT Profit {display_profit:.2f}% - Crash Mode {slt1:.2f}/{sl1:.2f} activated')
                return SL1
        
        # Rally mode: Not making new lows (price rising away from lows), allow trailing
        else:
            if SLT2 is not None and current_profit > SLT2:
                self.dp.send_msg(f'*** {pair} *** SHORT Profit {display_profit:.2f}% - {slt2:.2f}/{sl2:.2f} activated')
                logger.info(f'*** {pair} *** SHORT Profit {display_profit:.2f}% - {slt2:.2f}/{sl2:.2f} activated')
                return SL2
            if SLT1 is not None and current_profit > SLT1:
                self.dp.send_msg(f'*** {pair} *** SHORT Profit {display_profit:.2f}% - {slt1:.2f}/{sl1:.2f} activated')
                logger.info(f'*** {pair} *** SHORT Profit {display_profit:.2f}% - {slt1:.2f}/{sl1:.2f} activated')
                return SL1
        
        # Default: Return 1.0 to keep base stoploss active
        return 1.0

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Custom entry price to avoid order clustering.

        For shorts: DECREMENT price (enter lower) if same as last entry.
        This is the inverse of longs which increment (enter higher).

        Args:
            pair: Trading pair
            trade: Trade object (None for new entries)
            current_time: Current timestamp
            proposed_rate: Proposed entry rate
            entry_tag: Entry signal tag
            side: Trade direction

        Returns:
            float: Adjusted entry price
        """
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:
            # For shorts: INCREMENT to enter at a better (higher) price
            # When shorting, you SELL first, so entering at HIGHER price = better entry
            entry_price *= self.increment.value  # Multiply to enter higher
            logger.info(f"{pair} SHORT: Incremented entry price: {entry_price} based on previous entry price: {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit with crash mode logic (inverse of moon mode).

        For shorts:
        - Block ROI exit if NOT making new lows (min_l < 0.003)
        - Block exits below 0.3% profit

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

        # Block ROI exit if not making new lows (inverted from longs)
        # For shorts: min_l > 0.003 = price far from lows = not crashing = block exit
        if exit_reason == 'roi' and (last_candle['min_l'] > 0.003):
            return False

        # Handle freak events - block exits with low profit
        if exit_reason == 'Down Trend Soon' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} Waiting for Profit")
            self.dp.send_msg(f'{trade.pair} Waiting for Profit')
            return False

        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI is below 0.003")
            self.dp.send_msg(f'{trade.pair} ROI is below 0.003')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        """
        4-Layer Exit System for Shorts (UPDATED):
        
        Layer 1: Base stoploss (-0.99) = Safety net (almost never hit)
        Layer 2: custom_stoploss = ATR + Crash mode trailing
        Layer 3: custom_exit reversal detection (NEW) = Indicator-based early exits
        Layer 4: custom_exit unclog/zombie = Time-based cleanup
        
        Logic:
        - Reversal detection (NEW): Exit losing trades when indicators signal bullish reversal
        - Hours 0-48: No forced time-based exits, but reversal detection active
        - After 48 hours:
          - If losing > 4%: Force exit ('unclog_short')
          - If at breakeven (-0.5% to +0.5%): Force exit ('zombie_short')
          - Otherwise: Let crash mode handle it
        
        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current timestamp
            current_rate: Current rate
            current_profit: Current profit
            
        Returns:
            str: Exit reason or None
        """
        # Get current candle data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        # PHASE 2 (NEW): Indicator Reversal Detection (for losing shorts)
        # Exit early when indicators signal bullish reversal
        if current_profit < self.reversal_loss_threshold.value:
            
            # 1. RSI Reversal: RSI rises above threshold = bullish momentum returning
            if current_candle['rsi'] > self.reversal_rsi_threshold.value:
                logger.info(f"*** {pair} *** SHORT Reversal: RSI {current_candle['rsi']:.2f} > {self.reversal_rsi_threshold.value} at {current_profit*100:.2f}% loss")
                return 'reversal_rsi_short'
            
            # 2. EWO Reversal: EWO crosses above mean = upward momentum
            if current_candle['EWO'] > current_candle['EWO_MEAN_UP']:
                logger.info(f"*** {pair} *** SHORT Reversal: EWO {current_candle['EWO']:.2f} > mean {current_candle['EWO_MEAN_UP']:.2f} at {current_profit*100:.2f}% loss")
                return 'reversal_ewo_short'
            
            # 3. HMA Breakout: Price breaks above Hull MA = trend change (bullish for price, bad for short)
            if current_candle['close'] > current_candle['hma_50']:
                logger.info(f"*** {pair} *** SHORT Reversal: Price {current_candle['close']:.6f} > HMA50 {current_candle['hma_50']:.6f} at {current_profit*100:.2f}% loss")
                return 'reversal_hma_short'
        
        # Calculate trade duration in hours
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # Phase 3: First 48 hours - NO forced time-based exits (reversal detection still active above)
        if trade_duration_hours < 48:
            return None
        
        # Phase 4: After 48 hours - Unclog losing/zombie trades
        
        # Unclog: Force exit if losing > 4% (worst case scenario)
        if current_profit < -0.04:
            logger.info(f"*** {pair} *** SHORT Unclog: {current_profit*100:.2f}% loss after {trade_duration_hours:.1f} hours")
            return 'unclog_short'
        
        # Zombie: Force exit if stuck at breakeven after 48h
        if -0.005 <= current_profit <= 0.005:
            logger.info(f"*** {pair} *** SHORT Zombie: {current_profit*100:.2f}% (breakeven) after {trade_duration_hours:.1f} hours")
            return 'zombie_short'
        
        # Profitable trades: Let custom_stoploss (crash mode) handle trailing
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Fixed 3x leverage for all short trades.

        Returns:
            float: Leverage multiplier (3.0 = 3x)
        """
        return 3.0

    # Exit signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'

    position_adjustment_enable = False
    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate technical indicators.

        Same indicators as long strategy - only the interpretation differs.
        """
        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.low_offset.value)
        dataframe['ma_hi'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (self.high_offset.value)
        dataframe['ma_hi_2'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (self.high_offset_2.value)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        # HMA-BUY SQUEEZE
        dataframe['HMA_SQZ'] = (((dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] - dataframe['hma_50'])
            / dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) * 100)

        dataframe['zero'] = 0
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe.loc[dataframe['EWO'] > 0, "EWO_UP"] = dataframe['EWO']
        dataframe.loc[dataframe['EWO'] < 0, "EWO_DN"] = dataframe['EWO']
        dataframe['EWO_UP'].ffill()
        dataframe['EWO_DN'].ffill()
        dataframe['EWO_MEAN_UP'] = dataframe['EWO_UP'].mean()
        dataframe['EWO_MEAN_DN'] = dataframe['EWO_DN'].mean()
        dataframe['EWO_UP_FIB'] = dataframe['EWO_MEAN_UP'] * 1.618
        dataframe['EWO_DN_FIB'] = dataframe['EWO_MEAN_DN'] * 1.618

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # Check how far we are from min and max
        dataframe['max'] = dataframe['OHLC4'].rolling(self.mms.value).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(self.mms.value).min() / dataframe['OHLC4'] - 1)

        dataframe['max_l'] = dataframe['OHLC4'].rolling(self.mml.value).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(self.mml.value).min() / dataframe['OHLC4'] - 1)

        # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['OHLC4'].rolling(self.move.value)
        rolling_max = rolling_window.max()
        rolling_min = rolling_window.min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * 1.6
        dataframe['exit_mean'] = rolling_min * (1 + dataframe['move_mean'])
        dataframe['exit_mean_x'] = rolling_min * (1 + dataframe['move_mean_x'])
        dataframe['enter_mean'] = rolling_max * (1 - dataframe['move_mean'])
        dataframe['enter_mean_x'] = rolling_max * (1 - dataframe['move_mean_x'])
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=5) / dataframe['OHLC4'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define SHORT entry conditions (inverted from long strategy).

        All 4 entry signals are inverted for overbought conditions:
        1. lambo_short: Parabolic pumps
        2. sell1ewo_short: Rallies during downtrend
        3. sell2ewo_short: Extreme overbought
        4. cofi_short: Stochastic overbought crossdown
        """
        # Lambo Short: Parabolic pump (inverted from lambo2)
        # Long: close < ema_14 * 0.981, rsi_4 < 44, rsi_14 < 39
        # Short: close > ema_14 * (2 - 0.981) = 1.019, rsi_4 > (100-44) = 56, rsi_14 > (100-39) = 61
        lambo_short = (
            (dataframe['close'] > (dataframe['ema_14'] * (2 - self.lambo2_ema_14_factor.value))) &
            (dataframe['rsi_4'] > (100 - int(self.lambo2_rsi_4_limit.value))) &
            (dataframe['rsi_14'] > (100 - int(self.lambo2_rsi_14_limit.value))) &
            (dataframe['atr_pcnt'] > dataframe['min_l']) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[lambo_short, 'enter_short'] = 1
        dataframe.loc[lambo_short, 'enter_tag'] = 'lambo_short'

        # Sell1EWO Short: Rally during downtrend (inverted from buy1ewo)
        # Long: rsi_fast < 35, close < ma_lo, EWO > EWO_MEAN_UP
        # Short: rsi_fast > 65, close > ma_hi, EWO < EWO_MEAN_DN
        sell1ewo_short = (
                (dataframe['rsi_fast'] > 65) &
                (dataframe['close'] > dataframe['ma_hi']) &
                (dataframe['EWO'] < dataframe['EWO_MEAN_DN']) &
                (dataframe['close'] > dataframe['exit_mean_x']) &
                (dataframe['close'].shift() > dataframe['exit_mean_x'].shift()) &
                (dataframe['rsi'] > (100 - self.rsi_buy.value)) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0)
        )
        dataframe.loc[sell1ewo_short, 'enter_short'] = 1
        dataframe.loc[sell1ewo_short, 'enter_tag'] = 'sell1ewo_short'

        # Sell2EWO Short: Extreme overbought (inverted from buy2ewo)
        # Long: rsi_fast < 35, close < ma_lo, EWO < EWO_DN_FIB
        # Short: rsi_fast > 65, close > ma_hi, EWO > EWO_UP_FIB
        sell2ewo_short = (
                (dataframe['rsi_fast'] > 65) &
                (dataframe['close'] > dataframe['ma_hi']) &
                (dataframe['EWO'] > dataframe['EWO_UP_FIB']) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0)
        )
        dataframe.loc[sell2ewo_short, 'enter_short'] = 1
        dataframe.loc[sell2ewo_short, 'enter_tag'] = 'sell2ewo_short'

        # COFI Short: Overbought crossdown (inverted from cofi)
        # Long: open < ema_8 * 0.98, crossed_above(fastk, fastd), fastk < 22, fastd < 20
        # Short: open > ema_8 * (2 - 0.98) = 1.02, crossed_below(fastk, fastd), fastk > 78, fastd > 80
        cofi_short = (
                (dataframe['open'] > dataframe['ema_8'] * (2 - self.buy_ema_cofi.value)) &
                (qtpylib.crossed_below(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] > (100 - self.buy_fastk.value)) &
                (dataframe['fastd'] > (100 - self.buy_fastd.value)) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] < dataframe['EWO_MEAN_DN']) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0)
            )
        dataframe.loc[cofi_short, 'enter_short'] = 1
        dataframe.loc[cofi_short, 'enter_tag'] = 'cofi_short'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define SHORT exit conditions (inverted from long strategy).

        Exit when price falls below thresholds (profit target reached for shorts).
        """
        # Exit Short Condition 1 (inverted from condition5)
        # Long: close > hma_50 AND close > ma_hi_2 AND close > exit_mean_x
        # Short: close < hma_50 AND close < ma_lo AND close < enter_mean_x
        condition_short_1 = (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] < dataframe['ma_lo']) &
                (dataframe['close'] < dataframe['enter_mean_x']) &
                (dataframe['rsi'] < 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] < dataframe['rsi_slow'])
            )
        dataframe.loc[condition_short_1, 'exit_short'] = 1
        dataframe.loc[condition_short_1, 'exit_tag'] = 'Close < Offset Lo 1'

        # Exit Short Condition 2 (inverted from condition6)
        # Long: close < hma_50 AND close > ma_hi
        # Short: close > hma_50 AND close < ma_lo
        condition_short_2 = (
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] < dataframe['ma_lo']) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] < dataframe['rsi_slow'])
            )
        dataframe.loc[condition_short_2, 'exit_short'] = 1
        dataframe.loc[condition_short_2, 'exit_tag'] = 'Close < Offset Lo 2'

        return dataframe


def pct_change(a, b):
    return (b - a) / a
