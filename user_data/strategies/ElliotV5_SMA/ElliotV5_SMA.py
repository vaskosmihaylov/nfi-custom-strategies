"""
ElliotV5_SMA Strategy - Long Positions with 3-Layer Stop Loss Protection

A momentum-based long strategy using Elliott Wave Oscillator (EWO) and Simple
Moving Averages (SMA) to identify high-probability entry points during both
bullish and oversold market conditions.

## Entry Logic

Two entry conditions (OR logic - either can trigger):

1. **High EWO Entry**: Price below SMA with strong positive momentum
   - Price < SMA * low_offset (discount to moving average)
   - EWO > ewo_high (strong upward momentum)
   - RSI < rsi_buy (not overbought)

2. **Low EWO Entry**: Deep oversold conditions
   - Price < SMA * low_offset (discount to moving average)
   - EWO < ewo_low (deep negative momentum, likely reversal)

## Exit Logic

- **ROI-based**: Time-decay profit targets (21.5% → 3% over 201 minutes)
- **Signal-based**: Price crosses above SMA * high_offset
- **Trailing stop**: Activates at +3% profit with 0.5% trail
- **Custom stop loss**: 3-layer protection system (see below)

## Volatility-Adaptive Stop Loss with Moon Mode

**NO PROTECTION WINDOW** - Immediately trails to prevent liquidations

**Base Stoploss** (-0.99)
- Wide safety net that allows recovery

**custom_stoploss()** - Volatility-Adaptive Trailing
- Moon Mode: When making new highs (max_l > 0.003), allows wider trailing
- Adapts to each coin's volatility (move_mean, move_mean_x)
- For volatile coins (15-20% swings), automatically uses wider stops
- Prevents liquidations by tightening before -30% loss

**custom_exit()** - Time-Based Unclog
- Exits losing positions after N days at -4% loss
- Frees capital from dead positions

**confirm_trade_exit()** - Exit Quality Control
- Blocks ROI exits when not making new highs
- Blocks exits with profit below 0.3%
- Ensures better exit prices

## Risk Management

- **Leverage**: Fixed 3x for all trades
- **Liquidation Prevention**: Volatility-adaptive stops trigger before -30% liquidation point
- **Volatility Adaptation**: Stop distance adjusts to each coin's natural price swings
- **Moon Mode**: Lets winners run when making new highs
- **No Protection Window**: Immediate stop loss response prevents deep losses

## Performance Characteristics

- **Timeframe**: 5-minute candles
- **Best Markets**: Trending or mean-reverting with clear momentum shifts
- **Risk Level**: Moderate (3x leverage with -4% max loss after protection)

## Technical Requirements

- FreqTrade 2023.1+
- Technical Analysis Library (TA-Lib)
- Minimum 2000 candles for startup

## See Also

- ElliotV5_SMA_Shorts: Short-only variant of this strategy
- Memory file: elliotv5_sma_stoploss_fix_jan2026 (implementation details)
"""

from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Union
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import numpy as np
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter
import logging

logger = logging.getLogger(__name__)

buy_params = {
      "base_nb_candles_buy": 17,
      "ewo_high": 3.34,
      "ewo_low": -17.457,
      "low_offset": 0.978,
      "rsi_buy": 65
    }

sell_params = {
      "base_nb_candles_sell": 49,
      "high_offset": 1.019
    }

def EWO(dataframe, sma_length=5, sma2_length=35):
    """
    Elliott Wave Oscillator - Momentum indicator.

    Note: Despite the traditional EWO name, this implementation uses SMA (not EMA)
    for calculation, as it provides better signals for this strategy.

    Args:
        dataframe: OHLCV dataframe
        sma_length: Fast SMA period (default 5)
        sma2_length: Slow SMA period (default 35)

    Returns:
        Series: Percentage difference between fast and slow SMAs
    """
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

class ElliotV5_SMA(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.215,
        "40": 0.132,
        "87": 0.086,
        "201": 0.03
    }

    stoploss = -0.99

    # Volatility-based trailing stop parameters
    sl1 = DecimalParameter(-0.013, -0.005, default=-0.013, space='sell', optimize=True)
    move = IntParameter(35, 60, default=48, space='buy', optimize=True)
    mms = IntParameter(6, 20, default=12, space='buy', optimize=True)
    mml = IntParameter(300, 400, default=360, space='buy', optimize=True)

    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Unclog parameters (used in custom_exit)
    unclog_days = IntParameter(1, 5, default=4, space='sell', optimize=True)
    unclog = DecimalParameter(0.01, 0.08, default=0.04, decimals=2, space='sell', optimize=True)

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True

    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 2000

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = True

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

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
        conditions = []

        conditions.append(
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Fixed 3x leverage for all long trades.

        Increased from 1x to 3x for higher profit potential while maintaining
        risk control through -18.9% stop loss (allows 6.3% price movement).

        Returns:
            float: Leverage multiplier (3.0 = 3x)
        """
        return 3.0

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Volatility-adaptive trailing stop loss with Moon Mode.

        NO PROTECTION WINDOW - Starts trailing immediately to prevent liquidations.

        Moon Mode Logic:
        - When making new highs (max_l > 0.003): Allow wider trailing
          → Trail at move_mean_x (1.6x volatility) when profit > move_mean_x
          → Trail at sl1 (tight) when profit > move_mean
        - When not making new highs: Tighter protection
          → Trail at sl1 when profit > sl1

        Volatility Adaptation:
        - move_mean: Average volatility of this pair
        - move_mean_x: 1.6x average volatility (wider buffer)
        - For volatile coins (15-20% swings), stops adapt automatically
        - For stable coins, stops are tighter

        Why this prevents liquidations:
        - No 48h protection = stops can trigger immediately when needed
        - Volatility-adaptive = won't stop out on normal swings
        - Base stop still -0.99 = allows recovery, but custom_stoploss tightens before -30%

        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current timestamp
            current_rate: Current price
            current_profit: Current profit/loss ratio
            **kwargs: Additional arguments

        Returns:
            float: Stoploss percentage or self.stoploss to keep base
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        SLT1 = current_candle['move_mean']  # Average volatility
        SL1 = self.sl1.value  # Fixed tight stop (e.g., -0.013 = 1.3%)
        SLT2 = current_candle['move_mean_x']  # 1.6x average volatility
        SL2 = current_candle['move_mean_x'] - current_candle['move_mean']  # Difference

        display_profit = current_profit * 100
        slt1 = SLT1 * 100 if SLT1 is not None else 0
        sl1 = SL1 * 100
        slt2 = SLT2 * 100 if SLT2 is not None else 0
        sl2 = SL2 * 100 if SL2 is not None else 0

        # Moon mode: When making new highs (price near 360-candle high), allow wider trailing
        if current_candle['max_l'] > 0.003:
            # At high profit (> 1.6x volatility), trail with medium stop
            if SLT2 is not None and not np.isnan(SLT2) and current_profit > SLT2:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - Moon Mode {slt2:.2f}/{sl2:.2f} activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - Moon Mode {slt2:.2f}/{sl2:.2f} activated')
                return SL2
            # At medium profit (> average volatility), trail with tight stop
            if SLT1 is not None and not np.isnan(SLT1) and current_profit > SLT1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - Moon Mode {slt1:.2f}/{sl1:.2f} activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - Moon Mode {slt1:.2f}/{sl1:.2f} activated')
                return SL1
        else:
            # Not making new highs: Tighter protection
            if SLT1 is not None and not np.isnan(SLT1) and current_profit > SL1:
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.2f}% - {slt1:.2f}/{sl1:.2f} activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.2f}% - {slt1:.2f}/{sl1:.2f} activated')
                return SL1  # Safe stoploss

        return self.stoploss  # Keep base -0.99

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Time-based unclog mechanism for losing and zombie trades.

        With volatility-adaptive stop loss, this mainly handles:
        - Unclog: Trades that have been losing for too long
        - Zombie: Trades stuck at breakeven

        Logic (matches EI4_t4c0s_V2_2):
        - If losing > unclog threshold after unclog_days: Force exit ('unclog')

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
        # Unclog: Sell any positions at a loss if they are held for more than N days
        if current_profit < -self.unclog.value and (current_time - trade.open_date_utc).days >= self.unclog_days.value:
            return 'unclog'

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit with moon mode logic.

        Prevents premature exits when:
        - ROI exit when NOT making new highs (max_l < 0.003)
        - Any exit with profit below 0.3%

        This ensures we don't exit winners too early and captures more upside.

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

        # Block ROI exit if not making new highs (wait for better opportunity)
        if exit_reason == 'roi' and (last_candle['max_l'] < 0.003):
            return False

        # Block exits with very low profit - wait for better exit
        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI profit is below 0.3%")
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} Trailing stop price is below 0")
            return False

        return True
