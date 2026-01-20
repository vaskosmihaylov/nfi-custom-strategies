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

## 3-Layer Stop Loss Architecture

**Layer 1: Base Stoploss** (-0.99)
- Very wide safety net, almost never hit
- Allows maximum recovery time

**Layer 2: custom_stoploss()**
- 48-hour protection window (no tightening)
- After 48h: Trails profitable positions (>1%) using RSI/EWO indicators
- Graduated trailing: 1% at 2% profit, 1.5% at 5% profit

**Layer 3: custom_exit()**
- Unclog mechanism: Exits losing positions (-4%) after 48h
- Zombie detection: Exits breakeven trades (±0.5%) after 48h
- Prevents capital lockup in dead positions

## Risk Management

- **Leverage**: Fixed 3x for all trades
- **Max Loss**: -4% after 48-hour protection (-12% price movement)
- **Protection Period**: 48 hours to allow recovery from volatility
- **Improvement**: Prevents premature -18.9% stop losses

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
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter

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
        3-Layer stop loss architecture with 48-hour protection window.

        Layer 1: Base stoploss (-0.99) acts as wide safety net
        Layer 2: This method controls tightening for PROFITABLE trades only
        Layer 3: custom_exit() handles losing/zombie trades

        Logic:
        - Hours 0-48: Return 1.0 (keep base stoploss, don't tighten) - PROTECTION WINDOW
        - Hours 48+:  Trail ONLY profitable positions (>1%) using RSI/EWO indicators

        Why this works:
        - Prevents premature exits in first 48h (allows recovery from normal volatility)
        - FreqTrade limitation: custom_stoploss can ONLY tighten, never widen
        - Solution: Wide base stoploss + return 1.0 during protection = no tightening
        - After 48h, losing trades handled by custom_exit (unclog/zombie)
        - Profitable trades get smart indicator-based trailing

        CRITICAL: Must return 1.0 (not -0.99) during protection!
        - Returning -0.99 would try to widen (IGNORED by FreqTrade)
        - Returning 1.0 means "don't change stop" (keeps base -0.99)

        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current timestamp
            current_rate: Current price
            current_profit: Current profit/loss ratio
            **kwargs: Additional arguments

        Returns:
            float: Stoploss percentage or 1.0 to keep base stoploss
        """
        # Calculate trade duration in hours
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Phase 1: First 48 hours - PROTECTION WINDOW
        # Return 1.0 to keep base stoploss active (don't tighten)
        if trade_duration < 48:
            return 1.0

        # Phase 2: After 48 hours - Trail ONLY profitable trades
        # Losing trades will be handled by custom_exit (unclog/zombie)
        if current_profit > 0.01:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()

            # For positions > 2% profit - aggressive trailing
            if current_profit > 0.02:
                # Strong overbought with high EWO - likely top, trail tight
                if current_candle['rsi'] > 80 and current_candle['EWO'] > 8:
                    return -0.01  # 1% trailing stop

                # Extreme overbought - take profits
                if current_candle['rsi'] > 85:
                    return -0.01  # 1% trailing stop

                # At 5%+ profit, tighter trailing
                if current_profit >= 0.05:
                    if current_candle['rsi'] > 75:
                        return -0.015  # 1.5% trailing stop

            # For positions 1-2% profit - conservative trailing
            else:
                # Very extreme overbought only
                if current_candle['rsi'] > 85:
                    return -0.01

        # Default: Keep base stoploss (don't tighten)
        # Losing/breakeven trades handled by custom_exit
        return 1.0

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Layer 3 of stop loss architecture: Handle losing and zombie trades.

        Logic:
        - Hours 0-48: Return None (protection window, no forced exits)
        - Hours 48+:
          * If loss > 4%: Force exit ('unclog')
          * If at breakeven (±0.5%): Force exit ('zombie')
          * Otherwise: Continue normal exit logic

        Why this approach:
        - Separates concerns: custom_stoploss trails profits, custom_exit cuts losses
        - 48-hour protection allows recovery from normal volatility
        - -4% max loss = -12% price movement with 3x leverage (much better than -18.9%)
        - Zombie detection frees capital stuck in breakeven trades
        - Works with FreqTrade limitation (custom_exit can exit anytime)

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
        # Calculate trade duration in hours
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Phase 1: First 48 hours - PROTECTION WINDOW
        # No forced exits during protection period
        if trade_duration_hours < 48:
            return None

        # Phase 2: After 48 hours - Risk management for losing/zombie trades

        # Unclog: Force exit losing positions to prevent large losses
        if current_profit < -0.04:
            return 'unclog'

        # Zombie: Force exit breakeven trades to free up capital
        if -0.005 <= current_profit <= 0.005:
            return 'zombie'

        # No custom exit, continue normal logic (ROI, signals, trailing stop)
        return None
