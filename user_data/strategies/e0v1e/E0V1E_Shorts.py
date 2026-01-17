"""
E0V1E_Shorts Strategy

A shorts-only variant of the E0V1E strategy, designed to profit in bear markets
and during overbought conditions. This strategy mirrors the successful long-only approach
but inverts the logic for short positions.

Original E0V1E Strategy (Longs):
- Uses EWO (Elliott Wave Oscillator) for momentum divergence
- Simple RSI and EMA-based entries
- Fast 5m timeframe

Strategy Concept:
Uses Elliott Wave Oscillator (EWO) to identify momentum divergence for short entries.
EWO measures the percentage difference between fast and slow EMAs.

Entry Conditions (OR logic - either triggers entry):
1. Short EWO: Price above EMA during uptrend (inverted from long logic)
2. Short Buy_1: RSI and SMA-based mean reversion shorts

Exit Conditions:
- Signal: Price falls below EMA
- ROI: 7% initial target (more conservative than longs' 10%)
- Stop Loss: -18.9% (tighter than longs due to short squeeze risk)
- Custom Stoploss: Dynamic trailing

Key Differences from Long Strategy:
- Tighter stop loss: -18.9% vs -99% (shorts are riskier)
- Lower ROI targets: 7% vs 10% (faster profit-taking)
- Inverted entry/exit logic
- Max 4 short positions: Position limit enforcement via confirm_trade_entry

Author: Derived from E0V1E
Version: 1.0.0
"""

from datetime import datetime, timedelta
import logging
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce

logger = logging.getLogger(__name__)

def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class E0V1E_Shorts(IStrategy):
    """
    Shorts-only Elliott Wave Oscillator strategy for bear markets.

    This strategy is designed to run in PARALLEL with E0V1E (longs)
    in separate containers to evaluate short performance independently.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # More conservative ROI for shorts (30% lower than longs)
    minimal_roi = {
        "0": 0.07  # 7% profit target (vs 10% for longs)
    }

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 20

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    # Tighter stop loss for shorts (vs -0.99 for longs)
    stoploss = -0.189

    # Custom stoploss
    use_custom_stoploss = True

    # Position limits
    max_open_trades = 4
    max_short_trades = 4

    # Shorts-specific parameters (inverted from longs)
    is_optimize_ewo = True
    sell_rsi_fast = IntParameter(50, 65, default=55, space='buy', optimize=is_optimize_ewo)
    sell_rsi = IntParameter(65, 85, default=65, space='buy', optimize=is_optimize_ewo)
    sell_ewo = DecimalParameter(-5, 6.0, default=5.585, space='buy', optimize=is_optimize_ewo)
    sell_ema_low = DecimalParameter(1.01, 1.1, default=1.058, space='buy', optimize=is_optimize_ewo)
    sell_ema_high = DecimalParameter(0.8, 1.05, default=0.916, space='buy', optimize=is_optimize_ewo)

    is_optimize_32 = True
    sell_rsi_fast_32 = IntParameter(50, 80, default=54, space='buy', optimize=is_optimize_32)
    sell_rsi_32 = IntParameter(50, 85, default=81, space='buy', optimize=is_optimize_32)
    sell_sma15_32 = DecimalParameter(1.0, 1.1, default=1.058, decimals=3, space='buy', optimize=is_optimize_32)
    sell_cti_32 = DecimalParameter(0, 1, default=0.86, decimals=2, space='buy', optimize=is_optimize_32)

    is_optimize_deadfish = True
    cover_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    cover_deadfish_profit = DecimalParameter(0.05, 0.15, default=0.05, space='sell', optimize=is_optimize_deadfish)
    cover_deadfish_bb_factor = DecimalParameter(0.80, 1.10, default=1.0, space='sell', optimize=is_optimize_deadfish)
    cover_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='sell', optimize=is_optimize_deadfish)

    cover_fastx = IntParameter(0, 50, default=25, space='sell', optimize=True)

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # ewo indicators
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['EWO'] = ewo(dataframe, 50, 200)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # loss sell indicators
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # Inverted from long logic
        is_ewo_short = (
                (dataframe['rsi_fast'] > self.sell_rsi_fast.value) &
                (dataframe['close'] > dataframe['ema_8'] * self.sell_ema_low.value) &
                (dataframe['EWO'] < self.sell_ewo.value) &
                (dataframe['close'] > dataframe['ema_16'] * self.sell_ema_high.value) &
                (dataframe['rsi'] > self.sell_rsi.value)
        )

        # Inverted from buy_1 logic
        short_1 = (
                (dataframe['rsi_slow'] > dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] > self.sell_rsi_fast_32.value) &
                (dataframe['rsi'] < self.sell_rsi_32.value) &
                (dataframe['close'] > dataframe['sma_15'] * self.sell_sma15_32.value) &
                (dataframe['cti'] > self.sell_cti_32.value)
        )

        conditions.append(is_ewo_short)
        dataframe.loc[is_ewo_short, 'enter_tag'] += 'ewo_short'

        conditions.append(short_1)
        dataframe.loc[short_1, 'enter_tag'] += 'short_1'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        enter_tag = ''
        if hasattr(trade, 'enter_tag') and trade.enter_tag is not None:
            enter_tag = trade.enter_tag
        enter_tags = enter_tag.split()

        if "ewo_short" in enter_tags:
            if current_profit >= 0.05:
                return -0.005

        if current_profit > 0.01:
            if current_candle["fastk"] < self.cover_fastx.value:
                return -0.001

            if current_candle["rsi"] < 20:
                return -0.001

        if current_profit < 0.01:
            if current_candle["rsi"] < 10:
                return -0.001

        return self.stoploss

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # stoploss - deadfish (inverted for shorts)
        if ((current_profit < self.cover_deadfish_profit.value)
                and (current_candle['bb_width'] < self.cover_deadfish_bb_width.value)
                and (current_candle['close'] < current_candle['bb_middleband2'] * self.cover_deadfish_bb_factor.value)
                and (current_candle['volume_mean_12'] < current_candle[
                    'volume_mean_24'] * self.cover_deadfish_volume_factor.value)):
            logger.info(f"{pair} cover_stoploss_deadfish at {current_profit*100}")
            return "cover_stoploss_deadfish"

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #dataframe.loc[(), ['exit_short', 'exit_tag']] = (0, 'short_out')

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Fixed 3x leverage for all short trades.

        Returns:
            float: Leverage multiplier (3.0 = 3x)
        """
        return 3.0
