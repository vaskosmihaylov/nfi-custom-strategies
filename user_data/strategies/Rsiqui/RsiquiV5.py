import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Tuple, Union
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)
import logging

logger = logging.getLogger(__name__)

class RsiquiV5(IStrategy):
    """
    RsiquiV5 - RSI Gradient Strategy (Optimized)

    OPTIMIZATIONS (Jan 2026):
    1. Balanced entry thresholds: RSI 35/65 instead of 27/59 (more balanced long/short)
    2. ATR-based dynamic stoploss: Prevents catastrophic losses like -27%
    3. Kept existing ROI table and exit signals
    4. 3x leverage maintained

    Strategy Logic:
    - Entry: RSI gradient crosses zero at extreme RSI levels
    - Exit: ROI table, exit signals (gradient reversal), or ATR-based stops
    - Risk Management: Dynamic ATR stops replace fixed -27.3% stoploss
    """

    INTERFACE_VERSION = 3

    can_short = True
    timeframe = '5m'
    use_exit_signal = True
    exit_profit_only = True
    use_custom_stoploss = True

    # Buy hyperspace params (OPTIMIZED):
    buy_params = {
        "rsi_entry_long": 35,   # Easier to trigger (was 27)
        "rsi_entry_short": 65,  # Harder to trigger (was 59)
        "window": 24,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsi_exit_long": 18,
        "rsi_exit_short": 75,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.223,
        "34": 0.082,
        "82": 0.033,
        "109": 0
    }

    # Stoploss (managed by custom_stoploss):
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # Max Open Trades:
    max_open_trades = -1

    # Hyperparameters
    rsi_entry_long  = IntParameter(25, 40, default=buy_params.get('rsi_entry_long'),  space='buy',  optimize=True)
    rsi_exit_long   = IntParameter(10, 25, default=sell_params.get('rsi_exit_long'),   space='sell', optimize=True)
    rsi_entry_short = IntParameter(60, 75, default=buy_params.get('rsi_entry_short'), space='buy',  optimize=True)
    rsi_exit_short  = IntParameter(70, 85, default=sell_params.get('rsi_exit_short'),  space='sell', optimize=True)
    window          = IntParameter(5, 100, default=buy_params.get('window'),          space='buy',  optimize=False)

    # ATR-Based Dynamic Stop Loss (NEW)
    atr_stop_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='sell', optimize=True)
    max_atr_stop = DecimalParameter(-0.20, -0.10, default=-0.15, space='sell', optimize=False)
    min_atr_stop = DecimalParameter(-0.05, -0.02, default=-0.03, space='sell', optimize=False)

    @property
    def plot_config(self):
        plot_config = {}

        plot_config['main_plot'] = {
            'rsi_ema' : {}
        }
        plot_config['subplots'] = {
            'Misc': {
                'rsi': {},
                'rsi_gra' : {},
            },
        }

        return plot_config

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ATR-Based Dynamic Stop Loss (prevents catastrophic losses)

        Replaces fixed -27.3% stoploss with volatility-adjusted stops.
        Prevents trades from reaching -20%+ losses that wipe out many winners.

        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current timestamp
            current_rate: Current price
            current_profit: Current profit ratio

        Returns:
            float: Stoploss percentage (negative value)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # Calculate ATR-based stop: 2x ATR as percentage
        atr_stop = -1 * self.atr_stop_multiplier.value * current_candle['atr_pcnt']

        # Clamp between bounds to prevent too tight or too loose stops
        # max_atr_stop = -0.15 (maximum loss allowed)
        # min_atr_stop = -0.03 (minimum protection)
        atr_stop = max(atr_stop, self.max_atr_stop.value)
        atr_stop = min(atr_stop, self.min_atr_stop.value)

        # Log when stop activates
        if current_profit < atr_stop:
            logger.info(f"*** {pair} *** ATR Stop: {current_profit*100:.2f}% < {atr_stop*100:.2f}%")

        return atr_stop

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])

        # ATR for dynamic stoploss (NEW)
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=14) / dataframe['close'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals based on RSI gradient crosses.

        OPTIMIZED: Balanced RSI thresholds (35/65 instead of 27/59)
        - Longs: RSI < 35 (easier to trigger than 27)
        - Shorts: RSI > 65 (harder to trigger than 59)

        This should produce more balanced long/short ratio.
        """
        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_entry_long.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_entry_short.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_exit_long.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_exit_short.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 3
