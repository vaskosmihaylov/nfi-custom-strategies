# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
from typing import Dict, List
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.persistence import Trade
from datetime import datetime
# --------------------------------
import talib.abstract as ta
import warnings
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class SimpleRSI_Shorts(IStrategy):
    """
    SimpleRSI_Shorts - Shorts-only variant of SimpleRSI.

    Original enters long when RSI crosses above minRSI (default 80) - momentum breakout.
    This shorts variant enters short when RSI crosses below (100 - minRSI) = 20 - momentum breakdown.

    Same ROI, stoploss, and leverage as longs with only trading logic inverted.
    """

    can_short: bool = True

    INTERFACE_VERSION = 3

    rsiWindow = IntParameter(7, 21, default=14, space="buy", optimize=True)
    # Inverted: 100 - 80 = 20 (enter short when RSI drops into oversold)
    minRSI = DecimalParameter(1, 99, decimals=0, default=20, space="buy", optimize=True)

    use_custom_stoploss: bool = True
    process_only_new_candles: bool = True

    position_adjustment_enable: bool = False

    # Same as longs
    minimal_roi = {
        "0": 500.0
    }

    stoploss = -0.99

    trailing_stop = False

    timeframe = '1d'

    startup_candle_count: int = 50

    # Max short trades
    max_short_trades = 4

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Emergency backstop: prevent liquidation at 3x leverage
        if current_profit <= -0.20:
            return -0.21
        return 1

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        # Only allow shorts
        if side == "long":
            return False

        # Enforce max short positions
        short_count = sum(1 for t in Trade.get_trades_proxy(is_open=True) if t.is_short)
        if short_count >= self.max_short_trades:
            return False

        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=int(self.rsiWindow.value))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Inverted: enter short when RSI drops below threshold from above
        conditions = []
        conditions.append(dataframe['RSI'] <= self.minRSI.value)
        conditions.append(dataframe['RSI'].shift(1) > self.minRSI.value)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['RSI'] <= -5)  # never exits

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        return 3.0

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration": 10080
            }
        ]
