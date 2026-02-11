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
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from datetime import datetime
# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import warnings
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class SimpleRSI(IStrategy):

    can_short: bool = False

    INTERFACE_VERSION = 3

    rsiWindow = IntParameter(7, 21, default=14, space="buy", optimize=True)
    minRSI = DecimalParameter(1, 99, decimals=0, default=80, space="buy", optimize=True)

    use_custom_stoploss: bool = True
    process_only_new_candles: bool = True

    position_adjustment_enable: bool = False

    minimal_roi = {
        "0": 500.0
    }

    stoploss = -0.99

    trailing_stop = False

    timeframe = '1d'

    startup_candle_count: int = 50

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=int(self.rsiWindow.value))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['RSI'] >= self.minRSI.value)
        conditions.append(dataframe['RSI'].shift(1) < self.minRSI.value)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['RSI'] >= 105)  # never exits

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

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
