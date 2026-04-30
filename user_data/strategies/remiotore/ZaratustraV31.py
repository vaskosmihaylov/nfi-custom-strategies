import logging
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional
import talib.abstract as ta
from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter, BooleanParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
    
    
    
class ZaratustraV31(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    can_short = True

    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.01

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.0011
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_buy_sell_signals = False
    
    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {}
        plot_config['subplots'] = {
            'RSI' : {
                'rsi_7' : { 'color' : 'orange', },
                'rsi_14' : { 'color' : 'red' },
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi14'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi7'] > dataframe['rsi14']),
            ['enter_long', 'enter_tag']
        ] = (1, 'RSI7 > RSI14')

        dataframe.loc[
            (dataframe['rsi7'] < dataframe['rsi14']),
            ['enter_short', 'enter_tag']
        ] = (1, 'RSI7 < RSI14')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi7'] < dataframe['rsi14']),
            ['exit_long', 'exit_tag']
        ] = (1, 'RSI7 cross down')

        dataframe.loc[
            (dataframe['rsi7'] > dataframe['rsi14']),
            ['exit_short', 'exit_tag']
        ] = (1, 'RSI7 cross up')
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 1