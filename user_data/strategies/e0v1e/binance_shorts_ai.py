from datetime import datetime, timedelta
from typing import Optional, Union
import logging
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce


logger = logging.getLogger(__name__)


class binance_shorts_ai(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True

    minimal_roi = {
        "0": 0.03
    }

    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 120

    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_market_ratio": 0.99,
    }

    stoploss = -0.25
    use_custom_stoploss = True

    is_optimize_32 = False
    buy_rsi_fast_32 = IntParameter(20, 70, default=45, space="buy", optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=35, space="buy", optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.961, decimals=3, space="buy", optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.58, decimals=2, space="buy", optimize=is_optimize_32)

    sell_fastx = IntParameter(50, 100, default=75, space="sell", optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sma_15"] = ta.SMA(dataframe, timeperiod=15)
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["fastk"] = stoch_fast["fastk"]
        dataframe["rsi_shifted"] = dataframe["rsi_slow"].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        enter_short_conditions = [
            (dataframe["rsi_slow"] > dataframe["rsi_shifted"]),
            (dataframe["rsi_fast"] > self.buy_rsi_fast_32.value),
            (dataframe["rsi"] < self.buy_rsi_32.value),
            (dataframe["close"] > dataframe["sma_15"] * self.buy_sma15_32.value),
            (dataframe["cti"] > self.buy_cti_32.value),
        ]

        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"],
            ] = (1, "short_ai")

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_profit >= 0.08:
            return -0.01

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return -0.001

        if current_time - timedelta(hours=4) > trade.open_date_utc:
            if current_profit > -0.03:
                return -0.005

        return self.stoploss

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ["exit_short", "exit_tag"]] = (0, "short_out")
        return dataframe
