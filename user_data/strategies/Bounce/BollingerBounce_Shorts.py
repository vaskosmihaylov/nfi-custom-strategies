# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy
from freqtrade.strategy import CategoricalParameter, DecimalParameter
import Config


class BollingerBounce_Shorts(IStrategy):
    """
    Shorts-only inverse of BollingerBounce.
    Enters on rejection from upper Bollinger band and exits on lower-band flush.
    """

    INTERFACE_VERSION = 3
    can_short: bool = True

    # Keep parameter names aligned with long strategy for familiarity.
    buy_mfi = DecimalParameter(10, 40, decimals=0, default=13.0, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.81, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.04, space="buy")

    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.62, space="sell")
    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    startup_candle_count = 20

    minimal_roi = Config.minimal_roi
    trailing_stop = Config.trailing_stop
    trailing_stop_positive = Config.trailing_stop_positive
    trailing_stop_positive_offset = Config.trailing_stop_positive_offset
    trailing_only_offset_is_reached = Config.trailing_only_offset_is_reached
    stoploss = Config.stoploss
    timeframe = Config.timeframe
    process_only_new_candles = Config.process_only_new_candles
    use_exit_signal = Config.use_exit_signal
    exit_profit_only = Config.exit_profit_only
    ignore_roi_if_entry_signal = Config.ignore_roi_if_entry_signal
    order_types = Config.order_types

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_gain'] = ((dataframe['bb_upperband'] - dataframe['close']) / dataframe['close'])
        dataframe['bb_drop'] = ((dataframe['close'] - dataframe['bb_lowerband']) / dataframe['close'])

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] >= (100 - self.buy_mfi.value))

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] > (-1 * self.buy_fisher.value))

        # Potential downside room to lower band
        conditions.append(dataframe['bb_drop'] >= self.buy_bb_gain.value)

        # Red rejection candle at upper band
        conditions.append(dataframe['close'] < dataframe['open'])
        conditions.append(
            (dataframe['open'] > dataframe['bb_upperband']) &
            (dataframe['close'] <= dataframe['bb_upperband'])
        )

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull()), 'exit_short'] = 0
            return dataframe

        dataframe.loc[
            (
                (
                    (dataframe['open'] < dataframe['bb_lowerband']) |
                    (dataframe['close'] < dataframe['bb_lowerband'])
                ) |
                (
                    (dataframe['fisher_rsi'] < self.sell_fisher.value) &
                    (dataframe['sar'] < dataframe['close'])
                )
            ),
            'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side: str, **kwargs) -> float:
        return min(3.0, max_leverage)
