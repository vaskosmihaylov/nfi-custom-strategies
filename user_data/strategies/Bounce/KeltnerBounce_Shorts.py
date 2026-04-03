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


class KeltnerBounce_Shorts(IStrategy):
    """
    Shorts-only inverse of KeltnerBounce.
    Enters on rejection from upper Keltner channel and exits near lower channel.
    """

    INTERFACE_VERSION = 3
    can_short: bool = True

    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.03, space="buy")
    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")

    buy_mfi = DecimalParameter(10, 100, decimals=0, default=63, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=0.1, space="buy")

    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")

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

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": Config.stoploss_guard_lookback_candles,
                "trade_limit": Config.stoploss_guard_trade_limit,
                "stop_duration_candles": Config.stoploss_guard_duration_candles,
                "only_per_pair": True,
            }
        ]

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['mfi'] = ta.MFI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        keltner = qtpylib.keltner_channel(dataframe)
        dataframe['kc_upperband'] = keltner['upper']
        dataframe['kc_lowerband'] = keltner['lower']
        dataframe['kc_middleband'] = keltner['mid']

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_gain'] = ((dataframe['bb_upperband'] - dataframe['close']) / dataframe['close'])
        dataframe['bb_drop'] = ((dataframe['close'] - dataframe['bb_lowerband']) / dataframe['close'])

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['volume'] > 0)

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] >= (100 - self.buy_mfi.value))

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] >= (-1 * self.buy_fisher.value))

        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_drop'] >= self.buy_bb_gain.value)

        conditions.append(
            (
                (dataframe['open'] < dataframe['kc_upperband']) &
                (dataframe['open'] > dataframe['kc_middleband']) &
                (dataframe['close'] < dataframe['kc_upperband']) &
                (dataframe['close'] > dataframe['kc_middleband'])
            ) &
            (dataframe['close'] < dataframe['open']) &
            (
                (dataframe['open'].shift(1) >= dataframe['kc_upperband']) |
                (dataframe['close'].shift(1) >= dataframe['kc_upperband'])
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['open'] < dataframe['kc_lowerband']) |
                (dataframe['close'] < dataframe['kc_lowerband'])
            ),
            'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side: str, **kwargs) -> float:
        return min(Config.trade_leverage, max_leverage)
