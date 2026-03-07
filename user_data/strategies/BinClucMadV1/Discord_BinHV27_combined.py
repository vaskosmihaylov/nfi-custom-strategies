from datetime import datetime
from functools import reduce

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, merge_informative_pair, stoploss_from_open
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib

import talib.abstract as ta
import numpy  # noqa


class BinHV27_combined(IStrategy):
    """

        strategy sponsored by user BinH from slack

        this file commbines:
        https://github.com/ssssi/freqtrade_strs/blob/82baabd2099b5f592458678554ca7162148f1d7a/binance/BinHV27/long/BinHV27_long.py
        https://github.com/ssssi/freqtrade_strs/blob/82baabd2099b5f592458678554ca7162148f1d7a/binance/BinHV27/short/BinHV27_short.py
        https://github.com/ssssi/freqtrade_strs/blob/82baabd2099b5f592458678554ca7162148f1d7a/binance/BinHV27/short/BinHV27_short.json

        Switching Long/Short is done based on Talib TSF (time series forecast)


    """

    minimal_roi = {
        "0": 1
    }

    buy_params = {
        'entry_long_adx1': 25,
        'entry_long_emarsi1': 20,
        'entry_long_adx2': 30,
        'entry_long_emarsi2': 20,
        'entry_long_adx3': 35,
        'entry_long_emarsi3': 20,
        'entry_long_adx4': 30,
        'entry_long_emarsi4': 25,
        'entry_short_adx1': 62,
        'entry_short_emarsi1': 29,
        'entry_short_adx2': 29,
        'entry_short_emarsi2': 30,
        'entry_short_adx3': 33,
        'entry_short_emarsi3': 22,
        'entry_short_adx4': 88,
        'entry_short_emarsi4': 57
    }

    sell_params = {
        # custom stop loss params
        "pHSL_long": -0.25,
        "pPF_1_long": 0.012,
        "pPF_2_long": 0.05,
        "pSL_1_long": 0.01,
        "pSL_2_long": 0.04,
        "pHSL_short": -0.863,
        "pPF_1_short": 0.018,
        "pPF_2_short": 0.197,
        "pSL_1_short": 0.015,
        "pSL_2_short": 0.157,

        # leverage set
        "leverage_num": 1,

        # sell params
        'exit_long_emarsi1': 75,
        'exit_long_adx2': 30,
        'exit_long_emarsi2': 80,
        'exit_long_emarsi3': 75,
        'exit_short_emarsi1': 30,
        'exit_short_adx2': 21,
        'exit_short_emarsi2': 71,
        'exit_short_emarsi3': 72,

        # sell optional
        "exit_long_1": True,
        "exit_long_2": True,
        "exit_long_3": True,
        "exit_long_4": True,
        "exit_long_5": True,
        "exit_short_1": False,
        "exit_short_2": True,
        "exit_short_3": True,
        "exit_short_4": True,
        "exit_short_5": False,
    }

    stoploss = -0.99
    timeframe = '5m'
    inf_timeframe = '4h'

    process_only_new_candles = True
    startup_candle_count = 240

    # default False
    use_custom_stoploss = True

    can_short = True

    # buy params
    entry_optimize = True
    entry_long_adx1 = IntParameter(low=10, high=100, default=25, space='buy', optimize=entry_optimize)
    entry_long_emarsi1 = IntParameter(low=10, high=100, default=20, space='buy', optimize=entry_optimize)
    entry_long_adx2 = IntParameter(low=20, high=100, default=30, space='buy', optimize=entry_optimize)
    entry_long_emarsi2 = IntParameter(low=20, high=100, default=20, space='buy', optimize=entry_optimize)
    entry_long_adx3 = IntParameter(low=10, high=100, default=35, space='buy', optimize=entry_optimize)
    entry_long_emarsi3 = IntParameter(low=10, high=100, default=20, space='buy', optimize=entry_optimize)
    entry_long_adx4 = IntParameter(low=20, high=100, default=30, space='buy', optimize=entry_optimize)
    entry_long_emarsi4 = IntParameter(low=20, high=100, default=25, space='buy', optimize=entry_optimize)
    entry_short_adx1 = IntParameter(low=10, high=100, default=25, space='buy', optimize=entry_optimize)
    entry_short_emarsi1 = IntParameter(low=10, high=100, default=20, space='buy', optimize=entry_optimize)
    entry_short_adx2 = IntParameter(low=20, high=100, default=30, space='buy', optimize=entry_optimize)
    entry_short_emarsi2 = IntParameter(low=20, high=100, default=20, space='buy', optimize=entry_optimize)
    entry_short_adx3 = IntParameter(low=10, high=100, default=35, space='buy', optimize=entry_optimize)
    entry_short_emarsi3 = IntParameter(low=10, high=100, default=20, space='buy', optimize=entry_optimize)
    entry_short_adx4 = IntParameter(low=20, high=100, default=30, space='buy', optimize=entry_optimize)
    entry_short_emarsi4 = IntParameter(low=20, high=100, default=25, space='buy', optimize=entry_optimize)

    # trailing stoploss
    trailing_optimize = False
    pHSL_long = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_1_long = DecimalParameter(0.008, 0.100, default=0.016, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_1_long = DecimalParameter(0.008, 0.100, default=0.011, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_2_long = DecimalParameter(0.040, 0.200, default=0.080, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_2_long = DecimalParameter(0.040, 0.200, default=0.040, decimals=3, space='sell', optimize=trailing_optimize)
    pHSL_short = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_1_short = DecimalParameter(0.008, 0.100, default=0.016, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_1_short = DecimalParameter(0.008, 0.100, default=0.011, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_2_short = DecimalParameter(0.040, 0.200, default=0.080, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_2_short = DecimalParameter(0.040, 0.200, default=0.040, decimals=3, space='sell', optimize=trailing_optimize)

    # sell params
    exit_optimize = True
    exit_long_adx2 = IntParameter(low=10, high=100, default=30, space='sell', optimize=exit_optimize)
    exit_long_emarsi1 = IntParameter(low=10, high=100, default=75, space='sell', optimize=exit_optimize)
    exit_long_emarsi2 = IntParameter(low=20, high=100, default=80, space='sell', optimize=exit_optimize)
    exit_long_emarsi3 = IntParameter(low=20, high=100, default=75, space='sell', optimize=exit_optimize)
    exit_short_adx2 = IntParameter(low=10, high=100, default=30, space='sell', optimize=exit_optimize)
    exit_short_emarsi1 = IntParameter(low=10, high=100, default=75, space='sell', optimize=exit_optimize)
    exit_short_emarsi2 = IntParameter(low=20, high=100, default=80, space='sell', optimize=exit_optimize)
    exit_short_emarsi3 = IntParameter(low=20, high=100, default=75, space='sell', optimize=exit_optimize)

    exit2_optimize = True
    exit_long_1 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_long_2 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_long_3 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_long_4 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_long_5 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_short_1 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_short_2 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_short_3 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_short_4 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)
    exit_short_5 = CategoricalParameter([True, False], default=True, space="sell", optimize=exit2_optimize)

    leverage_optimize = False
    leverage_num = IntParameter(low=1, high=5, default=1, space='sell', optimize=leverage_optimize)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        if trade.is_short:
            HSL = self.pHSL_short.value
            PF_1 = self.pPF_1_short.value
            SL_1 = self.pSL_1_short.value
            PF_2 = self.pPF_2_short.value
            SL_2 = self.pSL_2_short.value
        else:
            HSL = self.pHSL_long.value
            PF_1 = self.pPF_1_long.value
            SL_1 = self.pSL_1_long.value
            PF_2 = self.pPF_2_long.value
            SL_2 = self.pSL_2_long.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if trade.is_short:
            if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
                return 1
        else:
            if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
                return 1

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = numpy.nan_to_num(ta.RSI(dataframe, timeperiod=5))
        rsiframe = DataFrame(dataframe['rsi']).rename(columns={'rsi': 'close'})
        dataframe['emarsi'] = numpy.nan_to_num(ta.EMA(rsiframe, timeperiod=5))
        dataframe['adx'] = numpy.nan_to_num(ta.ADX(dataframe))
        dataframe['minusdi'] = numpy.nan_to_num(ta.MINUS_DI(dataframe))
        minusdiframe = DataFrame(dataframe['minusdi']).rename(columns={'minusdi': 'close'})
        dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(minusdiframe, timeperiod=25))
        dataframe['plusdi'] = numpy.nan_to_num(ta.PLUS_DI(dataframe))
        plusdiframe = DataFrame(dataframe['plusdi']).rename(columns={'plusdi': 'close'})
        dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(plusdiframe, timeperiod=5))
        dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=60))
        dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=120))
        dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=120))
        dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=240))
        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & (
                (dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']
        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(
            dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe[
            'slowsma'].shift().gt(dataframe['slowsma'].shift(2))
        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        # informative timeframe
        inf_dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        inf_dataframe['hlc3'] = ta.TYPPRICE(inf_dataframe)
        inf_dataframe['tsf'] = ta.TSF(inf_dataframe['hlc3'], timeperiod=2)
        inf_dataframe['allow_long'] = ((inf_dataframe['tsf'] / inf_dataframe['hlc3']) > 1.01)
        inf_dataframe['allow_short'] = ((inf_dataframe['tsf'] / inf_dataframe['hlc3']) < 0.99)

        dataframe = merge_informative_pair(dataframe, inf_dataframe, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        long_conditions = []
        short_conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        long_entry_1 = (
                dataframe[f'allow_long_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_long_adx1.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.entry_long_emarsi1.value)
        )

        long_entry_2 = (
                dataframe[f'allow_long_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_long_adx2.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.entry_long_emarsi2.value)
        )

        long_entry_3 = (
                dataframe[f'allow_long_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_long_adx3.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.entry_long_emarsi3.value)
        )

        long_entry_4 = (
                dataframe[f'allow_long_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_long_adx4.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.entry_long_emarsi4.value)
        )

        long_conditions.append(long_entry_1)
        dataframe.loc[long_entry_1, 'enter_tag'] += 'long_entry_1'

        long_conditions.append(long_entry_2)
        dataframe.loc[long_entry_2, 'enter_tag'] += 'long_entry_2'

        long_conditions.append(long_entry_3)
        dataframe.loc[long_entry_3, 'enter_tag'] += 'long_entry_3'

        long_conditions.append(long_entry_4)
        dataframe.loc[long_entry_4, 'enter_tag'] += 'long_entry_4'

        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, long_conditions),
                'enter_long'] = 1
        else:
            dataframe.loc[(), ['enter_long', 'enter_tag']] = (0, 'no_long_entry')


        short_entry_1 = (
                dataframe[f'allow_short_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_short_adx1.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.entry_short_emarsi1.value)
        )

        short_entry_2 = (
                dataframe[f'allow_short_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_short_adx2.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.entry_short_emarsi2.value)
        )

        short_entry_3 = (
                dataframe[f'allow_short_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_short_adx3.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.entry_short_emarsi3.value)
        )

        short_entry_4 = (
                dataframe[f'allow_short_{self.inf_timeframe}'] &
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.entry_short_adx4.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.entry_short_emarsi4.value)
        )

        short_conditions.append(short_entry_1)
        dataframe.loc[short_entry_1, 'enter_tag'] += 'short_entry_1'

        short_conditions.append(short_entry_2)
        dataframe.loc[short_entry_2, 'enter_tag'] += 'short_entry_2'

        short_conditions.append(short_entry_3)
        dataframe.loc[short_entry_3, 'enter_tag'] += 'short_entry_3'

        short_conditions.append(short_entry_4)
        dataframe.loc[short_entry_4, 'enter_tag'] += 'short_entry_4'

        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, short_conditions),
                'enter_short'] = 1
        else:
            dataframe.loc[(), ['enter_short', 'enter_tag']] = (0, 'no_short_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), ['exit_short', 'exit_tag']] = (0, 'no_short_exit')
        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'no_long_exit')
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if (current_profit >= self.pPF_1_long.value) and not trade.is_short:
            return None

        if (current_profit >= self.pPF_1_short.value) and trade.is_short:
            return None

        if self.exit_long_1.value and not trade.is_short:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['lowsma'] or last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['bigdown'])
            ):
                return "exit_long_1"

        if self.exit_long_2.value and not trade.is_short:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['emarsi'] > self.exit_long_emarsi1.value or last_candle['close'] > last_candle['slowsma'])
                    and (last_candle['bigdown'])
            ):
                return "exit_long_2"

        if self.exit_long_3.value and not trade.is_short:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['adx'] > self.exit_long_adx2.value)
                    and (last_candle['emarsi'] >= self.exit_long_emarsi2.value)
                    and (last_candle['bigup'])
            ):
                return "exit_long_3"

        if self.exit_long_4.value and not trade.is_short:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['slowingdown'])
                    and (last_candle['emarsi'] >= self.exit_long_emarsi3.value)
                    and (last_candle['slowsma'] > 0)
            ):
                return "exit_long_4"

        if self.exit_long_5.value and not trade.is_short:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (last_candle['minusdi'] < last_candle['plusdi'])
                    and (last_candle['close'] > last_candle['lowsma'])
                    and (last_candle['slowsma'] > 0)
            ):
                return "exit_long_5"


        if self.exit_short_1.value and trade.is_short:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['lowsma'] or last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['bigdown'])
            ):
                return "exit_short_1"

        if self.exit_short_2.value and trade.is_short:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['emarsi'] > self.exit_short_emarsi1.value or last_candle['close'] > last_candle['slowsma'])
                    and (last_candle['bigdown'])
            ):
                return "exit_short_2"

        if self.exit_short_3.value and trade.is_short:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['adx'] > self.exit_short_adx2.value)
                    and (last_candle['emarsi'] >= self.exit_short_emarsi2.value)
                    and (last_candle['bigup'])
            ):
                return "exit_short_3"

        if self.exit_short_4.value and trade.is_short:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['slowingdown'])
                    and (last_candle['emarsi'] >= self.exit_short_emarsi3.value)
                    and (last_candle['slowsma'] > 0)
            ):
                return "exit_short_4"

        if self.exit_short_5.value and trade.is_short:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (last_candle['minusdi'] < last_candle['plusdi'])
                    and (last_candle['close'] > last_candle['lowsma'])
                    and (last_candle['slowsma'] > 0)
            ):
                return "exit_short_5"

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value
