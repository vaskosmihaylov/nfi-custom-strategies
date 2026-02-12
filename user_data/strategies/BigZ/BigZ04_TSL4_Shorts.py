import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce


###########################################################################################################
##    BigZ04_TSL4_Shorts - Shorts-only variant of BigZ04_TSL4                                           ##
##                                                                                                       ##
##    Original BigZ04_TSL4 by Perkmeister, based on BigZ04 by ilya                                      ##
##    Shorts conversion inverts all entry/exit signals for bear market profiting                         ##
##                                                                                                       ##
##    - 11 short entry signals (inverted from longs)                                                    ##
##    - Trailing stoploss with hard stoploss protection                                                 ##
##    - 3x leverage via leverage() callback                                                             ##
##    - Max 4 short positions via confirm_trade_entry()                                                 ##
##                                                                                                       ##
###########################################################################################################

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class BigZ04_TSL4_Shorts(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True

    minimal_roi = {
        "0": 0.10,
    }

    stoploss = -0.20

    timeframe = '5m'
    inf_1h = '1h'

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.001
    ignore_roi_if_entry_signal = False

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    # Max short trades
    max_short_trades = 4

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    sell_params = {
        #############
        # Enable/Disable conditions (reusing same param names for shorts)
        "sell_condition_0_enable": True,
        "sell_condition_1_enable": True,
        "sell_condition_2_enable": True,
        "sell_condition_3_enable": True,
        "sell_condition_4_enable": True,
        "sell_condition_5_enable": True,
        "sell_condition_6_enable": True,
        "sell_condition_7_enable": True,
        "sell_condition_8_enable": True,
        "sell_condition_9_enable": True,
        "sell_condition_10_enable": True,
        "sell_condition_11_enable": True,
        "sell_condition_12_enable": True,
        "sell_condition_13_enable": False,
    }

    buy_params = {
        "base_nb_candles_sell": 49,
        "high_offset": 1.006,
        "pHSL": -0.08,
        "pPF_1": 0.016,
        "pSL_1": 0.011,
        "pPF_2": 0.080,
        "pSL_2": 0.040,
    }

    ############################################################################

    # Short entry condition enables
    sell_condition_0_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_9_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_10_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_11_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_12_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_13_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)

    # Inverted thresholds: RSI (100 - value), BB offsets inverted
    sell_bb20_close_bbupperband_safe_1 = DecimalParameter(0.950, 1.050, default=1.011, decimals=3, space='sell', optimize=False, load=True)
    sell_bb20_close_bbupperband_safe_2 = DecimalParameter(0.900, 1.300, default=1.018, decimals=2, space='sell', optimize=False, load=True)

    sell_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space='sell', decimals=1, optimize=False, load=True)
    sell_volume_drop_1 = DecimalParameter(1, 10, default=3.8, space='sell', decimals=1, optimize=False, load=True)
    sell_volume_drop_2 = DecimalParameter(1, 10, default=3, space='sell', decimals=1, optimize=False, load=True)
    sell_volume_drop_3 = DecimalParameter(1, 10, default=2.7, space='sell', decimals=1, optimize=False, load=True)

    # RSI 1h thresholds inverted: 100 - original_value
    sell_rsi_1h_0 = DecimalParameter(15.0, 45.0, default=29.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1h_1a = DecimalParameter(22.0, 35.0, default=31.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1h_1 = DecimalParameter(60.0, 90.0, default=83.5, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1h_2 = DecimalParameter(60.0, 90.0, default=85.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1h_3 = DecimalParameter(60.0, 90.0, default=80.0, space='sell', decimals=1, optimize=True, load=True)
    sell_rsi_1h_4 = DecimalParameter(60.0, 90.0, default=65.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1h_5 = DecimalParameter(40.0, 90.0, default=61.0, space='sell', decimals=1, optimize=False, load=True)

    # RSI thresholds inverted: 100 - original_value
    sell_rsi_0 = DecimalParameter(60.0, 90.0, default=70.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1 = DecimalParameter(60.0, 90.0, default=72.0, space='sell', decimals=1, optimize=True, load=True)
    sell_rsi_2 = DecimalParameter(60.0, 93.0, default=90.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_3 = DecimalParameter(60.0, 93.0, default=85.8, space='sell', decimals=1, optimize=False, load=True)

    sell_macd_1 = DecimalParameter(0.01, 0.09, default=0.02, space='sell', decimals=2, optimize=False, load=True)
    sell_macd_2 = DecimalParameter(0.01, 0.09, default=0.03, space='sell', decimals=2, optimize=False, load=True)

    # Inverted dip factor: 1/1.024 ≈ 0.977
    sell_dip_0 = DecimalParameter(0.960, 0.990, default=0.977, space='sell', decimals=3, optimize=False, load=True)

    # hyperopt parameters for custom_stoploss() - same as longs
    trade_time = IntParameter(25, 65, default=35, space='buy', optimize=False, load=True)
    rsi_1h_val = IntParameter(25, 45, default=32, space='buy', optimize=False, load=True)
    narrow_stop = DecimalParameter(1.005, 1.030, default=1.020, space='buy', decimals=3, optimize=False, load=True)
    wide_stop = DecimalParameter(1.010, 1.045, default=1.035, space='buy', decimals=3, optimize=False, load=True)

    # hyperopt parameters for SMAOffsetProtect exit signal
    base_nb_candles_sell = IntParameter(5, 80, default=49, space='buy', optimize=False, load=True)
    high_offset = DecimalParameter(0.99, 1.1, default=1.006, space='buy', optimize=False, load=True)

    # trailing stoploss hyperopt parameters - same as longs
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='buy', optimize=False, load=True)
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='buy', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='buy', optimize=False, load=True)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='buy', optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='buy', optimize=False, load=True)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        return 3.0

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

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        return False

    # Same trailing stoploss logic as longs - current_profit is already adjusted for shorts by Freqtrade
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        informative_1h['ema_50'] = ta.SMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.SMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband'] = bollinger['lower']
        informative_1h['bb_middleband'] = bollinger['mid']
        informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()

        # EMA
        dataframe['ema_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        # MACD
        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # SMA
        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ------ ATR stuff
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # Condition 12: BB upperband breakout in downtrend (inverted from longs BB lowerband dip)
        conditions.append(
            (
                self.sell_condition_12_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['close'] < dataframe['ema_200_1h']) &

                (dataframe['close'] > dataframe['bb_upperband'] * 1.007) &
                (dataframe['high'] > dataframe['bb_upperband'] * 1.015) &
                (dataframe['close'].shift() < dataframe['bb_upperband']) &
                (dataframe['rsi_1h'] > self.sell_rsi_1h_0.value) &
                (dataframe['open'] < dataframe['close']) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                ((dataframe['open'] - dataframe['close']).abs() < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &

                (dataframe['volume'] > 0)
            )
        )

        # Condition 11: Momentum breakdown (inverted from longs momentum breakout)
        conditions.append(
            (
                self.sell_condition_11_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &

                (dataframe['hist'] < 0) &
                (dataframe['hist'].shift() < 0) &
                (dataframe['hist'].shift(2) < 0) &
                (dataframe['hist'].shift(3) < 0) &
                (dataframe['hist'].shift(5) < 0) &

                (dataframe['bb_middleband'].shift(5) - dataframe['bb_middleband'] > dataframe['close'] / 200) &
                (dataframe['bb_middleband'].shift(10) - dataframe['bb_middleband'] > dataframe['close'] / 100) &
                ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) < (dataframe['close'] * 0.1)) &
                ((dataframe['close'].shift() - dataframe['open'].shift()) < (dataframe['close'] * 0.018)) &
                (dataframe['rsi'] < 49) &

                (dataframe['open'] > dataframe['close']) &
                (dataframe['open'].shift() < dataframe['close'].shift()) &

                (dataframe['close'] < dataframe['bb_middleband']) &
                (dataframe['close'].shift() > dataframe['bb_middleband'].shift()) &
                (dataframe['low'].shift(2) < dataframe['bb_middleband'].shift(2)) &

                (dataframe['volume'] > 0)
            )
        )

        # Condition 0: Overbought RSI rally (inverted from longs dip buying)
        conditions.append(
            (
                self.sell_condition_0_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &

                (dataframe['rsi'] > self.sell_rsi_0.value) &

                ((dataframe['close'] * self.sell_dip_0.value > dataframe['open'].shift(3)) |
                (dataframe['close'] * self.sell_dip_0.value > dataframe['open'].shift(2)) |
                (dataframe['close'] * self.sell_dip_0.value > dataframe['open'].shift(1))) &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_0.value) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            )
        )

        # Condition 1: BB upperband in downtrend (inverted from longs BB lowerband)
        conditions.append(
            (
                self.sell_condition_1_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['close'] < dataframe['ema_200_1h']) &

                (dataframe['close'] > dataframe['bb_upperband'] * self.sell_bb20_close_bbupperband_safe_1.value) &
                (dataframe['rsi_1h'] > self.sell_rsi_1h_1a.value) &
                (dataframe['open'] < dataframe['close']) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                ((dataframe['close'] - dataframe['open']).abs() < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &

                (dataframe['volume'] > 0)
            )
        )

        # Condition 2: BB upperband touch in downtrend (inverted from longs)
        conditions.append(
            (
                self.sell_condition_2_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &

                (dataframe['close'] > dataframe['bb_upperband'] * self.sell_bb20_close_bbupperband_safe_2.value) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                ((dataframe['close'] - dataframe['open']).abs() < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &
                (dataframe['volume'] > 0)
            )
        )

        # Condition 3: BB upperband with overbought RSI (inverted from longs)
        conditions.append(
            (
                self.sell_condition_3_enable.value &

                (dataframe['close'] < dataframe['ema_200_1h']) &

                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['rsi'] > self.sell_rsi_3.value) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_3.value)) &

                (dataframe['volume'] > 0)
            )
        )

        # Condition 4: BB upperband with overbought 1h RSI (inverted from longs)
        conditions.append(
            (
                self.sell_condition_4_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_1.value) &

                (dataframe['close'] > dataframe['bb_upperband']) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                (dataframe['volume'] > 0)
            )
        )

        # Condition 5: EMA bullish cross with BB upperband (inverted: short the top)
        conditions.append(
            (
                self.sell_condition_5_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['close'] < dataframe['ema_200_1h']) &

                (dataframe['ema_12'] > dataframe['ema_26']) &
                ((dataframe['ema_12'] - dataframe['ema_26']) > (dataframe['open'] * self.sell_macd_1.value)) &
                ((dataframe['ema_12'].shift() - dataframe['ema_26'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] > (dataframe['bb_upperband'])) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            )
        )

        # Condition 6: EMA bullish cross with overbought 1h RSI (inverted from longs)
        conditions.append(
            (
                self.sell_condition_6_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_5.value) &

                (dataframe['ema_12'] > dataframe['ema_26']) &
                ((dataframe['ema_12'] - dataframe['ema_26']) > (dataframe['open'] * self.sell_macd_2.value)) &
                ((dataframe['ema_12'].shift() - dataframe['ema_26'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] > (dataframe['bb_upperband'])) &

                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                (dataframe['volume'] > 0)
            )
        )

        # Condition 7: EMA bullish cross with overbought 1h RSI (inverted from longs)
        conditions.append(
            (
                self.sell_condition_7_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_2.value) &

                (dataframe['ema_12'] > dataframe['ema_26']) &
                ((dataframe['ema_12'] - dataframe['ema_26']) > (dataframe['open'] * self.sell_macd_1.value)) &
                ((dataframe['ema_12'].shift() - dataframe['ema_26'].shift()) > (dataframe['open'] / 100)) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            )
        )

        # Condition 8: Overbought RSI with overbought 1h RSI (inverted from longs)
        conditions.append(
            (
                self.sell_condition_8_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_3.value) &
                (dataframe['rsi'] > self.sell_rsi_1.value) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &

                (dataframe['volume'] > 0)
            )
        )

        # Condition 9: Extreme overbought RSI with overbought 1h RSI (inverted from longs)
        conditions.append(
            (
                self.sell_condition_9_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_4.value) &
                (dataframe['rsi'] > self.sell_rsi_2.value) &

                (dataframe['volume'] < (dataframe['volume'].shift() * self.sell_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.sell_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.sell_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0)
            )
        )

        # Condition 10: 1h BB upperband + MACD histogram (inverted from longs)
        conditions.append(
            (
                self.sell_condition_10_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_4.value) &
                (dataframe['close_1h'] > dataframe['bb_upperband_1h']) &

                (dataframe['hist'] < 0) &
                (dataframe['hist'].shift(2) > 0) &
                (dataframe['rsi'] > 59.5) &
                (dataframe['hist'] < -dataframe['close'] * 0.0012) &
                (dataframe['open'] > dataframe['close']) &

                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_short'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_middleband'] * 0.99) &
                (dataframe['volume'] > 0)
            )
            ,
            'exit_short'
        ] = 1

        return dataframe
