# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta
from typing import Dict, List

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce
from technical.indicators import RMI, zema

# --------------------------------
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

class BB_RPB_TSL_RNG_TBS_GOLD_Shorts(IStrategy):
    '''
        BB_RPB_TSL_Shorts
        Shorts-only variant of BB_RPB_TSL_RNG_TBS_GOLD
        @author jilv220 (original), converted to shorts
        Simple bollinger brand strategy inspired by this blog  ( https://hacks-for-life.blogspot.com/2020/12/freqtrade-notes.html )
        RPB, which stands for Real Pull Back, taken from ( https://github.com/GeorgeMurAlkh/freqtrade-stuff/blob/main/user_data/strategies/TheRealPullbackV2.py )
        The trailing custom stoploss taken from BigZ04_TSL from Perkmeister ( modded by ilya )
        Converted to shorts with inverted entry/exit conditions.
    '''

    INTERFACE_VERSION = 3
    can_short = True

    ##########################################################################

    # Hyperopt result area

    # sell space (inverted from buy)
    sell_params = {
        ##
        "sell_btc_safe": -289,
        "sell_btc_safe_1d": -0.05,
        ##
        "sell_threshold": 0.003,
        "sell_bb_factor": 0.999,
        "sell_bb_delta": 0.025,
        "sell_bb_width": 0.095,
        ##
        "sell_cci": 116,  # Inverted: was -116
        "sell_cci_length": 25,
        "sell_rmi": 51,  # Inverted: 100 - 49
        "sell_rmi_length": 17,
        "sell_srsi_fk": 68,  # Inverted: 100 - 32
        ##
        "sell_closedelta": 12.148,
        "sell_ema_diff": 0.022,
        ##
        "sell_adx": 20,
        "sell_fastd": 78,  # Inverted: 100 - 22 (approximately 80)
        "sell_fastk": 78,  # Inverted: 100 - 22
        "sell_ema_cofi": 1.02,  # Inverted: 2 - 0.98
        "sell_ewo_high": -4.179,  # Inverted sign
        ##
        "sell_ema_high_2": 0.913,  # Inverted: 2 - 1.087
        "sell_ema_low_2": 1.030,  # Inverted: 2 - 0.970
        ##
    }

    # cover space (inverted from sell)
    cover_params = {
        "pHSL": -0.178,
        "pPF_1": 0.019,
        "pPF_2": 0.065,
        "pSL_1": 0.019,
        "pSL_2": 0.062,
        "cover_btc_safe": -389,
        "base_nb_candles_cover": 24,
        "low_offset": 1.009,  # Inverted: 2 - 0.991
        "low_offset_2": 1.003  # Inverted: 2 - 0.997
    }

    # Keep ROI unchanged as requested
    minimal_roi = {
        "0": 0.10,
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Keep stoploss unchanged as requested
    stoploss = -0.049

    # Custom stoploss
    use_custom_stoploss = True
    use_exit_signal = True
    process_only_new_candles = True

    # Short position limits
    max_short_trades = 4
    ############################################################################

    ## Sell params (shorts entry)

    is_optimize_pump = False
    sell_rmi = IntParameter(50, 70, default=65, optimize=is_optimize_pump)
    sell_cci = IntParameter(90, 135, default=116, optimize=is_optimize_pump)
    sell_srsi_fk = IntParameter(50, 70, default=68, optimize=is_optimize_pump)
    sell_cci_length = IntParameter(25, 45, default=25, optimize=is_optimize_pump)
    sell_rmi_length = IntParameter(8, 20, default=17, optimize=is_optimize_pump)

    is_optimize_break = False
    sell_bb_width = DecimalParameter(0.05, 0.2, default=0.095, optimize=is_optimize_break)
    sell_bb_delta = DecimalParameter(0.025, 0.08, default=0.025, optimize=is_optimize_break)

    is_optimize_local_pump = False
    sell_ema_diff = DecimalParameter(0.022, 0.027, default=0.022, optimize=is_optimize_local_pump)
    sell_bb_factor = DecimalParameter(0.999, 1.010, default=0.999, optimize=False)
    sell_closedelta = DecimalParameter(12.0, 18.0, default=12.148, optimize=is_optimize_local_pump)

    is_optimize_ewo = False
    sell_rsi_fast = IntParameter(50, 65, default=55, optimize=False)  # Inverted: 100 - 45
    sell_rsi = IntParameter(70, 85, default=65, optimize=False)  # Inverted: 100 - 35
    sell_ewo = DecimalParameter(-5, 6.0, default=5.585, optimize=is_optimize_ewo)  # Inverted sign
    sell_ema_low = DecimalParameter(1.01, 1.10, default=1.058, optimize=is_optimize_ewo)  # Inverted: 2 - 0.942
    sell_ema_high = DecimalParameter(0.80, 1.05, default=0.916, optimize=is_optimize_ewo)  # Inverted: 2 - 1.084

    is_optimize_ewo_2 = False
    sell_ema_low_2 = DecimalParameter(1.022, 1.04, default=1.04, optimize=is_optimize_ewo_2)  # Inverted: 2 - 0.96
    sell_ema_high_2 = DecimalParameter(0.80, 0.95, default=0.91, optimize=is_optimize_ewo_2)  # Inverted: 2 - 1.09

    is_optimize_cofi = False
    sell_ema_cofi = DecimalParameter(1.02, 1.04, default=1.03, optimize=is_optimize_cofi)  # Inverted: 2 - 0.97
    sell_fastk = IntParameter(70, 80, default=78, optimize=is_optimize_cofi)  # Inverted: 100 - 20
    sell_fastd = IntParameter(70, 80, default=78, optimize=is_optimize_cofi)  # Inverted: 100 - 20
    sell_adx = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    sell_ewo_high = DecimalParameter(-12, -2, default=-4.179, optimize=is_optimize_cofi)  # Inverted sign

    is_optimize_btc_safe = False
    sell_btc_safe = IntParameter(-300, 50, default=-289, optimize=is_optimize_btc_safe)
    sell_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize=is_optimize_btc_safe)
    sell_threshold = DecimalParameter(0.003, 0.012, default=0.003, optimize=is_optimize_btc_safe)

    # Sell params toggle
    sell_is_pump_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_is_break_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)

    ## Cover params (shorts exit)
    cover_btc_safe = IntParameter(-400, -300, default=-389, optimize=True)
    base_nb_candles_cover = IntParameter(5, 80, default=cover_params['base_nb_candles_cover'], space='buy', optimize=True)
    low_offset = DecimalParameter(0.9, 1.05, default=cover_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(0.5, 1.01, default=cover_params['low_offset_2'], space='buy', optimize=True)

    ## Trailing params

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.178, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.019, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.019, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.065, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.062, decimals=3, space='sell', load=True)

    ############################################################################

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        """
        Fixed 3x leverage for all shorts
        """
        return 3.0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        """
        Only allow shorts, block longs
        Enforce max 4 short positions
        """
        # Block all long entries
        if side == "long":
            return False

        # Count open shorts
        short_count = sum(1 for trade in Trade.get_trades_proxy(is_open=True) if trade.is_short)

        # Enforce max shorts limit
        if short_count >= self.max_short_trades:
            return False

        return True

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"
        informative_pairs.append((btc_info_pair, self.timeframe))

        return informative_pairs

    ############################################################################

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    ############################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."

        # Bollinger bands (hyperopt hard to implement)
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        ### BTC protection

        # BTC info
        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"
        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe(btc_info_pair, timeframe=inf_tf)
        informative_past = informative.copy().shift(1)                                                                                                   # Get recent BTC info

        # BTC 5m dump protection
        informative_past_source = (informative_past['open'] + informative_past['close'] + informative_past['high'] + informative_past['low']) / 4        # Get BTC price
        informative_threshold = informative_past_source * self.sell_threshold.value                                                                       # BTC pump n% in 5 min
        informative_past_delta = informative_past['close'].shift(1) - informative_past['close']                                                          # should be negative if pump
        informative_diff = informative_threshold - informative_past_delta                                                                                # Need be larger than 0
        dataframe['btc_threshold'] = informative_threshold
        dataframe['btc_diff'] = informative_diff

        # BTC 1d pump protection
        informative_past_1d = informative.copy().shift(288)
        informative_past_source_1d = (informative_past_1d['open'] + informative_past_1d['close'] + informative_past_1d['high'] + informative_past_1d['low']) / 4
        dataframe['btc_5m'] = informative_past_source
        dataframe['btc_1d'] = informative_past_source_1d

        ### Other checks

        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        dataframe['bb_delta'] = ((dataframe['bb_upperband3'] - dataframe['bb_upperband2']) / dataframe['bb_upperband2'])  # Inverted: measures gap between 3-stdev and 2-stdev upper bands
        dataframe['bb_top_cross'] = qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband3']).astype('int')  # Inverted: crossed_above instead of crossed_below

        # CCI hyperopt
        for val in self.sell_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)
        dataframe['cci_long'] = ta.CCI(dataframe, 170)

        # RMI hyperopt
        for val in self.sell_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        # SRSI hyperopt ?
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # SMA
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # EMA
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, 50, 200)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)

        # Calculate all ma_cover values
        for val in self.base_nb_candles_cover.range:
            dataframe[f'ma_cover_{val}'] = ta.EMA(dataframe, timeperiod=val)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        if self.sell_is_pump_enabled.value:

            is_pump = (
                (dataframe[f'rmi_length_{self.sell_rmi_length.value}'] > self.sell_rmi.value) &  # Inverted: > instead of <
                (dataframe[f'cci_length_{self.sell_cci_length.value}'] >= self.sell_cci.value) &  # Inverted: >= instead of <=
                (dataframe['srsi_fk'] > self.sell_srsi_fk.value)  # Inverted: > instead of <
            )

        if self.sell_is_break_enabled.value:

            is_break = (

                (   (dataframe['bb_delta'] > self.sell_bb_delta.value)
                    &
                    (dataframe['bb_width'] > self.sell_bb_width.value)
                )
                &
                (dataframe['closedelta'] > dataframe['close'] * self.sell_closedelta.value / 1000 ) &
                (dataframe['close'] > dataframe['bb_upperband3'] * self.sell_bb_factor.value)  # Inverted: > and upperband3
            )

        is_local_downtrend = (  # Inverted from is_local_uptrend

                (dataframe['ema_26'] < dataframe['ema_12']) &  # Inverted: < instead of >
                (dataframe['ema_12'] - dataframe['ema_26'] > dataframe['open'] * self.sell_ema_diff.value) &  # Flipped subtraction
                (dataframe['ema_12'].shift() - dataframe['ema_26'].shift() > dataframe['open'] / 100) &  # Flipped subtraction
                (dataframe['close'] > dataframe['bb_upperband2'] * self.sell_bb_factor.value) &  # Inverted: > and upperband2
                (dataframe['closedelta'] > dataframe['close'] * self.sell_closedelta.value / 1000 )
            )

        is_ewo = (  # Inverted thresholds

                (dataframe['rsi_fast'] > self.sell_rsi_fast.value) &  # Inverted: > instead of <
                (dataframe['close'] > dataframe['ema_8'] * self.sell_ema_low.value) &  # Inverted: > instead of <
                (dataframe['EWO'] < self.sell_ewo.value) &  # Inverted: < instead of >
                (dataframe['close'] > dataframe['ema_16'] * self.sell_ema_high.value) &  # Inverted: > instead of <
                (dataframe['rsi'] > self.sell_rsi.value)  # Inverted: > instead of <
            )

        is_ewo_2 = (
                (dataframe['rsi_fast'] > self.sell_rsi_fast.value) &  # Inverted: > instead of <
                (dataframe['close'] > dataframe['ema_8'] * self.sell_ema_low_2.value) &  # Inverted: > instead of <
                (dataframe['EWO'] < self.sell_ewo_high.value) &  # Inverted: < instead of >
                (dataframe['close'] > dataframe['ema_16'] * self.sell_ema_high_2.value) &  # Inverted: > instead of <
                (dataframe['rsi'] > self.sell_rsi.value)  # Inverted: > instead of <
            )

        is_cofi = (
                (dataframe['open'] > dataframe['ema_8'] * self.sell_ema_cofi.value) &  # Inverted: > instead of <
                (qtpylib.crossed_below(dataframe['fastk'], dataframe['fastd'])) &  # Inverted: crossed_below instead of crossed_above
                (dataframe['fastk'] > self.sell_fastk.value) &  # Inverted: > instead of <
                (dataframe['fastd'] > self.sell_fastd.value) &  # Inverted: > instead of <
                (dataframe['adx'] > self.sell_adx.value) &
                (dataframe['EWO'] < self.sell_ewo_high.value)  # Inverted: < instead of >
            )

        # NFI quick mode (inverted)

        is_nfi_32 = (
                (dataframe['rsi_slow'] > dataframe['rsi_slow'].shift(1)) &  # Inverted: > instead of <
                (dataframe['rsi_fast'] > 54) &  # Inverted: 100 - 46
                (dataframe['rsi'] < 81) &  # Inverted: 100 - 19
                (dataframe['close'] > dataframe['sma_15'] * 1.058) &  # Inverted: > and 2 - 0.942
                (dataframe['cti'] > 0.86)  # Inverted: > instead of <
            )

        is_nfi_33 = (
                (dataframe['close'] > (dataframe['ema_13'] * 1.022)) &  # Inverted: > and 2 - 0.978
                (dataframe['EWO'] < -8) &  # Inverted: < instead of >
                (dataframe['cti'] > 0.88) &  # Inverted: > instead of <
                (dataframe['rsi'] > 68) &  # Inverted: 100 - 32
                (dataframe['r_14'] > -2.0) &  # Inverted: williams_r near 0 (overbought)
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.5))
            )

        is_BB_checked = is_pump & is_break

        ## condition append
        conditions.append(is_BB_checked)
        dataframe.loc[is_BB_checked, 'enter_tag'] += 'bb_short '

        conditions.append(is_local_downtrend)
        dataframe.loc[is_local_downtrend, 'enter_tag'] += 'local_downtrend '

        conditions.append(is_ewo)
        dataframe.loc[is_ewo, 'enter_tag'] += 'ewo_short '

        conditions.append(is_ewo_2)
        dataframe.loc[is_ewo_2, 'enter_tag'] += 'ewo2_short '

        conditions.append(is_cofi)
        dataframe.loc[is_cofi, 'enter_tag'] += 'cofi_short '

        conditions.append(is_nfi_32)
        dataframe.loc[is_nfi_32, 'enter_tag'] += 'nfi_32_short '

        conditions.append(is_nfi_33)
        dataframe.loc[is_nfi_33, 'enter_tag'] += 'nfi_33_short '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] < dataframe['sma_9']) &  # Inverted: < instead of >
                (dataframe['close'] < (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.low_offset_2.value)) &  # Inverted: < and low_offset
                (dataframe['rsi'] < 50) &  # Inverted: < instead of >
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] < dataframe['rsi_slow'])  # Inverted: < instead of >

            )
            |
            (
                (dataframe['sma_9'] < (dataframe['sma_9'].shift(1) - dataframe['sma_9'].shift(1) * 0.005 )) &  # Inverted: < and subtraction
                (dataframe['close'] > dataframe['hma_50']) &  # Inverted: > instead of <
                (dataframe['close'] < (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.low_offset.value)) &  # Inverted: < and low_offset
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] < dataframe['rsi_slow'])  # Inverted: < instead of >
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_short'
            ] = 1

        return dataframe
