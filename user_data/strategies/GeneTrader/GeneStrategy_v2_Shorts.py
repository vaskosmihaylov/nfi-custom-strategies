import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from functools import reduce
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter,IntParameter,informative
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import math
import logging
from freqtrade.persistence import Trade
import pandas_ta as pta
from technical.indicators import RMI

logger = logging.getLogger(__name__)

# Elliot Wave Oscillator
def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif



def top_percent_change_dca(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

#EWO

def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
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
        name="{0} Williams %R".format(period),
        )

    return WR * -100



# VWAP bands
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    upper_band = rolling_mean + (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(upper_band)

#Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class GeneStrategy_v2_Shorts(IStrategy):
    """
    GeneStrategy_v2_Shorts - Short-only version

    Converted from GeneStrategy_v2.py (long strategy) to trade shorts.
    All entry/exit logic inverted for short positions.

    Date: 2026-01-28
    """
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "2026-01-28 (Shorts)"

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Shorts only
    can_short = True

    # DCA
    position_adjustment_enable = True

    # Stoploss:
    stoploss = -0.99  # Safety net

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Custom stoploss
    use_custom_stoploss = False

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': "market",
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    def is_support(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data)-1):
            if row < len(row_data)/2:
                conditions.append(row_data[row] > row_data[row+1])
            else:
                conditions.append(row_data[row] < row_data[row+1])
        return reduce(lambda x, y: x & y, conditions)

    # Protection (NFIX29) - inverted for shorts
    fast_ewo = 50
    slow_ewo = 200

    # NFINext44 - inverted for shorts
    sell_44_ma_offset = 1.018  # was buy_44_ma_offset = 0.982
    sell_44_ewo = 18.143  # was -18.143
    sell_44_cti = 0.8  # was -0.8
    sell_44_r_1h = 75.0  # was -75.0

    # NFINext37 - inverted for shorts
    sell_37_ma_offset = 1.02  # was 0.98
    sell_37_ewo = -9.8  # was 9.8
    sell_37_rsi = 44.0  # was 56.0 (inverted)
    sell_37_cti = 0.7  # was -0.7

    # NFINext7 - inverted for shorts
    sell_ema_open_mult_7 = 0.030
    sell_cti_7 = 0.89  # was -0.89

    # Sell parameters (for short entries - inverted from buy params)
    sell_rmi = IntParameter(50.0, 70.0, default=55, space='sell', optimize=True)
    sell_cci = IntParameter(90.0, 135.0, default=126, space='sell', optimize=True)
    sell_srsi_fk = IntParameter(50.0, 70.0, default=58, space='sell', optimize=True)
    sell_cci_length = IntParameter(25.0, 45.0, default=42, space='sell', optimize=True)
    sell_rmi_length = IntParameter(8.0, 20.0, default=11, space='sell', optimize=True)

    sell_bb_width = DecimalParameter(0.065, 0.135, default=0.097, space='sell', optimize=True)
    sell_bb_delta = DecimalParameter(0.018, 0.035, default=0.028, space='sell', optimize=True)

    sell_roc_1h = IntParameter(-200.0, 25.0, default=-13, space='sell', optimize=True)
    sell_bb_width_1h = DecimalParameter(0.3, 2.0, default=1.3, space='sell', optimize=True)

    # ClucHA - inverted for shorts
    is_optimize_clucha = False
    sell_clucha_bbdelta_close = DecimalParameter(0.0005, 0.02, default=0.001, space='sell', optimize=True)
    sell_clucha_bbdelta_tail = DecimalParameter(0.7, 1.0, default=1.0, space='sell', optimize=True)
    sell_clucha_close_bbupper = DecimalParameter(0.0005, 0.02, default=0.008, space='sell', optimize=True)
    sell_clucha_closedelta_close = DecimalParameter(0.0005, 0.02, default=0.014, space='sell', optimize=True)
    sell_clucha_rocr_1h = DecimalParameter(0.5, 1.0, default=0.51, space='sell', optimize=True)

    # Local_Downtrend (inverted from uptrend)
    sell_ema_diff = DecimalParameter(0.022, 0.027, default=0.026, space='sell', optimize=True)
    sell_bb_factor = DecimalParameter(1.001, 1.01, default=1.005, space='sell', optimize=True)
    sell_closedelta = DecimalParameter(12.0, 18.0, default=13.1, space='sell', optimize=True)

    # Sell params (inverted from buy params)
    rocr_1h = DecimalParameter(0.5, 1.0, default=0.51, space='sell', optimize=True)
    rocr1_1h = DecimalParameter(0.5, 1.0, default=0.59, space='sell', optimize=True)
    bbdelta_close = DecimalParameter(0.0005, 0.02, default=0.001, space='sell', optimize=True)
    closedelta_close = DecimalParameter(0.0005, 0.02, default=0.014, space='sell', optimize=True)
    bbdelta_tail = DecimalParameter(0.7, 1.0, default=1.0, space='sell', optimize=True)
    close_bbupper = DecimalParameter(0.0005, 0.02, default=0.008, space='sell', optimize=True)

    # Buy params (for short exits - inverted from sell params)
    buy_fisher = DecimalParameter(-0.5, -0.1, default=-0.5, space='buy', optimize=True)
    buy_bbmiddle_close = DecimalParameter(0.9, 1.03, default=0.933, space='buy', optimize=True)

    # Bullfish (inverted from Deadfish)
    buy_bullfish_bb_width = DecimalParameter(0.03, 0.75, default=0.06, space='buy', optimize=True)
    buy_bullfish_profit = DecimalParameter(-0.15, -0.05, default=-0.1, space='buy', optimize=True)
    buy_bullfish_bb_factor = DecimalParameter(0.9, 1.2, default=1.2, space='buy', optimize=True)
    buy_bullfish_volume_factor = DecimalParameter(1.0, 2.5, default=1.9, space='buy', optimize=True)

    # SMAOffset (inverted for shorts)
    base_nb_candles_sell = IntParameter(8.0, 20.0, default=13, space='sell', optimize=True)
    base_nb_candles_buy = IntParameter(8.0, 50.0, default=44, space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=1.009, space='sell', optimize=True)
    low_offset = DecimalParameter(0.985, 0.995, default=0.993, space='buy', optimize=True)
    low_offset_2 = DecimalParameter(0.98, 0.99, default=0.99, space='buy', optimize=True)

    # Trailing (inverted logic for shorts)
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.25, space='sell', optimize=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.5, space='sell', optimize=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.08, space='sell', optimize=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', optimize=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.08, space='sell', optimize=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.07, space='sell', optimize=True)

    # hard stoploss profit
    pHSL = DecimalParameter(-0.5, -0.04, default=-0.163, space='buy', optimize=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.02, default=0.01, space='buy', optimize=True)
    pSL_1 = DecimalParameter(0.008, 0.02, default=0.008, space='buy', optimize=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.04, 0.1, default=0.072, space='buy', optimize=True)
    pSL_2 = DecimalParameter(0.02, 0.07, default=0.054, space='buy', optimize=True)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        """
        Fixed 3x leverage for all shorts
        """
        return 3.0

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        informative_pairs += [("BTC/USDT", "5m"),
                             ]
        return informative_pairs

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_sells = trade.select_filled_orders('exit')
        count_of_sells = len(filled_sells)

        if (last_candle is not None):
            # Time-based exit for shorts (inverted logic)
            if (current_time - timedelta(minutes=30) > trade.open_date_utc) & (trade.open_date_utc + timedelta(minutes=1500) < current_time) & (last_candle['close'] > last_candle['ema_200']):
                return 'dlho_to_trva_short'

            # Trailing stops for shorts
            if (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (((trade.open_rate - trade.min_rate) / 100) > (current_profit + self.sell_trail_down_1.value)):
                return 'trail_target_1_short'
            elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (((trade.open_rate - trade.min_rate) / 100) > (current_profit + self.sell_trail_down_2.value)):
                return 'trail_target_2_short'
            elif (current_profit > 3) & (last_candle['rsi'] < 15):  # Inverted from >85
                 return 'RSI-15 target'

            # Exit signals for profitable shorts
            if (current_profit > 0) & (count_of_sells < 4) & (last_candle['close'] < last_candle['hma_50']) & (last_candle['close'] < (last_candle[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) & (last_candle['rsi']<50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] < last_candle['rsi_slow']):
                return 'buy signal1_short'
            if (current_profit > 0) & (count_of_sells >= 4) & (last_candle['close'] < last_candle['hma_50'] * 0.99) & (last_candle['close'] < (last_candle[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) & (last_candle['rsi']<50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] < last_candle['rsi_slow']):
                return 'buy signal1 * 0.99_short'
            if (current_profit > 0) & (last_candle['close'] < last_candle['hma_50']) & (last_candle['close'] < (last_candle[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &  (last_candle['volume'] > 0) & (last_candle['rsi_fast'] < last_candle['rsi_slow']):
                return 'buy signal2_short'

            # Bullfish protection (inverted from Deadfish)
            if (    (current_profit < self.buy_bullfish_profit.value)
                and (last_candle['close'] > last_candle['ema_200'])
                and (last_candle['bb_width'] < self.buy_bullfish_bb_width.value)
                and (last_candle['close'] < last_candle['bb_middleband2'] * self.buy_bullfish_bb_factor.value)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.buy_bullfish_volume_factor.value)
                and (last_candle['cmf'] > 0.0)
            ):
                return f"buy_stoploss_bullfish"


    # come from BB_RPB_TSL
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

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        info_tf = '5m'

        informative = self.dp.get_pair_dataframe('BTC/USDT', timeframe=info_tf)
        informative_btc = informative.copy().shift(1)

        dataframe['btc_close'] = informative_btc['close']
        dataframe['btc_ema_fast'] = ta.EMA(informative_btc, timeperiod=20)
        dataframe['btc_ema_slow'] = ta.EMA(informative_btc, timeperiod=25)
        dataframe['up'] = (dataframe['btc_ema_fast'] > dataframe['btc_ema_slow']).astype('int')

        # Calculate all ma_buy values (for short exits)
        for val in self.base_nb_candles_buy.range:
             dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        ## BB 40
        bollinger2_40 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        #EMA
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # ClucHA
        dataframe['bb_delta_cluc'] = (dataframe['bb_upperband2_40'] - dataframe['bb_middleband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()

        # SRSI hyperopt
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # Set Up Bollinger Bands (use upper band for shorts)
        mid, upper = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['upper'] = upper
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (dataframe['upper'] - mid).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_high'] - dataframe['ha_close']).abs()

        dataframe['bb_upperband'] = dataframe['upper']
        dataframe['bb_middleband'] = dataframe['mid']

        # DIP (inverted to PUMP for shorts)
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        dataframe['bb_delta'] = ((dataframe['bb_upperband3'] - dataframe['bb_upperband2']) / dataframe['bb_upperband2'])

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

         # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)

        dataframe['vwap_high'] = vwap_high
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100
        # Diff
        dataframe['ema_vwap_diff_50'] = ( ( dataframe['vwap_upperband'] - dataframe['ema_50'] ) / dataframe['ema_50'] )

        # Pump protection (inverted from dip)
        dataframe['tpct_change_0']   = top_percent_change_dca(dataframe,0)
        dataframe['tpct_change_1']   = top_percent_change_dca(dataframe,1)
        dataframe['tcp_percent_4'] =   top_percent_change_dca(dataframe , 4)

        # NFINEXT44
        dataframe['ewo'] = ewo(dataframe, 50, 200)

        # SMA
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        # RMI hyperopt
        for val in self.sell_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        # CCI hyperopt
        for val in self.sell_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        #CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # NFIX39
        dataframe['bb_delta_cluc'] = (dataframe['bb_upperband2_40'] - dataframe['bb_middleband2_40']).abs()

        # NFIX29
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # local_downtrend
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        # insta_signal
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # rebuy check if EMA is falling
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)

        # Profit Maximizer - PMAX (NFINext37)
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)
        informative['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        informative['cmf'] = chaikin_money_flow(dataframe, 20)

        # Resistance levels (inverted from support)
        res_series = informative['high'].rolling(window = 5, center=True).apply(lambda row: not self.is_support(row), raw=True).shift(2)
        informative['res_level'] = Series(np.where(res_series, np.where(informative['close'] > informative['open'], informative['close'], informative['open']), float('NaN'))).ffill()
        informative['roc'] = ta.ROC(informative, timeperiod=9)

        informative['r_480'] = williams_r(informative, period=480)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_lowerband2'] = bollinger2['lower']
        informative['bb_middleband2'] = bollinger2['mid']
        informative['bb_upperband2'] = bollinger2['upper']
        informative['bb_width'] = ((informative['bb_upperband2'] - informative['bb_lowerband2']) / informative['bb_middleband2'])

        informative['r_84'] = williams_r(informative, period=84)
        informative['cti_40'] = pta.cti(informative["close"], length=40)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Short entry signals (inverted from long entry signals)
        """

        # BTC pump (inverted from dump)
        btc_pump = (
                (dataframe['btc_close'].rolling(24).min() <= (dataframe['btc_close'] * 0.97 ))
            )
        rsi_check = (
                (dataframe['rsi_84'] > 40) &
                (dataframe['rsi_112'] > 40)
            )

        # PUMP signal (inverted from DIP)
        dataframe.loc[
                ((dataframe[f'rmi_length_{self.sell_rmi_length.value}'] > self.sell_rmi.value) &
                (dataframe[f'cci_length_{self.sell_cci_length.value}'] >= self.sell_cci.value) &
                (dataframe['srsi_fk'] > self.sell_srsi_fk.value) &
                (dataframe['bb_delta'] > self.sell_bb_delta.value) &
                (dataframe['bb_width'] > self.sell_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.sell_closedelta.value / 1000 ) &
                (dataframe['close'] > dataframe['bb_upperband3'] * self.sell_bb_factor.value)&
                (dataframe['roc_1h'] > self.sell_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.sell_bb_width_1h.value)
            ),
        ['enter_short', 'enter_tag']] = (1, 'PUMP signal')

        # Break signal (inverted)
        dataframe.loc[

                ((dataframe['bb_delta'] > self.sell_bb_delta.value) &
                (dataframe['bb_width'] > self.sell_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.sell_closedelta.value / 1000 ) &
                (dataframe['close'] > dataframe['bb_upperband3'] * self.sell_bb_factor.value)&
                (dataframe['roc_1h'] > self.sell_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.sell_bb_width_1h.value)

            ),
        ['enter_short', 'enter_tag']] = (1, 'Break signal short')

        # cluc_HA (inverted)
        dataframe.loc[

                    ((dataframe['rocr_1h'] < self.sell_clucha_rocr_1h.value ) &

                        (dataframe['bb_upperband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.sell_clucha_bbdelta_close.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.sell_clucha_closedelta_close.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.sell_clucha_bbdelta_tail.value) &
                        (dataframe['ha_close'] > dataframe['bb_upperband2_40'].shift()) &
                        (dataframe['close'] < (dataframe['res_level_1h'] * 1.12)) &
                        (dataframe['ha_close'] > dataframe['ha_close'].shift())

                    ),
        ['enter_short', 'enter_tag']] = (1, 'cluc_HA_short')

        # NFIX39 (inverted)
        dataframe.loc[
                ((dataframe['ema_200'] < (dataframe['ema_200'].shift(12) * 0.99)) &
                (dataframe['ema_200'] < (dataframe['ema_200'].shift(48) * 0.93)) &
                (dataframe['bb_upperband2_40'].shift().gt(0)) &
                (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) &
                (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)) &
                (dataframe['close'].gt(dataframe['bb_upperband2_40'].shift())) &
                (dataframe['close'].ge(dataframe['close'].shift())) &
                (dataframe['close'] < dataframe['ema_50'] * 1.088)

            ),
        ['enter_short', 'enter_tag']] = (1, 'NFIX39_short')

        # NFIX29 (inverted)
        dataframe.loc[
                ((dataframe['close'] < (dataframe['res_level_1h'] * 1.28)) &
                (dataframe['close'] > (dataframe['ema_16'] * 1.018)) &
                (dataframe['EWO'] > 10.0) &
                (dataframe['cti'] > 0.9)

            ),
        ['enter_short', 'enter_tag']] = (1, 'NFIX29_short')

        # local_downtrend (inverted from uptrend)
        dataframe.loc[
                ((dataframe['ema_26'] < dataframe['ema_12']) &
                (dataframe['ema_12'] - dataframe['ema_26'] > dataframe['open'] * self.sell_ema_diff.value) &
                (dataframe['ema_12'].shift() - dataframe['ema_26'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] > dataframe['bb_upperband2'] * self.sell_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.sell_closedelta.value / 1000 )

             ),
        ['enter_short', 'enter_tag']] = (1, 'local_downtrend')

        # vwap (inverted)
        dataframe.loc[
                (

                (dataframe['close'] > dataframe['vwap_high']) &
                (dataframe['tcp_percent_4'] > 0.053) &
                (dataframe['cti'] > 0.8) &
                (dataframe['rsi'] > 65) &
                (dataframe['rsi_84'] > 40) &
                (dataframe['rsi_112'] > 40) &
                (dataframe['volume'] > 0)
           ),
        ['enter_short', 'enter_tag']] = (1, 'vwap_short')

        # insta_signal (inverted)
        dataframe.loc[
                ((dataframe['bb_width_1h'] > 0.131) &
                (dataframe['r_14'] > 51) &
                (dataframe['r_84_1h'] > 70) &
                (dataframe['cti'] > 0.845) &
                (dataframe['cti_40_1h'] > 0.735)
                &
                ( (dataframe['close'].rolling(48).min() <= (dataframe['close'] * 0.9 )) ) &
                (dataframe['btc_close'].rolling(24).min() <= (dataframe['btc_close'] * 0.97 ))
          ),
        ['enter_short', 'enter_tag']] = (1, 'insta_signal_short')

        # NFINext44 (inverted)
        dataframe.loc[
            ((dataframe['close'] > (dataframe['ema_16'] * self.sell_44_ma_offset))&
            (dataframe['ewo'] > self.sell_44_ewo)&
            (dataframe['cti'] > self.sell_44_cti)&
            (dataframe['r_480_1h'] > self.sell_44_r_1h)&
            (dataframe['volume'] > 0)
          ),
        ['enter_short', 'enter_tag']] = (1, 'NFINext44_short')


        # NFINext37 (inverted)
        dataframe.loc[
            ((dataframe['pm'] < dataframe['pmax_thresh'])&
            (dataframe['close'] > dataframe['sma_75'] * self.sell_37_ma_offset)&
            (dataframe['ewo'] < self.sell_37_ewo)&
            (dataframe['rsi'] > self.sell_37_rsi)&
            (dataframe['cti'] > self.sell_37_cti)
        ),
        ['enter_short', 'enter_tag']] = (1, 'NFINext37_short')

        # NFINext7 (inverted)
        dataframe.loc[
            ((dataframe['ema_26'] < dataframe['ema_12'])&
            ((dataframe['ema_12'] - dataframe['ema_26']) > (dataframe['open'] * self.sell_ema_open_mult_7))&
            ((dataframe['ema_12'].shift() - dataframe['ema_26'].shift()) > (dataframe['open'] / 100))&
            (dataframe['cti'] > self.sell_cti_7)

        ),
        ['enter_short', 'enter_tag']] = (1, 'NFINext7_short')

        # newstrat52 (inverted)
        dataframe.loc[
                ((dataframe['rsi_slow'] > dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] > 54) &
                (dataframe['rsi'] < 81) &
                (dataframe['close'] > dataframe['sma_15'] * 1.058) &
                (dataframe['cti'] > 0.86)
        ),
        ['enter_short', 'enter_tag']] = (1, 'NFINext32_short')

        # sma_3 (inverted)
        dataframe.loc[
                ((dataframe['bb_upperband2_40'].shift() > 0) &
                (dataframe['bb_delta_cluc'] > dataframe['close'] * 0.059) &
                (dataframe['ha_closedelta'] > dataframe['close'] * 0.023) &
                (dataframe['tail'] < dataframe['bb_delta_cluc'] * 0.24) &
                (dataframe['close'] > dataframe['bb_upperband2_40'].shift()) &
                (dataframe['close'] > dataframe['close'].shift()) &
                (btc_pump == 0)
        ),
        ['enter_short', 'enter_tag']] = (1, 'sma_3_short')

        # WVAP (inverted)
        dataframe.loc[
                ((dataframe['close'] > dataframe['vwap_upperband']) &
                (dataframe['tpct_change_1'] > 0.04) &
                (dataframe['cti'] > 0.8) &
                (dataframe['rsi'] > 65) &
                (rsi_check) &
                (btc_pump == 0)
        ),
        ['enter_short', 'enter_tag']] = (1, 'WVAP_short')


        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Short exit signals (inverted from long exit signals)
        """

        dataframe.loc[
            (dataframe['fisher'] < self.buy_fisher.value) &
            (dataframe['ha_low'].ge(dataframe['ha_low'].shift(1))) &
            (dataframe['ha_low'].shift(1).ge(dataframe['ha_low'].shift(2))) &
            (dataframe['ha_close'].ge(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] < dataframe['ha_close']) &
            ((dataframe['ha_close'] * self.buy_bbmiddle_close.value) < dataframe['bb_middleband']) &
            (dataframe['volume'] > 0),
            'exit_short'
        ] = 1

        return dataframe

    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4

    def top_percent_change_dca(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None

        # credits to reinuvader for not blindly executing safety orders
        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only sell when it seems it's dropping back down (for shorts)
        last_candle = dataframe.iloc[-1].squeeze()

        filled_sells = trade.select_filled_orders('entry')
        count_of_sells = len(filled_sells)

        # DCA logic for shorts (inverted conditions)
        if count_of_sells == 1 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open']) :
            return None
        elif count_of_sells == 2 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
            return None
        elif count_of_sells == 3 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open'])and (last_candle['ema_vwap_diff_50'] < 0.215) :
            return None
        elif count_of_sells == 4 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open'])and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5']) <= (last_candle['ema_10']):
            return None
        elif count_of_sells == 5 and (last_candle['cmf_1h'] > 0.00) and (last_candle['close'] > last_candle['open']) and (last_candle['rsi_14_1h'] > 70) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5']) <= (last_candle['ema_10']):
            logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to drop below 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to drop below 70")
            return None
        elif count_of_sells == 6 and (last_candle['cmf_1h'] > 0.00) and (last_candle['close'] > last_candle['open']) and (last_candle['rsi_14_1h'] > 70) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open'] and (last_candle['ema_vwap_diff_50'] < 0.215)) and (last_candle['ema_5']) <= (last_candle['ema_10']):
            logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to drop below 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to drop below 70")
            return None
        elif count_of_sells == 7 and (last_candle['cmf_1h'] > 0.00) and (last_candle['close'] > last_candle['open']) and (last_candle['rsi_14_1h'] > 70) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open'] and (last_candle['ema_vwap_diff_50'] < 0.215)) and (last_candle['ema_5']) <= (last_candle['ema_10']):
            logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to drop below 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to drop below 70")
            return None
        elif count_of_sells == 8 and (last_candle['cmf_1h'] > 0.00) and (last_candle['close'] > last_candle['open']) and (last_candle['rsi_14_1h'] > 70) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open'] and (last_candle['ema_vwap_diff_50'] < 0.215)) and (last_candle['ema_5']) <= (last_candle['ema_10']):
            logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to drop below 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to drop below 70")
            return None

        if 1 <= count_of_sells <= self.max_safety_orders:
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_sells)
            if (self.safety_order_step_scale > 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_sells - 1)) - 1) / (self.safety_order_step_scale - 1))
            elif (self.safety_order_step_scale < 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale,(count_of_sells - 1))) / (1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    # This returns first order stake size
                    stake_amount = filled_sells[0].cost
                    # This then calculates current safety order size
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale,(count_of_sells - 1))
                    amount = stake_amount / current_rate
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = 'MA_' + str(MAtype) + '_' + str(length)
    atr = 'ATR_' + str(period)
    pm = 'pm_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
    pmx = 'pmX_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)

    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    # 1 = up, -1 = down, 0 = no trend
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), -1, 1), 0)

    return pm, pmx
