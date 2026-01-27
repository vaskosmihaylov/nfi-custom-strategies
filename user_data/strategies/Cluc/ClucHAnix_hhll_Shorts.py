import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import time
import logging

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter
from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
from typing import Optional

logger = logging.getLogger(__name__)

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(upper_band), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class ClucHAnix_hhll_Shorts(IStrategy):
    """
    ClucHAnix_hhll Shorts Strategy

    Converted from ClucHAnix_hhll long strategy with inverted logic for short positions.
    Uses Heikin Ashi candles, Bollinger Bands, and multiple indicators for overbought
    entry detection and bullish exit signals.

    Key Features:
    - Entries on overbought conditions (price above upper Bollinger Band)
    - Advanced risk management with pump detection
    - Slippage protection via confirm_trade_entry
    - Max 4 short positions enforced
    - Progressive trailing stop loss

    Author: Converted from ClucHAnix_hhll (long strategy)
    Version: 1.0.0
    """

    INTERFACE_VERSION = 3
    can_short = True

    # Max short positions
    max_short_trades = 8

    # Hyperparameters (inverted from long strategy)
    buy_params = {
        ##
        "max_slip": 0.73,
        ##
        "bbdelta_close": 0.01846,
        "bbdelta_tail": 0.98973,
        "close_bbupper": 0.00785,  # Using upper band instead of lower
        "closedelta_close": 0.01009,
        "rocr_1h": 0.4589,  # Inverted: 1 - 0.5411 for overbought detection
        ##
        "short_ll_diff_48": 6.867,  # Short at lows (inverted from hh_diff)
        "short_hh_diff_48": -12.884,  # Short away from highs (inverted from ll_diff)
    }

    # Sell hyperspace params (same trailing logic):
    sell_params = {
        "pPF_1": 0.011,
        "pPF_2": 0.064,
        "pSL_1": 0.011,
        "pSL_2": 0.062,

        # exit signal params (inverted)
        "low_offset": 0.907,  # Exit when price drops (inverse of high_offset)
        "low_offset_2": 1.211,
        "sell_bbmiddle_close": 1.02714,  # Inverted: 2 - 0.97286
        "sell_fisher": -0.48492,  # Negative Fisher for bearish
    }

    # ROI table:
    minimal_roi = {
        "0": 0.103,
        "3": 0.05,
        "5": 0.033,
        "61": 0.027,
        "125": 0.011,
        "292": 0.005,
    }

    # Stoploss:
    stoploss = -0.99  # use custom stoploss

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """

    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergencyexit': 'market',
        'forceentry': "market",
        'forceexit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # Entry params (shorts enter on overbought)
    is_optimize_clucHA = False
    rocr_1h = RealParameter(0.4, 0.9, default=buy_params['rocr_1h'], space='buy', optimize=is_optimize_clucHA)
    bbdelta_close = RealParameter(0.0005, 0.02, default=buy_params['bbdelta_close'], space='buy', optimize=is_optimize_clucHA)
    closedelta_close = RealParameter(0.0005, 0.02, default=buy_params['closedelta_close'], space='buy', optimize=is_optimize_clucHA)
    bbdelta_tail = RealParameter(0.7, 1.0, default=buy_params['bbdelta_tail'], space='buy', optimize=is_optimize_clucHA)
    close_bbupper = RealParameter(0.0005, 0.02, default=buy_params['close_bbupper'], space='buy', optimize=is_optimize_clucHA)

    is_optimize_hh_ll = False
    short_ll_diff_48 = DecimalParameter(0.0, 15, default=buy_params['short_ll_diff_48'], space='buy', optimize=is_optimize_hh_ll)
    short_hh_diff_48 = DecimalParameter(-23, 40, default=buy_params['short_hh_diff_48'], space='buy', optimize=is_optimize_hh_ll)

    ## Slippage params
    is_optimize_slip = False
    max_slip = DecimalParameter(0.33, 0.80, default=buy_params['max_slip'], decimals=3, optimize=is_optimize_slip, space='buy', load=True)

    # exit params (shorts exit on bullish signals)
    is_optimize_sell = False
    sell_fisher = RealParameter(-0.5, -0.1, default=sell_params['sell_fisher'], space='sell', optimize=is_optimize_sell)
    sell_bbmiddle_close = RealParameter(0.9, 1.3, default=sell_params['sell_bbmiddle_close'], space='sell', optimize=is_optimize_sell)
    low_offset = DecimalParameter(0.80, 1.1, default=sell_params['low_offset'], space='sell', optimize=is_optimize_sell)
    low_offset_2 = DecimalParameter(0.50, 0.85, default=sell_params['low_offset_2'], space='sell', optimize=is_optimize_sell)

    is_optimize_trailing = False
    pPF_1 = DecimalParameter(0.011, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize=is_optimize_trailing)
    pSL_1 = DecimalParameter(0.011, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize=is_optimize_trailing)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize=is_optimize_trailing)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize=is_optimize_trailing)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    # come from BB_RPB_TSL
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        sl_profit = -0.99

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = -0.99

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    ## Confirm Entry - enforces max short positions and slippage check
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Enforce maximum short position limit and check slippage.
        """
        # Only allow shorts in this strategy
        if side == "long":
            return False  # Reject any long signals

        # Count current open short positions
        short_count = 0
        trades = Trade.get_trades_proxy(is_open=True)
        for trade in trades:
            if trade.is_short:
                short_count += 1

        # Check if we can open another short
        if short_count >= self.max_short_trades:
            return False  # Already at max short positions

        # Slippage check
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        max_slip = self.max_slip.value

        if len(dataframe) < 1:
            return False

        dataframe = dataframe.iloc[-1].squeeze()
        if rate > dataframe['close']:
            slippage = ((rate / dataframe['close']) - 1) * 100

            if slippage < max_slip:
                return True
            else:
                return False

        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        Custom exit logic for shorts - inverted from long strategy.
        Exits on bullish reversal signals or rally conditions.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]
        previous_candle_2 = dataframe.iloc[-3]

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate) if not trade.is_short else ((trade.open_rate - trade.min_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate) if not trade.is_short else ((trade.max_rate - trade.open_rate) / trade.open_rate)

        # stoploss - rally (inverse of deadfish for shorts)
        # Exit short if price rallies above EMA200 with narrow BB
        if (    (current_profit < -0.063)
                and (last_candle['close'] > last_candle['ema_200'])
                and (last_candle['bb_width'] < 0.043)
                and (last_candle['close'] < last_candle['bb_middleband2'] * 1.046)  # Inverted: 2 - 0.954
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * 2.37)
            ):
            return 'exit_stoploss_rally'

        # stoploss - dump recovery (inverted pump logic)
        # Exit if price recovers after dump (hl_pct_change_48 negative = dump)
        if (last_candle['hl_pct_change_48_1h'] < -0.95):  # Inverted: dump instead of pump
            if (
                    (-0.04 > current_profit > -0.08)
                    and (max_profit < 0.005)
                    and (max_loss < 0.08)
                    and (last_candle['close'] > last_candle['ema_200'])  # Above EMA = bullish
                    and (last_candle['sma_200_dec_20'] == False)  # SMA rising
                    and (last_candle['ema_vwma_osc_32'] > 0.0)  # Positive oscillators = bullish
                    and (last_candle['ema_vwma_osc_64'] > 0.0)
                    and (last_candle['ema_vwma_osc_96'] > 0.0)
                    and (last_candle['cmf'] > 0.25)  # Positive money flow
                    and (last_candle['cmf_1h'] > 0.0)
            ):
                return 'exit_stoploss_d_48_1_1'
            elif (
                    (-0.04 > current_profit > -0.08)
                    and (max_profit < 0.01)
                    and (max_loss < 0.08)
                    and (last_candle['close'] > last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'] == False)
                    and (last_candle['ema_vwma_osc_32'] > 0.0)
                    and (last_candle['ema_vwma_osc_64'] > 0.0)
                    and (last_candle['ema_vwma_osc_96'] > 0.0)
                    and (last_candle['cmf'] > 0.25)
                    and (last_candle['cmf_1h'] > 0.0)
            ):
                return 'exit_stoploss_d_48_1_2'

        if (last_candle['hl_pct_change_36_1h'] < -0.7):
            if (
                    (-0.04 > current_profit > -0.08)
                    and (max_loss < 0.08)
                    and (max_profit > (current_profit + 0.1))
                    and (last_candle['close'] > last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'] == False)
                    and (last_candle['sma_200_dec_20_1h'] == False)
                    and (last_candle['ema_vwma_osc_32'] > 0.0)
                    and (last_candle['ema_vwma_osc_64'] > 0.0)
                    and (last_candle['ema_vwma_osc_96'] > 0.0)
                    and (last_candle['cmf'] > 0.25)
                    and (last_candle['cmf_1h'] > 0.0)
            ):
                return 'exit_stoploss_d_36_1_1'

        if (last_candle['hl_pct_change_36_1h'] < -0.5):
            if (
                    (-0.05 > current_profit > -0.08)
                    and (max_loss < 0.08)
                    and (max_profit > (current_profit + 0.1))
                    and (last_candle['close'] > last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'] == False)
                    and (last_candle['sma_200_dec_20_1h'] == False)
                    and (last_candle['ema_vwma_osc_32'] > 0.0)
                    and (last_candle['ema_vwma_osc_64'] > 0.0)
                    and (last_candle['ema_vwma_osc_96'] > 0.0)
                    and (last_candle['cmf'] > 0.25)
                    and (last_candle['cmf_1h'] > 0.0)
                    and (last_candle['rsi'] > 60.0)  # Overbought RSI
            ):
                return 'exit_stoploss_d_36_2_1'

        if (last_candle['hl_pct_change_24_1h'] < -0.6):
            if (
                    (-0.04 > current_profit > -0.08)
                    and (max_loss < 0.08)
                    and (last_candle['close'] > last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'] == False)
                    and (last_candle['sma_200_dec_20_1h'] == False)
                    and (last_candle['ema_vwma_osc_32'] > 0.0)
                    and (last_candle['ema_vwma_osc_64'] > 0.0)
                    and (last_candle['ema_vwma_osc_96'] > 0.0)
                    and (last_candle['cmf'] > 0.25)
                    and (last_candle['cmf_1h'] > 0.0)
            ):
                return 'exit_stoploss_d_24_1_1'

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands (need both upper and lower for shorts)
        mid, upper, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['upper'] = upper
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (dataframe['upper'] - mid).abs()  # Distance from upper for shorts
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_high'] - dataframe['ha_close']).abs()  # Upper tail for shorts

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']
        dataframe['bb_upperband'] = dataframe['upper']

        # BB 20
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['ema_24'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['ema_200'] = ta.EMA(dataframe['close'], timeperiod=200)

        # SMA
        dataframe['sma_9'] = ta.SMA(dataframe['close'], timeperiod=9)
        dataframe['sma_200'] = ta.SMA(dataframe['close'], timeperiod=200)

        # HMA
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # volume
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        # ROCR
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # hh48 and ll48 (for shorts, we care about being at lows)
        dataframe['hh_48'] = ta.MAX(dataframe['high'], 48)
        dataframe['hh_48_diff'] = (dataframe['hh_48'] - dataframe['close']) / dataframe['hh_48'] * 100

        dataframe['ll_48'] = ta.MIN(dataframe['low'], 48)
        dataframe['ll_48_diff'] = (dataframe['close'] - dataframe['ll_48']) / dataframe['ll_48'] * 100

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # RSI
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # sma dec 20
        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)

        # EMA of VWMA Oscillator
        dataframe['ema_vwma_osc_32'] = ema_vwma_osc(dataframe, 32)
        dataframe['ema_vwma_osc_64'] = ema_vwma_osc(dataframe, 64)
        dataframe['ema_vwma_osc_96'] = ema_vwma_osc(dataframe, 96)

        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # 1h tf
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        informative['sma_200'] = ta.SMA(informative['close'], timeperiod=200)

        informative['hl_pct_change_48'] = range_percent_change(informative, 'HL', 48)
        informative['hl_pct_change_36'] = range_percent_change(informative, 'HL', 36)
        informative['hl_pct_change_24'] = range_percent_change(informative, 'HL', 24)
        informative['sma_200_dec_20'] = informative['sma_200'] < informative['sma_200'].shift(20)

        # CMF
        informative['cmf'] = chaikin_money_flow(informative, 20)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        # Fix pandas FutureWarning about downcasting
        dataframe = dataframe.infer_objects(copy=False)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic for shorts - inverted from long strategy.
        Enters when price is overbought (above upper Bollinger Band, high RSI, etc.)
        """
        dataframe.loc[
            ( dataframe['rocr_1h'].lt(self.rocr_1h.value) )  # Inverted: low ROCR = bearish 1h trend
            &
            (   (
                    (dataframe['upper'].shift().gt(0)) &
                    (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value)) &
                    (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value)) &
                    (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value)) &
                    (dataframe['ha_close'].gt(dataframe['upper'].shift())) &  # Above upper BB
                    (dataframe['ha_close'].ge(dataframe['ha_close'].shift()))  # Rising or flat
                )
                |
                (
                    (dataframe['ha_close'] > dataframe['ema_slow']) &  # Above EMA = overbought
                    (dataframe['ha_close'] > (2 - self.close_bbupper.value) * dataframe['bb_upperband'])  # Far above upper BB
                )
            )
            &
            (dataframe['ll_48_diff'] > self.short_ll_diff_48.value)  # Not at extreme lows
            &
            (dataframe['hh_48_diff'] > self.short_hh_diff_48.value)  # Distance from highs
        ,'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic for shorts - inverted from long strategy.
        Exits when price shows bullish signals (bearish Fisher, price below MA, etc.)
        """
        dataframe.loc[
            (   (
                    (dataframe['fisher'] < self.sell_fisher.value) &  # Bearish Fisher
                    (dataframe['ha_low'].ge(dataframe['ha_low'].shift(1))) &  # Lows rising
                    (dataframe['ha_low'].shift(1).ge(dataframe['ha_low'].shift(2))) &  # Lows rising trend
                    (dataframe['ha_close'].ge(dataframe['ha_close'].shift(1))) &  # Close rising
                    (dataframe['ema_fast'] < dataframe['ha_close']) &  # Fast EMA below close = bullish
                    ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) < dataframe['bb_middleband'])  # Below midband
                )
                |
                (
                    (dataframe['close'] < dataframe['sma_9']) &  # Below SMA = bearish
                    (dataframe['close'] < (dataframe['ema_24'] * self.low_offset_2.value)) &  # Below offset
                    (dataframe['rsi'] < 50) &  # RSI low
                    (dataframe['rsi_fast'] < dataframe['rsi_slow'])  # RSI declining
                )
                |
                (
                    (dataframe['sma_9'] < (dataframe['sma_9'].shift(1) - dataframe['sma_9'].shift(1) * 0.005 )) &  # SMA declining
                    (dataframe['close'] > dataframe['hma_50']) &  # Above HMA
                    (dataframe['close'] < (dataframe['ema_24'] * self.low_offset.value)) &  # Below offset
                    (dataframe['rsi_fast'] < dataframe['rsi_slow'])  # RSI declining
                )
            )
            &
            (dataframe['volume'] > 0)

        ,'exit_short'] = 1

        return dataframe

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma_result = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    vwma_result = vwma_result.fillna(0)
    return vwma_result

# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

# Chaikin Money Flow
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
