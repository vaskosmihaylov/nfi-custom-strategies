# --- Do not remove these libs ---
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date

import logging
from datetime import timedelta, datetime
import datetime

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (IStrategy, BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, merge_informative_pair, stoploss_from_open,
                                stoploss_from_absolute, merge_informative_pair)
from freqtrade.persistence import Trade
from typing import List, Tuple, Optional
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque

#import ta as taold

import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class PlotConfig():

    def __init__(self):
        self.config = {
            'main_plot': {
                resample('bollinger_upperband') : {'color': 'rgba(4,137,122,0.7)'},
                resample('kc_upperband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_middleband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_lowerband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('bollinger_lowerband') : {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': resample('bollinger_upperband'),
                    'fill_color': 'rgba(4,137,122,0.07)'
                },
                resample('ema9') : {'color': 'purple'},
                resample('ema20') : {'color': 'yellow'},
                resample('ema50') : {'color': 'red'},
                resample('ema200') : {'color': 'white'},
            },
            'subplots': {
                "ATR" : {
                    resample('atr'):{'color':'firebrick'}
                }
            }
        }

    def add_pivots_in_config(self):
        self.config['main_plot']["pivot_lows"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'olive'
                }
            }
        }
        self.config['main_plot']["pivot_highs"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'violet'
                }
            }
        }
        self.config['main_plot']["pivot_highs"] = {
            "plotly": {
                'mode': 'markers',
                'marker': {
                    'symbol': 'diamond-open',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'violet'
                }
            }
        }
        return self

    def add_divergence_in_config(self, indicator:str):
        # self.config['main_plot']["bullish_divergence_" + indicator + "_occurence"] = {
        #     "plotly": {
        #         'mode': 'markers',
        #         'marker': {
        #             'symbol': 'diamond',
        #             'size': 11,
        #             'line': {
        #                 'width': 2
        #             },
        #             'color': 'orange'
        #         }
        #     }
        # }
        # self.config['main_plot']["bearish_divergence_" + indicator + "_occurence"] = {
        #     "plotly": {
        #         'mode': 'markers',
        #         'marker': {
        #             'symbol': 'diamond',
        #             'size': 11,
        #             'line': {
        #                 'width': 2
        #             },
        #             'color': 'purple'
        #         }
        #     }
        # }
        for i in range(3):
            self.config['main_plot']["bullish_divergence_" + indicator + "_line_" + str(i)] = {
                "plotly": {
                    'mode': 'lines',
                    'line' : {
                        'color': 'green',
                        'dash' :'dash'
                    }
                }
            }
            self.config['main_plot']["bearish_divergence_" + indicator + "_line_" + str(i)] = {
                "plotly": {
                    'mode': 'lines',
                    'line' : {
                        "color":'crimson',
                        'dash' :'dash'
                    }
                }
            }
        return self

    def add_total_divergences_in_config(self, dataframe):
        total_bullish_divergences_count = dataframe[resample("total_bullish_divergences_count")]
        total_bullish_divergences_names = dataframe[resample("total_bullish_divergences_names")]
        self.config['main_plot'][resample("total_bullish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': total_bullish_divergences_count,
                'hovertext': total_bullish_divergences_names,
                'textfont':{'size': 11, 'color':'green'},
                'textposition':'bottom center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'green'
                }
            }
        }
        total_bearish_divergences_count = dataframe[resample("total_bearish_divergences_count")]
        total_bearish_divergences_names = dataframe[resample("total_bearish_divergences_names")]
        self.config['main_plot'][resample("total_bearish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': total_bearish_divergences_count,
                'hovertext': total_bearish_divergences_names,
                'textfont':{'size': 11, 'color':'crimson'},
                'textposition':'top center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'crimson'
                }
            }
        }
        return self

class AlexBandSniper(IStrategy):

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v1.2.3"



    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.20, -0.01, decimals=2, name='stoploss')]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(6, 20, name='max_open_trades'),
            ]
        def trailing_space() -> List[Dimension]:
            # All parameters here are mandatory, you can only modify their type or the range.
            return [
                # Fixed to true, if optimizing trailing_stop we assume to use trailing stop at all times.
                Categorical([True], name='trailing_stop'),

                SKDecimal(0.02, 0.5, decimals=2, name='trailing_stop_positive'),
                # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
                # so this intermediate parameter is used as the value of the difference between
                # them. The value of the 'trailing_stop_positive_offset' is constructed in the
                # generate_trailing_params() method.
                # This is similar to the hyperspace dimensions used for constructing the ROI tables.
                SKDecimal(0.03, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),

                Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]
    
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    
    minimal_roi = {
        "0": 0.07,
        "5": 0.06,
        "10": 0.03,
        "30": 0.01
    }
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    
    stoploss = -0.25
    can_short = True
    use_custom_stoploss = False
    #max_open_trades = 10
    leverage_value = 7.0

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.45
    trailing_stop_positive_offset = 0.53
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '15m'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "exit_pricing" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 0
    
    #count_bullish_divergences = IntParameter(0, 11, default=0, space='buy', optimize=True, load=True)
    #count_bearish_divergences = IntParameter(0, 11, default=0, space='buy', optimize=True, load=True)
    window = IntParameter(5, 29, default=5, space="buy", optimize=True, load=True)
    index_range = IntParameter(30, 100, default=30, space='buy', optimize=True, load=True)

    #trailing_stop_buf = BooleanParameter(default=False, space='buy', optimize=True, load=True)
    #trailing_stop = trailing_stop_buf.value
    #use_chop = BooleanParameter(default=False, space='buy', optimize=True, load=True)
    #use_chop_up_smooth = IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)    # if 0 not used
    #use_chop_down_smooth = IntParameter(1, 100, default=100, space='buy', optimize=True, load=True) # if 100 not used
    #use_natr_direction_change = BooleanParameter(default=False, space='sell', optimize=True, load=True)
    #use_natr_combined_change = BooleanParameter(default=False, space='buy', optimize=True, load=True)

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Get the informative pair
        # informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')
        # informative = resample_to_interval(dataframe, self.get_ticker_indicator() * 15)
        informative = dataframe
        # Momentum Indicators
        # ------------------------------------

        # RSI
        informative['rsi'] = ta.RSI(informative)
        # Stochastic Slow
        informative['stoch'] = ta.STOCH(informative)['slowk']
        # ROC
        informative['roc'] = ta.ROC(informative)
        # Ultimate Oscillator
        informative['uo'] = ta.ULTOSC(informative)
        # Awesome Oscillator
        informative['ao'] = qtpylib.awesome_oscillator(informative)
        # MACD
        informative['macd'] = ta.MACD(informative)['macd']
        # Commodity Channel Index
        informative['cci'] = ta.CCI(informative)
        # CMF
        informative['cmf'] = chaikin_money_flow(informative, 20)
        # OBV
        informative['obv'] = ta.OBV(informative)
        # MFI
        informative['mfi'] = ta.MFI(informative)
        # ADX
        informative['adx'] = ta.ADX(informative)

        # ATR
        informative['atr'] = qtpylib.atr(informative, window=14, exp=False)

        # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        keltner = emaKeltner(informative)
        informative["kc_upperband"] = keltner["upper"]
        informative["kc_middleband"] = keltner["mid"]
        informative["kc_lowerband"] = keltner["lower"]

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bollinger_upperband'] = bollinger['upper']
        informative['bollinger_lowerband'] = bollinger['lower']

        # EMA - Exponential Moving Average
        informative['ema9'] = ta.EMA(informative, timeperiod=9)
        informative['ema20'] = ta.EMA(informative, timeperiod=20)
        informative['ema50'] = ta.EMA(informative, timeperiod=50)
        informative['ema200'] = ta.EMA(informative, timeperiod=200)

        pivots = pivot_points(informative, self.window.value)
        informative['pivot_lows'] = pivots['pivot_lows']
        informative['pivot_highs'] = pivots['pivot_highs']

        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below


        self.initialize_divergences_lists(informative)
        (high_iterator, low_iterator) = self.get_iterators(informative)
        self.add_divergences(informative, 'rsi',high_iterator, low_iterator)
        self.add_divergences(informative, 'stoch',high_iterator, low_iterator)
        self.add_divergences(informative, 'roc',high_iterator, low_iterator)
        self.add_divergences(informative, 'uo',high_iterator, low_iterator)
        self.add_divergences(informative, 'ao',high_iterator, low_iterator)
        self.add_divergences(informative, 'macd',high_iterator, low_iterator)
        self.add_divergences(informative, 'cci',high_iterator, low_iterator)
        self.add_divergences(informative, 'cmf',high_iterator, low_iterator)
        self.add_divergences(informative, 'obv',high_iterator, low_iterator)
        self.add_divergences(informative, 'mfi',high_iterator, low_iterator)
        self.add_divergences(informative, 'adx',high_iterator, low_iterator)

        # print("-------------------informative-------------------")
        # print(informative)
        # print("-------------------dataframe-------------------")
        # print(dataframe)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '15m', ffill=True)

        # dataframe = resampled_merge(dataframe, informative)
        # print(dataframe[resample("total_bullish_divergences_count")])
        # for index, value in enumerate(dataframe[resample("total_bullish_divergences_count")]):
        #     if value < 0.5:
        #         dataframe[resample("total_bullish_divergences_count")][index] = None
        #         dataframe[resample("total_bullish_divergences")][index] = None
        #         dataframe[resample("total_bullish_divergences_names")][index] = None
        #     else:
        #         print(value)
        #         print(dataframe[resample("total_bullish_divergences")][index])
        #         print(dataframe[resample("total_bullish_divergences_names")][index])
        self.plot_config = (
            PlotConfig()
            # .add_pivots_in_config()
            # .add_divergence_in_config('rsi')
            # .add_divergence_in_config('stoch')
            # .add_divergence_in_config('roc')
            # .add_divergence_in_config('uo')
            # .add_divergence_in_config('ao')
            # .add_divergence_in_config('macd')
            # .add_divergence_in_config('cci')
            # .add_divergence_in_config('cmf')
            # .add_divergence_in_config('obv')
            # .add_divergence_in_config('mfi')
            # .add_divergence_in_config('adx')
            .add_total_divergences_in_config(dataframe)
            .config)

        dataframe['chop'] = choppiness_index(dataframe['high'], dataframe['low'], dataframe['close'], window=14) 
        
        dataframe['natr'] = ta.NATR(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
    
        dataframe['natr_diff'] = dataframe['natr'] - dataframe['natr'].shift(1)  # Change in ATR
        
        dataframe['natr_direction_change'] = (dataframe['natr_diff'] * dataframe['natr_diff'].shift(1) < 0)
        # Define the rolling mean of NATR
        natr_mean = dataframe['natr'].rolling(window=12).mean()

        # Check for an upward change in NATR (NATR crosses above its moving average)
        dataframe['natr_upward_change'] = dataframe['natr'] > natr_mean

        # Check for a downward change in NATR (NATR crosses below its moving average)
        dataframe['natr_downward_change'] = dataframe['natr'] < natr_mean

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Lookahead-sichere Entry-Bedingungen mit .shift(1)
        long_condition = (
            (dataframe[resample('total_bullish_divergences')].shift(1) > 0) &
            two_bands_check_long(dataframe) &
            (dataframe['volume'] > 0)
        )

        short_condition = (
            (dataframe[resample('total_bearish_divergences')].shift(1) > 0) &
            two_bands_check_short(dataframe) &
            (dataframe['volume'] > 0)
        )

        # Setze Entry-Signale
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1

        # Tags getrennt, um Konflikte zu vermeiden
        dataframe['enter_tag_long'] = ''
        dataframe['enter_tag_short'] = ''
        dataframe.loc[long_condition, 'enter_tag_long'] = 'Bull_D_Long'
        dataframe.loc[short_condition, 'enter_tag_short'] = 'Bear_D_Short'

        # Kombinierte Tag-Spalte für Visualisierung
        dataframe['enter_tag'] = dataframe['enter_tag_long'] + dataframe['enter_tag_short']

        # Debug-Logging für letzte Candle
        candle = dataframe.iloc[-1]
        logger.info(
            f"[{metadata['pair']}] {candle['date']} | Close: {candle['close']} | "
            f"Long: {candle.get('enter_long', 0)} ({candle.get('enter_tag_long', '')}) | "
            f"Short: {candle.get('enter_short', 0)} ({candle.get('enter_tag_short', '')})"
        )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit signals
        """

        # Lookahead-sichere Bedingungen: nutze .shift(1) statt .shift(-1)
        short_exit_condition = (
            (dataframe[resample('total_bullish_divergences')].shift(1) > 0)
            #& (dataframe['total_bullish_divergences_count'].iloc[-1] > self.count_bullish_divergences.value)
            # & (keltner_middleband_check(dataframe) & (ema_check(dataframe)) & (green_candle(dataframe)))
            # | (keltner_lowerband_check(dataframe) & (ema_check(dataframe)))
            # | (bollinger_lowerband_check(dataframe) & (ema_check(dataframe)))
            & two_bands_check_long(dataframe)
            & (dataframe['volume'] > 0)
            #& (self.use_natr_direction_change.value & dataframe['natr_direction_change'])
        )

        long_exit_condition = (
            (dataframe[resample('total_bearish_divergences')].shift(1) > 0)
            #& (dataframe['total_bearish_divergences_count'].iloc[-1] > self.count_bearish_divergences.value)
            # & (keltner_middleband_check(dataframe) & (ema_check(dataframe)) & (green_candle(dataframe)))
            # | (keltner_lowerband_check(dataframe) & (ema_check(dataframe)))
            # | (bollinger_lowerband_check(dataframe) & (ema_check(dataframe)))
            & two_bands_check_short(dataframe)
            & (dataframe['volume'] > 0)
            #& (self.use_natr_direction_change.value & dataframe['natr_direction_change'])
        )

        # Setze Exit-Signale
        dataframe.loc[short_exit_condition, 'exit_short'] = 1
        dataframe.loc[long_exit_condition, 'exit_long'] = 1

        # Setze Tags getrennt
        dataframe['exit_tag_short'] = ''
        dataframe['exit_tag_long'] = ''

        dataframe.loc[short_exit_condition, 'exit_tag_short'] = 'Exit_Short_Div'
        dataframe.loc[long_exit_condition, 'exit_tag_long'] = 'Exit_Long_Div'

        # Kombinierte Tag-Spalte für Plot
        dataframe['exit_tag'] = dataframe['exit_tag_long'] + dataframe['exit_tag_short']

        # Debug-Logging für letzte Candle
        candle = dataframe.iloc[-1]
        logger.info(
            f"[{metadata['pair']}] {candle['date']} | Close: {candle['close']} | "
            f"Exit Long: {candle.get('exit_long', 0)} ({candle.get('exit_tag_long', '')}) | "
            f"Exit Short: {candle.get('exit_short', 0)} ({candle.get('exit_tag_short', '')})"
        )

        return dataframe


    '''
    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        takeprofit_long = 999999
        takeprofit_short = -999999
        self.trailing_stop = False

        entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))
        signal_candle = dataframe.loc[dataframe['date'] == signal_time]
        if not signal_candle.empty:
            signal_candle = signal_candle.iloc[-1].squeeze()
            if not trade.is_short:
                takeprofit_long = signal_candle[resample('high')] + signal_candle[resample('atr')]
            else:
                takeprofit_short = signal_candle[resample('low')] - signal_candle[resample('atr')]

        if not trade.is_short and takeprofit_long < current_rate:
            self.trailing_stop = True
        elif trade.is_short and takeprofit_short > current_rate:
            self.trailing_stop = True
        
        return False
    '''

    # default leverage
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, length: int = 14, multiplier: float = 1.5, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        takeprofit_long = 999999
        takeprofit_short = -999999
        entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))
        signal_candle = dataframe.loc[dataframe['date'] == signal_time]

        if not signal_candle.empty:
            signal_candle = signal_candle.iloc[-1].squeeze()
            if not trade.is_short:
                takeprofit_long = signal_candle[resample('high')] + signal_candle[resample('atr')]
            else:
                takeprofit_short = signal_candle[resample('low')] - signal_candle[resample('atr')]

            if (current_profit > abs(self.trailing_stop_positive_offset)) and ((not trade.is_short and takeprofit_long < current_rate) or (trade.is_short and takeprofit_short > current_rate)):
                return stoploss_from_open(abs(self.trailing_stop_positive), current_profit, is_short=trade.is_short, leverage=trade.leverage)
 
        #entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        #signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))
        #signal_candle = dataframe.loc[dataframe['date'] == signal_time]
        #if not signal_candle.empty:
            #signal_candle = signal_candle.iloc[-1].squeeze()
            if trade.is_short:
                stoploss = signal_candle[resample('high')] + signal_candle[resample('atr')]
            else:
                stoploss = signal_candle[resample('low')] - signal_candle[resample('atr')]

            if (stoploss < current_rate and not trade.is_short) or (stoploss > current_rate and trade.is_short):
                return stoploss_from_absolute(stoploss, current_rate, is_short=trade.is_short, leverage=trade.leverage)

        # return maximum stoploss value, keeping current stoploss price unchanged
        # https://www.freqtrade.io/en/stable/strategy-callbacks/#trailing-stoploss-with-positive-offset
        return None #stoploss_from_open(abs(self.stoploss), current_profit, is_short=trade.is_short, leverage=trade.leverage)

    def initialize_divergences_lists(self, dataframe: DataFrame):
        # Bullish Divergenzen
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ''

        # Bearish Divergenzen
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ''

    def get_iterators(self, dataframe):
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
            else:
                high_iterator.append(index)
        
        # TODO move indexes back for not lookahead!
        # for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        #     if np.isnan(row.pivot_lows):
        #         low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
        #     else:
        #         if index-self.window.value-1 > 0 and dataframe['pivot_lows'][index-self.window.value-1] != dataframe['pivot_lows'][index]:
        #             low_iterator.append(index)
        #         else:
        #             low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])

        #     if np.isnan(row.pivot_highs):
        #         high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
        #     else:
        #         if index-self.window.value-1 > 0 and dataframe['pivot_highs'][index-self.window.value-1] != dataframe['pivot_highs'][index]:
        #             high_iterator.append(index)
        #         else:
        #             high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
        # # print(high_iterator)

        return high_iterator, low_iterator

    def add_divergences(self, dataframe: DataFrame, indicator: str, high_iterator, low_iterator):
        (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = self.divergence_finder_dataframe(dataframe, indicator, high_iterator, low_iterator)
        dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
        # for index, bearish_line in enumerate(bearish_lines):
        #     dataframe['bearish_divergence_' + indicator + '_line_'+ str(index)] = bearish_line
        dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences
        # for index, bullish_line in enumerate(bullish_lines):
        #     dataframe['bullish_divergence_' + indicator + '_line_'+ str(index)] = bullish_line

    def divergence_finder_dataframe(self, dataframe: DataFrame, indicator_source: str, high_iterator, low_iterator) -> Tuple[pd.Series, pd.Series]:
        bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
        bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bullish_divergences = np.empty(len(dataframe['close'])) * np.nan

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

            bearish_occurence = self.bearish_divergence_finder(dataframe,
                                                        dataframe[indicator_source],
                                                        high_iterator,
                                                        index)

            if bearish_occurence != None:
                (prev_pivot , current_pivot) = bearish_occurence
                bearish_prev_pivot = dataframe['close'][prev_pivot]
                bearish_current_pivot = dataframe['close'][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                length = current_pivot - prev_pivot
                bearish_lines_index = 0
                can_exist = True
                while(True):
                    can_draw = True
                    if bearish_lines_index <= len(bearish_lines):
                        bearish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                    actual_bearish_lines = bearish_lines[bearish_lines_index]
                    for i in range(length + 1):
                        point = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                        indicator_point =  bearish_ind_prev_pivot + (bearish_ind_current_pivot - bearish_ind_prev_pivot) * i / length
                        if i != 0 and i != length:
                            if (point <= dataframe['close'][prev_pivot + i]
                                    or indicator_point <= dataframe[indicator_source][prev_pivot + i]):
                                can_exist = False
                        if not np.isnan(actual_bearish_lines[prev_pivot + i]):
                            can_draw = False
                    if not can_exist:
                        break
                    if can_draw:
                        for i in range(length + 1):
                            actual_bearish_lines[prev_pivot + i] = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                        break
                    bearish_lines_index = bearish_lines_index + 1
                if can_exist:
                    bearish_divergences[index] = row.close
                    dataframe.loc[index,"total_bearish_divergences"] = row.close
                    # dataframe["total_bearish_divergences"][index] = row.close
                    if index > self.index_range.value:
                        dataframe.loc[index-self.index_range.value,"total_bearish_divergences_count"] += 1
                        dataframe.loc[index-self.index_range.value,"total_bearish_divergences_names"] += indicator_source.upper() + '<br>'
                        # dataframe[index-30"total_bearish_divergences_count"][index-30] = dataframe["total_bearish_divergences_count"][index-30] + 1
                        # dataframe["total_bearish_divergences_names"][index-30] = dataframe["total_bearish_divergences_names"][index-30] + indicator_source.upper() + '<br>'

            bullish_occurence = self.bullish_divergence_finder(dataframe,
                                                        dataframe[indicator_source],
                                                        low_iterator,
                                                        index)

            if bullish_occurence != None:
                (prev_pivot , current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe['close'][prev_pivot]
                bullish_current_pivot = dataframe['close'][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                length = current_pivot - prev_pivot
                bullish_lines_index = 0
                can_exist = True
                while(True):
                    can_draw = True
                    if bullish_lines_index <= len(bullish_lines):
                        bullish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                    actual_bullish_lines = bullish_lines[bullish_lines_index]
                    for i in range(length + 1):
                        point = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                        indicator_point =  bullish_ind_prev_pivot + (bullish_ind_current_pivot - bullish_ind_prev_pivot) * i / length
                        if i != 0 and i != length:
                            if (point >= dataframe['close'][prev_pivot + i]
                                    or indicator_point >= dataframe[indicator_source][prev_pivot + i]):
                                can_exist = False
                        if not np.isnan(actual_bullish_lines[prev_pivot + i]):
                            can_draw = False
                    if not can_exist:
                        break
                    if can_draw:
                        for i in range(length + 1):
                            actual_bullish_lines[prev_pivot + i] = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                        break
                    bullish_lines_index = bullish_lines_index + 1
                if can_exist:
                    bullish_divergences[index] = row.close
                    dataframe.loc[index,"total_bullish_divergences"] = row.close
                    # dataframe["total_bullish_divergences"][index] = row.close
                    if index > self.index_range.value:
                        dataframe.loc[index-self.index_range.value, "total_bullish_divergences_count"] += 1
                        dataframe.loc[index-self.index_range.value, "total_bullish_divergences_names"] = dataframe.loc[index-30, "total_bullish_divergences_names"] + indicator_source.upper() + '<br>'
                        # dataframe["total_bullish_divergences_count"][index-30] = dataframe["total_bullish_divergences_count"][index-30] + 1
                        # dataframe["total_bullish_divergences_names"][index-30] = dataframe["total_bullish_divergences_names"][index-30] + indicator_source.upper() + '<br>'

        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index):
        #try:
        if high_iterator[index] == index:
            current_pivot = high_iterator[index]
            occurences = list(dict.fromkeys(high_iterator))
            current_index = occurences.index(high_iterator[index])
            for i in range(current_index-1,current_index - self.window.value - 1,-1):
                if i < 0 or i >= len(occurences):
                    continue
                prev_pivot = occurences[i]
                if np.isnan(prev_pivot):
                    return
                if ((dataframe['pivot_highs'][current_pivot] < dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
                        or (dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                    return (prev_pivot , current_pivot)
        #except:
        #    pass
        return None

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index):
        #try:
        if low_iterator[index] == index:
            current_pivot = low_iterator[index]
            occurences = list(dict.fromkeys(low_iterator))
            current_index = occurences.index(low_iterator[index])
            for i in range(current_index-1,current_index - self.window.value - 1,-1):
                if i < 0 or i >= len(occurences):
                    continue
                prev_pivot = occurences[i]
                if np.isnan(prev_pivot):
                    return
                if ((dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
                        or (dataframe['pivot_lows'][current_pivot] > dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                    return (prev_pivot, current_pivot)
        #except:
        #    pass
        return None

def choppiness_index(high, low, close, window=14):
    #atr = ta.NATR(high, low, close, window=1)
    #atr = taold.volatility.average_true_range(high, low, close, window=1)

    # Calculate ATR and convert to pandas Series to enable rolling operations
    natr = pd.Series(ta.NATR(high, low, close, window=window))  # Use the pandas version here

    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()
    
    choppiness = 100 * np.log10((natr.rolling(window=window).sum()) / (high_max - low_min)) / np.log10(window)
    return choppiness

def resample(indicator):
    # return "resample_15_" + indicator
    return indicator

def two_bands_check_long(dataframe):
    check = (
        # ((dataframe['low'] < dataframe['bollinger_lowerband']) & (dataframe['high'] > dataframe['kc_lowerband'])) |
        ((dataframe[resample('low')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('kc_upperband')])) # 1
        #  ((dataframe['low'] < dataframe['kc_lowerband']) & (dataframe['high'] > dataframe['kc_middleband'])) # 2
        # | ((dataframe['low'] < dataframe['kc_middleband']) & (dataframe['high'] > dataframe['kc_upperband'])) # 2
    )
    return ~check

def two_bands_check_short(dataframe):
    check = (
        # ((dataframe['low'] < dataframe['bollinger_lowerband']) & (dataframe['high'] > dataframe['kc_lowerband'])) |
        ((dataframe[resample('high')] > dataframe[resample('kc_upperband')]) & (dataframe[resample('low')] < dataframe[resample('kc_lowerband')])) # 1
        #  ((dataframe['low'] < dataframe['kc_lowerband']) & (dataframe['high'] > dataframe['kc_middleband'])) # 2
        # | ((dataframe['low'] < dataframe['kc_middleband']) & (dataframe['high'] > dataframe['kc_upperband'])) # 2
    )
    return ~check
        
    # def ema_cross_check_long(dataframe):
    #     dataframe['ema20_50_cross'] = qtpylib.crossed_below(dataframe[resample('ema20')],dataframe[resample('ema50')])
    #     dataframe['ema20_200_cross'] = qtpylib.crossed_below(dataframe[resample('ema20')],dataframe[resample('ema200')])
    #     dataframe['ema50_200_cross'] = qtpylib.crossed_below(dataframe[resample('ema50')],dataframe[resample('ema200')])
    #     return ~(
    #             dataframe['ema20_50_cross']
    #             | dataframe['ema20_200_cross']
    #             | dataframe['ema50_200_cross']
    #     )

    # def ema_cross_check_short(dataframe):
    #     dataframe['ema20_50_cross'] = qtpylib.crossed_above(dataframe[resample('ema20')],dataframe[resample('ema50')])
    #     dataframe['ema20_200_cross'] = qtpylib.crossed_above(dataframe[resample('ema20')],dataframe[resample('ema200')])
    #     dataframe['ema50_200_cross'] = qtpylib.crossed_above(dataframe[resample('ema50')],dataframe[resample('ema200')])
    #     return ~(
    #             dataframe['ema20_50_cross']
    #             | dataframe['ema20_200_cross']
    #             | dataframe['ema50_200_cross']
    #     )

    # def green_candle(dataframe):
    #     return dataframe[resample('open')] < dataframe[resample('close')]

    # def keltner_middleband_check(dataframe):
    #     return (dataframe[resample('low')] < dataframe[resample('kc_middleband')]) & (dataframe[resample('high')] > dataframe[resample('kc_middleband')])

    # def keltner_lowerband_check(dataframe):
    #     return (dataframe[resample('low')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('kc_lowerband')])

    # def bollinger_lowerband_check(dataframe):
    #     return (dataframe[resample('low')] < dataframe[resample('bollinger_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('bollinger_lowerband')])

    # def bollinger_keltner_check(dataframe):
    #     return (dataframe[resample('bollinger_lowerband')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('bollinger_upperband')] > dataframe[resample('kc_upperband')])

    # def ema_check(dataframe):
    #     check = (
    #             (dataframe[resample('ema9')] < dataframe[resample('ema20')])
    #             & (dataframe[resample('ema20')] < dataframe[resample('ema50')])
    #             & (dataframe[resample('ema50')] < dataframe[resample('ema200')]))
    #     return ~check


from enum import Enum
class PivotSource(Enum):
    HighLow = 0
    Close = 1

def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source: PivotSource = PivotSource.Close) -> DataFrame:
    high_source = None
    low_source = None

    if pivot_source == PivotSource.Close:
        high_source = 'close'
        low_source = 'close'
    elif pivot_source == PivotSource.HighLow:
        high_source = 'high'
        low_source = 'low'

    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()

    # find pivot points
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index - window] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index - window] = getattr(current_value, low_source)
            last_values.popleft()

    # find last one
    if len(last_values) >= window + 2:
        current_value = last_values[-2]
        is_greater = True
        is_less = True
        for window_index in range(0, window):
            left = last_values[-2 - window_index - 1]
            right = last_values[-1]
            local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
            is_greater &= local_is_greater
            is_less &= local_is_less
        if is_greater:
            pivot_points_highs[index - 1] = getattr(current_value, high_source)
        if is_less:
            pivot_points_lows[index - 1] = getattr(current_value, low_source)

    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    is_greater = True
    is_less = True
    if (getattr(current_value, high_source) < getattr(left, high_source) or
            getattr(current_value, high_source) < getattr(right, high_source)):
        is_greater = False

    if (getattr(current_value, low_source) > getattr(left, low_source) or
            getattr(current_value, low_source) > getattr(right, low_source)):
        is_less = False
    return (is_greater, is_less)

def emaKeltner(dataframe):
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner

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
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')