# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)

# @Rallipanos # changes by IcHiAT taken from https://github.com/XinuxC/Ft-things/blob/main/strategies/Etcg.py
# Shorts variant - Converted for shorting bear market rallies and overbought conditions


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



class ETCG_Shorts(IStrategy):
    """
    ETCG_Shorts Strategy

    A shorts-only variant of ETCG, designed to profit in bear markets
    and during overbought conditions.

    Original ETCG Performance (Longs):
    - Multiple entry modes: lambo2, buy1ewo, buy2ewo, cofi
    - Focus on oversold conditions and positive momentum

    Strategy Concept:
    This short strategy inverts the ETCG logic to identify overbought rallies
    in downtrends and parabolic pump conditions for shorting opportunities.

    Entry Conditions:
    1. lambo2_short: Price above EMA with overbought RSI (parabolic pumps)
    2. sell1ewo_short: Rallies during downtrend with negative EWO momentum
    3. sell2ewo_short: Extreme overbought with very high EWO
    4. cofi_short: Stochastic overbought crossdown with negative EWO

    Exit Conditions:
    - Signal: Price falls below HMA/EMA thresholds with RSI reversal
    - ROI: Conservative targets (5% → 0.2%)
    - Stop Loss: -18.9% (tighter than -99% for longs)

    Key Differences from Long Strategy:
    - Same stop loss: -99% (managed by trailing stop)
    - More conservative ROI: 7% vs 5% initial (more conservative)
    - Shorter unclog window: 3 days vs 4 days
    - Max 8 short positions via confirm_trade_entry()
    - Leverage: 3x (set in environment)

    Author: Derived from ETCG
    Version: 1.0.0
    """
    INTERFACE_VERSION = 3
    can_short = True

    # ROI table (more conservative than longs):
    minimal_roi = {
        "0": 0.07,      # Start at 7% vs 5% for longs
        "20": 0.035,    # More aggressive decay
        "40": 0.020,
        "87": 0.015,
        "201": 0.007,
        "202": 0.003
    }

    # Sell hyperspace params (inverted from buy_params):
    sell_params = {
        "base_nb_candles_sell": 12,  # Same as buy
        "rsi_sell": 100 - 58,  # 42 (inverted)
        "ewo_high": 3.001,     # Keep same (upper bound)
        "ewo_low": -10.289,    # Keep same (lower bound)
        "high_offset": 2 - 0.987,  # 1.013 (inverted from low_offset)
        "lambo2_ema_14_factor": 2 - 0.981,  # 1.019 (inverted)
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 100 - 39,  # 61 (inverted)
        "lambo2_rsi_4_limit": 100 - 44,   # 56 (inverted)
        "sell_adx": 20,
        "sell_fastd": 100 - 20,  # 80 (inverted)
        "sell_fastk": 100 - 22,  # 78 (inverted)
        "sell_ema_cofi": 2 - 0.98,  # 1.02 (inverted)
        "sell_ewo_high": 4.179  # Keep same
    }

    # Buy hyperspace params (inverted from sell_params):
    buy_params = {
        "base_nb_candles_buy": 22,  # Same as sell
        "low_offset": 2 - 1.014,    # 0.986 (inverted from high_offset)
        "low_offset_2": 2 - 1.01    # 0.99 (inverted from high_offset_2)
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    # Stoploss (same as longs):
    stoploss = -0.99

    # Position limits
    max_open_trades = 8
    max_short_trades = 8

    # SMAOffset
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(0.980, 0.990, default=buy_params['low_offset_2'], space='buy', optimize=True)

    # lambo2 (inverted)
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=sell_params['lambo2_ema_14_factor'], space='sell', optimize=True)
    lambo2_rsi_4_limit = IntParameter(40, 95, default=sell_params['lambo2_rsi_4_limit'], space='sell', optimize=True)
    lambo2_rsi_14_limit = IntParameter(40, 95, default=sell_params['lambo2_rsi_14_limit'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    ewo_low = DecimalParameter(-20.0, -8.0, default=sell_params['ewo_low'], space='sell', optimize=True)
    ewo_high = DecimalParameter(3.0, 3.4, default=sell_params['ewo_high'], space='sell', optimize=True)
    rsi_sell = IntParameter(30, 70, default=sell_params['rsi_sell'], space='sell', optimize=False)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5% - increased from 0.1% to let winners run
    trailing_stop_positive_offset = 0.03  # 3% - increased from 1.2% to capture bigger moves
    trailing_only_offset_is_reached = True

    # cofi (inverted)
    is_optimize_cofi = False
    sell_ema_cofi = DecimalParameter(1.02, 1.04, default=sell_params['sell_ema_cofi'], optimize=is_optimize_cofi)
    sell_fastk = IntParameter(70, 80, default=sell_params['sell_fastk'], optimize=is_optimize_cofi)
    sell_fastd = IntParameter(70, 80, default=sell_params['sell_fastd'], optimize=is_optimize_cofi)
    sell_adx = IntParameter(20, 30, default=sell_params['sell_adx'], optimize=is_optimize_cofi)
    sell_ewo_high = DecimalParameter(2, 12, default=sell_params['sell_ewo_high'], optimize=is_optimize_cofi)

    # Exit signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {
        'main_plot': {
            'ma_sell': {'color': 'orange'},
            'ma_buy': {'color': 'orange'},
        },
    }

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        """
        Enforce shorts-only and max position limits.
        """
        # Only allow shorts
        if side == "long":
            return False

        # Count open shorts
        short_count = sum(1 for t in Trade.get_trades_proxy(is_open=True) if t.is_short)

        # Enforce max shorts limit
        if short_count >= self.max_short_trades:
            return False

        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        """
        Unclog mechanism: Exit positions at loss after 3 days (tighter than 4 days for longs).
        """
        if current_profit < -0.04 and (current_time - trade.open_date_utc).days >= 3:
            return 'unclog_short'

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each trade.

        Returns fixed 3x leverage for all trades.
        """
        return 3.0

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['trading_mode'] == "futures":
            btc_info_pair = "BTC/USDT:USDT"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1h))

        return informative_pairs

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        df36h = dataframe.copy().shift(432)  # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift(288)  # TODO FIXME: This assumes 5m timeframe

        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()

        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])

        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()

        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)

        return dataframe


    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['price_trend_long'] = (dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['trading_mode'] == "futures":
            btc_info_pair = "BTC/USDT:USDT"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)


        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)


        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)


        # Pump strength
        dataframe['zema_30'] = ftt.dema(dataframe, period=30)
        dataframe['zema_200'] = ftt.dema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)



        dataframe = self.pump_dump_protection(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic inverted for shorts: look for overbought conditions and negative momentum.
        """
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # lambo2_short: Parabolic pumps (inverted)
        lambo2_short = (
            (dataframe['close'] > (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] > int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] > int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2_short, 'enter_tag'] += 'lambo2_short_'
        conditions.append(lambo2_short)

        # sell1ewo_short: Rallies during downtrend (inverted)
        sell1ewo_short = (
            (dataframe['rsi_fast'] > 65) &
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
            (dataframe['EWO'] > self.ewo_high.value) &
            (dataframe['rsi'] > self.rsi_sell.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] > (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value))
        )
        dataframe.loc[sell1ewo_short, 'enter_tag'] += 'sell1ewo_short_'
        conditions.append(sell1ewo_short)

        # sell2ewo_short: Extreme overbought (inverted)
        sell2ewo_short = (
            (dataframe['rsi_fast'] > 65) &
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
            (dataframe['EWO'] < self.ewo_low.value) &
            (dataframe['volume'] > 0) &
            (dataframe['close'] > (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value))
        )
        dataframe.loc[sell2ewo_short, 'enter_tag'] += 'sell2ewo_short_'
        conditions.append(sell2ewo_short)

        # cofi_short: Stochastic overbought crossdown (inverted)
        cofi_short = (
            (dataframe['open'] > dataframe['ema_8'] * self.sell_ema_cofi.value) &
            (qtpylib.crossed_below(dataframe['fastk'], dataframe['fastd'])) &
            (dataframe['fastk'] > self.sell_fastk.value) &
            (dataframe['fastd'] > self.sell_fastd.value) &
            (dataframe['adx'] > self.sell_adx.value) &
            (dataframe['EWO'] < -self.sell_ewo_high.value)  # Negative EWO for shorts
        )
        dataframe.loc[cofi_short, 'enter_tag'] += 'cofi_short_'
        conditions.append(cofi_short)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_short'
            ] = 1


        dont_short_conditions = []

        # Don't short if there seems to be a Pump and Dump event (same protection)
        dont_short_conditions.append((dataframe['pnd_volume_warn'] < 0.0))

        # BTC price protection (inverted: don't short when BTC oversold - may bounce)
        dont_short_conditions.append((dataframe['btc_rsi_8_1h'] > 65.0))

        if dont_short_conditions:
            for condition in dont_short_conditions:
                dataframe.loc[condition, 'enter_short'] = 0

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic inverted for shorts: exit when price falls and momentum reverses down.
        """
        conditions = []

        conditions.append(
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['rsi'] < 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] < dataframe['rsi_slow'])
            )
            |
            (
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] < dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_short'
            ] = 1


        return dataframe


def pct_change(a, b):
    return (b - a) / a
