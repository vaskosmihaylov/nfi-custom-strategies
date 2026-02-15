# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, DecimalParameter, stoploss_from_open, IntParameter
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
from freqtrade.persistence import Trade
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta, timezone
from freqtrade.vendor.qtpylib.indicators import heikinashi, tdi, awesome_oscillator, sma
import math
import logging
from technical.indicators import ichimoku
logger = logging.getLogger(__name__)

def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

class AwesomeEWOLambo_Shorts(IStrategy):
    """
    AwesomeEWOLambo_Shorts Strategy

    A shorts-only variant of AwesomeEWOLambo, designed to profit during overbought
    conditions and bear markets using Elliott Wave Oscillator momentum detection.

    Original AwesomeEWOLambo (Longs):
    - Author: xNighbloodx Natblida
    - Entry: 4 modes (EWO high/low, lambo2, RSI divergence)
    - Risk: -70% stoploss, 2% ROI target
    - DCA: 8 safety orders with 1.4x scaling

    Strategy Concept (Shorts):
    Enter shorts when market is overbought (high RSI, price above EMAs, negative EWO momentum).
    Uses 4 entry modes mirroring the long strategy but inverted for bearish conditions.

    Entry Conditions:
    1. short_ewo_high_rsi: High RSI (>65) + Price above MA + Negative EWO
    2. short_ewo2_high_rsi: Very high RSI (>75) + Price above MA + Very negative EWO
    3. short_lambo2: Price above EMA_14 + High RSI_4/RSI_14
    4. short_ewo_low_rsi: High RSI + Price above MA + Positive EWO (overheated)

    Exit Conditions:
    - Signal: Red candle + negative difference signal
    - ROI: 7% → 1.5% over 60 minutes
    - Stop Loss: -18.9%
    - Trailing Stop: Activates at +1.2% profit
    - Unclog: -4% after 3 days

    Key Differences from Long Strategy:
    - Tighter stop loss: -18.9% vs -70%
    - More aggressive ROI: 7% vs 2%
    - Shorter unclog: 3 days vs 10 days
    - Max 8 short positions (same as longs)
    - All RSI thresholds inverted (100 - value)
    - All EMA offsets inverted (2 - value)

    Author: Derived from AwesomeEWOLambo by xNighbloodx
    Version: 1.0.0
    """

    INTERFACE_VERSION: int = 3
    can_short = True

    # More conservative ROI for shorts
    minimal_roi = {"0": 0.07, "20": 0.05, "40": 0.03, "60": 0.015}

    # Tighter stoploss for shorts
    stoploss = -0.12

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Short-specific
    max_short_trades = 8

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    # Sell hyperspace params (INVERTED from buy_params)
    sell_params = {
        "base_nb_candles_sell": 12,
        "ewo_high": -1.001,  # Inverted sign
        "ewo_high_2": 3.585,  # Inverted sign
        "low_offset": 1.013,  # 2 - 0.987 = 1.013
        "low_offset_2": 1.058,  # 2 - 0.942 = 1.058
        "ewo_low": 2.289,  # Inverted sign
        "rsi_sell": 42,  # 100 - 58 = 42
        "lambo2_ema_14_factor": 1.03,
        "lambo2_rsi_14_limit": 66,
        "lambo2_rsi_4_limit": 70,
    }

    # Exit hyperspace params (INVERTED from sell_params of longs)
    cover_params = {
        "base_nb_candles_cover": 22,
        "high_offset": 0.986,  # 2 - 1.014 = 0.986
        "high_offset_2": 0.99,  # 2 - 1.01 = 0.99
    }

    # SMAOffset
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    base_nb_candles_cover = IntParameter(8, 20, default=cover_params['base_nb_candles_cover'], space='sell', optimize=False)
    low_offset = DecimalParameter(1.005, 1.015, default=sell_params['low_offset'], space='sell', optimize=True)
    low_offset_2 = DecimalParameter(1.01, 1.1, default=sell_params['low_offset_2'], space='sell', optimize=False)
    high_offset = DecimalParameter(0.985, 0.995, default=cover_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.980, 0.990, default=cover_params['high_offset_2'], space='sell', optimize=True)

    ewo_low = DecimalParameter(0.0, 20.0, default=sell_params['ewo_low'], space='sell', optimize=True)
    ewo_high = DecimalParameter(-5.0, 0.0, default=sell_params['ewo_high'], space='sell', optimize=True)
    rsi_sell = IntParameter(30, 70, default=sell_params['rsi_sell'], space='sell', optimize=False)
    ewo_high_2 = DecimalParameter(-12.0, 6.0, default=sell_params['ewo_high_2'], space='sell', optimize=False)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=sell_params['lambo2_ema_14_factor'], space='sell', optimize=True)
    lambo2_rsi_4_limit = IntParameter(40, 95, default=sell_params['lambo2_rsi_4_limit'], space='sell', optimize=True)
    lambo2_rsi_14_limit = IntParameter(40, 95, default=sell_params['lambo2_rsi_14_limit'], space='sell', optimize=True)

    # trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.004
    trailing_stop_positive_offset = 0.015

    # run "populate_indicators" only for new candle
    process_only_new_candles = True
    startup_candle_count = 96

    # Experimental settings
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True

    # DCA settings (same as longs)
    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4
    position_adjustment_enable = True
    threshold = 0.3

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
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

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df36h = dataframe.copy().shift(432)
        df24h = dataframe.copy().shift(288)
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()
        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])
        dataframe['rsi_mean'] = dataframe['rsi_14'].rolling(48).mean()
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        max_reached_price = trade.max_rate
        trailing_percentage = 0.05
        new_stoploss = max_reached_price * (1 - trailing_percentage)
        return max(new_stoploss, self.stoploss)

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Tighter unclog: 3 days vs 10 days for longs (shorts pay funding fees)
        if current_profit < -0.04 and (current_time - trade.open_date_utc).days >= 3:
            return 'unclog_short'

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        # Only allow shorts
        if side == "long":
            return False

        # Count open shorts
        short_count = sum(1 for t in Trade.get_trades_proxy(is_open=True) if t.is_short)

        # Enforce max shorts limit
        if short_count >= self.max_short_trades:
            return False

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['exit_signal']):
                # INVERTED: Block exit if price is ABOVE ema100 (shorts still profitable)
                if (last_candle['hma_50']*0.851 < last_candle['ema100']) and (last_candle['close'] > last_candle['ema100']*1.049):
                    return False

        # Slippage protection (same logic)
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()
        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0
        return True

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Convert to Heikin Ashi candles
        heikin_ashi_df = heikinashi(dataframe)
        dataframe['ha_close'] = heikin_ashi_df['close']
        dataframe['ha_open'] = heikin_ashi_df['open']

        # EMA
        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_cover values
        for val in self.base_nb_candles_cover.range:
            dataframe[f'ma_cover_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Stoch
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # Bollinger Bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['sellsignal'] = (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset.value)
        dataframe['coversignal'] = (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.high_offset.value)

        dataframe['difference_signal'] = (dataframe['ha_close'] - dataframe[f'ma_cover_{self.base_nb_candles_cover.value}']).sub(dataframe['ha_close'].sub(dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']).mean()).div(dataframe['ha_close'].sub(dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']).std())
        dataframe['close_sell_signal'] = (dataframe['ha_close'] - dataframe['sellsignal']).sub(dataframe['ha_close'].sub(dataframe['sellsignal']).mean()).div(dataframe['ha_close'].sub(dataframe['sellsignal']).std())
        dataframe['distance'] = (dataframe['ha_close'] - dataframe['sellsignal']) / dataframe['ha_close'].std()
        dataframe['sell_signal_distance'] = dataframe['distance'].abs() < self.threshold

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # Add TDI (Traders Dynamic Index)
        tdi_df = tdi(dataframe['close'])
        dataframe['tdi_rsi'] = tdi_df['rsi']
        dataframe['tdi_signal'] = tdi_df['rsi_signal']

        # Add Awesome Oscillator
        dataframe['ao'] = awesome_oscillator(dataframe)

        # Add Simple Moving Average for comparison
        dataframe['sma'] = sma(dataframe['close'], window=14)

        dataframe = self.pump_dump_protection(dataframe, metadata)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Short signal 1: EWO high (negative for shorts)
        sell1ewo = (
            (dataframe['rsi_fast'] > 65) &  # INVERTED: 100 - 35
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset.value)) &  # INVERTED: >
            (dataframe['EWO'] < self.ewo_high.value) &  # INVERTED: < (ewo_high is negative)
            (dataframe['rsi_14'] > self.rsi_sell.value) &  # INVERTED: >
            (dataframe['volume'] > 0) &
            (dataframe['close'] > (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.high_offset.value))  # INVERTED: >
        )
        dataframe.loc[sell1ewo, 'enter_tag'] += 'short_ewo_high_rsi_'
        conditions.append(sell1ewo)

        # Short signal 2: EWO very high with very high RSI
        sell2ewo = (
            (dataframe['rsi_fast'] > 65) &  # INVERTED
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset_2.value)) &  # INVERTED
            (dataframe['EWO'] < self.ewo_high_2.value) &  # INVERTED
            (dataframe['rsi_14'] > self.rsi_sell.value) &  # INVERTED
            (dataframe['volume'] > 0) &
            (dataframe['close'] > (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.high_offset.value)) &  # INVERTED
            (dataframe['rsi_14'] > 75)  # INVERTED: 100 - 25 = 75
        )
        dataframe.loc[sell2ewo, 'enter_tag'] += 'short_ewo2_high_rsi_'
        conditions.append(sell2ewo)

        # Lambo2 short: Price above EMA with high RSI
        lambo2_short = (
            (dataframe['close'] > (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &  # INVERTED: >
            (dataframe['rsi_4'] > int(self.lambo2_rsi_4_limit.value)) &  # INVERTED: >
            (dataframe['rsi_14'] > int(self.lambo2_rsi_14_limit.value))  # INVERTED: >
        )
        dataframe.loc[lambo2_short, 'enter_tag'] += 'short_lambo2_'
        conditions.append(lambo2_short)

        # Short signal: EWO low (positive for shorts = market overheated)
        sellewolow = (
            (dataframe['rsi_fast'] > 65) &  # INVERTED
            (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset.value)) &  # INVERTED
            (dataframe['EWO'] > self.ewo_low.value) &  # INVERTED: > (ewo_low is positive)
            (dataframe['volume'] > 0) &
            (dataframe['close'] > (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.high_offset.value))  # INVERTED
        )
        dataframe.loc[sellewolow, 'enter_tag'] += 'short_ewo_low_rsi_'
        conditions.append(sellewolow)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_short'] = 1  # INVERTED: enter_short

        # Pump/dump protection still applies
        dont_sell_conditions = []
        dont_sell_conditions.append((dataframe['pnd_volume_warn'] == -1))
        if dont_sell_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, dont_sell_conditions), 'enter_short'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Exit short when downtrend signal appears
        coversignal = (
            (dataframe['ha_close'] < dataframe['ha_open']) &  # INVERTED: Red candle = exit short
            (dataframe['difference_signal'] <= -1.9)  # INVERTED: <= (threshold inverted)
        )
        dataframe.loc[coversignal, 'exit_tag'] += 'cover_signal'
        conditions.append(coversignal)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'exit_short'] = 1  # INVERTED: exit_short

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None

        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # For shorts, add only after upward spike starts reverting down.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] > previous_candle['close']:
            return None

        count_of_entries = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'sell':
                continue
            if order.status == "closed":
                count_of_entries += 1

        if 1 <= count_of_entries <= self.max_safety_orders:
            safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_entries - 1)) - 1) / (self.safety_order_step_scale - 1))
            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                except Exception as exception:
                    # Backtesting can provide no live rate for wallet stake calculation.
                    logger.info(f'Fallback stake sizing for {trade.pair} after wallet stake error: {str(exception)}')
                    stake_amount = trade.stake_amount
                stake_amount = stake_amount * math.pow(self.safety_order_volume_scale,(count_of_entries - 1))
                logger.info(f"Initiating safety order short add #{count_of_entries} for {trade.pair} with stake amount {stake_amount}")
                return stake_amount

        return None
