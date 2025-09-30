import talib.abstract as ta
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import (
  IStrategy,
  CategoricalParameter,
  DecimalParameter,
  IntParameter,
  informative,
  merge_informative_pair
)
from freqtrade.persistence import Trade

# OPTIMIZED VERSION V2 - Enhanced Trend Filtering & Bidirectional Balance
# Focus: Proper trend filtering, balanced long/short entries, improved risk management
class BandtasticFiboHyper_opt314_v2(IStrategy):
  INTERFACE_VERSION = 3
  can_short = True
  timeframe = '5m'
  informative_timeframe = '1h'

  # ROI table - Keep optimized values
  minimal_roi = {
    "0": 0.195,
    "21": 0.074,
    "43": 0.028,
    "109": 0
  }

  # ============= OPTIMIZED STOP LOSS CONFIGURATION =============
  stoploss = -0.15  # 15% initial stop (dynamically adjusted)

  startup_candle_count = 999

  # ============= OPTIMIZED TRAILING STOP =============
  trailing_stop = True
  trailing_stop_positive = 0.01  # Start trailing at 1% profit
  trailing_stop_positive_offset = 0.015  # Activate at 1.5% profit
  trailing_only_offset_is_reached = True

  # Max open trades
  max_open_trades = 1

  # ========= LEVERAGE PARAMETERS =========
  max_leverage = DecimalParameter(1.0, 5.0, default=2.295, space='protection', optimize=True)
  max_short_leverage = DecimalParameter(1.0, 3.0, default=2.953, space='protection', optimize=True)
  atr_threshold_low = DecimalParameter(0.005, 0.03, default=0.019, space='protection', optimize=True)
  atr_threshold_high = DecimalParameter(0.02, 0.08, default=0.026, space='protection', optimize=True)

  # ========= DYNAMIC STOP LOSS PARAMETERS =========
  atr_stop_multiplier_long = DecimalParameter(1.5, 4.0, default=2.5, space='protection', optimize=True)
  atr_stop_multiplier_short = DecimalParameter(1.0, 3.0, default=1.8, space='protection', optimize=True)

  min_stop_loss_long = DecimalParameter(0.03, 0.08, default=0.05, space='protection', optimize=True)
  min_stop_loss_short = DecimalParameter(0.02, 0.06, default=0.035, space='protection', optimize=True)

  max_stop_loss_long = DecimalParameter(0.10, 0.20, default=0.15, space='protection', optimize=True)
  max_stop_loss_short = DecimalParameter(0.06, 0.12, default=0.08, space='protection', optimize=True)

  short_stop_time_hours = IntParameter(2, 8, default=4, space='protection', optimize=True)
  short_stop_tighten_factor = DecimalParameter(0.5, 0.9, default=0.7, space='protection', optimize=True)

  # ========= ENHANCED LONG PARAMETERS =========
  buy_fastema = IntParameter(1, 236, default=191, space='buy', optimize=True)
  buy_slowema = IntParameter(1, 250, default=128, space='buy', optimize=True)
  buy_rsi = IntParameter(15, 70, default=56, space='buy', optimize=True)
  buy_mfi = IntParameter(15, 70, default=40, space='buy', optimize=True)
  buy_rsi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
  buy_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
  buy_ema_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
  buy_trigger = CategoricalParameter(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'], 
                                     default='bb_lower4', space='buy', optimize=True)
  buy_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
  buy_fib_level = CategoricalParameter(['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'], 
                                       default='fib_382', space='buy', optimize=True)

  # NEW: Higher timeframe confirmation for longs
  buy_1h_trend_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
  buy_volume_threshold = DecimalParameter(0.8, 1.5, default=1.0, space='buy', optimize=True)

  # ========= ENHANCED SHORT PARAMETERS =========
  short_fastema = IntParameter(1, 250, default=53, space='sell', optimize=True)
  short_slowema = IntParameter(1, 250, default=168, space='sell', optimize=True)
  short_rsi = IntParameter(50, 85, default=70, space='sell', optimize=True)  # OPTIMIZED: More realistic threshold
  short_mfi = IntParameter(50, 85, default=65, space='sell', optimize=True)  # OPTIMIZED: Higher threshold
  short_rsi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
  short_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
  short_ema_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)  # ENABLED
  short_trigger = CategoricalParameter(['bb_upper2', 'bb_upper3', 'bb_upper4'], 
                                       default='bb_upper3', space='sell', optimize=True)  # OPTIMIZED: Tighter

  # NEW: Critical - Higher timeframe downtrend confirmation
  short_1h_trend_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
  short_volume_threshold = DecimalParameter(1.0, 2.0, default=1.2, space='sell', optimize=True)
  short_adx_threshold = IntParameter(15, 35, default=20, space='sell', optimize=True)

  # ========= SELL/COVER PARAMETERS =========
  sell_fastema = IntParameter(1, 365, default=222, space='sell', optimize=True)
  sell_slowema = IntParameter(1, 365, default=192, space='sell', optimize=True)
  sell_rsi = IntParameter(30, 100, default=47, space='sell', optimize=True)
  sell_mfi = IntParameter(30, 100, default=46, space='sell', optimize=True)
  sell_rsi_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
  sell_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
  sell_ema_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
  sell_trigger = CategoricalParameter(['sell-bb_upper1', 'sell-bb_upper2', 'sell-bb_upper3', 'sell-bb_upper4'], 
                                      default='sell-bb_upper2', space='sell', optimize=True)

  cover_fastema = IntParameter(1, 250, default=97, space='buy', optimize=True)
  cover_slowema = IntParameter(1, 250, default=191, space='buy', optimize=True)
  cover_rsi = IntParameter(10, 70, default=42, space='buy', optimize=True)
  cover_mfi = IntParameter(10, 70, default=12, space='buy', optimize=True)
  cover_rsi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
  cover_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
  cover_ema_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
  cover_trigger = CategoricalParameter(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'], 
                                       default='bb_lower4', space='buy', optimize=True)
  cover_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
  cover_fib_level = CategoricalParameter(['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'], 
                                         default='fib_382', space='buy', optimize=True)

  def informative_pairs(self):
    """Required for 1h timeframe data."""
    pairs = self.dp.current_whitelist()
    informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
    return informative_pairs

  @informative('1h')
  def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """Calculate 1h trend indicators for higher timeframe confirmation."""
    # EMAs for trend detection
    dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=21)
    dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=55)
    dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=200)

    # RSI and ADX for trend strength
    dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
    dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

    # Define clear uptrend and downtrend
    dataframe['uptrend'] = (
      (dataframe['close'] > dataframe['ema_fast']) &
      (dataframe['ema_fast'] > dataframe['ema_slow']) &
      (dataframe['close'] > dataframe['ema_long']) &
      (dataframe['rsi'] > 45) &
      (dataframe['adx'] > 15)
    )

    dataframe['downtrend'] = (
      (dataframe['close'] < dataframe['ema_fast']) &
      (dataframe['ema_fast'] < dataframe['ema_slow']) &
      (dataframe['close'] < dataframe['ema_long']) &
      (dataframe['rsi'] < 55) &
      (dataframe['adx'] > 15)
    )

    return dataframe

  def leverage(self, pair: str, current_time: datetime, current_rate: float,
               proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
    """
    Dynamic leverage based on normalized ATR volatility.
    Lower volatility = higher leverage, higher volatility = lower leverage.
    """
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    if dataframe is None or len(dataframe) < 20:
      return 2.0  # Fallback default

    close = dataframe['close'].iloc[-1]
    atr = ta.ATR(dataframe, timeperiod=14).iloc[-1]
    normalized_atr = atr / close if close > 0 else 0

    if normalized_atr < self.atr_threshold_low.value:
      lev = 4.0
    elif normalized_atr < self.atr_threshold_high.value:
      lev = 2.5
    else:
      lev = 1.5

    # Apply leverage limits
    if side == 'short':
      lev = min(lev, self.max_short_leverage.value)

    return min(lev, float(self.max_leverage.value))

  def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                     current_rate: float, current_profit: float, **kwargs) -> float:
    """
    CRITICAL: Dynamic stop loss based on ATR and trade direction.

    Key Strategy:
    - Shorts get TIGHTER stops (preventing large losses)
    - Longs get normal stops (maintaining performance)
    - Both use ATR-based dynamic adjustment
    - Time-based tightening for underwater shorts
    """
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    if dataframe is None or len(dataframe) < 20:
      return self.stoploss

    # Get current ATR and calculate normalized ATR
    last_candle = dataframe.iloc[-1]
    atr = last_candle['atr'] if 'atr' in last_candle else 0
    normalized_atr = last_candle['normalized_atr'] if 'normalized_atr' in last_candle else 0.02

    # Determine if this is a long or short trade
    is_short = trade.is_short

    # ========= CALCULATE ATR-BASED STOP LOSS =========
    if is_short:
      # SHORTS: Use tighter multiplier
      atr_stop = normalized_atr * float(self.atr_stop_multiplier_short.value)
      min_stop = float(self.min_stop_loss_short.value)
      max_stop = float(self.max_stop_loss_short.value)

      # Time-based tightening for shorts that aren't profitable
      if current_profit < 0:
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600  # hours
        if trade_duration > float(self.short_stop_time_hours.value):
          # Tighten the stop as time passes
          time_factor = float(self.short_stop_tighten_factor.value)
          atr_stop = atr_stop * time_factor
          min_stop = min_stop * time_factor
    else:
      # LONGS: Use normal multiplier
      atr_stop = normalized_atr * float(self.atr_stop_multiplier_long.value)
      min_stop = float(self.min_stop_loss_long.value)
      max_stop = float(self.max_stop_loss_long.value)

    # Clamp the stop loss between min and max
    dynamic_stop = max(min_stop, min(atr_stop, max_stop))

    # Return negative value (stop loss is always negative)
    return -dynamic_stop

  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """Calculate all technical indicators."""
    # RSI and MFI
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['mfi'] = ta.MFI(dataframe)

    # ATR and normalized ATR (critical for dynamic stops)
    dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
    dataframe['normalized_atr'] = dataframe['atr'] / dataframe['close']

    # ADX for trend strength (critical for shorts)
    dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

    # Volume analysis
    dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

    # Bollinger Bands (multiple standard deviations)
    for std in range(1, 5):
      bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=float(std), nbdevdn=float(std))
      dataframe[f'bb_lowerband{std}'] = bb['lowerband']
      dataframe[f'bb_middleband{std}'] = bb['middleband']
      dataframe[f'bb_upperband{std}'] = bb['upperband']

    # EMAs (collect all unique periods)
    ema_periods = set([
      int(self.buy_fastema.value), int(self.buy_slowema.value),
      int(self.sell_fastema.value), int(self.sell_slowema.value),
      int(self.short_fastema.value), int(self.short_slowema.value),
      int(self.cover_fastema.value), int(self.cover_slowema.value)
    ])
    for period in ema_periods:
      if period > 0 and len(dataframe) >= period:
        dataframe[f'EMA_{period}'] = ta.EMA(dataframe, timeperiod=period)

    # Fibonacci Retracement Levels
    lookback = 50
    if len(dataframe) >= lookback:
      recent_max = dataframe['high'].rolling(lookback).max()
      recent_min = dataframe['low'].rolling(lookback).min()
      diff = recent_max - recent_min
      dataframe['fib_236'] = recent_max - diff * 0.236
      dataframe['fib_382'] = recent_max - diff * 0.382
      dataframe['fib_5'] = recent_max - diff * 0.5
      dataframe['fib_618'] = recent_max - diff * 0.618
      dataframe['fib_786'] = recent_max - diff * 0.786

    return dataframe

  def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """Define entry conditions for long and short positions with trend filtering."""

    # ============= LONG ENTRY CONDITIONS =============
    long_conditions = []

    if self.buy_rsi_enabled.value:
      long_conditions.append(dataframe['rsi'] < self.buy_rsi.value)

    if self.buy_mfi_enabled.value:
      long_conditions.append(dataframe['mfi'] < self.buy_mfi.value)

    if self.buy_ema_enabled.value:
      fast_col = f'EMA_{self.buy_fastema.value}'
      slow_col = f'EMA_{self.buy_slowema.value}'
      if fast_col in dataframe and slow_col in dataframe:
        long_conditions.append(dataframe[fast_col] > dataframe[slow_col])

    if self.buy_trigger.value.startswith('bb_lower'):
      bb_col = f'bb_lowerband{self.buy_trigger.value[-1]}'
      long_conditions.append(dataframe['close'] < dataframe[bb_col])

    if self.buy_trigger.value == 'fibonacci' and self.buy_fib_enabled.value:
      fib_col = self.buy_fib_level.value
      if fib_col in dataframe.columns:
        long_conditions.append(dataframe['close'] < dataframe[fib_col])

    # NEW: 1h uptrend confirmation
    if self.buy_1h_trend_enabled.value and 'uptrend_1h' in dataframe.columns:
      long_conditions.append(dataframe['uptrend_1h'] == True)

    # NEW: Volume confirmation
    long_conditions.append(dataframe['volume'] > dataframe['volume_mean'] * self.buy_volume_threshold.value)

    if long_conditions:
      dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1

    # ============= SHORT ENTRY CONDITIONS (HEAVILY OPTIMIZED) =============
    short_conditions = []

    # CRITICAL: More realistic RSI threshold
    if self.short_rsi_enabled.value:
      short_conditions.append(dataframe['rsi'] > self.short_rsi.value)

    # CRITICAL: Higher MFI threshold
    if self.short_mfi_enabled.value:
      short_conditions.append(dataframe['mfi'] > self.short_mfi.value)

    # CRITICAL: EMA bearish alignment
    if self.short_ema_enabled.value:
      fast_col = f'EMA_{self.short_fastema.value}'
      slow_col = f'EMA_{self.short_slowema.value}'
      if fast_col in dataframe and slow_col in dataframe:
        short_conditions.append(dataframe[fast_col] < dataframe[slow_col])

    # Using tighter trigger (bb_upper3 or bb_upper4)
    if self.short_trigger.value.startswith('bb_upper'):
      bb_col = f'bb_upperband{self.short_trigger.value[-1]}'
      short_conditions.append(dataframe['close'] > dataframe[bb_col])

    # CRITICAL: 1h downtrend confirmation (prevents shorts in uptrends!)
    if self.short_1h_trend_enabled.value and 'downtrend_1h' in dataframe.columns:
      short_conditions.append(dataframe['downtrend_1h'] == True)

    # CRITICAL: Stronger volume confirmation
    short_conditions.append(dataframe['volume'] > dataframe['volume_mean'] * self.short_volume_threshold.value)

    # CRITICAL: ADX strength filter
    short_conditions.append(dataframe['adx'] > self.short_adx_threshold.value)

    if short_conditions:
      dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_short'] = 1

    return dataframe

  def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """Define exit conditions for long and short positions."""

    # ============= LONG EXIT CONDITIONS =============
    long_exit = []

    if self.sell_rsi_enabled.value:
      long_exit.append(dataframe['rsi'] > self.sell_rsi.value)

    if self.sell_mfi_enabled.value:
      long_exit.append(dataframe['mfi'] > self.sell_mfi.value)

    if self.sell_ema_enabled.value:
      fast_col = f'EMA_{self.sell_fastema.value}'
      slow_col = f'EMA_{self.sell_slowema.value}'
      if fast_col in dataframe and slow_col in dataframe:
        long_exit.append(dataframe[fast_col] < dataframe[slow_col])

    if self.sell_trigger.value.startswith('sell-bb_upper'):
      bb_col = f'bb_upperband{self.sell_trigger.value[-1]}'
      long_exit.append(dataframe['close'] > dataframe[bb_col])

    # Exit on 1h downtrend
    if 'downtrend_1h' in dataframe.columns:
      long_exit.append(dataframe['downtrend_1h'] == True)

    long_exit.append(dataframe['volume'] > 0)

    if long_exit:
      dataframe.loc[reduce(lambda x, y: x & y, long_exit), 'exit_long'] = 1

    # ============= SHORT EXIT CONDITIONS =============
    short_exit = []

    if self.cover_rsi_enabled.value:
      short_exit.append(dataframe['rsi'] < self.cover_rsi.value)

    if self.cover_mfi_enabled.value:
      short_exit.append(dataframe['mfi'] < self.cover_mfi.value)

    if self.cover_ema_enabled.value:
      fast_col = f'EMA_{self.cover_fastema.value}'
      slow_col = f'EMA_{self.cover_slowema.value}'
      if fast_col in dataframe and slow_col in dataframe:
        short_exit.append(dataframe[fast_col] > dataframe[slow_col])

    if self.cover_trigger.value.startswith('bb_lower'):
      bb_col = f'bb_lowerband{self.cover_trigger.value[-1]}'
      short_exit.append(dataframe['close'] < dataframe[bb_col])

    if self.cover_trigger.value == 'fibonacci' and self.cover_fib_enabled.value:
      fib_col = self.cover_fib_level.value
      if fib_col in dataframe.columns:
        short_exit.append(dataframe['close'] < dataframe[fib_col])

    # Exit on 1h uptrend
    if 'uptrend_1h' in dataframe.columns:
      short_exit.append(dataframe['uptrend_1h'] == True)

    short_exit.append(dataframe['volume'] > 0)

    if short_exit:
      dataframe.loc[reduce(lambda x, y: x & y, short_exit), 'exit_short'] = 1

    return dataframe
