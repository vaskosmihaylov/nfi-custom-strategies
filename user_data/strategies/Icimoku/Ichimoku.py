# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
from datetime import datetime  # noqa
from typing import Optional, Union  # noqa
from functools import reduce  # noqa
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Ichimoku_improved(IStrategy):
  """
  Improved Ichimoku Strategy with better exit logic and cloud-based filters.
  
  Key improvements:
  - Adjusted Ichimoku parameters for 4h timeframe
  - Cloud breakout confirmation (no trades inside cloud)
  - RSI momentum filter
  - Improved exit signals using TK cross and cloud re-entry
  - Better trailing stop logic with profit-based activation
  - ATR-based stop loss and take profit
  """
  INTERFACE_VERSION = 3

  timeframe = '4h'

  USE_TALIB = False

  # Can this strategy go short?
  can_short: bool = True

  # ROI table - more realistic targets
  minimal_roi = {
    "0": 0.08,    # 8% profit target
    "720": 0.05,  # After 12h (3 candles), reduce to 5%
    "1440": 0.03, # After 24h (6 candles), reduce to 3%
    "2880": 0.015 # After 48h (12 candles), reduce to 1.5%
  }

  # Hard stop loss - will be overridden by custom_stoploss
  stoploss = -0.10  # 10% hard stop

  # Disable built-in trailing stop - we use custom_stoploss
  trailing_stop = False
  
  process_only_new_candles: bool = True
  use_exit_signal = True
  exit_profit_only = False
  ignore_roi_if_entry_signal = False

  use_custom_stoploss: bool = True

  # Number of candles the strategy requires before producing valid signals
  startup_candle_count: int = 60  # Increased for Senkou Span B calculation

  # Optional order type mapping
  order_types = {
    'entry': 'market',
    'exit': 'market',
    'stoploss': 'market',
    'stoploss_on_exchange': False
  }

  # Optional order time in force
  order_time_in_force = {
    'entry': 'gtc',
    'exit': 'gtc'
  }

  # Ichimoku parameters optimized for 4h timeframe
  # Standard: 9, 26, 52 - we use slightly faster settings
  TS = IntParameter(7, 12, default=9, space="buy", optimize=True)
  KS = IntParameter(20, 30, default=26, space="buy", optimize=True)
  SS = IntParameter(40, 60, default=52, space="buy", optimize=True)

  # ATR parameters for stop loss and take profit
  ATR_length = IntParameter(10, 20, default=14, space="buy", optimize=True)
  ATR_SL_Multip = DecimalParameter(1.5, 3.5, decimals=1, default=2.0, space="buy", optimize=True)
  ATR_TP_Multip = DecimalParameter(2.0, 5.0, decimals=1, default=3.0, space="buy", optimize=True)
  
  # Trailing stop activation threshold (% profit)
  trailing_activation = DecimalParameter(0.01, 0.05, decimals=2, default=0.02, space="buy", optimize=True)
  trailing_distance = DecimalParameter(0.005, 0.02, decimals=3, default=0.01, space="buy", optimize=True)

  # RSI filter
  rsi_enabled = BooleanParameter(default=True, space="buy", optimize=True)
  rsi_length = IntParameter(10, 20, default=14, space="buy", optimize=True)
  rsi_buy_threshold = IntParameter(40, 60, default=50, space="buy", optimize=True)
  rsi_sell_threshold = IntParameter(40, 60, default=50, space="buy", optimize=True)

  # Cloud filter - require price to be outside cloud for entry
  cloud_filter = BooleanParameter(default=True, space="buy", optimize=True)

  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Populate all indicators needed for the strategy.
    """
    if self.dp.runmode.value in ('live', 'dry_run'):
      # Use pandas_ta for live/dry-run for stability
      self.USE_TALIB = False
    else:
      self.USE_TALIB = True

    # Calculate Ichimoku indicator
    ichimo = pta.ichimoku(
      high=dataframe['high'], 
      low=dataframe['low'], 
      close=dataframe['close'],
      tenkan=int(self.TS.value), 
      kijun=int(self.KS.value), 
      senkou=int(self.SS.value),
      include_chikou=True
    )[0]
    
    dataframe['tenkan'] = ichimo[f'ITS_{int(self.TS.value)}'].copy()
    dataframe['kijun'] = ichimo[f'IKS_{int(self.KS.value)}'].copy()
    dataframe['senkanA'] = ichimo[f'ISA_{int(self.TS.value)}'].copy()
    dataframe['senkanB'] = ichimo[f'ISB_{int(self.KS.value)}'].copy()
    dataframe['chiko'] = ichimo[f'ICS_{int(self.KS.value)}'].copy()

    # Calculate ATR for stop loss and take profit
    dataframe['ATR'] = pta.atr(
      dataframe['high'], 
      dataframe['low'], 
      dataframe['close'],
      length=int(self.ATR_length.value), 
      talib=self.USE_TALIB
    )

    # Calculate RSI for momentum filter
    dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=int(self.rsi_length.value))

    # Determine if price is above or below cloud
    dataframe['cloud_top'] = dataframe[['senkanA', 'senkanB']].max(axis=1)
    dataframe['cloud_bottom'] = dataframe[['senkanA', 'senkanB']].min(axis=1)
    dataframe['above_cloud'] = dataframe['close'] > dataframe['cloud_top']
    dataframe['below_cloud'] = dataframe['close'] < dataframe['cloud_bottom']
    dataframe['in_cloud'] = ~(dataframe['above_cloud'] | dataframe['below_cloud'])

    # Cloud color (green when senkanA > senkanB, red otherwise)
    dataframe['cloud_green'] = dataframe['senkanA'] > dataframe['senkanB']
    dataframe['cloud_red'] = dataframe['senkanA'] < dataframe['senkanB']

    # TK cross signals
    dataframe['tk_cross_up'] = qtpylib.crossed_above(dataframe['tenkan'], dataframe['kijun'])
    dataframe['tk_cross_down'] = qtpylib.crossed_below(dataframe['tenkan'], dataframe['kijun'])

    return dataframe

  def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Entry signals based on Ichimoku cloud breakout and TK cross.
    
    Long entry conditions:
    - Price above cloud (cloud breakout)
    - Cloud is green (bullish)
    - Tenkan > Kijun (bullish momentum)
    - Optional: RSI > threshold (momentum confirmation)
    
    Short entry conditions:
    - Price below cloud (cloud breakdown)
    - Cloud is red (bearish)
    - Tenkan < Kijun (bearish momentum)
    - Optional: RSI < threshold (momentum confirmation)
    """
    # Long entry conditions
    long_conditions = [
      (dataframe['above_cloud']),  # Price above cloud
      (dataframe['cloud_green']),  # Cloud is green (bullish)
      (dataframe['tenkan'] > dataframe['kijun']),  # TK bullish alignment
      (dataframe['close'] > dataframe['tenkan']),  # Price above tenkan
    ]

    # Add cloud filter if enabled
    if self.cloud_filter.value:
      long_conditions.append(~dataframe['in_cloud'])

    # Add RSI filter if enabled
    if self.rsi_enabled.value:
      long_conditions.append(dataframe['rsi'] > self.rsi_buy_threshold.value)

    dataframe.loc[
      reduce(lambda x, y: x & y, long_conditions),
      'enter_long'
    ] = 1

    # Short entry conditions
    short_conditions = [
      (dataframe['below_cloud']),  # Price below cloud
      (dataframe['cloud_red']),  # Cloud is red (bearish)
      (dataframe['tenkan'] < dataframe['kijun']),  # TK bearish alignment
      (dataframe['close'] < dataframe['tenkan']),  # Price below tenkan
    ]

    # Add cloud filter if enabled
    if self.cloud_filter.value:
      short_conditions.append(~dataframe['in_cloud'])

    # Add RSI filter if enabled
    if self.rsi_enabled.value:
      short_conditions.append(dataframe['rsi'] < self.rsi_sell_threshold.value)

    dataframe.loc[
      reduce(lambda x, y: x & y, short_conditions),
      'enter_short'
    ] = 1

    return dataframe

  def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Exit signals based on TK cross or cloud re-entry.
    
    Long exit:
    - TK bearish cross (tenkan crosses below kijun)
    - Price enters cloud from above
    - Price drops below kijun
    
    Short exit:
    - TK bullish cross (tenkan crosses above kijun)
    - Price enters cloud from below
    - Price rises above kijun
    """
    # Long exit conditions
    dataframe.loc[
      (
        (dataframe['tk_cross_down']) |  # TK bearish cross
        (dataframe['in_cloud']) |  # Price entered cloud
        (dataframe['close'] < dataframe['kijun'])  # Price below kijun
      ),
      'exit_long'
    ] = 1

    # Short exit conditions
    dataframe.loc[
      (
        (dataframe['tk_cross_up']) |  # TK bullish cross
        (dataframe['in_cloud']) |  # Price entered cloud
        (dataframe['close'] > dataframe['kijun'])  # Price above kijun
      ),
      'exit_short'
    ] = 1

    return dataframe

  def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                      current_rate: float, current_profit: float, **kwargs) -> float:
    """
    Custom stop-loss function with ATR-based stops and trailing stop after profit threshold.
    
    Logic:
    1. Initial stop loss: ATR-based distance below/above entry
    2. After reaching trailing_activation profit, activate trailing stop
    3. Trailing stop follows price at trailing_distance below/above current price
    4. Stop loss can only move in favorable direction (never widens)
    """
    # Get analyzed dataframe
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    
    # Get the candle where trade was opened
    trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
    trade_candle = dataframe.loc[dataframe['date'] == trade_date]

    if trade_candle.empty:
      # If we can't find the trade candle, return the default stoploss
      return self.stoploss

    trade_candle = trade_candle.squeeze()
    atr = trade_candle['ATR']
    
    if pd.isna(atr) or atr == 0:
      return self.stoploss

    # Calculate initial ATR-based stop loss distance
    initial_sl_distance = atr * float(self.ATR_SL_Multip.value)
    
    # Check if trailing stop should be activated
    if current_profit >= float(self.trailing_activation.value):
      # Activate trailing stop at trailing_distance from current price
      trailing_sl = float(self.trailing_distance.value)
      
      # Ensure we don't widen the stop loss
      # For longs: stop should be moving up, for shorts: stop should be moving down
      if not trade.is_short:
        # For longs, stop moves up with price
        initial_sl_profit = -initial_sl_distance / trade.open_rate
        return max(trailing_sl, initial_sl_profit)
      else:
        # For shorts, stop moves down with price
        initial_sl_profit = -initial_sl_distance / trade.open_rate
        return max(trailing_sl, initial_sl_profit)
    else:
      # Use initial ATR-based stop loss
      if not trade.is_short:
        # For long positions, SL is below entry price
        sl_price = trade.open_rate - initial_sl_distance
        sl_profit = (sl_price - trade.open_rate) / trade.open_rate
      else:
        # For short positions, SL is above entry price
        sl_price = trade.open_rate + initial_sl_distance
        sl_profit = (trade.open_rate - sl_price) / trade.open_rate
      
      # Return the stop loss as a ratio (negative value)
      # Ensure it doesn't exceed the hard stoploss
      return max(sl_profit, self.stoploss)

  def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                  current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
    """
    Custom exit logic for take profit based on ATR.
    
    Take profit at ATR_TP_Multip * ATR distance from entry.
    """
    # Get analyzed dataframe
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    
    # Get the candle where trade was opened
    trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
    trade_candle = dataframe.loc[dataframe['date'] == trade_date]

    if trade_candle.empty:
      return None

    trade_candle = trade_candle.squeeze()
    atr = trade_candle['ATR']
    
    if pd.isna(atr) or atr == 0:
      return None

    # Calculate take profit distance
    tp_distance = atr * float(self.ATR_TP_Multip.value)
    
    if not trade.is_short:
      # For long positions, TP is above entry price
      tp_price = trade.open_rate + tp_distance
      if current_rate >= tp_price:
        return 'take_profit_atr'
    else:
      # For short positions, TP is below entry price
      tp_price = trade.open_rate - tp_distance
      if current_rate <= tp_price:
        return 'take_profit_atr'

    return None
