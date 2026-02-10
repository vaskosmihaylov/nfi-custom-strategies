"""
FrankenStrat_Shorts Strategy

A shorts-only variant of FrankenStrat, designed to profit in bear markets
and during overbought conditions.

Strategy Concept:
Multi-signal short entry strategy combining SMAOffset, TEMA, ClucMay, MACD,
and SSL channel signals - all inverted from the long version.

Entry Conditions (7 signals):
1. SMAOffset Short: Price above EMA with moderate EWO
2. TEMA Short: Price above TEMA with moderate EWO
3. ClucMay Bear Guard: Downtrend with price above upper BB
4. ClucMay No Guard: Deep above upper BB with extreme overbought RSI
5. MACD Bear Guard: Downtrend with bullish MACD crossover above upper BB
6. MACD No Guard: Bullish MACD crossover above upper BB
7. SSL Short: Price above SMA5 with bearish SSL and RSI divergence

Exit Conditions:
- Signal: Price below BB middleband or below EMA
- ROI: Same as longs (2.9% tiered)
- Custom Stoploss: 4-hour timeout + emergency -20% backstop

Key Differences from Long Strategy:
- INTERFACE_VERSION 3 with enter_short/exit_short
- 3x leverage via leverage() callback
- Max 4 short positions via confirm_trade_entry()
- Emergency backstop at -20%

Author: Derived from FrankenStrat
Version: 1.0.0
"""

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter

from functools import reduce


def SSLChannels(dataframe, length=7):
  df = dataframe.copy()
  df["ATR"] = ta.ATR(df, timeperiod=14)
  df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
  df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
  df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.nan))
  df["hlv"] = df["hlv"].ffill()
  df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
  df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
  return df["sslDown"], df["sslUp"]


def ewo(dataframe, ema_length=5, ema2_length=35):
  df = dataframe.copy()
  ema1 = ta.EMA(df, timeperiod=ema_length)
  ema2 = ta.EMA(df, timeperiod=ema2_length)
  emadif = (ema1 - ema2) / df["close"] * 100
  return emadif


class FrankenStrat_Shorts(IStrategy):
  INTERFACE_VERSION = 3
  can_short = True

  minimal_roi = {
    "0": 0.029,
    "10": 0.021,
    "30": 0.01,
    "40": 0.005,
  }

  # SMAOffset - same values as longs, logic inverted in populate_entry_trend
  base_nb_candles_buy = IntParameter(5, 80, default=12, space="buy", optimize=True)
  base_nb_candles_sell = IntParameter(5, 80, default=31, space="sell", optimize=True)
  low_offset = DecimalParameter(0.9, 0.99, default=0.975, space="buy", optimize=True)
  high_offset = DecimalParameter(0.95, 1.1, default=1.0, space="sell", optimize=True)

  # TEMA
  tema_low_offset = DecimalParameter(0.9, 0.99, default=0.95, space="buy", optimize=True)

  # Protection
  fast_ewo = 50
  slow_ewo = 200

  stoploss = -0.99  # effectively disabled, managed by custom_stoploss

  timeframe = "5m"
  inf_1h = "1h"

  # Exit signal
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
  process_only_new_candles = False

  # Number of candles the strategy requires before producing valid signals
  startup_candle_count: int = 200

  # Max short positions
  max_short_trades = 4

  # Optional order type mapping.
  order_types = {
    "entry": "limit",
    "exit": "limit",
    "stoploss": "market",
    "stoploss_on_exchange": False,
  }

  def leverage(self, pair: str, current_time: datetime, current_rate: float,
               proposed_leverage: float, max_leverage: float, entry_tag: str,
               side: str, **kwargs) -> float:
    return 3.0

  def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: str,
                          side: str, **kwargs) -> bool:
    # Block all long entries
    if side == "long":
      return False

    # Count open shorts
    short_count = sum(1 for trade in Trade.get_trades_proxy(is_open=True) if trade.is_short)

    # Enforce max shorts limit
    if short_count >= self.max_short_trades:
      return False

    return True

  def custom_stoploss(self, pair: str, trade: "Trade", current_time: datetime,
                      current_rate: float, current_profit: float, **kwargs) -> float:
    # Emergency backstop - prevent liquidation at 3x leverage
    if current_profit <= -0.20:
      return -0.21

    # Manage losing trades and open room for better ones.
    if (current_profit < 0) & (current_time - timedelta(minutes=240) > trade.open_date_utc):
      return 0.01
    return 0.99

  def informative_pairs(self):
    pairs = self.dp.current_whitelist()
    informative_pairs = [(pair, "1h") for pair in pairs]
    return informative_pairs

  def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    assert self.dp, "DataProvider is required for multiple timeframes."
    # Get the informative pair
    informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_1h)
    # EMA
    informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
    informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
    # RSI
    informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)

    # SSL Channels
    ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
    informative_1h["ssl_down"] = ssl_down_1h
    informative_1h["ssl_up"] = ssl_up_1h

    return informative_1h

  def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    for length in set(list(self.base_nb_candles_buy.range) + list(self.base_nb_candles_sell.range)):
      dataframe[f"ema_{length}"] = ta.EMA(dataframe, timeperiod=length)

    dataframe["ewo"] = ewo(dataframe, self.fast_ewo, self.slow_ewo)

    # RSI
    dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
    dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)
    dataframe["hma_50"] = qtpylib.hull_moving_average(dataframe["close"], window=50)

    # TEMA
    dataframe["tema"] = ta.TEMA(dataframe, length=14)

    # strategy ClucMay72018
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    dataframe["bb_lowerband"] = bollinger["lower"]
    dataframe["bb_middleband"] = bollinger["mid"]
    dataframe["bb_upperband"] = bollinger["upper"]
    dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=50)
    dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30).mean()

    # EMA
    dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

    dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
    dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)

    # SMA
    dataframe["sma_5"] = ta.EMA(dataframe, timeperiod=5)

    # RSI
    dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

    return dataframe

  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # The indicators for the 1h informative timeframe
    informative_1h = self.informative_1h_indicators(dataframe, metadata)
    dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

    # The indicators for the normal (5m) timeframe
    dataframe = self.normal_tf_indicators(dataframe, metadata)

    return dataframe

  def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
      (
        # Signal 1: SMAOffset Short - price above EMA (overbought)
        (dataframe["close"] > (dataframe[f"ema_{self.base_nb_candles_buy.value}"] * (2 - self.low_offset.value)))
        & (dataframe["ewo"] < 1)
        & (dataframe["volume"] > 0)
      )
      | (
        # Signal 2: TEMA Short - price above TEMA (overbought)
        (dataframe["close"] > (dataframe["tema"] * (2 - self.tema_low_offset.value)))
        & (dataframe["ewo"] < 1)
        & (dataframe["volume"] > 0)
      )
      | (
        # Signal 3: ClucMay Short with bear guard (inverted bull guard)
        (dataframe["close"] < dataframe["ema_200"])
        & (dataframe["close"] < dataframe["ema_200_1h"])
        & (dataframe["close"] > dataframe["ema_slow"])
        & (dataframe["close"] > 1.01 * dataframe["bb_upperband"])
        & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4)
        & (dataframe["volume"] > 0)
      )
      | (
        # Signal 4: ClucMay Short without guard (inverted bear no guard)
        (dataframe["close"] > dataframe["ema_slow"])
        & (dataframe["close"] > 1.025 * dataframe["bb_upperband"])
        & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
        & (dataframe["rsi_1h"] > 85)
        & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4)
        & (dataframe["volume"] > 0)
      )
      | (
        # Signal 5: MACD Short with bear guard (inverted MACD bull guard)
        (dataframe["close"] < dataframe["ema_200"])
        & (dataframe["close"] < dataframe["ema_200_1h"])
        & (dataframe["ema_12"] > dataframe["ema_26"])
        & ((dataframe["ema_12"] - dataframe["ema_26"]) > (dataframe["open"] * 0.02))
        & ((dataframe["ema_12"].shift() - dataframe["ema_26"].shift()) > (dataframe["open"] / 100))
        & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
        & (dataframe["close"] > dataframe["bb_upperband"])
        & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4)
        & (dataframe["volume"] > 0)
      )
      | (
        # Signal 6: MACD Short without guard (inverted MACD bear no guard)
        (dataframe["ema_12"] > dataframe["ema_26"])
        & ((dataframe["ema_12"] - dataframe["ema_26"]) > (dataframe["open"] * 0.03))
        & ((dataframe["ema_12"].shift() - dataframe["ema_26"].shift()) > (dataframe["open"] / 100))
        & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
        & (dataframe["close"] > dataframe["bb_upperband"])
        & (dataframe["volume"] > 0)
      )
      | (
        # Signal 7: SSL Short - price above SMA5 with bearish SSL
        (dataframe["close"] > dataframe["sma_5"])
        & (dataframe["ssl_down_1h"] > dataframe["ssl_up_1h"])
        & (dataframe["ema_slow"] < dataframe["ema_200"])
        & (dataframe["ema_50_1h"] < dataframe["ema_200_1h"])
        & (dataframe["rsi"] > dataframe["rsi_1h"] + 43.276)
        & (dataframe["volume"] > 0)
      ),
      "enter_short",
    ] = 1

    return dataframe

  def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    conditions = []

    conditions.append(
      (
        # Price below BB middleband (inverted from above)
        (dataframe["close"] < dataframe["bb_middleband"] * 0.99)
        & (dataframe["volume"] > 0)
      )
    )

    conditions.append(
      (
        (dataframe["close"] < (dataframe[f"ema_{self.base_nb_candles_sell.value}"] * (2 - self.high_offset.value)))
        & (dataframe["ewo"] > -20)
        & (dataframe["volume"] > 0)
      )
    )
    if conditions:
      dataframe.loc[
        reduce(lambda x, y: x | y, conditions),
        "exit_short",
      ] = 1

    return dataframe
