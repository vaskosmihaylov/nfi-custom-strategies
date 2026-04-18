# -*- coding: utf-8 -*-
"""
NostalgiaForInfinityNext_ChangeToTower_V6_Short

Short-selling strategy for Freqtrade futures trading.
Companion to the V6 long strategy.

Key design principles (V2 rewrite):
- BTC downtrend is a GLOBAL GATE: no shorts unless BTC 1h is in confirmed downtrend
- Each condition requires 6-8 simultaneous filters (not 4-5)
- Safe-pump detection prevents shorting already-crashed pairs
- 1h RSI confirmation layer on every condition
- Strict volume filters to avoid low-liquidity traps
- Only 5 focused conditions (not 10 loose ones)

Timeframe: 5m (with 1h informative)
Trading Mode: Futures (shorting enabled)
Target Exchange: Bybit
"""

import logging
import numpy as np
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from technical.indicators import zema, VIDYA
import pandas_ta as pta

log = logging.getLogger(__name__)


class NostalgiaForInfinityNext_ChangeToTower_V6_Short(IStrategy):
    INTERFACE_VERSION = 3

    # ===== CONSERVATIVE RISK PARAMETERS FOR SHORTS =====
    # Shorts are riskier (unlimited upside risk), so we use tighter stops
    # and lower ROI targets compared to the long side.
    minimal_roi = {
        "0": 0.06,  # 6% immediate target
        "30": 0.035,  # 3.5% after 30 candles (2.5h)
        "60": 0.02,  # 2% after 60 candles (5h)
        "120": 0.01,  # 1% after 120 candles (10h)
    }

    stoploss = -0.06  # 6% hard stoploss

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.008  # 0.8% trail
    trailing_stop_positive_offset = 0.02  # Activate at 2% profit

    use_custom_stoploss = True

    timeframe = "5m"
    info_timeframe = "1h"

    has_BTC_info_tf = True

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = True  # Match long strategy: hold for recovery
    ignore_roi_if_entry_signal = True

    startup_candle_count: int = 480

    can_short = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "trailing_stop_loss": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }

    #############################################################
    # SHORT ENTRY CONDITION ENABLES
    #############################################################

    short_params = {
        "short_condition_1_enable": True,  # RSI Overbought Exhaustion
        "short_condition_2_enable": True,  # BB Upper Rejection + Divergence
        "short_condition_3_enable": True,  # Volume Climax Reversal
        "short_condition_4_enable": True,  # EMA Breakdown Confirmation
        "short_condition_5_enable": True,  # Extreme Overbought Multi-Signal
    }

    #############################################################
    # SAFE-PUMP / SAFE-DIP THRESHOLDS FOR SHORTS
    #############################################################

    # Don't short if price already dumped recently (avoid catching falling knives going short)
    short_safe_dump_threshold = DecimalParameter(0.1, 0.4, default=0.20, space="buy", optimize=False, load=True)

    # Require minimum pump before shorting (we want to short the top, not the bottom)
    short_min_pump_24h = DecimalParameter(0.02, 0.15, default=0.05, space="buy", optimize=False, load=True)

    #############################################################
    # CONDITION 1: RSI OVERBOUGHT EXHAUSTION
    # RSI 5m > 80 AND RSI declining AND RSI 1h > 65 AND CTI > 0.9
    # AND close above EMA200 AND MFI > 80 AND volume declining
    #############################################################
    short_rsi_min_1 = DecimalParameter(76.0, 92.0, default=80.0, space="buy", optimize=False, load=True)
    short_rsi_1h_min_1 = DecimalParameter(58.0, 78.0, default=65.0, space="buy", optimize=False, load=True)
    short_cti_min_1 = DecimalParameter(0.80, 0.99, default=0.90, space="buy", optimize=False, load=True)
    short_mfi_min_1 = DecimalParameter(72.0, 95.0, default=80.0, space="buy", optimize=False, load=True)

    #############################################################
    # CONDITION 2: BB UPPER REJECTION + DIVERGENCE
    # Close above upper BB AND RSI > 75 AND prev close > prev upper BB
    # AND close < prev close (bearish engulf) AND CMF negative AND 1h RSI > 60
    #############################################################
    short_bb_mult_2 = DecimalParameter(1.00, 1.06, default=1.01, space="buy", optimize=False, load=True)
    short_rsi_min_2 = DecimalParameter(70.0, 88.0, default=75.0, space="buy", optimize=False, load=True)
    short_rsi_1h_min_2 = DecimalParameter(55.0, 75.0, default=60.0, space="buy", optimize=False, load=True)

    #############################################################
    # CONDITION 3: VOLUME CLIMAX REVERSAL
    # Volume > 3x 30-candle avg AND RSI > 82 AND RSI declining
    # AND StochRSI > 95 AND 1h RSI > 60 AND close > EMA100
    #############################################################
    short_vol_mult_3 = DecimalParameter(2.0, 5.0, default=3.0, space="buy", decimals=1, optimize=False, load=True)
    short_rsi_min_3 = DecimalParameter(78.0, 92.0, default=82.0, space="buy", optimize=False, load=True)
    short_stochrsi_min_3 = DecimalParameter(90.0, 99.0, default=95.0, space="buy", optimize=False, load=True)
    short_rsi_1h_min_3 = DecimalParameter(55.0, 75.0, default=60.0, space="buy", optimize=False, load=True)

    #############################################################
    # CONDITION 4: EMA BREAKDOWN CONFIRMATION
    # EMA26 crosses below EMA50 (not the fast/micro cross like 12/26)
    # AND close < EMA100 AND SMA200 declining AND RSI 1h < 55
    # AND moderi_96 == False (bearish elder ray)
    #############################################################
    short_rsi_1h_max_4 = DecimalParameter(40.0, 58.0, default=55.0, space="buy", optimize=False, load=True)

    #############################################################
    # CONDITION 5: EXTREME OVERBOUGHT MULTI-SIGNAL
    # RSI > 85 AND StochRSI fastk > 98 AND Williams %R > -5
    # AND CTI > 0.92 AND 1h RSI > 68
    # Ultra-selective: all oscillators pegged to ceiling simultaneously
    #############################################################
    short_rsi_min_5 = DecimalParameter(82.0, 95.0, default=85.0, space="buy", optimize=False, load=True)
    short_stochrsi_min_5 = DecimalParameter(95.0, 99.9, default=98.0, space="buy", optimize=False, load=True)
    short_wr_max_5 = DecimalParameter(-8.0, -1.0, default=-5.0, space="buy", optimize=False, load=True)
    short_cti_min_5 = DecimalParameter(0.88, 0.99, default=0.92, space="buy", optimize=False, load=True)
    short_rsi_1h_min_5 = DecimalParameter(60.0, 78.0, default=68.0, space="buy", optimize=False, load=True)

    #############################################################

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """Tiered stoploss for shorts. Conservative — let trailing handle profit side."""
        if current_profit > 0.03:
            return -0.015  # Lock 1.5% once 3% is reached
        elif current_profit > 0.0:
            return -1  # Use default stoploss / trailing
        elif current_profit > -0.03:
            return -1  # Allow recovery up to -3%
        elif current_profit > -0.05:
            return -0.055  # Tighten near hard stop
        else:
            return -0.04  # Emergency

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 3:
            return None
        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]

        # ===== PROFIT-TAKING EXITS =====
        # Take profit on RSI oversold (target reached for short)
        if (current_profit > 0.02) and (last_candle["rsi_14"] < 30):
            return "exit_short_rsi_oversold"

        # Take profit if RSI 1h goes oversold during our short
        rsi_1h_col = "rsi_14_1h_1h" if "rsi_14_1h_1h" in last_candle.index else "rsi_14_1h"
        if (current_profit > 0.015) and (last_candle.get(rsi_1h_col, 50) < 35):
            return "exit_short_rsi_1h_oversold"

        # ===== TRAILING PROFIT PROTECTION =====
        max_profit = ((trade.open_rate - trade.min_rate) / trade.open_rate) if trade.min_rate else 0
        # If we had > 3% profit and gave back > 1.5%, exit
        if (max_profit > 0.03) and (current_profit < max_profit - 0.015) and (current_profit > 0):
            return "exit_short_trail_protect"

        # ===== STALE TRADE EXIT =====
        # Cut losing shorts after 24h (shorts shouldn't be held long)
        if (current_profit < -0.02) and (current_time - timedelta(hours=24) > trade.open_date_utc):
            return "exit_short_stale_24h"

        # ===== REVERSAL DETECTION =====
        # BTC recovering while we're short — exit on small loss
        btc_dn_col = "btc_btc_not_downtrend_1h" if "btc_btc_not_downtrend_1h" in last_candle.index else None
        if btc_dn_col and last_candle.get(btc_dn_col, True):
            # BTC is no longer in downtrend
            if current_profit > -0.005:
                return "exit_short_btc_recovered"

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ===== BTC informative (1h) =====
        if self.has_BTC_info_tf:
            btc_info_tf = self.dp.get_pair_dataframe("BTC/USDT:USDT", self.info_timeframe)
            btc_info_tf["btc_rsi_14"] = ta.RSI(btc_info_tf, timeperiod=14)
            btc_info_tf["btc_ema_50"] = ta.EMA(btc_info_tf, timeperiod=50)
            btc_info_tf["btc_ema_200"] = ta.EMA(btc_info_tf, timeperiod=200)
            btc_info_tf["btc_sma_200"] = ta.SMA(btc_info_tf, timeperiod=200)
            # Strict downtrend: close < ema50 AND rsi < 50 AND sma200 declining
            btc_info_tf["btc_not_downtrend"] = (btc_info_tf["close"] > btc_info_tf["btc_ema_50"]) | (
                btc_info_tf["btc_rsi_14"] > 55
            )
            btc_info_tf["btc_downtrend_confirmed"] = (
                (btc_info_tf["close"] < btc_info_tf["btc_ema_50"])
                & (btc_info_tf["btc_rsi_14"] < 48)
                & (btc_info_tf["btc_sma_200"] < btc_info_tf["btc_sma_200"].shift(12))
            )
            ignore_columns = ["date", "open", "high", "low", "close", "volume"]
            btc_info_tf.rename(
                columns=lambda s: s if s in ignore_columns else "btc_" + s,
                inplace=True,
            )
            dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.info_timeframe, ffill=True)
            drop_columns = [
                (s + "_" + self.info_timeframe) for s in ["date", "open", "high", "low", "close", "volume"]
            ]
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        # ===== Pair informative 1h =====
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.info_timeframe)
        informative_1h["ema_50_1h"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_100_1h"] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h["ema_200_1h"] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h["sma_200_1h"] = ta.SMA(informative_1h, timeperiod=200)
        informative_1h["rsi_14_1h"] = ta.RSI(informative_1h, timeperiod=14)
        informative_1h["cmf_1h"] = chaikin_money_flow(informative_1h, 20)
        informative_1h["r_480_1h"] = williams_r(informative_1h, period=480)
        # 1h pump detection: how much has price risen in last 24 candles (24h)
        informative_1h["pump_24_1h"] = (informative_1h["close"] - informative_1h["close"].shift(24)) / informative_1h[
            "close"
        ].shift(24)
        # 1h dump detection: how much has price fallen in last 24 candles (24h)
        informative_1h["dump_24_1h"] = (informative_1h["close"].shift(24) - informative_1h["close"]) / informative_1h[
            "close"
        ].shift(24)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.info_timeframe, ffill=True)
        drop_columns = [(s + "_" + self.info_timeframe) for s in ["date"]]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        # ===== Normal TF (5m) indicators =====
        # Bollinger Bands 20,2
        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb20_2_low"] = bb_20_std2["lower"]
        dataframe["bb20_2_mid"] = bb_20_std2["mid"]
        dataframe["bb20_2_upp"] = bb_20_std2["upper"]

        # EMAs
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # SMAs
        dataframe["sma_30"] = ta.SMA(dataframe, timeperiod=30)
        dataframe["sma_75"] = ta.SMA(dataframe, timeperiod=75)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["sma_200_dec_20"] = dataframe["sma_200"] < dataframe["sma_200"].shift(20)

        # RSI
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe)

        # CMF
        dataframe["cmf"] = chaikin_money_flow(dataframe, 20)

        # Volume means
        dataframe["volume_mean_4"] = dataframe["volume"].rolling(4).mean()
        dataframe["volume_mean_30"] = dataframe["volume"].rolling(30).mean()

        # StochRSI 96
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=96, fastk_period=3, fastd_period=3, fastd_matype=0)
        dataframe["stochrsi_fastk_96"] = stochrsi["fastk"]
        dataframe["stochrsi_fastd_96"] = stochrsi["fastd"]

        # Williams %R
        dataframe["r_14"] = williams_r(dataframe, period=14)
        dataframe["r_480"] = williams_r(dataframe, period=480)

        # CTI
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)

        # Moderi 96
        dataframe["moderi_96"] = moderi(dataframe, 96)

        # EWO
        dataframe["ewo"] = ewo(dataframe, 50, 200)

        # ===== SAFE-PUMP / SAFE-DUMP COLUMNS =====
        # Detect if pair already dumped > threshold in last 24h on 1h
        # (don't short something that already crashed)
        dump_col = "dump_24_1h_1h" if "dump_24_1h_1h" in dataframe.columns else None
        if dump_col:
            dataframe["safe_to_short_dump"] = dataframe[dump_col] < self.short_safe_dump_threshold.value
        else:
            dataframe["safe_to_short_dump"] = True

        # Detect if pair pumped enough to justify a short (we want overextended, not flat)
        pump_col = "pump_24_1h_1h" if "pump_24_1h_1h" in dataframe.columns else None
        if pump_col:
            dataframe["has_recent_pump"] = dataframe[pump_col] > self.short_min_pump_24h.value
        else:
            dataframe["has_recent_pump"] = True

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""

        # ===== GLOBAL GATES — ALL conditions must pass these =====
        vol_filter = dataframe["volume"] > 0

        # BTC must be in downtrend (this is the primary filter that prevents overtrading)
        btc_dn_col = "btc_btc_downtrend_confirmed_1h"
        if btc_dn_col in dataframe.columns:
            btc_gate = dataframe[btc_dn_col] == True
        else:
            btc_gate = True  # Fallback if column missing

        # Safe-dump gate: don't short something that already dumped
        safe_dump_gate = dataframe["safe_to_short_dump"]

        # 1h RSI gate: don't short if 1h RSI is already oversold
        rsi_1h_col = "rsi_14_1h_1h" if "rsi_14_1h_1h" in dataframe.columns else "rsi_14_1h"
        rsi_1h_gate = dataframe[rsi_1h_col] > 40

        # Common global filter
        global_gate = vol_filter & btc_gate & safe_dump_gate & rsi_1h_gate

        # ===== CONDITION 1: RSI OVERBOUGHT EXHAUSTION =====
        # All oscillators show exhaustion + price above EMA200 + 1h confirms
        if self.short_params["short_condition_1_enable"]:
            c1 = (
                global_gate
                & (dataframe["rsi_14"] > self.short_rsi_min_1.value)
                & (dataframe["rsi_14"] < dataframe["rsi_14"].shift(1))  # RSI declining
                & (dataframe["rsi_14"].shift(1) > dataframe["rsi_14"].shift(2))  # Was rising, now turning
                & (dataframe["mfi"] > self.short_mfi_min_1.value)
                & (dataframe["cti"] > self.short_cti_min_1.value)
                & (dataframe["close"] > dataframe["ema_200"])
                & (dataframe[rsi_1h_col] > self.short_rsi_1h_min_1.value)
                & (dataframe["has_recent_pump"])
            )
            conditions.append(c1)
            dataframe.loc[c1, "enter_tag"] += "s_rsi_exhaust "

        # ===== CONDITION 2: BB UPPER REJECTION + DIVERGENCE =====
        # Close above upper BB, previous also above BB, now closing lower
        # CMF turning negative = money flow reversing + 1h RSI confirmation
        if self.short_params["short_condition_2_enable"]:
            c2 = (
                global_gate
                & (dataframe["close"] > dataframe["bb20_2_upp"] * self.short_bb_mult_2.value)
                & (dataframe["close"].shift(1) > dataframe["bb20_2_upp"].shift(1))  # Prev also above BB
                & (dataframe["close"] < dataframe["close"].shift(1))  # Bearish candle
                & (dataframe["rsi_14"] > self.short_rsi_min_2.value)
                & (dataframe["cmf"] < 0)  # Money flow negative
                & (dataframe[rsi_1h_col] > self.short_rsi_1h_min_2.value)
                & (dataframe["close"] > dataframe["ema_100"])
                & (dataframe["has_recent_pump"])
            )
            conditions.append(c2)
            dataframe.loc[c2, "enter_tag"] += "s_bb_reject "

        # ===== CONDITION 3: VOLUME CLIMAX REVERSAL =====
        # Massive volume spike + extreme RSI + StochRSI pegged
        # This is a blow-off top pattern
        if self.short_params["short_condition_3_enable"]:
            c3 = (
                global_gate
                & (dataframe["volume"] > dataframe["volume_mean_30"] * self.short_vol_mult_3.value)
                & (dataframe["rsi_14"] > self.short_rsi_min_3.value)
                & (dataframe["rsi_14"] < dataframe["rsi_14"].shift(1))  # RSI turning down
                & (dataframe["stochrsi_fastk_96"] > self.short_stochrsi_min_3.value)
                & (dataframe[rsi_1h_col] > self.short_rsi_1h_min_3.value)
                & (dataframe["close"] > dataframe["ema_100"])
                & (dataframe["close"] > dataframe["bb20_2_upp"])  # Above upper BB
            )
            conditions.append(c3)
            dataframe.loc[c3, "enter_tag"] += "s_vol_climax "

        # ===== CONDITION 4: EMA BREAKDOWN CONFIRMATION =====
        # Slower EMA cross (26/50) for confirmed trend change
        # Plus SMA200 declining + elder ray bearish + 1h weak
        if self.short_params["short_condition_4_enable"]:
            c4 = (
                global_gate
                & qtpylib.crossed_below(dataframe["ema_26"], dataframe["ema_50"])
                & (dataframe["close"] < dataframe["ema_100"])
                & (dataframe["sma_200_dec_20"])  # SMA200 declining
                & (dataframe["moderi_96"] == False)  # Elder ray bearish
                & (dataframe[rsi_1h_col] < self.short_rsi_1h_max_4.value)  # 1h not too strong
                & (dataframe["rsi_14"] > 35)  # Not already oversold on 5m
                & (dataframe["rsi_14"] < 55)  # But not overbought either
            )
            conditions.append(c4)
            dataframe.loc[c4, "enter_tag"] += "s_ema_break "

        # ===== CONDITION 5: EXTREME OVERBOUGHT MULTI-SIGNAL =====
        # Every oscillator pegged to ceiling: RSI + StochRSI + WR + CTI + 1h RSI
        # Ultra-rare but high-conviction
        if self.short_params["short_condition_5_enable"]:
            c5 = (
                global_gate
                & (dataframe["rsi_14"] > self.short_rsi_min_5.value)
                & (dataframe["stochrsi_fastk_96"] > self.short_stochrsi_min_5.value)
                & (dataframe["r_480"] > self.short_wr_max_5.value)
                & (dataframe["cti"] > self.short_cti_min_5.value)
                & (dataframe[rsi_1h_col] > self.short_rsi_1h_min_5.value)
                & (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["bb20_2_upp"])
            )
            conditions.append(c5)
            dataframe.loc[c5, "enter_tag"] += "s_extreme_ob "

        # ===== COMBINE =====
        if conditions:
            dataframe.loc[:, "enter_short"] = reduce(lambda x, y: x | y, conditions)
        else:
            dataframe.loc[:, "enter_short"] = 0

        dataframe.loc[:, "enter_long"] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Signal-based exits for shorts.
        Kept minimal — ROI, trailing stop, and custom_exit do most of the work.
        """
        conditions = []

        # Exit short: RSI deeply oversold + below lower BB (target reached)
        conditions.append((dataframe["rsi_14"] < 25) & (dataframe["close"] < dataframe["bb20_2_low"]))

        # Exit short: EMA golden cross on meaningful timeframe
        conditions.append(qtpylib.crossed_above(dataframe["ema_26"], dataframe["ema_50"]) & (dataframe["rsi_14"] < 50))

        if conditions:
            dataframe.loc[:, "exit_short"] = reduce(lambda x, y: x | y, conditions)
        else:
            dataframe.loc[:, "exit_short"] = 0

        dataframe.loc[:, "exit_long"] = 0

        return dataframe


# ===== HELPER FUNCTIONS =====


def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    mfv = ((dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])) / (
        dataframe["high"] - dataframe["low"]
    )
    mfv = mfv.fillna(0.0)
    mfv *= dataframe["volume"]
    cmf = mfv.rolling(n, min_periods=0).sum() / dataframe["volume"].rolling(n, min_periods=0).sum()
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name="cmf")


def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name="{0} Williams %R".format(period),
    )
    return WR * -100


def vwma(dataframe: DataFrame, length: int = 10) -> Series:
    pv = dataframe["close"] * dataframe["volume"]
    return Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe["volume"], timeperiod=length))


def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)


def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    return (sma1 - sma2) / dataframe["close"] * 100
