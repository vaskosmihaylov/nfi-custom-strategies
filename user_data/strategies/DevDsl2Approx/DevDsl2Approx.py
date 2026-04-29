import logging
from datetime import datetime

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import talib.abstract as ta
import technical.indicators as ftt
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, stoploss_from_open
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def vwap_bands(dataframe: DataFrame, window: int = 20, num_of_std: float = 1.0):
    df = dataframe.copy()
    df["vwap"] = qtpylib.rolling_vwap(df, window=window)
    rolling_std = df["vwap"].rolling(window=window).std()
    df["vwap_low"] = df["vwap"] - (rolling_std * num_of_std)
    df["vwap_high"] = df["vwap"] + (rolling_std * num_of_std)
    return df["vwap_low"], df["vwap"], df["vwap_high"]


def vwma(series: Series, volume: Series, length: int) -> Series:
    vol_sum = volume.rolling(length).sum()
    return (series * volume).rolling(length).sum() / vol_sum.replace(0, np.nan)


def smma(series: Series, period: int) -> Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if len(series) < period:
        return result

    result.iloc[period - 1] = series.iloc[:period].mean()
    for idx in range(period, len(series)):
        result.iloc[idx] = ((result.iloc[idx - 1] * (period - 1)) + series.iloc[idx]) / period
    return result


def dsl_rsi_level(rsi: Series, length: int) -> Series:
    upper = pd.Series(np.nan, index=rsi.index, dtype=float)
    lower = pd.Series(np.nan, index=rsi.index, dtype=float)
    alpha = 2.0 / (length + 1.0)
    upper_prev = 50.0
    lower_prev = 50.0

    for idx, value in enumerate(rsi.fillna(50.0)):
        if value > 50.0:
            upper_prev = upper_prev + alpha * (value - upper_prev)
        if value < 50.0:
            lower_prev = lower_prev + alpha * (value - lower_prev)
        upper.iloc[idx] = upper_prev
        lower.iloc[idx] = lower_prev

    return upper - lower


class DevDsl2Approx(IStrategy):
    """
    A public-fingerprint reconstruction of `Dev_dsl2` tuned for local Bybit futures backtesting.

    The strategy keeps the original indicator family but now runs as a balanced long/short
    5m futures strategy so it can be evaluated against the available 2026 Bybit USDT dataset.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = True
    startup_candle_count = 240
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    BUY_DEFAULTS = {
        "bb_factor": 0.992,
        "vwap_factor": 0.998,
        "rsi_14_max": 34,
        "rsi_slow_max": 42,
        "rsi_84_max": 58,
        "dc_percent_max": 0.32,
        "bbdelta_close_min": 0.011,
        "close_delta_min": 0.006,
        "dsl_delta_min": 0.2,
        "goodloss_max": 0.06,
        "donchian_room_min": 0.010,
        "volume_factor": 0.80,
        "smma_length": 14,
        "dc_length": 20,
        "trend_relax_factor": 0.975,
        "donchian_smma_factor": 0.97,
    }

    SELL_DEFAULTS = {
        "rsi_exit_min": 62,
        "fastk_exit_min": 78,
        "high_offset": 1.012,
        "vwap_factor": 1.002,
        "profit_fade_min": 0.022,
        "trend_fail_min": 0.012,
        "band_exhaustion_rsi_mfi_rmi_min": 60,
    }

    SHORT_DEFAULTS = {
        "bb_factor": 1.000,
        "vwap_factor": 1.000,
        "rsi_14_min": 70,
        "rsi_slow_min": 56,
        "rsi_84_min": 50,
        "dc_percent_min": 0.85,
        "dsl_delta_max": 0.0,
        "goodrise_max": 0.12,
        "donchian_room_min": 0.010,
        "volume_factor": 0.80,
        "trend_relax_factor": 0.995,
        "donchian_smma_factor": 1.01,
        "cover_rsi_max": 38,
        "cover_fastk_max": 22,
        "low_offset": 0.988,
        "cover_vwap_factor": 0.998,
        "band_compression_rsi_mfi_rmi_max": 40,
    }

    RISK_DEFAULTS = {
        "hard_stoploss": -0.08,
        "profit_floor_1": 0.016,
        "profit_floor_2": 0.06,
        "stop_1": 0.012,
        "stop_2": 0.04,
        "time_stop_minutes": 16 * 60,
        "time_stop_profit": 0.002,
    }

    buy_bb_factor = DecimalParameter(
        0.985,
        0.999,
        default=BUY_DEFAULTS["bb_factor"],
        decimals=3,
        space="buy",
        optimize=True,
    )
    buy_vwap_factor = DecimalParameter(
        0.990,
        1.000,
        default=BUY_DEFAULTS["vwap_factor"],
        decimals=3,
        space="buy",
        optimize=True,
    )
    buy_rsi_14_max = IntParameter(24, 40, default=BUY_DEFAULTS["rsi_14_max"], space="buy", optimize=True)
    buy_rsi_slow_max = IntParameter(30, 50, default=BUY_DEFAULTS["rsi_slow_max"], space="buy", optimize=True)
    buy_rsi_84_max = IntParameter(45, 70, default=BUY_DEFAULTS["rsi_84_max"], space="buy", optimize=True)
    buy_dc_percent_max = DecimalParameter(
        0.12,
        0.45,
        default=BUY_DEFAULTS["dc_percent_max"],
        decimals=2,
        space="buy",
        optimize=True,
    )
    buy_bbdelta_close_min = DecimalParameter(
        0.004,
        0.03,
        default=BUY_DEFAULTS["bbdelta_close_min"],
        decimals=3,
        space="buy",
        optimize=True,
    )
    buy_close_delta_min = DecimalParameter(
        0.001,
        0.02,
        default=BUY_DEFAULTS["close_delta_min"],
        decimals=3,
        space="buy",
        optimize=True,
    )
    buy_dsl_delta_min = DecimalParameter(
        -2.0,
        5.0,
        default=BUY_DEFAULTS["dsl_delta_min"],
        decimals=1,
        space="buy",
        optimize=True,
    )
    buy_goodloss_max = DecimalParameter(
        0.02,
        0.12,
        default=BUY_DEFAULTS["goodloss_max"],
        decimals=2,
        space="buy",
        optimize=True,
    )
    buy_donchian_room_min = DecimalParameter(
        0.004,
        0.04,
        default=BUY_DEFAULTS["donchian_room_min"],
        decimals=3,
        space="buy",
        optimize=True,
    )
    short_bb_factor = DecimalParameter(
        1.001,
        1.03,
        default=SHORT_DEFAULTS["bb_factor"],
        decimals=3,
        space="sell",
        optimize=True,
    )
    short_vwap_factor = DecimalParameter(
        1.000,
        1.02,
        default=SHORT_DEFAULTS["vwap_factor"],
        decimals=3,
        space="sell",
        optimize=True,
    )
    short_rsi_14_min = IntParameter(55, 86, default=SHORT_DEFAULTS["rsi_14_min"], space="sell", optimize=True)
    short_rsi_slow_min = IntParameter(45, 80, default=SHORT_DEFAULTS["rsi_slow_min"], space="sell", optimize=True)
    short_rsi_84_min = IntParameter(40, 75, default=SHORT_DEFAULTS["rsi_84_min"], space="sell", optimize=True)
    short_dc_percent_min = DecimalParameter(
        0.55,
        0.92,
        default=SHORT_DEFAULTS["dc_percent_min"],
        decimals=2,
        space="sell",
        optimize=True,
    )
    short_dsl_delta_max = DecimalParameter(
        -5.0,
        2.0,
        default=SHORT_DEFAULTS["dsl_delta_max"],
        decimals=1,
        space="sell",
        optimize=True,
    )
    short_goodrise_max = DecimalParameter(
        0.03,
        0.14,
        default=SHORT_DEFAULTS["goodrise_max"],
        decimals=2,
        space="sell",
        optimize=True,
    )
    short_donchian_room_min = DecimalParameter(
        0.004,
        0.04,
        default=SHORT_DEFAULTS["donchian_room_min"],
        decimals=3,
        space="sell",
        optimize=True,
    )

    sell_rsi_exit_min = IntParameter(55, 78, default=SELL_DEFAULTS["rsi_exit_min"], space="sell", optimize=True)
    sell_fastk_exit_min = IntParameter(68, 95, default=SELL_DEFAULTS["fastk_exit_min"], space="sell", optimize=True)
    sell_high_offset = DecimalParameter(
        1.004,
        1.03,
        default=SELL_DEFAULTS["high_offset"],
        decimals=3,
        space="sell",
        optimize=True,
    )
    sell_vwap_factor = DecimalParameter(
        1.000,
        1.02,
        default=SELL_DEFAULTS["vwap_factor"],
        decimals=3,
        space="sell",
        optimize=True,
    )

    plot_config = {
        "main_plot": {
            "vwap_low": {"color": "#7a8ca5"},
            "vwap_upperband": {"color": "#d17b5f"},
            "dc_upper": {"color": "#47a447"},
            "dc_lower": {"color": "#c95f5f"},
            "tenkan_sen": {"color": "#f2b134"},
            "high_offset_sma": {"color": "#b06ab3"},
            "low_offset_sma": {"color": "#3a7f55"},
        },
        "subplots": {
            "Momentum": {
                "rsi_14": {"color": "#4b8bbe"},
                "rsi_slow": {"color": "#306998"},
                "dsl_lvl": {"color": "#ff7f0e"},
            },
            "StochRSI": {
                "fastk_rsi": {"color": "#2ca02c"},
                "fastd_rsi": {"color": "#d62728"},
            },
        },
    }

    def version(self) -> str:
        return "2026.04.23-devdsl2-bybit-v2"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]

        dataframe["volume_mean_4"] = dataframe["volume"].rolling(4).mean().shift(1)
        dataframe["sma_15"] = ta.SMA(dataframe, timeperiod=15)
        dataframe["high_offset_sma"] = dataframe["sma_15"] * self.sell_high_offset.value
        dataframe["low_offset_sma"] = dataframe["sma_15"] * self.SHORT_DEFAULTS["low_offset"]
        dataframe["vwma"] = vwma(dataframe["close"], dataframe["volume"], 20)
        dataframe["smma_smma_length_value"] = smma(dataframe["close"], self.BUY_DEFAULTS["smma_length"])

        dataframe["trend_close_5m"] = dataframe["close"]
        dataframe["trend_close_15m"] = ta.EMA(dataframe["close"], timeperiod=3)
        dataframe["trend_close_30m"] = ta.EMA(dataframe["close"], timeperiod=6)
        dataframe["trend_close_1h"] = ta.EMA(dataframe["close"], timeperiod=12)
        dataframe["trend_close_2h"] = ta.EMA(dataframe["close"], timeperiod=24)
        dataframe["trend_close_4h"] = ta.EMA(dataframe["close"], timeperiod=48)
        dataframe["trend_close_6h"] = ta.EMA(dataframe["close"], timeperiod=72)
        dataframe["trend_open_6h"] = ta.EMA(dataframe["open"], timeperiod=72)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband2"] = bollinger["lower"]
        dataframe["bb_middleband2"] = bollinger["mid"]
        dataframe["bb_upperband2"] = bollinger["upper"]
        dataframe["bbdelta"] = (dataframe["bb_middleband2"] - dataframe["bb_lowerband2"]).abs()
        dataframe["close_delta2"] = (dataframe["close"] - dataframe["close"].shift()).abs()
        dataframe["lower_far"] = (dataframe["bb_lowerband2"] - dataframe["close"]) / dataframe["close"]
        dataframe["upper_near"] = (dataframe["bb_upperband2"] - dataframe["close"]) / dataframe["close"]
        dataframe["upper_far"] = (dataframe["close"] - dataframe["bb_upperband2"]) / dataframe["close"]
        dataframe["lower_near"] = (dataframe["close"] - dataframe["bb_lowerband2"]) / dataframe["close"]

        vwap_low, vwap_mid, vwap_high = vwap_bands(dataframe, 20, 1)
        dataframe["vwap_low"] = vwap_low
        dataframe["vwap_high"] = vwap_high
        dataframe["vwap_upperband"] = vwap_high
        dataframe["vwap_middleband"] = vwap_mid

        dc_length = self.BUY_DEFAULTS["dc_length"]
        dataframe["dc_upper"] = dataframe["high"].shift(1).rolling(dc_length).max()
        dataframe["dc_lower"] = dataframe["low"].shift(1).rolling(dc_length).min()
        dataframe["dc_middle"] = (dataframe["dc_upper"] + dataframe["dc_lower"]) / 2.0
        dc_range = (dataframe["dc_upper"] - dataframe["dc_lower"]).replace(0, np.nan)
        dataframe["dc_percent"] = (dataframe["close"] - dataframe["dc_lower"]) / dc_range
        dataframe["dc_chf"] = (dataframe["close"] - dataframe["dc_middle"]) / dataframe["dc_middle"].replace(0, np.nan)

        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)
        dataframe["rsi_84"] = ta.RSI(dataframe, timeperiod=84)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        rsi_min = dataframe["rsi_14"].rolling(14).min()
        rsi_max = dataframe["rsi_14"].rolling(14).max()
        stoch_rsi = 100 * (dataframe["rsi_14"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        dataframe["fastk_rsi"] = stoch_rsi.rolling(3).mean()
        dataframe["fastd_rsi"] = dataframe["fastk_rsi"].rolling(3).mean()

        rmi_source = dataframe["close"] - dataframe["close"].shift(4)
        dataframe["rmi"] = ta.RSI(rmi_source.fillna(0), timeperiod=14)
        dataframe["rsi_mfi_rmi_length"] = (dataframe["rsi_14"] + dataframe["mfi"] + dataframe["rmi"]) / 3.0

        momentum = dataframe["close"] - dataframe["close"].shift(5)
        positive_mom = momentum.clip(lower=0)
        negative_mom = (-momentum).clip(lower=0)
        pmom = positive_mom.rolling(3).mean()
        nmom = negative_mom.rolling(3).mean()
        dataframe["positive_mom_pmom_nmom"] = ((positive_mom > negative_mom) & (pmom > nmom)).astype(int)
        dataframe["positive_pmom_nmom_prev"] = (
            (pmom.shift(1) > nmom.shift(1)) & (pmom.shift(2) > nmom.shift(2))
        ).astype(int)
        dataframe["negative_mom_pmom_nmom"] = ((negative_mom > positive_mom) & (nmom > pmom)).astype(int)
        dataframe["negative_pmom_nmom_prev"] = (
            (nmom.shift(1) > pmom.shift(1)) & (nmom.shift(2) > pmom.shift(2))
        ).astype(int)

        ichimoku = ftt.ichimoku(
            dataframe,
            conversion_line_period=20,
            base_line_periods=60,
            laggin_span=120,
            displacement=30,
        )
        dataframe["tenkan_sen"] = ichimoku["tenkan_sen"]
        dataframe["cloud_red"] = ichimoku["cloud_red"].astype(int)

        dataframe["dsl_lvl"] = dsl_rsi_level(dataframe["rsi_14"], 7)
        dataframe["strike_strike"] = np.sign(dataframe["close"].diff()).rolling(3).sum()
        dataframe["averimpet_mma_length_value"] = (
            (dataframe["close"] - dataframe["smma_smma_length_value"]).abs()
            / dataframe["smma_smma_length_value"].replace(0, np.nan)
        )

        rsi_mean = dataframe["rsi_14"].rolling(20).mean()
        rsi_std = dataframe["rsi_14"].rolling(20).std()
        dataframe["dev_ma_period_rsi_period_stdev_multiplier"] = rsi_mean + (rsi_std * 1.6)
        dataframe["disp_up_ma_period_rsi_period_stdev_multiplier_dispersion"] = (
            rsi_std / rsi_mean.abs().replace(0, np.nan)
        )

        dataframe["goodloss"] = ((dataframe["trend_open_6h"] - dataframe["close"]) / dataframe["close"]).clip(lower=0)
        dataframe["goodrise"] = ((dataframe["close"] - dataframe["trend_open_6h"]) / dataframe["close"]).clip(lower=0)
        dataframe["dsl_delta"] = dataframe["dsl_lvl"] - dataframe["dsl_lvl"].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        general_guard = (
            (dataframe["volume"] > 0)
            & (dataframe["volume"] > dataframe["volume_mean_4"] * self.BUY_DEFAULTS["volume_factor"])
            & (dataframe["goodloss"] < self.buy_goodloss_max.value)
            & (dataframe["upper_near"] > self.buy_donchian_room_min.value)
        )

        trend_guard = (
            (dataframe["trend_close_6h"] > dataframe["trend_open_6h"] * self.BUY_DEFAULTS["trend_relax_factor"])
            | ((dataframe["close"] > dataframe["tenkan_sen"]) & (dataframe["cloud_red"] == 0))
        )
        bear_guard = (
            (
                (dataframe["trend_close_6h"] < dataframe["trend_open_6h"] * self.SHORT_DEFAULTS["trend_relax_factor"])
                & (dataframe["close"] < dataframe["trend_open_6h"])
            )
            | ((dataframe["close"] < dataframe["tenkan_sen"]) & (dataframe["cloud_red"] == 1))
        )

        deep_reversion = (
            (dataframe["close"] < dataframe["vwap_low"] * self.buy_vwap_factor.value)
            & (dataframe["close"] < dataframe["bb_lowerband2"] * self.buy_bb_factor.value)
            & ((dataframe["bbdelta"] / dataframe["close"]) > self.buy_bbdelta_close_min.value)
            & ((dataframe["close_delta2"] / dataframe["close"]) > self.buy_close_delta_min.value)
            & (dataframe["rsi_14"] < self.buy_rsi_14_max.value)
            & (dataframe["rsi_slow"] < self.buy_rsi_slow_max.value)
            & (dataframe["rsi_84"] < self.buy_rsi_84_max.value)
            & (dataframe["dc_percent"] < self.buy_dc_percent_max.value)
            & (dataframe["fastk_rsi"] > dataframe["fastd_rsi"])
            & (dataframe["dsl_delta"] > self.buy_dsl_delta_min.value)
            & (dataframe["positive_mom_pmom_nmom"] == 1)
        )

        donchian_reclaim = (
            (dataframe["close"] <= dataframe["dc_lower"] * 1.01)
            & (dataframe["dc_chf"] < -0.01)
            & (dataframe["lower_far"] < -0.005)
            & (dataframe["fastk_rsi"] > dataframe["fastd_rsi"])
            & (dataframe["positive_pmom_nmom_prev"] == 1)
            & (dataframe["strike_strike"] <= 0)
            & (dataframe["dsl_delta"] > 0)
            & (
                dataframe["close"]
                > dataframe["smma_smma_length_value"] * self.BUY_DEFAULTS["donchian_smma_factor"]
            )
        )
        short_general_guard = (
            (dataframe["volume"] > 0)
            & (dataframe["volume"] > dataframe["volume_mean_4"] * self.SHORT_DEFAULTS["volume_factor"])
            & (dataframe["goodrise"] < self.short_goodrise_max.value)
        )
        short_reversion = (
            (dataframe["close"] > dataframe["vwap_high"] * self.short_vwap_factor.value)
            & (dataframe["close"] > dataframe["bb_upperband2"] * self.short_bb_factor.value)
            & (dataframe["upper_far"] > 0)
            & (dataframe["rsi_14"] > self.short_rsi_14_min.value)
            & (dataframe["rsi_slow"] > self.short_rsi_slow_min.value)
            & (dataframe["rsi_84"] > self.short_rsi_84_min.value)
            & (dataframe["dc_percent"] > self.short_dc_percent_min.value)
            & (dataframe["fastk_rsi"] < dataframe["fastd_rsi"])
            & (dataframe["dsl_delta"] < self.short_dsl_delta_max.value)
            & (dataframe["close"] > dataframe["trend_close_30m"])
        )
        donchian_short = (
            (dataframe["close"] >= dataframe["dc_upper"] * 0.99)
            & (dataframe["dc_chf"] > 0.01)
            & (dataframe["upper_far"] > 0.005)
            & (dataframe["fastk_rsi"] < dataframe["fastd_rsi"])
            & (dataframe["strike_strike"] >= 0)
            & (dataframe["dsl_delta"] < 0)
            & (
                dataframe["close"]
                > dataframe["smma_smma_length_value"] * self.SHORT_DEFAULTS["donchian_smma_factor"]
            )
        )
        entry_dsl_vwap = general_guard & trend_guard & deep_reversion
        entry_donchian = general_guard & donchian_reclaim
        entry_short_reversion = short_general_guard & bear_guard & short_reversion
        entry_short_donchian = short_general_guard & bear_guard & donchian_short

        dataframe.loc[entry_dsl_vwap, ["enter_long", "enter_tag"]] = (1, "dsl_vwap_reclaim")
        dataframe.loc[entry_donchian, ["enter_long", "enter_tag"]] = (1, "donchian_dsl_reclaim")
        dataframe.loc[entry_short_reversion, ["enter_short", "enter_tag"]] = (1, "dsl_vwap_short")
        dataframe.loc[entry_short_donchian, ["enter_short", "enter_tag"]] = (1, "donchian_dsl_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        vwap_extension = (
            (dataframe["close"] > dataframe["vwap_upperband"] * self.sell_vwap_factor.value)
            & (dataframe["rsi_14"] > self.sell_rsi_exit_min.value)
            & (dataframe["fastk_rsi"] > self.sell_fastk_exit_min.value)
        )

        band_exhaustion = (
            (dataframe["close"] > dataframe["high_offset_sma"])
            & (dataframe["close"] > dataframe["bb_upperband2"])
            & (dataframe["fastk_rsi"] < dataframe["fastd_rsi"])
            & (dataframe["rsi_mfi_rmi_length"] > self.SELL_DEFAULTS["band_exhaustion_rsi_mfi_rmi_min"])
        )

        donchian_take = (
            (dataframe["close"] > dataframe["dc_upper"])
            & (dataframe["rsi_14"] > self.sell_rsi_exit_min.value)
            & (dataframe["dsl_lvl"] < dataframe["dsl_lvl"].shift(1))
        )
        vwap_cover = (
            (dataframe["close"] < dataframe["vwap_low"] * self.SHORT_DEFAULTS["cover_vwap_factor"])
            & (dataframe["rsi_14"] < self.SHORT_DEFAULTS["cover_rsi_max"])
            & (dataframe["fastk_rsi"] < self.SHORT_DEFAULTS["cover_fastk_max"])
            & (dataframe["dsl_lvl"] > dataframe["dsl_lvl"].shift(1))
        )
        band_compression = (
            (dataframe["close"] < dataframe["low_offset_sma"])
            & (dataframe["close"] < dataframe["bb_lowerband2"])
            & (dataframe["fastk_rsi"] > dataframe["fastd_rsi"])
            & (dataframe["dc_percent"] < 0.2)
            & (dataframe["rsi_mfi_rmi_length"] < self.SHORT_DEFAULTS["band_compression_rsi_mfi_rmi_max"])
        )
        donchian_cover = (
            (dataframe["close"] < dataframe["dc_lower"])
            & (dataframe["rsi_14"] < self.SHORT_DEFAULTS["cover_rsi_max"])
            & (dataframe["dsl_lvl"] > dataframe["dsl_lvl"].shift(1))
        )

        dataframe.loc[vwap_extension, ["exit_long", "exit_tag"]] = (1, "vwap_extension")
        dataframe.loc[band_exhaustion, ["exit_long", "exit_tag"]] = (1, "band_exhaustion")
        dataframe.loc[donchian_take, ["exit_long", "exit_tag"]] = (1, "donchian_take")
        dataframe.loc[vwap_cover, ["exit_short", "exit_tag"]] = (1, "vwap_cover")
        dataframe.loc[band_compression, ["exit_short", "exit_tag"]] = (1, "band_compression")
        dataframe.loc[donchian_cover, ["exit_short", "exit_tag"]] = (1, "donchian_cover")

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        hard_stoploss = self.RISK_DEFAULTS["hard_stoploss"]
        profit_floor_1 = self.RISK_DEFAULTS["profit_floor_1"]
        profit_floor_2 = self.RISK_DEFAULTS["profit_floor_2"]
        stop_1 = self.RISK_DEFAULTS["stop_1"]
        stop_2 = self.RISK_DEFAULTS["stop_2"]

        if current_profit > profit_floor_2:
            stop_profit = stop_2 + (current_profit - profit_floor_2)
        elif current_profit > profit_floor_1:
            stop_profit = stop_1 + (
                (current_profit - profit_floor_1)
                * (stop_2 - stop_1)
                / (profit_floor_2 - profit_floor_1)
            )
        else:
            stop_profit = hard_stoploss

        if stop_profit >= current_profit:
            return -0.99

        return stoploss_from_open(
            stop_profit,
            current_profit,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        if self.dp is None:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        previous_candle = dataframe.iloc[-2] if len(dataframe) > 1 else last_candle
        held_minutes = (current_time - trade.open_date_utc).total_seconds() / 60.0

        if trade.is_short:
            if (
                current_profit > self.SELL_DEFAULTS["trend_fail_min"]
                and last_candle["cloud_red"] == 0
                and last_candle["close"] > last_candle["trend_open_6h"]
            ):
                return "trend_fail_short"

            if (
                current_profit > self.SELL_DEFAULTS["profit_fade_min"]
                and last_candle["dsl_lvl"] > previous_candle["dsl_lvl"]
            ):
                return "profit_fade_short"

            if (
                held_minutes > self.RISK_DEFAULTS["time_stop_minutes"]
                and current_profit > self.RISK_DEFAULTS["time_stop_profit"]
                and last_candle["close"] > last_candle["trend_close_6h"]
            ):
                return "time_stop_short"
        else:
            if (
                current_profit > self.SELL_DEFAULTS["trend_fail_min"]
                and last_candle["cloud_red"] == 1
                and last_candle["close"] < last_candle["trend_open_6h"]
            ):
                return "trend_fail"

            if (
                current_profit > self.SELL_DEFAULTS["profit_fade_min"]
                and last_candle["dsl_lvl"] < previous_candle["dsl_lvl"]
            ):
                return "profit_fade"

            if (
                held_minutes > self.RISK_DEFAULTS["time_stop_minutes"]
                and current_profit > self.RISK_DEFAULTS["time_stop_profit"]
                and last_candle["close"] < last_candle["trend_close_6h"]
            ):
                return "time_stop"

        return None
