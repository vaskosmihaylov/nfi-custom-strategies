# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# --- Do not remove these libs ---
import datetime
from typing import List, Tuple
import numpy as np  # noqa
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.persistence import Trade
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque


class HarmonicDivergence_Shorts(IStrategy):
    """
    HarmonicDivergence_Shorts Strategy

    A shorts-only variant of HarmonicDivergence_fix, designed to profit from
    bearish divergences detected across 11 momentum indicators.

    Entry Conditions:
    - Bearish divergence detected (price making higher highs, indicators making lower highs)
    - Two bands volatility filter (Keltner channel)
    - Volume > 0

    Exit Conditions:
    - ROI: 0.7%
    - Stop Loss: -2%
    - Custom stoploss: ATR-based (entry candle high + ATR)

    Key Differences from Long Strategy:
    - Uses bearish divergences instead of bullish
    - Custom stoploss inverted: high + ATR (above entry) instead of low - ATR
    - Max 4 short positions via confirm_trade_entry()
    - 3x leverage via leverage() callback

    Version: 1.0.0
    """
    INTERFACE_VERSION = 3
    can_short = True

    minimal_roi = {"0": 0.007}
    stoploss = -0.02
    use_custom_stoploss = False

    trailing_stop = False
    timeframe = "15m"
    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30
    max_short_trades = 4

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}
    plot_config = None

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative = dataframe
        # Momentum Indicators
        informative["rsi"] = ta.RSI(informative)
        informative["stoch"] = ta.STOCH(informative)["slowk"]
        informative["roc"] = ta.ROC(informative)
        informative["uo"] = ta.ULTOSC(informative)
        informative["ao"] = qtpylib.awesome_oscillator(informative)
        informative["macd"] = ta.MACD(informative)["macd"]
        informative["cci"] = ta.CCI(informative)
        informative["cmf"] = chaikin_money_flow(informative, 20)
        informative["obv"] = ta.OBV(informative)
        informative["mfi"] = ta.MFI(informative)
        informative["adx"] = ta.ADX(informative)
        informative["atr"] = qtpylib.atr(informative, window=14, exp=False)
        # Keltner Channel
        keltner = emaKeltner(informative)
        informative["kc_upperband"] = keltner["upper"]
        informative["kc_middleband"] = keltner["mid"]
        informative["kc_lowerband"] = keltner["lower"]
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(informative), window=20, stds=2
        )
        informative["bollinger_upperband"] = bollinger["upper"]
        informative["bollinger_lowerband"] = bollinger["lower"]
        # EMA
        informative["ema9"] = ta.EMA(informative, timeperiod=9)
        informative["ema20"] = ta.EMA(informative, timeperiod=20)
        informative["ema50"] = ta.EMA(informative, timeperiod=50)
        informative["ema200"] = ta.EMA(informative, timeperiod=200)
        # Pivot Points
        pivots = pivot_points(informative)
        informative["pivot_lows"] = pivots["pivot_lows"]
        informative["pivot_highs"] = pivots["pivot_highs"]
        # Divergences
        initialize_divergences_lists(informative)
        add_divergences(informative, "rsi")
        add_divergences(informative, "stoch")
        add_divergences(informative, "roc")
        add_divergences(informative, "uo")
        add_divergences(informative, "ao")
        add_divergences(informative, "macd")
        add_divergences(informative, "cci")
        add_divergences(informative, "cmf")
        add_divergences(informative, "obv")
        add_divergences(informative, "mfi")
        add_divergences(informative, "adx")
        HarmonicDivergence_Shorts.plot_config = (
            PlotConfig().add_total_divergences_in_config(dataframe).config
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Short on bearish divergences (price higher highs, indicator lower highs)
        dataframe.loc[
            (dataframe[resample("total_bearish_divergences")].shift() > 0)
            & two_bands_check(dataframe)
            & (dataframe["volume"] > 0),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["volume"] > 0, "exit_short"] = 0
        return dataframe

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: str,
        side: str, **kwargs,
    ) -> bool:
        if side == "long":
            return False
        short_count = sum(
            1 for t in Trade.get_trades_proxy(is_open=True) if t.is_short
        )
        if short_count >= self.max_short_trades:
            return False
        return True

    def custom_exit(
        self, pair: str, trade: "Trade", current_time: "datetime",
        current_rate: float, current_profit: float, **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        takeprofit = 0
        for i in range(1, len(dataframe["close"])):
            if (
                dataframe.iloc[-i]["date"]
                .to_pydatetime()
                .replace(tzinfo=datetime.timezone.utc)
                == trade.open_date_utc
            ):
                entry_candle = dataframe.iloc[-i - 1].squeeze()
                # For shorts, take profit is below entry: low - ATR
                takeprofit = entry_candle[resample("low")] - entry_candle[resample("atr")]
                break

    def leverage(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_leverage: float, max_leverage: float, entry_tag: str,
        side: str, **kwargs,
    ) -> float:
        return 3.0

    def custom_stoploss(
        self, pair: str, trade: "Trade", current_time: datetime,
        current_rate: float, current_profit: float, **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        stoploss = 0
        for i in range(1, len(dataframe["close"])):
            if (
                dataframe.iloc[-i]["date"]
                .to_pydatetime()
                .replace(tzinfo=datetime.timezone.utc)
                == trade.open_date_utc
            ):
                entry_candle = dataframe.iloc[-i - 1].squeeze()
                # For shorts, stoploss is above entry: high + ATR
                stoploss = entry_candle[resample("high")] + entry_candle[resample("atr")]
                break
        # For shorts, if stoploss price is above current rate, calculate distance
        if stoploss > current_rate:
            return -(stoploss / current_rate - 1)
        # Stoploss already breached, return max to keep current stoploss unchanged
        return 1


class PlotConfig:

    def __init__(self):
        self.config = {
            "main_plot": {
                resample("bollinger_upperband"): {"color": "rgba(4,137,122,0.7)"},
                resample("kc_upperband"): {"color": "rgba(4,146,250,0.7)"},
                resample("kc_middleband"): {"color": "rgba(4,146,250,0.7)"},
                resample("kc_lowerband"): {"color": "rgba(4,146,250,0.7)"},
                resample("bollinger_lowerband"): {
                    "color": "rgba(4,137,122,0.7)",
                    "fill_to": resample("bollinger_upperband"),
                    "fill_color": "rgba(4,137,122,0.07)",
                },
                resample("ema9"): {"color": "purple"},
                resample("ema20"): {"color": "yellow"},
                resample("ema50"): {"color": "red"},
                resample("ema200"): {"color": "white"},
            },
            "subplots": {"ATR": {resample("atr"): {"color": "firebrick"}}},
        }

    def add_pivots_in_config(self):
        self.config["main_plot"]["pivot_lows"] = {
            "plotly": {
                "mode": "markers",
                "marker": {
                    "symbol": "diamond-open",
                    "size": 11,
                    "line": {"width": 2},
                    "color": "olive",
                },
            }
        }
        self.config["main_plot"]["pivot_highs"] = {
            "plotly": {
                "mode": "markers",
                "marker": {
                    "symbol": "diamond-open",
                    "size": 11,
                    "line": {"width": 2},
                    "color": "violet",
                },
            }
        }
        return self

    def add_divergence_in_config(self, indicator: str):
        for i in range(3):
            self.config["main_plot"][
                "bullish_divergence_" + indicator + "_line_" + str(i)
            ] = {"plotly": {"mode": "lines", "line": {"color": "green", "dash": "dash"}}}
            self.config["main_plot"][
                "bearish_divergence_" + indicator + "_line_" + str(i)
            ] = {"plotly": {"mode": "lines", "line": {"color": "crimson", "dash": "dash"}}}
        return self

    def add_total_divergences_in_config(self, dataframe):
        total_bullish_divergences_count = dataframe[
            resample("total_bullish_divergences_count")
        ]
        total_bullish_divergences_names = dataframe[
            resample("total_bullish_divergences_names")
        ]
        self.config["main_plot"][resample("total_bullish_divergences")] = {
            "plotly": {
                "mode": "markers+text",
                "text": total_bullish_divergences_count,
                "hovertext": total_bullish_divergences_names,
                "textfont": {"size": 11, "color": "green"},
                "textposition": "bottom center",
                "marker": {
                    "symbol": "diamond",
                    "size": 11,
                    "line": {"width": 2},
                    "color": "green",
                },
            }
        }
        total_bearish_divergences_count = dataframe[
            resample("total_bearish_divergences_count")
        ]
        total_bearish_divergences_names = dataframe[
            resample("total_bearish_divergences_names")
        ]
        self.config["main_plot"][resample("total_bearish_divergences")] = {
            "plotly": {
                "mode": "markers+text",
                "text": total_bearish_divergences_count,
                "hovertext": total_bearish_divergences_names,
                "textfont": {"size": 11, "color": "crimson"},
                "textposition": "top center",
                "marker": {
                    "symbol": "diamond",
                    "size": 11,
                    "line": {"width": 2},
                    "color": "crimson",
                },
            }
        }
        return self


def resample(indicator):
    return indicator


def two_bands_check(dataframe):
    check = (dataframe[resample("low")] < dataframe[resample("kc_lowerband")]) & (
        dataframe[resample("high")] > dataframe[resample("kc_upperband")]
    )
    return ~check


def ema_cross_check(dataframe):
    dataframe["ema20_50_cross"] = qtpylib.crossed_below(
        dataframe[resample("ema20")], dataframe[resample("ema50")]
    )
    dataframe["ema20_200_cross"] = qtpylib.crossed_below(
        dataframe[resample("ema20")], dataframe[resample("ema200")]
    )
    dataframe["ema50_200_cross"] = qtpylib.crossed_below(
        dataframe[resample("ema50")], dataframe[resample("ema200")]
    )
    return ~(
        dataframe["ema20_50_cross"]
        | dataframe["ema20_200_cross"]
        | dataframe["ema50_200_cross"]
    )


def green_candle(dataframe):
    return dataframe[resample("open")] < dataframe[resample("close")]


def keltner_middleband_check(dataframe):
    return (dataframe[resample("low")] < dataframe[resample("kc_middleband")]) & (
        dataframe[resample("high")] > dataframe[resample("kc_middleband")]
    )


def keltner_lowerband_check(dataframe):
    return (dataframe[resample("low")] < dataframe[resample("kc_lowerband")]) & (
        dataframe[resample("high")] > dataframe[resample("kc_lowerband")]
    )


def bollinger_lowerband_check(dataframe):
    return (
        dataframe[resample("low")] < dataframe[resample("bollinger_lowerband")]
    ) & (dataframe[resample("high")] > dataframe[resample("bollinger_lowerband")])


def bollinger_keltner_check(dataframe):
    return (
        dataframe[resample("bollinger_lowerband")]
        < dataframe[resample("kc_lowerband")]
    ) & (
        dataframe[resample("bollinger_upperband")]
        > dataframe[resample("kc_upperband")]
    )


def ema_check(dataframe):
    check = (
        (dataframe[resample("ema9")] < dataframe[resample("ema20")])
        & (dataframe[resample("ema20")] < dataframe[resample("ema50")])
        & (dataframe[resample("ema50")] < dataframe[resample("ema200")])
    )
    return ~check


def initialize_divergences_lists(dataframe: DataFrame):
    dataframe["total_bullish_divergences"] = np.empty(len(dataframe["close"])) * np.nan
    dataframe["total_bullish_divergences_count"] = np.empty(
        len(dataframe["close"])
    ) * np.nan
    dataframe["total_bullish_divergences_count"] = [
        0 if x != x else x for x in dataframe["total_bullish_divergences_count"]
    ]
    dataframe["total_bullish_divergences_names"] = np.empty(
        len(dataframe["close"])
    ) * np.nan
    dataframe["total_bullish_divergences_names"] = [
        "" if x != x else x for x in dataframe["total_bullish_divergences_names"]
    ]
    dataframe["total_bearish_divergences"] = np.empty(len(dataframe["close"])) * np.nan
    dataframe["total_bearish_divergences_count"] = np.empty(
        len(dataframe["close"])
    ) * np.nan
    dataframe["total_bearish_divergences_count"] = [
        0 if x != x else x for x in dataframe["total_bearish_divergences_count"]
    ]
    dataframe["total_bearish_divergences_names"] = np.empty(
        len(dataframe["close"])
    ) * np.nan
    dataframe["total_bearish_divergences_names"] = [
        "" if x != x else x for x in dataframe["total_bearish_divergences_names"]
    ]


def add_divergences(dataframe: DataFrame, indicator: str):
    (
        bearish_divergences,
        bearish_lines,
        bullish_divergences,
        bullish_lines,
    ) = divergence_finder_dataframe(dataframe, indicator)
    dataframe["bearish_divergence_" + indicator + "_occurence"] = bearish_divergences
    dataframe["bullish_divergence_" + indicator + "_occurence"] = bullish_divergences


def divergence_finder_dataframe(
    dataframe: DataFrame, indicator_source: str
) -> Tuple[pd.Series, pd.Series]:
    bearish_lines = [np.empty(len(dataframe["close"])) * np.nan]
    bearish_divergences = np.empty(len(dataframe["close"])) * np.nan
    bullish_lines = [np.empty(len(dataframe["close"])) * np.nan]
    bullish_divergences = np.empty(len(dataframe["close"])) * np.nan
    low_iterator = []
    high_iterator = []
    for index, row in enumerate(dataframe.itertuples(index=True, name="Pandas")):
        if np.isnan(row.pivot_lows):
            low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
        else:
            low_iterator.append(index)
        if np.isnan(row.pivot_highs):
            high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
        else:
            high_iterator.append(index)
    for index, row in enumerate(dataframe.itertuples(index=True, name="Pandas")):
        bearish_occurence = bearish_divergence_finder(
            dataframe, dataframe[indicator_source], high_iterator, index
        )
        if bearish_occurence is not None:
            prev_pivot, current_pivot = bearish_occurence
            bearish_prev_pivot = dataframe["close"][prev_pivot]
            bearish_current_pivot = dataframe["close"][current_pivot]
            bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
            bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
            length = current_pivot - prev_pivot
            bearish_lines_index = 0
            can_exist = True
            while True:
                can_draw = True
                if bearish_lines_index <= len(bearish_lines):
                    bearish_lines.append(np.empty(len(dataframe["close"])) * np.nan)
                actual_bearish_lines = bearish_lines[bearish_lines_index]
                for i in range(length + 1):
                    point = (
                        bearish_prev_pivot
                        + (bearish_current_pivot - bearish_prev_pivot) * i / length
                    )
                    indicator_point = (
                        bearish_ind_prev_pivot
                        + (bearish_ind_current_pivot - bearish_ind_prev_pivot)
                        * i
                        / length
                    )
                    if i != 0 and i != length:
                        if (
                            point <= dataframe["close"][prev_pivot + i]
                            or indicator_point
                            <= dataframe[indicator_source][prev_pivot + i]
                        ):
                            can_exist = False
                    if not np.isnan(actual_bearish_lines[prev_pivot + i]):
                        can_draw = False
                if not can_exist:
                    break
                if can_draw:
                    for i in range(length + 1):
                        actual_bearish_lines[prev_pivot + i] = (
                            bearish_prev_pivot
                            + (bearish_current_pivot - bearish_prev_pivot) * i / length
                        )
                    break
                bearish_lines_index = bearish_lines_index + 1
            if can_exist:
                bearish_divergences[index] = row.close
                dataframe.loc[index, "total_bearish_divergences"] = row.close
                if index > 30:
                    dataframe.loc[
                        index - 30, "total_bearish_divergences_count"
                    ] = (
                        dataframe.loc[index - 30, "total_bearish_divergences_count"] + 1
                    )
                    dataframe.loc[index - 30, "total_bearish_divergences_names"] = (
                        dataframe.loc[index - 30, "total_bearish_divergences_names"]
                        + indicator_source.upper()
                        + "<br>"
                    )
        bullish_occurence = bullish_divergence_finder(
            dataframe, dataframe[indicator_source], low_iterator, index
        )
        if bullish_occurence is not None:
            prev_pivot, current_pivot = bullish_occurence
            bullish_prev_pivot = dataframe["close"][prev_pivot]
            bullish_current_pivot = dataframe["close"][current_pivot]
            bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
            bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
            length = current_pivot - prev_pivot
            bullish_lines_index = 0
            can_exist = True
            while True:
                can_draw = True
                if bullish_lines_index <= len(bullish_lines):
                    bullish_lines.append(np.empty(len(dataframe["close"])) * np.nan)
                actual_bullish_lines = bullish_lines[bullish_lines_index]
                for i in range(length + 1):
                    point = (
                        bullish_prev_pivot
                        + (bullish_current_pivot - bullish_prev_pivot) * i / length
                    )
                    indicator_point = (
                        bullish_ind_prev_pivot
                        + (bullish_ind_current_pivot - bullish_ind_prev_pivot)
                        * i
                        / length
                    )
                    if i != 0 and i != length:
                        if (
                            point >= dataframe["close"][prev_pivot + i]
                            or indicator_point
                            >= dataframe[indicator_source][prev_pivot + i]
                        ):
                            can_exist = False
                    if not np.isnan(actual_bullish_lines[prev_pivot + i]):
                        can_draw = False
                if not can_exist:
                    break
                if can_draw:
                    for i in range(length + 1):
                        actual_bullish_lines[prev_pivot + i] = (
                            bullish_prev_pivot
                            + (bullish_current_pivot - bullish_prev_pivot) * i / length
                        )
                    break
                bullish_lines_index = bullish_lines_index + 1
            if can_exist:
                bullish_divergences[index] = row.close
                dataframe.loc[index, "total_bullish_divergences"] = row.close
                if index > 30:
                    dataframe.loc[
                        index - 30, "total_bullish_divergences_count"
                    ] = (
                        dataframe.loc[index - 30, "total_bullish_divergences_count"] + 1
                    )
                    dataframe.loc[index - 30, "total_bullish_divergences_names"] = (
                        dataframe["total_bullish_divergences_names"][index - 30]
                        + indicator_source.upper()
                        + "<br>"
                    )
    return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)


def bearish_divergence_finder(dataframe, indicator, high_iterator, index):
    if high_iterator[index] == index:
        current_pivot = high_iterator[index]
        occurences = list(dict.fromkeys(high_iterator))
        current_index = occurences.index(high_iterator[index])
        for i in range(current_index - 1, current_index - 6, -1):
            prev_pivot = occurences[i]
            if np.isnan(prev_pivot):
                return
            if (
                dataframe["pivot_highs"][current_pivot]
                < dataframe["pivot_highs"][prev_pivot]
                and indicator[current_pivot] > indicator[prev_pivot]
            ) or (
                dataframe["pivot_highs"][current_pivot]
                > dataframe["pivot_highs"][prev_pivot]
                and indicator[current_pivot] < indicator[prev_pivot]
            ):
                return (prev_pivot, current_pivot)
    return None


def bullish_divergence_finder(dataframe, indicator, low_iterator, index):
    if low_iterator[index] == index:
        current_pivot = low_iterator[index]
        occurences = list(dict.fromkeys(low_iterator))
        current_index = occurences.index(low_iterator[index])
        for i in range(current_index - 1, current_index - 6, -1):
            prev_pivot = occurences[i]
            if np.isnan(prev_pivot):
                return
            if (
                dataframe["pivot_lows"][current_pivot]
                < dataframe["pivot_lows"][prev_pivot]
                and indicator[current_pivot] > indicator[prev_pivot]
            ) or (
                dataframe["pivot_lows"][current_pivot]
                > dataframe["pivot_lows"][prev_pivot]
                and indicator[current_pivot] < indicator[prev_pivot]
            ):
                return (prev_pivot, current_pivot)
    return None


from enum import Enum


class PivotSource(Enum):
    HighLow = 0
    Close = 1


def pivot_points(
    dataframe: DataFrame,
    window: int = 5,
    pivot_source: PivotSource = PivotSource.Close,
) -> DataFrame:
    high_source = None
    low_source = None
    if pivot_source == PivotSource.Close:
        high_source = "close"
        low_source = "close"
    elif pivot_source == PivotSource.HighLow:
        high_source = "high"
        low_source = "low"
    pivot_points_lows = np.empty(len(dataframe["close"])) * np.nan
    pivot_points_highs = np.empty(len(dataframe["close"])) * np.nan
    last_values = deque()
    for index, row in enumerate(dataframe.itertuples(index=True, name="Pandas")):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(
                    current_value, high_source, low_source, left, right
                )
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index - window] = getattr(
                    current_value, high_source
                )
            if is_less:
                pivot_points_lows[index - window] = getattr(current_value, low_source)
            last_values.popleft()
    if len(last_values) >= window + 2:
        current_value = last_values[-2]
        is_greater = True
        is_less = True
        for window_index in range(0, window):
            left = last_values[-2 - window_index - 1]
            right = last_values[-1]
            local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(
                current_value, high_source, low_source, left, right
            )
            is_greater &= local_is_greater
            is_less &= local_is_less
        if is_greater:
            pivot_points_highs[index - 1] = getattr(current_value, high_source)
        if is_less:
            pivot_points_lows[index - 1] = getattr(current_value, low_source)
    return pd.DataFrame(
        index=dataframe.index,
        data={"pivot_lows": pivot_points_lows, "pivot_highs": pivot_points_highs},
    )


def check_if_pivot_is_greater_or_less(
    current_value, high_source: str, low_source: str, left, right
) -> Tuple[bool, bool]:
    is_greater = True
    is_less = True
    if getattr(current_value, high_source) < getattr(
        left, high_source
    ) or getattr(current_value, high_source) < getattr(right, high_source):
        is_greater = False
    if getattr(current_value, low_source) > getattr(
        left, low_source
    ) or getattr(current_value, low_source) > getattr(right, low_source):
        is_less = False
    return (is_greater, is_less)


def emaKeltner(dataframe):
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner["upper"] = ema20 + atr
    keltner["mid"] = ema20
    keltner["lower"] = ema20 - atr
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
    mfv = (df["close"] - df["low"] - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    mfv = mfv.fillna(0.0)
    mfv *= df["volume"]
    cmf = mfv.rolling(n, min_periods=0).sum() / df["volume"].rolling(
        n, min_periods=0
    ).sum()
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name="cmf")
