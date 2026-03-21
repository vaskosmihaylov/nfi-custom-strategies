"""
Triple Supertrend with ADX/RSI Confirmation (Enhanced):
* Description: This strategy generates three separate Supertrend indicators for both buy and sell conditions.
               A long (buy) signal is generated when all three buy Supertrend indicators are in an uptrend,
               and additional trend/momentum filters (ADX and RSI) confirm bullish conditions.
               Conversely, a short (sell) signal is generated when all three sell Supertrend indicators are in a downtrend,
               and the ADX/RSI filters confirm bearish conditions.
               Exits are triggered when a secondary Supertrend indicator signals a reversal or if RSI indicates overbought/oversold conditions.
* Author: @juankysoriano (Juan Carlos Soriano) - enhanced by modification
* github: https://github.com/juankysoriano/
*** NOTE: This enhanced Supertrend strategy now includes ADX and RSI filters as additional confirmation.
          Use at your own risk.
"""

import logging
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IntParameter, IStrategy
from numpy.lib import math
from pandas import DataFrame


class TripleSuperTrendADXRSI(IStrategy):
    INTERFACE_VERSION: int = 3

    # Enable short trades
    can_short: bool = True

    # Hyperparameters for generating the Supertrend indicators (Buy side)
    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
    }

    # Hyperparameters for generating the Supertrend indicators (Sell side)
    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
    }

    # ROI table: defines the minimal return on investment for different time periods
    minimal_roi = {"0": 0.1, "30": 0.75, "60": 0.05, "120": 0.025}
    # Stoploss threshold
    stoploss = -0.265

    # Trailing stop configuration
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Strategy timeframe and startup candle count
    timeframe = "1h"
    startup_candle_count = 18

    # ----- SUPER TREND PARAMETERS FOR BUY SIGNALS -----
    buy_m1 = IntParameter(1, 7, default=1)
    buy_m2 = IntParameter(1, 7, default=3)
    buy_m3 = IntParameter(1, 7, default=4)
    buy_p1 = IntParameter(7, 21, default=14)
    buy_p2 = IntParameter(7, 21, default=10)
    buy_p3 = IntParameter(7, 21, default=10)

    # ----- SUPER TREND PARAMETERS FOR SELL SIGNALS -----
    sell_m1 = IntParameter(1, 7, default=1)
    sell_m2 = IntParameter(1, 7, default=3)
    sell_m3 = IntParameter(1, 7, default=4)
    sell_p1 = IntParameter(7, 21, default=14)
    sell_p2 = IntParameter(7, 21, default=10)
    sell_p3 = IntParameter(7, 21, default=10)

    # ----- NEW PARAMETERS FOR TREND / MOMENTUM CONFIRMATION -----
    # ADX filter settings to avoid low-trend environments
    adx_period = IntParameter(10, 30, default=14, space="buy")
    adx_threshold = IntParameter(20, 40, default=25, space="buy")

    # RSI filter settings for entry confirmation and exit signals
    rsi_period = IntParameter(10, 30, default=14, space="buy")
    # For longs, require RSI to be above this value (bullish momentum confirmation)
    rsi_long_threshold = IntParameter(40, 60, default=50, space="buy")
    # For shorts, require RSI to be below this value (bearish momentum confirmation)
    rsi_short_threshold = IntParameter(40, 60, default=50, space="buy")
    # RSI exit thresholds (to catch reversals)
    rsi_long_exit = IntParameter(70, 80, default=75, space="sell")
    rsi_short_exit = IntParameter(20, 30, default=25, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds multiple Supertrend, ADX, and RSI indicators to the dataframe.
        Generates Supertrend indicators over different parameter ranges for both buy and sell signals.
        Also computes ADX for trend strength and RSI for momentum.
        """
        # Generate Supertrend indicators for BUY signals with different multipliers and periods
        for multiplier in self.buy_m1.range:
            for period in self.buy_p1.range:
                dataframe[f"supertrend_1_buy_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.buy_m2.range:
            for period in self.buy_p2.range:
                dataframe[f"supertrend_2_buy_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.buy_m3.range:
            for period in self.buy_p3.range:
                dataframe[f"supertrend_3_buy_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        # Generate Supertrend indicators for SELL signals with different multipliers and periods
        for multiplier in self.sell_m1.range:
            for period in self.sell_p1.range:
                dataframe[f"supertrend_1_sell_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.sell_m2.range:
            for period in self.sell_p2.range:
                dataframe[f"supertrend_2_sell_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        for multiplier in self.sell_m3.range:
            for period in self.sell_p3.range:
                dataframe[f"supertrend_3_sell_{multiplier}_{period}"] = self.supertrend(
                    dataframe, multiplier, period
                )["STX"]

        # Calculate ADX to measure the strength of the trend; helps to filter out weak trends
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        # Calculate RSI to determine momentum and possible overbought/oversold conditions
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Creates entry signals based on a combination of Supertrend, ADX, and RSI indicators.
        A long entry is signaled if:
          - All three Supertrend BUY indicators are in an uptrend ('up').
          - The trading volume is positive.
          - The ADX is above the threshold indicating a strong trend.
          - RSI is above the long threshold to confirm bullish momentum.
        A short entry is signaled similarly with sell conditions.
        """
        # LONG ENTRY CONDITION
        dataframe.loc[
            (
                (dataframe[f"supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}"] == "up")
                & (dataframe[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "up")
                & (dataframe[f"supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}"] == "up")
                & (dataframe["volume"] > 0)
                & (dataframe["adx"] > self.adx_threshold.value)
                & (dataframe["rsi"] > self.rsi_long_threshold.value)
            ),
            "enter_long",
        ] = 1

        # SHORT ENTRY CONDITION
        dataframe.loc[
            (
                (dataframe[f"supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}"] == "down")
                & (dataframe[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "down")
                & (dataframe[f"supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}"] == "down")
                & (dataframe["volume"] > 0)
                & (dataframe["adx"] > self.adx_threshold.value)
                & (dataframe["rsi"] < self.rsi_short_threshold.value)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sets exit conditions for trades based on a secondary Supertrend indicator or extreme RSI readings.
        For longs: exit if the secondary SELL Supertrend indicator is down OR if RSI is above the long exit threshold.
        For shorts: exit if the secondary BUY Supertrend indicator is up OR if RSI is below the short exit threshold.
        """
        # LONG EXIT CONDITION
        dataframe.loc[
            (
                (dataframe[f"supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}"] == "down")
                | (dataframe["rsi"] > self.rsi_long_exit.value)
            ),
            "exit_long",
        ] = 1

        # SHORT EXIT CONDITION
        dataframe.loc[
            (
                (dataframe[f"supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}"] == "up")
                | (dataframe["rsi"] < self.rsi_short_exit.value)
            ),
            "exit_short",
        ] = 1

        return dataframe

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        """
        Calculates the Supertrend indicator based on the Average True Range (ATR).
        The indicator computes basic upper and lower bands and then adjusts them to create final bands.
        It then determines the trend direction ('up' or 'down') based on the closing price relative to these bands.
        """
        df = dataframe.copy()

        # Calculate True Range and its moving average (ATR)
        df["TR"] = ta.TRANGE(df)
        df["ATR"] = ta.SMA(df["TR"], period)

        # Define column names for the Supertrend value and its direction
        st = "ST_" + str(period) + "_" + str(multiplier)
        stx = "STX_" + str(period) + "_" + str(multiplier)

        # Compute the basic upper and lower bands
        df["basic_ub"] = (df["high"] + df["low"]) / 2 + multiplier * df["ATR"]
        df["basic_lb"] = (df["high"] + df["low"]) / 2 - multiplier * df["ATR"]

        # Initialize final bands
        df["final_ub"] = 0.00
        df["final_lb"] = 0.00
        close_col = df.columns.get_loc("close")
        basic_ub_col = df.columns.get_loc("basic_ub")
        basic_lb_col = df.columns.get_loc("basic_lb")
        final_ub_col = df.columns.get_loc("final_ub")
        final_lb_col = df.columns.get_loc("final_lb")
        # Adjust the final bands based on previous values and the closing price
        for i in range(period, len(df)):
            df.iat[i, final_ub_col] = (
                df.iat[i, basic_ub_col]
                if df.iat[i, basic_ub_col] < df.iat[i - 1, final_ub_col]
                or df.iat[i - 1, close_col] > df.iat[i - 1, final_ub_col]
                else df.iat[i - 1, final_ub_col]
            )
            df.iat[i, final_lb_col] = (
                df.iat[i, basic_lb_col]
                if df.iat[i, basic_lb_col] > df.iat[i - 1, final_lb_col]
                or df.iat[i - 1, close_col] < df.iat[i - 1, final_lb_col]
                else df.iat[i - 1, final_lb_col]
            )

        # Initialize the Supertrend column
        df[st] = 0.00
        st_col = df.columns.get_loc(st)
        # Determine the Supertrend value based on the relationship between closing price and the final bands
        for i in range(period, len(df)):
            df.iat[i, st_col] = (
                df.iat[i, final_ub_col]
                if df.iat[i - 1, st_col] == df.iat[i - 1, final_ub_col]
                and df.iat[i, close_col] <= df.iat[i, final_ub_col]
                else df.iat[i, final_lb_col]
                if df.iat[i - 1, st_col] == df.iat[i - 1, final_ub_col]
                and df.iat[i, close_col] > df.iat[i, final_ub_col]
                else df.iat[i, final_lb_col]
                if df.iat[i - 1, st_col] == df.iat[i - 1, final_lb_col]
                and df.iat[i, close_col] >= df.iat[i, final_lb_col]
                else df.iat[i, final_ub_col]
                if df.iat[i - 1, st_col] == df.iat[i - 1, final_lb_col]
                and df.iat[i, close_col] < df.iat[i, final_lb_col]
                else 0.00
            )
        # Mark the trend direction as 'up' or 'down'
        df[stx] = np.where(
            (df[st] > 0.00), np.where((df["close"] < df[st]), "down", "up"), None
        )

        # Cleanup: remove temporary band columns and fill any missing values
        df.drop(["basic_ub", "basic_lb", "final_ub", "final_lb"], inplace=True, axis=1)
        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={"ST": df[st], "STX": df[stx]})
