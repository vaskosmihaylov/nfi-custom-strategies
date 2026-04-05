"""
5-8-13 EMA Trend Engine (MTF Optimized)
========================================
Trend-Following / Fibonacci EMA / MTF Confirmation

Identifies explosive crypto momentum by requiring a specific hierarchical
alignment of fast-moving EMAs on the execution timeframe, validated by
structural alignment on a 4-hour anchor.

Source: Quant Tactics
Timeframe: 1H (with 4H MTF confirmation)
Performance: 200% Return / 16% MDD
"""

from datetime import datetime
from typing import Optional, Union
import logging

from freqtrade.strategy import IStrategy, Trade, informative
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
)
from pandas import DataFrame
import talib.abstract as ta

logger = logging.getLogger(__name__)


class FibonacciEMATrendStrategy(IStrategy):
    """
    5-8-13 EMA Trend Engine (MTF Optimized)

    Entry Long: 5>8>13 EMA stack (1H) + 5>8>13 EMA stack (4H) + Price > PSAR
    Entry Short: 5<8<13 EMA stack (1H) + 5<8<13 EMA stack (4H) + Price < PSAR
    Exit: Swing High/Low SL + 3:1 RR TP
    """

    INTERFACE_VERSION = 3
    can_short = True

    # ROI handled by custom exit
    minimal_roi = {"0": 100}

    # Wide stoploss - handled by custom_stoploss
    stoploss = -0.99

    # Timeframe
    timeframe = "1h"

    # Process only new candles
    process_only_new_candles = True

    # Startup candle count
    startup_candle_count: int = 50

    # ==================== HYPEROPTABLE PARAMETERS ====================

    # Fibonacci EMA Lengths
    ema_fast = IntParameter(3, 8, default=5, space="buy", optimize=True)
    ema_mid = IntParameter(5, 13, default=8, space="buy", optimize=True)
    ema_slow = IntParameter(8, 21, default=13, space="buy", optimize=True)

    # Parabolic SAR
    psar_af = DecimalParameter(0.01, 0.05, default=0.02, decimals=2, space="buy", optimize=False)
    psar_max_af = DecimalParameter(0.1, 0.3, default=0.2, decimals=1, space="buy", optimize=False)

    # Swing Point Lookback
    swing_lookback = IntParameter(3, 10, default=5, space="sell", optimize=True)

    # Stop Loss Buffer
    sl_buffer_pct = DecimalParameter(0.1, 0.8, default=0.3, decimals=1, space="sell", optimize=True)

    # Risk Reward Ratio
    risk_reward_ratio = DecimalParameter(2.0, 4.0, default=3.0, decimals=1, space="sell", optimize=True)

    # ==================== INFORMATIVE TIMEFRAME ====================

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate 4H EMAs for MTF confirmation."""
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_mid"] = ta.EMA(dataframe, timeperiod=self.ema_mid.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        return dataframe

    # ==================== INDICATORS ====================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate 1H EMAs, PSAR, and Swing Levels."""

        # 1H EMAs
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_mid"] = ta.EMA(dataframe, timeperiod=self.ema_mid.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(dataframe, acceleration=self.psar_af.value, maximum=self.psar_max_af.value)

        # Swing High/Low (for stop loss calculation)
        dataframe["swing_high"] = dataframe["high"].rolling(window=self.swing_lookback.value).max()
        dataframe["swing_low"] = dataframe["low"].rolling(window=self.swing_lookback.value).min()

        return dataframe

    # ==================== ENTRY LOGIC ====================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Long: 1H Stack (5>8>13) + 4H Stack (5>8>13) + Price > PSAR
        Short: 1H Stack (5<8<13) + 4H Stack (5<8<13) + Price < PSAR
        """
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # 1H Bullish Stack
        stack_1h_bull = (dataframe["ema_fast"] > dataframe["ema_mid"]) & (dataframe["ema_mid"] > dataframe["ema_slow"])

        # 4H Bullish Stack (MTF confirmation)
        stack_4h_bull = (dataframe["ema_fast_4h"] > dataframe["ema_mid_4h"]) & (
            dataframe["ema_mid_4h"] > dataframe["ema_slow_4h"]
        )

        # Long Entry
        long_conditions = (
            stack_1h_bull & stack_4h_bull & (dataframe["close"] > dataframe["sar"]) & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_conditions, "enter_long"] = 1
        dataframe.loc[long_conditions, "enter_tag"] = "fib_ema_mtf_long"

        # 1H Bearish Stack
        stack_1h_bear = (dataframe["ema_fast"] < dataframe["ema_mid"]) & (dataframe["ema_mid"] < dataframe["ema_slow"])

        # 4H Bearish Stack
        stack_4h_bear = (dataframe["ema_fast_4h"] < dataframe["ema_mid_4h"]) & (
            dataframe["ema_mid_4h"] < dataframe["ema_slow_4h"]
        )

        # Short Entry
        short_conditions = (
            stack_1h_bear & stack_4h_bear & (dataframe["close"] < dataframe["sar"]) & (dataframe["volume"] > 0)
        )

        dataframe.loc[short_conditions, "enter_short"] = 1
        dataframe.loc[short_conditions, "enter_tag"] = "fib_ema_mtf_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """No indicator-based exit - handled by custom_exit."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ==================== RISK MANAGEMENT ====================

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[Union[str, bool]]:
        """
        Swing-based SL + 3:1 RR TP
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        # Get entry candle data
        entry_candle = dataframe[dataframe["date"] <= trade.open_date_utc].iloc[-1]

        if not trade.is_short:
            # Long: SL = Swing Low - Buffer
            sl_price = entry_candle["swing_low"] * (1 - self.sl_buffer_pct.value / 100)
            risk = trade.open_rate - sl_price
            tp_price = trade.open_rate + (risk * self.risk_reward_ratio.value)

            if current_rate >= tp_price:
                return f"tp_rr_{self.risk_reward_ratio.value}"
            if current_rate <= sl_price:
                return "sl_swing_low"
        else:
            # Short: SL = Swing High + Buffer
            sl_price = entry_candle["swing_high"] * (1 + self.sl_buffer_pct.value / 100)
            risk = sl_price - trade.open_rate
            tp_price = trade.open_rate - (risk * self.risk_reward_ratio.value)

            if current_rate <= tp_price:
                return f"tp_rr_{self.risk_reward_ratio.value}"
            if current_rate >= sl_price:
                return "sl_swing_high"

        return None

    # ==================== PLOT CONFIG ====================

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "green"},
            "ema_mid": {"color": "blue"},
            "ema_slow": {"color": "red"},
            "sar": {"color": "orange"},
        },
    }
