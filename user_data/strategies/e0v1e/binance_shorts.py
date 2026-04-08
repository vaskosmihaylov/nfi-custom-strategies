"""
binance_shorts - Short-only variant of binance.py (E0V1E long strategy for Binance)

Mirrors the successful long-only binance.py strategy with inverted logic for short positions.
Designed to profit in bear markets and overbought conditions.

Entry Conditions:
1. short_1: RSI/SMA/CTI mean-reversion shorts (inverted from long buy_1)
   (ewo_short removed: net loser in BT1 -2855 and BT2 -3167 USDT)

Exit Layers:
1. Base stoploss: -0.18 hard cap (6% price move at 3x leverage)
2. ROI: 7% target (faster profit-taking than long's ~disabled ROI)
3. custom_stoploss: Indicator-based trailing (fastk < cover_fastx when oversold)
4. custom_exit: 48h protection → unclog/zombie/deadfish detection

Parameter defaults sourced from E0V1E_Shorts.py (live-tested on Bybit futures, 3x leverage).
"""

from datetime import datetime, timedelta
import logging
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter

logger = logging.getLogger(__name__)


def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["low"] * 100
    return emadif


class binance_shorts(IStrategy):

    INTERFACE_VERSION = 3
    can_short = True

    # Tighter ROI for shorts - proven in E0V1E_Shorts live runs
    minimal_roi = {"0": 0.07}  # 7% (vs essentially disabled for longs)

    timeframe = "5m"

    process_only_new_candles = True
    startup_candle_count = 20

    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_market_ratio": 0.99,
    }

    # Hard cap at -18% loss (6% price move at 3x leverage)
    # Prevents catastrophic losses from pump tokens (PROMPT -99%, Q -84%)
    stoploss = -0.18

    use_custom_stoploss = True

    # Max simultaneous short positions
    max_short_trades = 4

    # --- Short entry parameters (inverted from binance.py longs) ---
    # Defaults from E0V1E_Shorts.py (live-tested)
    # ewo_short params removed: net loser in backtests (BT1: -2855, BT2: -3167 USDT)

    is_optimize_32 = True
    sell_rsi_fast_32 = IntParameter(50, 80, default=54, space="buy", optimize=is_optimize_32)
    sell_rsi_32 = IntParameter(50, 85, default=81, space="buy", optimize=is_optimize_32)
    sell_sma15_32 = DecimalParameter(1.0, 1.1, default=1.058, decimals=3, space="buy", optimize=is_optimize_32)
    sell_cti_32 = DecimalParameter(0, 1, default=0.86, decimals=2, space="buy", optimize=is_optimize_32)

    # --- Short exit parameters (cover = close short position) ---

    is_optimize_deadfish = True
    cover_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space="sell", optimize=is_optimize_deadfish)
    cover_deadfish_profit = DecimalParameter(0.05, 0.15, default=0.05, space="sell", optimize=is_optimize_deadfish)
    cover_deadfish_bb_factor = DecimalParameter(0.80, 1.10, default=1.0, space="sell", optimize=is_optimize_deadfish)
    cover_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space="sell", optimize=is_optimize_deadfish)

    cover_fastx = IntParameter(0, 50, default=25, space="sell", optimize=True)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        # Only allow shorts
        if side == "long":
            return False

        # Enforce max short position limit
        short_count = 0
        trades = Trade.get_trades_proxy(is_open=True)
        for trade in trades:
            if trade.is_short:
                short_count += 1

        if short_count >= self.max_short_trades:
            return False

        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI family
        dataframe["sma_15"] = ta.SMA(dataframe, timeperiod=15)
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        # EMA / EWO
        dataframe["ema_8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema_16"] = ta.EMA(dataframe, timeperiod=16)
        dataframe["EWO"] = ewo(dataframe, 50, 200)

        # Stochastic
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

        # Bollinger Bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband2"] = bollinger2["lower"]
        dataframe["bb_middleband2"] = bollinger2["mid"]
        dataframe["bb_upperband2"] = bollinger2["upper"]

        dataframe["bb_width"] = (dataframe["bb_upperband2"] - dataframe["bb_lowerband2"]) / dataframe["bb_middleband2"]

        # Volume
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12).mean().shift(1)
        dataframe["volume_mean_24"] = dataframe["volume"].rolling(24).mean().shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, "enter_tag"] = ""

        # --- short_1: Inverted RSI/SMA/CTI mean-reversion ---
        # Long buy_1: rsi_slow declining, rsi_fast < X, rsi > X, close < sma_15 * X, cti < X
        # Short short_1: rsi_slow rising, rsi_fast > X, rsi < X, close > sma_15 * X, cti > X
        short_1 = (
            (dataframe["rsi_slow"] > dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] > self.sell_rsi_fast_32.value)
            & (dataframe["rsi"] < self.sell_rsi_32.value)
            & (dataframe["close"] > dataframe["sma_15"] * self.sell_sma15_32.value)
            & (dataframe["cti"] > self.sell_cti_32.value)
        )

        dataframe.loc[short_1, "enter_tag"] = "short_1"
        dataframe.loc[short_1, "enter_short"] = 1

        return dataframe

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:
        """
        Indicator-based trailing stoploss for shorts.

        Base stoploss (-0.18) handles catastrophic loss prevention.
        This method only tightens the stop for profitable/recoverable trades.
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # After 60 min: tight stop if oversold (fastk < cover_fastx) and small loss
        if current_time - timedelta(minutes=60) > trade.open_date_utc:
            if (current_candle["fastk"] < self.cover_fastx.value) and (current_profit > -0.01):
                return -0.001

        # After 1 day: tight stop if oversold even with moderate loss
        if current_time - timedelta(days=1) > trade.open_date_utc:
            if (current_candle["fastk"] < self.cover_fastx.value) and (current_profit > -0.05):
                return -0.001

        # Profitable shorts: exit when oversold (price bottom)
        if current_profit > 0:
            if current_candle["fastk"] < self.cover_fastx.value:
                return -0.001

        # Keep base stoploss (-0.18)
        return 1.0

    def custom_exit(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> Optional[Union[str, bool]]:
        """
        3-layer exit for shorts (proven in E0V1E_Shorts):
        - Hours 0-48: No forced exits, let position develop
        - After 48h: unclog (>4% loss), zombie (breakeven), deadfish (low vol)
        """

        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Phase 1: First 48 hours - no forced exits
        if trade_duration_hours < 48:
            return None

        # Phase 2: After 48 hours

        # Unclog: force exit if losing > 3%
        if current_profit < -0.03:
            return "unclog"

        # Zombie: force exit if stuck at breakeven
        if -0.005 <= current_profit <= 0.005:
            return "zombie"

        # Deadfish: low volatility dead trade (inverted BB logic for shorts)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if (
            (current_profit < self.cover_deadfish_profit.value)
            and (current_candle["bb_width"] < self.cover_deadfish_bb_width.value)
            and (current_candle["close"] < current_candle["bb_middleband2"] * self.cover_deadfish_bb_factor.value)
            and (
                current_candle["volume_mean_12"]
                < current_candle["volume_mean_24"] * self.cover_deadfish_volume_factor.value
            )
        ):
            return "cover_stoploss_deadfish"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, ["exit_short", "exit_tag"]] = (0, "short_out")

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        return 3.0
