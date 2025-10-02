from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    informative,
    merge_informative_pair,
)
from pandas import DataFrame, Series
import talib.abstract as ta
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from freqtrade.persistence import Trade
import logging

logger = logging.getLogger(__name__)

class FVGAdvancedStrategy_V2(IStrategy):
    """
    Advanced FVG strategy – dual sided trading with refined signal, market regime, and risk control logic.
    FIXED VERSION: Relaxed entry conditions to allow trades while maintaining good risk management.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    informative_timeframe = "1h"

    # Enable shorting explicitly.
    can_short = True

    # Base risk parameters calibrated for futures dry-run.
    minimal_roi = {
        "0": 0.06,
        "60": 0.04,
        "180": 0.02,
        "360": 0
    }
    stoploss = -0.09

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Leverage parameters.
    max_leverage = 3.0
    min_leverage = 1.0
    default_leverage = 2.0

    # DCA / scale-out parameters.
    max_dca_orders = 1
    max_dca_multiplier = 1.5
    minimal_dca_profit = -0.04
    max_dca_profit = -0.12

    position_adjustment_enable = True
    max_entry_position_adjustment = -1
    max_exit_position_adjustment = 3
    exit_portion_size = 0.25

    # Tunable indicator parameters - RELAXED VALUES
    filter_width = DecimalParameter(0.1, 2.0, default=0.4, space="buy")  # Reduced from 0.6
    tp_mult = DecimalParameter(0.8, 3.0, default=1.6, space="sell")
    sl_mult = DecimalParameter(0.5, 2.0, default=0.9, space="sell")
    fvg_confirmation_bars = IntParameter(1, 3, default=1, space="buy")  # Reduced from 2

    # Core indicator lookbacks.
    atr_period = 14
    ema_fast_period = 21
    ema_slow_period = 55
    ema_long_period = 200
    bb_period = 20
    bb_std = 2.0
    adx_period = 14
    rsi_period = 14
    volatility_period = 100
    market_score_window = 240
    volume_window = 40

    # Market score thresholds - RELAXED VALUES
    market_score_long = 55  # Lowered from 65
    market_score_short = 45  # Raised from 35
    
    # Filter enable/disable flags
    use_strict_mode = False  # Set to True for more restrictive trading

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        mode = "STRICT" if self.use_strict_mode else "RELAXED"
        logger.info(f"FVG Advanced Strategy V2 initialised in {mode} mode")

    @staticmethod
    def _minmax(series: Series, window: int) -> Series:
        """Normalise a rolling window to 0-1 while guarding against zero division."""
        rolling_min = series.rolling(window).min()
        rolling_max = series.rolling(window).max()
        normalised = (series - rolling_min) / (rolling_max - rolling_min + 1e-9)
        return normalised.clip(lower=0.0, upper=1.0)

    def informative_pairs(self):
        pairs = getattr(self.dp, "current_whitelist", lambda: [])()
        return [(pair, self.informative_timeframe) for pair in pairs]

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Indicators on the higher timeframe used for confirmation and leverage control."""
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)

        dataframe["uptrend"] = (
            (dataframe["close"] > dataframe["ema"]) &
            (dataframe["ema"] > dataframe["ema_slow"]) &
            (dataframe["rsi"] > 50) &
            (dataframe["adx"] > 20)
        )

        dataframe["downtrend"] = (
            (dataframe["close"] < dataframe["ema"]) &
            (dataframe["ema"] < dataframe["ema_slow"]) &
            (dataframe["rsi"] < 50) &
            (dataframe["adx"] > 20)
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]).replace([np.inf, -np.inf], np.nan)

        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.ema_long_period)

        dataframe["trend_up"] = dataframe["ema_fast"] > dataframe["ema_slow"]
        dataframe["trend_down"] = dataframe["ema_fast"] < dataframe["ema_slow"]

        dataframe["volume_mean"] = dataframe["volume"].rolling(self.volume_window).mean()
        dataframe["volume_ratio"] = (dataframe["volume"] / dataframe["volume_mean"]).replace([np.inf, -np.inf], np.nan)

        bollinger = ta.BBANDS(dataframe, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
        dataframe["bb_upper"] = bollinger["upperband"]
        dataframe["bb_middle"] = bollinger["middleband"]
        dataframe["bb_lower"] = bollinger["lowerband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"].replace(0, np.nan)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period)
        dataframe["volatility"] = dataframe["close"].pct_change().rolling(self.volatility_period).std()

        prev2_high = dataframe["high"].shift(2)
        prev_high = dataframe["high"].shift(1)
        prev2_low = dataframe["low"].shift(2)
        prev_low = dataframe["low"].shift(1)

        gap_min = dataframe["atr"] * self.filter_width.value

        bull_gap = (prev2_high < prev_low) & (prev_low < dataframe["low"]) & (dataframe["low"] - prev2_high > gap_min)
        bear_gap = (prev2_low > prev_high) & (prev_high > dataframe["high"]) & (prev2_low - dataframe["high"] > gap_min)

        confirm_window = int(self.fvg_confirmation_bars.value)
        dataframe["bull_fvg"] = bull_gap.rolling(confirm_window).max().fillna(0).astype(bool)
        dataframe["bear_fvg"] = bear_gap.rolling(confirm_window).max().fillna(0).astype(bool)

        dataframe["bull_mid"] = (prev2_high + dataframe["low"]) / 2
        dataframe["bear_mid"] = (prev2_low + dataframe["high"]) / 2

        dataframe["atr_pct_rank"] = self._minmax(dataframe["atr_pct"], self.market_score_window)
        dataframe["bb_width_rank"] = self._minmax(dataframe["bb_width"], self.market_score_window)
        dataframe["volume_ratio"] = dataframe["volume_ratio"].fillna(0)

        dataframe["market_score"] = self.calculate_market_score(dataframe)

        informative = self.dp.get_pair_dataframe(metadata["pair"], self.informative_timeframe)
        if informative is not None and not informative.empty:
            dataframe = merge_informative_pair(
                dataframe,
                informative,
                self.timeframe,
                self.informative_timeframe,
                ffill=True
            )

        for column in ["uptrend_1h", "downtrend_1h"]:
            if column in dataframe:
                dataframe[column] = dataframe[column].where(dataframe[column].notna(), False).astype(bool)
            else:
                dataframe[column] = False

        dataframe["trend_up"] = dataframe["trend_up"].where(dataframe["trend_up"].notna(), False).astype(bool)
        dataframe["trend_down"] = dataframe["trend_down"].where(dataframe["trend_down"].notna(), False).astype(bool)
        dataframe["volume_ratio"] = dataframe["volume_ratio"].clip(lower=0)

        return dataframe

    def calculate_market_score(self, dataframe: DataFrame) -> Series:
        """Combine trend, momentum, and volatility signals into a bounded 0-100 regime score."""
        adx_component = np.clip(dataframe["adx"] / 50.0, 0.0, 1.0)
        rsi_component = 1.0 - np.clip(np.abs(dataframe["rsi"] - 50.0) / 50.0, 0.0, 1.0)
        volatility_component = 1.0 - self._minmax(dataframe["atr_pct"].ffill(), self.market_score_window)
        bb_component = 1.0 - dataframe["bb_width_rank"].fillna(0.5)
        momentum_component = np.where(dataframe["trend_up"], 1.0, np.where(dataframe["trend_down"], 0.0, 0.5))

        composite = (adx_component + rsi_component + volatility_component + bb_component + momentum_component) / 5.0
        return Series(composite * 100.0, index=dataframe.index).clip(lower=0.0, upper=100.0)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        if self.use_strict_mode:
            # STRICT MODE: All 7 conditions (original modified version)
            long_conditions = (
                dataframe["bull_fvg"] &
                dataframe["trend_up"] &
                dataframe["uptrend_1h"] &
                (dataframe["market_score"] > self.market_score_long) &
                (dataframe["volume_ratio"] > 1.05) &
                (dataframe["atr_pct_rank"] < 0.85) &
                (dataframe["close"] > dataframe["ema_long"])
            )

            short_conditions = (
                dataframe["bear_fvg"] &
                dataframe["trend_down"] &
                dataframe["downtrend_1h"] &
                (dataframe["market_score"] < self.market_score_short) &
                (dataframe["volume_ratio"] > 1.05) &
                (dataframe["atr_pct_rank"] < 0.85) &
                (dataframe["close"] < dataframe["ema_long"])
            )
        else:
            # RELAXED MODE: Only core 4-5 conditions for better trade frequency
            long_conditions = (
                dataframe["bull_fvg"] &
                (dataframe["trend_up"] | dataframe["uptrend_1h"]) &  # Either 5m OR 1h trend
                (dataframe["market_score"] > self.market_score_long) &
                (dataframe["volume_ratio"] > 0.95)  # Slightly above average volume
            )

            short_conditions = (
                dataframe["bear_fvg"] &
                (dataframe["trend_down"] | dataframe["downtrend_1h"]) &  # Either 5m OR 1h trend
                (dataframe["market_score"] < self.market_score_short) &
                (dataframe["volume_ratio"] > 0.95)
            )

        dataframe.loc[long_conditions, "enter_long"] = 1
        dataframe.loc[short_conditions, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        long_exit_conditions = (
            dataframe["bear_fvg"] |
            (~dataframe["trend_up"] & ~dataframe["uptrend_1h"]) |  # Both trends turned bearish
            (dataframe["market_score"] < self.market_score_long - 20) |  # Significant score drop
            (dataframe["volume_ratio"] < 0.7)  # Very low volume
        )

        short_exit_conditions = (
            dataframe["bull_fvg"] |
            (~dataframe["trend_down"] & ~dataframe["downtrend_1h"]) |  # Both trends turned bullish
            (dataframe["market_score"] > self.market_score_short + 20) |  # Significant score rise
            (dataframe["volume_ratio"] < 0.7)
        )

        dataframe.loc[long_exit_conditions, "exit_long"] = 1
        dataframe.loc[short_exit_conditions, "exit_short"] = 1

        return dataframe

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        **kwargs
    ) -> Optional[float]:
        """Scale positions in line with drawdown or trim after extended runs."""
        if not self.position_adjustment_enable:
            return None

        # Scale in on controlled drawdown.
        if self.max_dca_profit <= current_profit <= self.minimal_dca_profit:
            if trade.nr_of_successful_entries <= self.max_dca_orders:
                stake = min(trade.stake_amount * self.max_dca_multiplier, max_stake)
                return max(stake, min_stake)

        # Scale out partial profits.
        if current_profit > 0.03:
            reduction = -abs(trade.amount) * self.exit_portion_size
            if abs(reduction) * current_rate > min_stake:
                return reduction

        return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Dynamic leverage control: safer leverage during uncertain markets.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or len(dataframe) == 0:
                return self.default_leverage

            last_candle = dataframe.iloc[-1].squeeze()
            market_score = last_candle.get("market_score", 50)
            adx_1h = last_candle.get("adx_1h", 20)
            atr_pct_rank = last_candle.get("atr_pct_rank", 0.5)

            # Base leverage on market conditions
            if market_score > 70 or market_score < 30:
                # Strong directional bias
                leverage = self.max_leverage
            elif market_score > 60 or market_score < 40:
                # Moderate directional bias
                leverage = (self.max_leverage + self.default_leverage) / 2
            else:
                # Neutral/choppy market
                leverage = self.default_leverage

            # Reduce leverage during high volatility
            if atr_pct_rank > 0.8:
                leverage = max(self.min_leverage, leverage * 0.7)

            # Reduce leverage during weak trends
            if adx_1h < 20:
                leverage = max(self.min_leverage, leverage * 0.8)

            return max(self.min_leverage, min(leverage, self.max_leverage))

        except Exception as e:
            logger.warning(f"Error in leverage calculation: {e}")
            return self.default_leverage
