import sys
import os
from pathlib import Path


# Add the 'strategies' root directory to sys.path to enable absolute imports
def _setup_path():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == "strategies":
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_setup_path()
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib
import logging

# Intelligent & Decision Layers (with mock fallback for backtesting)
try:
    from signal_layer.src.providers.whale_signal_provider import (
        get_whale_signal_provider,
    )
    from analysis_layer.src.regime.llm_market_analyst import get_llm_market_analyst
    from decision_layer.src.engine.decision_engine import get_decision_engine
    from decision_layer.src.protocol.models import TechnicalSignal, TradeAction
except ImportError:
    # Mock providers for standalone backtesting
    class MockProvider:
        def get_signal(self, pair):
            return None

    class MockAnalyst:
        is_enabled = False

        def get_market_regime(self, *args):
            return None

    class MockDecisionEngine:
        def analyze_entry(self, **kwargs):
            wallet = kwargs.get("wallet_balance", 10000)

            class Decision:
                action = None
                position_size_usd = wallet * 0.1  # 10% of wallet per trade
                leverage = 3.0
                entry_tag = "mock"

            return Decision()

    def get_whale_signal_provider():
        return MockProvider()

    def get_llm_market_analyst():
        return MockAnalyst()

    def get_decision_engine():
        return MockDecisionEngine()

    class TechnicalSignal:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class TradeAction:
        HOLD = "HOLD"
        BUY = "BUY"
        SELL = "SELL"


logger = logging.getLogger(__name__)


class Donchian_ADX_CHOPStrategy(IStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.decision_engine = get_decision_engine()
        self.whale_provider = get_whale_signal_provider()
        self.llm_analyst = get_llm_market_analyst()
        self._active_decision = None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Decision Layer: Determine leverage via DecisionEngine."""
        if self._active_decision and self._active_decision.entry_tag == entry_tag:
            return self._active_decision.leverage
        return 3.0

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Decision Layer: Synthesize trade size."""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            wallet_balance = self.wallets.get_total_stake_amount()

            tech_signal = TechnicalSignal(
                should_enter=True,
                side="long" if side == "long" else "short",
                score=0.85 if "whale" in (entry_tag or "") else 0.75,
                indicators={},
                reasoning=f"Donchian Breakout ({entry_tag})",
            )

            decision = self.decision_engine.analyze_entry(
                pair=pair,
                technical_signal=tech_signal,
                wallet_balance=wallet_balance,
                market_data=dataframe.tail(20).to_dict(orient="records"),
            )
            self._active_decision = decision

            if decision.action == TradeAction.HOLD:
                return 0.0
            return max(min_stake or 0, min(decision.position_size_usd, max_stake))
        except Exception as e:
            logger.error(f"Stake error: {e}")
            return proposed_stake

    """
    A breakout trend-following strategy using Donchian Channel for perpetual futures.
    
    Concept:
    - Trade breakouts of the Donchian Channel (highest high / lowest low)
    - Filter with ADX for trend strength and CHOP for market state
    - Use EMA for trend direction confirmation
    - Exit via ATR-based trailing stop-loss
    
    Indicators Used:
    - Donchian Channel: Upper/Lower breakout levels
    - ADX: Trend strength filter (> 20)
    - Choppiness Index (CHOP): Market state filter (< 40 = trending)
    - EMA 50: Trend direction confirmation
    - ATR: Trailing stop-loss calculation
    
    Entry Logic:
    - LONG: Close breaks above DC upper + ADX > 20 + CHOP < 40 + Close > EMA50
    - SHORT: Close breaks below DC lower + ADX > 20 + CHOP < 40 + Close < EMA50
    
    Exit Logic:
    - Trailing Stop: ATR-based (3x ATR trailing from highest high/lowest low)
    
    Timeframe: 1h (designed for perpetual futures)
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI (disabled; exits handled by custom_exit and custom_stoploss)
    minimal_roi = {"0": 100}

    # Initial stoploss: wide enough to let custom_stoploss ATR logic take control
    stoploss = -0.50

    # Trailing stoploss (disabled; handled in custom_stoploss)
    trailing_stop = False

    # Optimal timeframe for the strategy
    timeframe = "1h"

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters - Donchian
    donchian_period = IntParameter(low=10, high=30, default=20, space="buy", optimize=True, load=True)

    # Hyperoptable parameters - ADX
    adx_threshold = IntParameter(low=15, high=30, default=25, space="buy", optimize=True, load=True)

    # Hyperoptable parameters - Choppiness
    chop_threshold = DecimalParameter(
        low=30.0,
        high=61.8,
        default=45.0,
        decimals=1,
        space="buy",
        optimize=True,
        load=True,
    )
    chop_period = IntParameter(low=10, high=20, default=14, space="buy", optimize=True, load=True)

    # Hyperoptable parameters - Trailing SL
    atr_multiplier = DecimalParameter(
        low=1.5,
        high=3.0,
        default=2.0,
        decimals=1,
        space="sell",
        optimize=True,
        load=True,
    )

    # Hyperoptable parameters - Risk:Reward ratio for TP
    rr_ratio = DecimalParameter(
        low=1.0,
        high=4.0,
        default=2.0,
        decimals=1,
        space="sell",
        optimize=True,
        load=True,
    )

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Plot configuration for FreqUI visualization
    plot_config = {
        "main_plot": {
            "dc_upper": {"color": "green"},
            "dc_lower": {"color": "red"},
            "dc_middle": {"color": "gray", "type": "line"},
            "dc_exit_upper": {"color": "lightgreen", "type": "line"},
            "dc_exit_lower": {"color": "salmon", "type": "line"},
            "ema50": {"color": "blue"},
        },
        "subplots": {
            "ADX": {
                "adx": {"color": "purple"},
                "plus_di": {"color": "green"},
                "minus_di": {"color": "red"},
            },
            "CHOP": {
                "chop": {"color": "orange"},
            },
        },
    }

    def informative_pairs(self):
        """Define additional informative pair/interval combinations."""
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds Donchian Channel, ADX, Choppiness Index, EMA, and ATR indicators.

        Indicators:
        - dc_upper: Donchian Channel upper (highest high of period)
        - dc_lower: Donchian Channel lower (lowest low of period)
        - adx: Average Directional Index for trend strength
        - chop: Choppiness Index (< 40 = trending, > 60 = choppy)
        - ema50: 50-period EMA for trend direction
        - atr: Average True Range for trailing SL
        """
        # Donchian Channel (hyperoptable period)
        # Exclude current candle (.shift(1)) so close can break above/below the band
        period = self.donchian_period.value
        dataframe["dc_upper"] = dataframe["high"].shift(1).rolling(window=period).max()
        dataframe["dc_lower"] = dataframe["low"].shift(1).rolling(window=period).min()
        dataframe["dc_middle"] = (dataframe["dc_upper"] + dataframe["dc_lower"]) / 2

        # Turtle-style exit channel: half-period counter-breakout
        exit_period = max(period // 2, 5)
        dataframe["dc_exit_upper"] = dataframe["high"].shift(1).rolling(window=exit_period).max()
        dataframe["dc_exit_lower"] = dataframe["low"].shift(1).rolling(window=exit_period).min()

        # ADX + Directional Indicators (14 periods)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        # Choppiness Index (hyperoptable period)
        chop_period = self.chop_period.value
        tr = pd.Series(
            ta.TRANGE(dataframe["high"], dataframe["low"], dataframe["close"]),
            index=dataframe.index,
        )
        sum_tr = tr.rolling(chop_period).sum()
        hh = dataframe["high"].rolling(chop_period).max()
        ll = dataframe["low"].rolling(chop_period).min()
        denominator = hh - ll
        denominator = denominator.replace(0, np.nan)  # Avoid division by zero
        chop = 100 * np.log10(sum_tr / denominator) / np.log10(chop_period)
        dataframe["chop"] = chop

        # 50-period EMA for trend direction
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        # Volume SMA for breakout confirmation
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()

        # ATR (14 periods) for trailing SL
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry logic: Donchian breakout + Trend Filter + Whale Boost."""
        pair = metadata["pair"]
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # 1. Standard technical conditions
        long_technical = (
            qtpylib.crossed_above(dataframe["close"], dataframe["dc_upper"])
            & (dataframe["adx"] > self.adx_threshold.value)
            & (dataframe["plus_di"] > dataframe["minus_di"])
            & (dataframe["chop"] < self.chop_threshold.value)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["volume"] > dataframe["volume_sma"])
        )
        short_technical = (
            qtpylib.crossed_below(dataframe["close"], dataframe["dc_lower"])
            & (dataframe["adx"] > self.adx_threshold.value)
            & (dataframe["minus_di"] > dataframe["plus_di"])
            & (dataframe["chop"] < self.chop_threshold.value)
            & (dataframe["close"] < dataframe["ema50"])
            & (dataframe["volume"] > dataframe["volume_sma"])
        )

        dataframe.loc[long_technical, "enter_long"] = 1
        dataframe.loc[long_technical, "enter_tag"] = "technical_breakout"
        dataframe.loc[short_technical, "enter_short"] = 1
        dataframe.loc[short_technical, "enter_tag"] = "technical_breakdown"

        # 2. Intelligent Layer: Whale Boost
        whale_signal = self.whale_provider.get_signal(pair)
        if whale_signal:
            if whale_signal.signal_type == "BUY":
                # Relaxed: Only need close > EMA50 and breakout if whale buy
                whale_long = qtpylib.crossed_above(dataframe["close"], dataframe["dc_upper"]) & (
                    dataframe["close"] > dataframe["ema50"]
                )
                dataframe.loc[whale_long, "enter_long"] = 1
                dataframe.loc[whale_long, "enter_tag"] = "whale_boost_breakout"
            elif whale_signal.signal_type == "SELL":
                whale_short = qtpylib.crossed_below(dataframe["close"], dataframe["dc_lower"]) & (
                    dataframe["close"] < dataframe["ema50"]
                )
                dataframe.loc[whale_short, "enter_short"] = 1
                dataframe.loc[whale_short, "enter_tag"] = "whale_boost_breakdown"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Turtle-style exit: close breaks the shorter-period counter-DC channel.
        Long exit: close drops below the exit-period lowest low.
        Short exit: close rises above the exit-period highest high.
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Exit long: close crosses below the shorter-period DC lower
        dataframe.loc[
            qtpylib.crossed_below(dataframe["close"], dataframe["dc_exit_lower"]),
            "exit_long",
        ] = 1

        # Exit short: close crosses above the shorter-period DC upper
        dataframe.loc[
            qtpylib.crossed_above(dataframe["close"], dataframe["dc_exit_upper"]),
            "exit_short",
        ] = 1

        return dataframe

    def _get_entry_atr(self, pair: str, trade: Trade):
        """Get the entry candle ATR and dataframe slice from entry to current."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        candle_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        entry_candles = dataframe.loc[dataframe["date"] == candle_date]
        if entry_candles.empty:
            return None

        entry_candle = entry_candles.iloc[-1]
        entry_atr = entry_candle["atr"]
        if pd.isna(entry_atr) or entry_atr <= 0:
            return None

        current_candle = dataframe.iloc[-1]
        current_atr = current_candle["atr"]
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = entry_atr

        entry_idx = dataframe.index.get_loc(entry_candles.index[0])
        current_idx = len(dataframe) - 1
        slice_df = dataframe.iloc[entry_idx : current_idx + 1]

        return {
            "entry_atr": entry_atr,
            "current_atr": current_atr,
            "slice_df": slice_df,
            "entry_idx": entry_idx,
            "current_idx": current_idx,
        }

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
        ATR-based R:R take-profit.

        Risk = atr_multiplier * entry_ATR (same as initial SL distance).
        TP triggers when unrealized profit >= rr_ratio * risk.
        """
        info = self._get_entry_atr(pair, trade)
        if info is None:
            return None

        entry_atr = info["entry_atr"]
        multiplier = self.atr_multiplier.value
        risk_distance = multiplier * entry_atr
        tp_distance = self.rr_ratio.value * risk_distance

        if not trade.is_short:
            tp_price = trade.open_rate + tp_distance
            if current_rate >= tp_price:
                return f"tp_rr_{self.rr_ratio.value}"
        else:
            tp_price = trade.open_rate - tp_distance
            if current_rate <= tp_price:
                return f"tp_rr_{self.rr_ratio.value}"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """
        ATR-based trailing stoploss with breakeven logic.

        Phases:
        1. Initial: SL at entry - (ATR * multiplier)
        2. Breakeven: once price moves 1x ATR in favor, move SL to entry (breakeven)
        3. Trailing: trail from highest high (long) / lowest low (short) - (ATR * multiplier)

        Returns relative stoploss (negative value).
        """
        info = self._get_entry_atr(pair, trade)
        if info is None:
            return -1

        entry_atr = info["entry_atr"]
        current_atr = info["current_atr"]
        slice_df = info["slice_df"]
        multiplier = self.atr_multiplier.value

        if not trade.is_short:
            max_high = slice_df["high"].max()
            favorable_move = max_high - trade.open_rate

            if favorable_move >= entry_atr:
                # Phase 2+3: at least breakeven, trailing from max_high
                sl_abs = max(trade.open_rate, max_high - (multiplier * current_atr))
            else:
                # Phase 1: initial ATR stop
                sl_abs = trade.open_rate - (multiplier * current_atr)
        else:
            min_low = slice_df["low"].min()
            favorable_move = trade.open_rate - min_low

            if favorable_move >= entry_atr:
                # Phase 2+3: at least breakeven, trailing from min_low
                sl_abs = min(trade.open_rate, min_low + (multiplier * current_atr))
            else:
                # Phase 1: initial ATR stop
                sl_abs = trade.open_rate + (multiplier * current_atr)

        sl_rel = stoploss_from_absolute(sl_abs, current_rate, is_short=trade.is_short)
        return sl_rel
