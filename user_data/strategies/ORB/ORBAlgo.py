import logging
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

logger = logging.getLogger(__name__)


class ORBAlgo(IStrategy):
    """
    Opening Range Breakout (ORB) strategy for crypto trading.

    Adapted from the "ORB Algo | Flux Charts" Pine Script indicator.

    Concept:
    - Defines an Opening Range (OR) from the first N minutes of each daily session.
    - Waits for price to break above OR high (long) or below OR low (short).
    - Confirms breakout via retests of the ORB level (sensitivity-based).
    - Manages exits with dynamic (EMA) or ATR-based take profits.
    - Uses adaptive stop loss based on the ORB midpoint.

    Since crypto markets are 24/7, "sessions" are defined as daily periods
    starting at a configurable UTC hour (default 0 = midnight UTC).
    """

    INTERFACE_VERSION = 3

    # --- Strategy settings ---
    timeframe = "5m"
    can_short = True
    startup_candle_count = 200
    process_only_new_candles = True
    use_exit_signal = True
    use_custom_stoploss = True

    stoploss = -0.10

    minimal_roi = {"0": 100}

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # --- Tunable parameters ---
    orb_period_minutes = IntParameter(
        15,
        120,
        default=30,
        space="buy",
        optimize=True,
        load=True,
    )
    session_start_hour = IntParameter(
        0,
        23,
        default=0,
        space="buy",
        optimize=True,
        load=True,
    )
    sensitivity = CategoricalParameter(
        ["High", "Medium", "Low", "Lowest"],
        default="Medium",
        space="buy",
        optimize=True,
        load=True,
    )
    breakout_condition = CategoricalParameter(
        ["Close", "EMA"],
        default="Close",
        space="buy",
        optimize=True,
        load=True,
    )
    tp_method = CategoricalParameter(
        ["Dynamic", "ATR"],
        default="Dynamic",
        space="buy",
        optimize=True,
        load=True,
    )
    ema_length = IntParameter(
        4,
        34,
        default=9,
        space="buy",
        optimize=True,
        load=True,
    )
    sl_method = CategoricalParameter(
        ["Safer", "Balanced", "Risky"],
        default="Balanced",
        space="buy",
        optimize=True,
        load=True,
    )
    adaptive_sl = CategoricalParameter(
        [True, False],
        default=True,
        space="buy",
        optimize=True,
        load=True,
    )

    # --- Constants (from Pine Script) ---
    MIN_PROFIT_PCT = 0.20
    MIN_PROFIT_INCREMENT_PCT = 0.075
    ATR_TP1_MULT = 0.75
    ATR_TP2_MULT = 1.50
    ATR_TP3_MULT = 2.25

    # --- Sensitivity -> retests mapping ---
    RETESTS_MAP = {"High": 0, "Medium": 1, "Low": 2, "Lowest": 3}

    def informative_pairs(self):
        return []

    @staticmethod
    def _timeframe_to_minutes(tf: str) -> int:
        """Convert timeframe string like '5m', '1h' to minutes."""
        multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
        suffix = tf[-1]
        return int(tf[:-1]) * multipliers.get(suffix, 1)

    # ---------------------------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tf_minutes = self._timeframe_to_minutes(self.timeframe)
        orb_candles = max(1, self.orb_period_minutes.value // tf_minutes)
        session_start = self.session_start_hour.value

        # --- Core indicators ---
        # EMA on HL2 (matches Pine: ta.ema((high + low) / 2.0, emaLength))
        hl2 = (dataframe["high"] + dataframe["low"]) / 2.0
        dataframe["ema_hl2"] = ta.EMA(hl2, timeperiod=self.ema_length.value)

        # ATR(12) for TP/SL calculations
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=12)

        # --- Session identification ---
        session_offset = pd.Timedelta(hours=session_start)
        dataframe["session_id"] = (dataframe["date"] - session_offset).dt.date

        # Candle index within each session
        dataframe["session_idx"] = dataframe.groupby("session_id").cumcount()

        # Mark ORB period candles (first N candles of the session)
        dataframe["is_orb"] = dataframe["session_idx"] < orb_candles

        # --- ORB levels per session ---
        # Compute from only the ORB period candles, then broadcast to all session candles
        orb_highs = dataframe.loc[dataframe["is_orb"]].groupby("session_id")["high"].max()
        orb_lows = dataframe.loc[dataframe["is_orb"]].groupby("session_id")["low"].min()
        dataframe["orb_high"] = dataframe["session_id"].map(orb_highs)
        dataframe["orb_low"] = dataframe["session_id"].map(orb_lows)

        # Invalidate ORB levels during the ORB period (no signals until range is set)
        dataframe.loc[dataframe["is_orb"], ["orb_high", "orb_low"]] = np.nan

        # ORB midpoint
        dataframe["orb_mid"] = (dataframe["orb_high"] + dataframe["orb_low"]) / 2.0

        # --- SL levels based on ORB ---
        dataframe = self._compute_sl_levels(dataframe)

        # --- Breakout / retest detection ---
        dataframe = self._detect_entries(dataframe, orb_candles)

        # Cleanup helper columns
        dataframe.drop(
            columns=["session_idx", "is_orb"],
            errors="ignore",
            inplace=True,
        )

        return dataframe

    # ---------------------------------------------------------------------------
    # SL level computation
    # ---------------------------------------------------------------------------
    def _compute_sl_levels(self, dataframe: DataFrame) -> DataFrame:
        """Compute ORB-based stop loss prices for long and short entries."""
        sl = self.sl_method.value
        h = dataframe["orb_high"]
        l = dataframe["orb_low"]  # noqa: E741
        mid = dataframe["orb_mid"]

        if sl == "Safer":
            # Tightest SL: halfway between ORB midpoint and the breakout boundary
            dataframe["orb_sl_long"] = (mid + h) / 2.0
            dataframe["orb_sl_short"] = (mid + l) / 2.0
        elif sl == "Risky":
            # Widest SL: halfway between ORB midpoint and the opposite boundary
            dataframe["orb_sl_long"] = (mid + l) / 2.0
            dataframe["orb_sl_short"] = (mid + h) / 2.0
        else:
            # Balanced (default): SL at the ORB midpoint
            dataframe["orb_sl_long"] = mid
            dataframe["orb_sl_short"] = mid

        return dataframe

    # ---------------------------------------------------------------------------
    # Breakout & retest state machine
    # ---------------------------------------------------------------------------
    def _detect_entries(self, dataframe: DataFrame, orb_candles: int) -> DataFrame:
        """
        Detect ORB breakouts and retests, produce entry signals.

        Implements the Pine Script state machine:
          Opening Range -> Waiting For Breakouts -> In Breakout -> Entry Taken

        For each session, at most one entry (long or short) is generated.
        """
        retests_needed = self.RETESTS_MAP[self.sensitivity.value]
        n = len(dataframe)

        enter_long = np.zeros(n, dtype=np.int8)
        enter_short = np.zeros(n, dtype=np.int8)
        entry_atr = np.full(n, np.nan)

        # Pre-extract arrays for fast iteration
        post_orb = (~dataframe["is_orb"] & dataframe["orb_high"].notna()).values
        atr_arr = dataframe["atr"].values
        if self.breakout_condition.value == "EMA":
            cond_price = dataframe["ema_hl2"].values
        else:
            cond_price = dataframe["close"].values

        orb_high = dataframe["orb_high"].values
        orb_low = dataframe["orb_low"].values
        close_arr = dataframe["close"].values
        low_arr = dataframe["low"].values
        high_arr = dataframe["high"].values
        session_ids = dataframe["session_id"].values

        # State tracking (reset per session)
        current_session = None
        state = "orb"  # orb | waiting | breakout | entered
        is_bull = False
        retest_count = 0
        breakout_idx = -1

        for i in range(n):
            # New session -> reset state
            if session_ids[i] != current_session:
                current_session = session_ids[i]
                state = "orb"
                retest_count = 0
                breakout_idx = -1

            # Skip ORB period candles
            if not post_orb[i]:
                continue

            # Transition from ORB period to waiting
            if state == "orb":
                state = "waiting"

            if state == "waiting":
                # Detect initial breakout
                if cond_price[i] > orb_high[i]:
                    state = "breakout"
                    is_bull = True
                    retest_count = 0
                    breakout_idx = i
                elif cond_price[i] < orb_low[i]:
                    state = "breakout"
                    is_bull = False
                    retest_count = 0
                    breakout_idx = i

                # For High sensitivity (0 retests): entry can trigger on the breakout bar
                if state == "breakout" and retests_needed == 0:
                    state = "entered"
                    entry_atr[i] = atr_arr[i]
                    if is_bull:
                        enter_long[i] = 1
                    else:
                        enter_short[i] = 1
                    continue

                continue

            if state == "breakout":
                # Failed breakout: price retreats back inside ORB range
                if is_bull and close_arr[i] < orb_high[i]:
                    state = "waiting"
                    continue
                if not is_bull and close_arr[i] > orb_low[i]:
                    state = "waiting"
                    continue

                # Retest: candle dips to the ORB boundary but closes beyond it
                if i > breakout_idx:
                    if is_bull and close_arr[i] > orb_high[i] and low_arr[i] <= orb_high[i]:
                        retest_count += 1
                    elif not is_bull and close_arr[i] < orb_low[i] and high_arr[i] >= orb_low[i]:
                        retest_count += 1

                # Enough retests -> entry
                if retest_count >= retests_needed:
                    state = "entered"
                    entry_atr[i] = atr_arr[i]
                    if is_bull:
                        enter_long[i] = 1
                    else:
                        enter_short[i] = 1
                continue

            # state == "entered": only one entry per session, skip remaining bars

        dataframe["orb_enter_long"] = enter_long
        dataframe["orb_enter_short"] = enter_short
        dataframe["entry_atr"] = entry_atr
        # Forward-fill entry ATR so it persists through the session for TP calcs
        dataframe["entry_atr"] = dataframe.groupby("session_id")["entry_atr"].ffill()
        return dataframe

    # ---------------------------------------------------------------------------
    # Entry / exit trends
    # ---------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["orb_enter_long"] == 1, ["enter_long", "enter_tag"]] = (1, "orb_long")
        dataframe.loc[dataframe["orb_enter_short"] == 1, ["enter_short", "enter_tag"]] = (1, "orb_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

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
        """Store entry-time ATR via dataframe for use in ATR-based TP."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            current_candle = dataframe.iloc[-1]
            # entry_atr column was set at the entry bar and forward-filled
            atr_val = current_candle.get("entry_atr")
            if not pd.isna(atr_val):
                # Store in a strategy-level dict keyed by pair+time for
                # retrieval in confirm_trade_exit / custom_exit
                self._pending_entry_atr[(pair, current_time.isoformat())] = float(atr_val)
        return True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Tracks entry-time ATR per trade (pair, open_date_iso) -> atr_value
        self._pending_entry_atr: dict[tuple[str, str], float] = {}

    # ---------------------------------------------------------------------------
    # Custom stoploss
    # ---------------------------------------------------------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        ORB-based dynamic stoploss with staged adaptive SL.

        Base SL is placed at the ORB midpoint (or shifted per sl_method).
        When adaptive_sl is enabled and TP1 has been reached, SL moves to
        breakeven (entry price), matching the Pine Script behavior.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return 1.0

        current_candle = dataframe.iloc[-1]

        # Pick the correct SL price for the trade direction
        if trade.is_short:
            sl_price = current_candle.get("orb_sl_short")
        else:
            sl_price = current_candle.get("orb_sl_long")

        if pd.isna(sl_price) or sl_price <= 0:
            return 1.0

        # Adaptive SL: move to breakeven once TP1 has been reached
        # Pine: after tp1 is hit with adaptiveSL, slPrice = entryPrice
        if self.adaptive_sl.value:
            tp_stage = trade.get_custom_data(key="orb_tp_stage")
            if tp_stage is not None and tp_stage >= 1:
                if not trade.is_short:
                    sl_price = max(sl_price, trade.open_rate)
                else:
                    sl_price = min(sl_price, trade.open_rate)

        # SL already breached -> exit immediately
        if not trade.is_short and current_rate <= sl_price:
            return -0.001
        if trade.is_short and current_rate >= sl_price:
            return -0.001

        # Return negative distance ratio from current rate
        sl_distance = abs(current_rate - sl_price) / current_rate
        return -sl_distance

    # ---------------------------------------------------------------------------
    # Custom exit (TP + session-end)
    # ---------------------------------------------------------------------------
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[Union[str, bool]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return None

        current_candle = dataframe.iloc[-1]

        # --- Session-end exit ---
        # Pine closes at previous bar's close when a new session begins.
        # Detect the last candle of the session: the *next* candle belongs to
        # a different session. We check whether the current candle is the last
        # of its session by looking at wether a new session is about to start.
        trade_session = self._get_session_date(trade.open_date_utc)
        current_session = current_candle.get("session_id")
        if trade_session is not None and current_session is not None:
            if current_session != trade_session:
                return "orb_session_end"

        # --- Staged Take Profit ---
        tp_stage = trade.get_custom_data(key="orb_tp_stage")
        if tp_stage is None:
            tp_stage = 0

        if self.tp_method.value == "Dynamic":
            return self._dynamic_tp_staged(trade, current_candle, tp_stage)

        if self.tp_method.value == "ATR":
            return self._atr_tp_staged(trade, current_candle, current_rate, tp_stage)

        return None

    def _dynamic_tp_staged(
        self,
        trade: Trade,
        candle,
        tp_stage: int,
    ) -> Optional[str]:
        """
        Staged Dynamic TP matching Pine Script logic.

        Pine tracks TP1/TP2/TP3 sequentially:
        - TP1: EMA profit >= MIN_PROFIT_PCT and close crosses back through EMA.
                Stores tp1Price = ema. Adaptive SL -> breakeven.
        - TP2: TP1 already hit, EMA > tp1Price by >= MIN_PROFIT_INCREMENT_PCT,
                and close crosses back.
        - TP3: TP2 already hit, same increment check. Full exit.

        Only TP3 triggers a full exit. TP1/TP2 update trade custom data
        and tighten the SL via custom_stoploss.
        """
        ema = candle.get("ema_hl2")
        close = candle.get("close")

        if pd.isna(ema) or pd.isna(close):
            return None

        is_long = not trade.is_short
        is_profitable = (ema > trade.open_rate) if is_long else (ema < trade.open_rate)
        ema_crossback = (close < ema) if is_long else (close > ema)
        profit_pct = abs(ema - trade.open_rate) / trade.open_rate * 100.0

        if not is_profitable or not ema_crossback:
            return None

        last_tp_price = trade.get_custom_data(key="orb_last_tp_price")

        # TP1: first take-profit level
        if tp_stage == 0 and profit_pct >= self.MIN_PROFIT_PCT:
            trade.set_custom_data(key="orb_tp_stage", value=1)
            trade.set_custom_data(key="orb_last_tp_price", value=float(ema))
            logger.info(f"{trade.pair} ORB TP1 hit at EMA={ema:.6f} (profit {profit_pct:.2f}%)")
            return None  # No exit yet — only SL tightens

        # TP2: second take-profit level
        if tp_stage == 1 and last_tp_price is not None:
            ema_beyond_tp1 = (ema > last_tp_price) if is_long else (ema < last_tp_price)
            increment_pct = abs(ema - last_tp_price) / last_tp_price * 100.0
            if ema_beyond_tp1 and increment_pct >= self.MIN_PROFIT_INCREMENT_PCT:
                trade.set_custom_data(key="orb_tp_stage", value=2)
                trade.set_custom_data(key="orb_last_tp_price", value=float(ema))
                logger.info(f"{trade.pair} ORB TP2 hit at EMA={ema:.6f} (increment {increment_pct:.2f}%)")
                return None  # No exit yet

        # TP3: third take-profit level -> full exit
        if tp_stage == 2 and last_tp_price is not None:
            ema_beyond_tp2 = (ema > last_tp_price) if is_long else (ema < last_tp_price)
            increment_pct = abs(ema - last_tp_price) / last_tp_price * 100.0
            if ema_beyond_tp2 and increment_pct >= self.MIN_PROFIT_INCREMENT_PCT:
                trade.set_custom_data(key="orb_tp_stage", value=3)
                logger.info(f"{trade.pair} ORB TP3 hit at EMA={ema:.6f} -> full exit")
                return "orb_tp3_dynamic"

        return None

    def _atr_tp_staged(
        self,
        trade: Trade,
        candle,
        current_rate: float,
        tp_stage: int,
    ) -> Optional[str]:
        """
        Staged ATR-based TP matching Pine Script logic.

        Uses entry-time ATR (stored in entry_atr column, persisted via
        _pending_entry_atr dict). Pine: lastORB.entryATR := atr at entry.

        TP levels are computed once from entry ATR:
        - TP1 = entry + entryATR * 0.75
        - TP2 = entry + entryATR * 1.50
        - TP3 = entry + entryATR * 2.25

        Only TP3 triggers a full exit. TP1 tightens SL to breakeven.
        """
        # Retrieve entry-time ATR
        entry_atr = self._get_entry_atr(trade, candle)
        if entry_atr is None or entry_atr <= 0:
            return None

        direction = -1 if trade.is_short else 1
        tp1 = trade.open_rate + entry_atr * self.ATR_TP1_MULT * direction
        tp2 = trade.open_rate + entry_atr * self.ATR_TP2_MULT * direction
        tp3 = trade.open_rate + entry_atr * self.ATR_TP3_MULT * direction

        is_long = not trade.is_short

        # TP1 check
        if tp_stage < 1:
            tp1_hit = (current_rate >= tp1) if is_long else (current_rate <= tp1)
            if tp1_hit:
                trade.set_custom_data(key="orb_tp_stage", value=1)
                logger.info(f"{trade.pair} ORB ATR TP1 hit at {current_rate:.6f} (target {tp1:.6f})")
                return None  # SL tightens via custom_stoploss

        # TP2 check
        if tp_stage == 1:
            tp2_hit = (current_rate >= tp2) if is_long else (current_rate <= tp2)
            if tp2_hit:
                trade.set_custom_data(key="orb_tp_stage", value=2)
                logger.info(f"{trade.pair} ORB ATR TP2 hit at {current_rate:.6f} (target {tp2:.6f})")
                return None

        # TP3 check -> full exit
        if tp_stage == 2:
            tp3_hit = (current_rate >= tp3) if is_long else (current_rate <= tp3)
            if tp3_hit:
                trade.set_custom_data(key="orb_tp_stage", value=3)
                logger.info(f"{trade.pair} ORB ATR TP3 hit at {current_rate:.6f} -> full exit")
                return "orb_tp3_atr"

        return None

    def _get_entry_atr(self, trade: Trade, candle) -> Optional[float]:
        """
        Retrieve entry-time ATR for the given trade.

        First checks the strategy-level cache (_pending_entry_atr), then
        falls back to the forward-filled entry_atr column in the dataframe.
        On first access, persists to trade custom data for durability.
        """
        # Check trade custom data first (persisted)
        stored = trade.get_custom_data(key="orb_entry_atr")
        if stored is not None:
            return float(stored)

        # Check strategy-level cache from confirm_trade_entry
        key = (trade.pair, trade.open_date_utc.isoformat() if trade.open_date_utc else None)
        if key in self._pending_entry_atr:
            val = self._pending_entry_atr.pop(key)
            trade.set_custom_data(key="orb_entry_atr", value=val)
            return val

        # Fallback: use forward-filled entry_atr from current candle
        atr_val = candle.get("entry_atr")
        if not pd.isna(atr_val):
            trade.set_custom_data(key="orb_entry_atr", value=float(atr_val))
            return float(atr_val)

        return None

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------
    def _get_session_date(self, dt: datetime):
        """Return the session date for a given datetime, adjusted by session_start_hour."""
        if dt is None:
            return None
        session_offset = pd.Timedelta(hours=self.session_start_hour.value)
        timestamp = pd.Timestamp(dt)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        adjusted = timestamp - session_offset
        return adjusted.date()

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
        return 1.0
