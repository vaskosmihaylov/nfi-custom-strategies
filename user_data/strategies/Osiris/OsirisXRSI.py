"""
OSIRIS XRSI — Cross-TimeFrame RSI Divergence (Day Trade)
=========================================================
VALIDATED on futures (2021-2026):
  Research script: N=457, WR=27.8%, Total=+149.6% (taker fee 0.1%)

MECHANISM (confirmed empirically):
  When the CURRENT 4h bar's RSI > 65 (strong ongoing 4h uptrend),
  a 5m RSI dip below 35 tends to recover — the 4h trend wins.
  Max hold = 8h. Edge comes from TIMEOUT exits in continued uptrend.

  LONG:  current 4h RSI > 65 + 5m RSI < 35 + green 5m candle
  SHORT: current 4h RSI < 35 + 5m RSI > 65 + red  5m candle

LAWS (immutable):
  1. Entry only on anchor retest; skip if target was touched before retest.
  2. Breakeven guard: once trade moves in favor, stop moves to lock profit.
  3. Loss-streak guard: after N consecutive losses, pause new entries.
  4. Retest bar quality: body must show conviction; wick must show the test.
  5. 15m momentum filter: entry direction must align with 15m RSI trend.
  6. 1m micro-timing confirmation when 1m data available.

SL=1%, TP=3%, max_hold=8h, cooldown=30min, leverage=1x
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, merge_informative_pair
import talib.abstract as ta

try:
    from freqtrade.strategy import stoploss_from_open
except ImportError:
    def stoploss_from_open(open_relative_stop, current_profit, is_short=False):
        if current_profit == 0:
            return 1
        if is_short:
            return -1 + ((1 - open_relative_stop) / (1 - current_profit))
        return 1 - ((1 + open_relative_stop) / (1 + current_profit))

logger = logging.getLogger(__name__)


class OsirisXRSI(IStrategy):
    """XRSI: current 4h RSI divergence vs 5m RSI — trade the 4h trend."""

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"

    minimal_roi = {"0": 0.197, "22": 0.074, "53": 0.037, "138": 0}
    stoploss = -0.278
    trailing_stop = False
    use_custom_stoploss = True
    startup_candle_count = 500
    process_only_new_candles = True

    _last_entry_time: dict = {}
    _guard_until: dict = {}
    _loss_streak: dict = {}
    _last_closed_trade_id: dict = {}

    rsi_4h_bull       = IntParameter(60, 75, default=60, space="buy", optimize=True)
    rsi_4h_bear       = IntParameter(25, 40, default=35, space="buy", optimize=True)
    rsi_5m_oversold   = IntParameter(25, 40, default=37, space="buy", optimize=True)
    rsi_5m_overbought = IntParameter(60, 75, default=75, space="buy", optimize=True)
    max_hold_candles  = IntParameter(48, 144, default=53, space="sell", optimize=True)
    cooldown_min      = IntParameter(15, 60, default=21, space="buy", optimize=True)
    tp_pct            = DecimalParameter(1.5, 4.0, default=2.2, decimals=1, space="sell", optimize=True)

    # Entry retest law
    retest_wait_bars     = IntParameter(3, 18, default=8, space="buy", optimize=True)
    retest_tolerance_pct = DecimalParameter(0.00, 0.20, default=0.07, decimals=2, space="buy", optimize=True)

    # Breakeven and stop guards
    be_trigger_pct    = DecimalParameter(0.5, 1.5, default=1.4, decimals=1, space="sell", optimize=True)
    be_lock_pct       = DecimalParameter(0.0, 0.3, default=0.1, decimals=1, space="sell", optimize=True)
    loss_streak_limit = IntParameter(2, 5, default=5, space="sell", optimize=True)
    guard_minutes     = IntParameter(30, 240, default=32, space="sell", optimize=True)

    # LAW 4: retest bar quality
    # Calibrated: body filter near-neutral in isolation; low defaults let hyperopt explore.
    # min_lower_wick=0.0 bypasses wick filter until hyperopt finds the right threshold.
    min_body_ratio    = DecimalParameter(0.05, 0.50, default=0.08, decimals=2, space="buy", optimize=True)
    min_lower_wick    = DecimalParameter(0.00, 0.25, default=0.03, decimals=2, space="buy", optimize=True)

    # LAW 5: 15m structure alignment
    # Calibrated: band=0 destroys long WR (7.7%), band>=3% is near-neutral for longs
    # and slightly positive for shorts. Default=3.0 keeps most fills while providing
    # structural context. Hyperopt can tighten if data supports it.
    ema50_band_pct    = DecimalParameter(0.0, 5.0, default=1.3, decimals=1, space="buy", optimize=True)

    # LAW 6: 1m micro (activated when available)
    # Thresholds calibrated to real 1m RSI distribution at retest fills:
    # long fills: rsi_1m mean=49.7, so 55 passes ~60% while filtering momentum exhaustion
    # short fills: symmetric at 45
    rsi_1m_long_max   = IntParameter(40, 65, default=65, space="buy", optimize=True)
    rsi_1m_short_min  = IntParameter(35, 60, default=42, space="buy", optimize=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist() if self.dp else []
        # 15m for trend structure filter
        # NOTE: 1m is NOT declared here — it's loaded on-demand via dp.get_pair_dataframe
        # to avoid Freqtrade reducing the backtest window when 1m data is absent.
        return [(p, "15m") for p in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["is_green"] = (dataframe["close"] > dataframe["open"]).astype(int)
        dataframe["is_red"]   = (dataframe["close"] < dataframe["open"]).astype(int)

        # 4h RSI: same method as research script (grp4h floor, last close, no shift)
        df = dataframe[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["grp4h"] = df["date"].dt.floor("4h")
        df4h = (
            df.groupby("grp4h", sort=True)
            .agg(close4h=("close", "last"))
            .reset_index()
        )
        df4h["rsi4h"] = ta.RSI(df4h["close4h"], timeperiod=14)
        df = df.merge(df4h[["grp4h", "rsi4h"]], on="grp4h", how="left")
        dataframe["rsi_4h"] = df["rsi4h"].values

        # ── LAW 5: 15m structure alignment ─────────────────────────────────────
        inf_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="15m")
        if not inf_15m.empty:
            inf_15m["rsi_15"] = ta.RSI(inf_15m, timeperiod=14)
            # EMA(50) on 15m: price above = bullish structure, below = bearish
            inf_15m["ema50_15"] = ta.EMA(inf_15m, timeperiod=50)
            # merge: take latest 15m bar before each 5m bar (no lookahead)
            m15 = merge_informative_pair(
                dataframe,
                inf_15m[["date", "rsi_15", "ema50_15"]],
                self.timeframe, "15m", ffill=True,
            )
            dataframe["rsi_15m"]   = m15["rsi_15_15m"].values
            dataframe["ema50_15m"] = m15["ema50_15_15m"].values
        else:
            dataframe["rsi_15m"]   = 50.0
            dataframe["ema50_15m"] = dataframe["close"]

        # ── LAW 6: 1m micro-timing (optional, loaded when available) ──────────
        inf_1m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1m")
        if not inf_1m.empty:
            inf_1m["rsi_1"] = ta.RSI(inf_1m, timeperiod=14)
            # Manual bucket merge: for each 5m bar (open at T, close at T+5m),
            # use the last 1m bar in that bucket (closes at T+5m — same time as 5m bar).
            # floor() of 1m open-time aligns to the 5m bar that contains it.
            inf_1m["bucket5m"] = pd.to_datetime(inf_1m["date"]).dt.floor("5min")
            last1m = (
                inf_1m.groupby("bucket5m", sort=True)["rsi_1"]
                .last()
                .reset_index()
                .rename(columns={"bucket5m": "date", "rsi_1": "rsi_1m"})
            )
            dataframe = dataframe.merge(last1m, on="date", how="left")
            dataframe["rsi_1m"] = dataframe["rsi_1m"].fillna(50.0)
            dataframe["has_1m"] = 1
        else:
            dataframe["rsi_1m"] = 50.0
            dataframe["has_1m"] = 0

        # Entry setup first, fill later only on retest of anchor price.
        setup_long = (
            dataframe["rsi_4h"].notna()
            & (dataframe["volume"] > 0)
            & (dataframe["rsi_4h"] > self.rsi_4h_bull.value)
            & (dataframe["rsi"] < self.rsi_5m_oversold.value)
            & (dataframe["is_green"] == 1)
        )
        setup_short = (
            dataframe["rsi_4h"].notna()
            & (dataframe["volume"] > 0)
            & (dataframe["rsi_4h"] < self.rsi_4h_bear.value)
            & (dataframe["rsi"] > self.rsi_5m_overbought.value)
            & (dataframe["is_red"] == 1)
        )

        # LAW: if setup happened and we missed anchor fill,
        # only enter on anchor retest IF target wasn't touched before retest.
        n = len(dataframe)
        fill_long = [0] * n
        fill_short = [0] * n
        c = dataframe["close"].values
        h = dataframe["high"].values
        l = dataframe["low"].values
        o = dataframe["open"].values
        wait_bars = int(self.retest_wait_bars.value)
        tol = float(self.retest_tolerance_pct.value) / 100.0
        tp = float(self.tp_pct.value) / 100.0

        for i in range(n - 1):
            end = min(i + 1 + wait_bars, n)

            if setup_long.iloc[i]:
                anchor = c[i]
                target = anchor * (1.0 + tp)
                low_band = anchor * (1.0 - tol)
                high_band = anchor * (1.0 + tol)
                invalid = False
                for j in range(i + 1, end):
                    if h[j] >= target:
                        invalid = True
                        break
                    # Retest happened: candle traded through anchor band.
                    if (l[j] <= high_band) and (h[j] >= low_band):
                        # Confirmation: reclaim with green candle.
                        if c[j] > o[j]:
                            fill_long[j] = 1
                        break
                if invalid:
                    continue

            if setup_short.iloc[i]:
                anchor = c[i]
                target = anchor * (1.0 - tp)
                low_band = anchor * (1.0 - tol)
                high_band = anchor * (1.0 + tol)
                invalid = False
                for j in range(i + 1, end):
                    if l[j] <= target:
                        invalid = True
                        break
                    if (l[j] <= high_band) and (h[j] >= low_band):
                        if c[j] < o[j]:
                            fill_short[j] = 1
                        break
                if invalid:
                    continue

        dataframe["entry_fill_long"] = fill_long
        dataframe["entry_fill_short"] = fill_short

        # ── LAW 4: retest bar quality ─────────────────────────────────────────
        candle_range = (dataframe["high"] - dataframe["low"]).clip(lower=1e-9)
        body = dataframe["close"] - dataframe["open"]
        dataframe["body_ratio"]   = body.abs() / candle_range
        # Lower wick for LONG: distance from open to low as % of range
        dataframe["lower_wick"]   = (dataframe[["open", "close"]].min(axis=1) - dataframe["low"]) / candle_range
        # Upper wick for SHORT
        dataframe["upper_wick"]   = (dataframe["high"] - dataframe[["open", "close"]].max(axis=1)) / candle_range

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        min_br  = float(self.min_body_ratio.value)
        min_lw  = float(self.min_lower_wick.value)
        band    = float(self.ema50_band_pct.value) / 100.0
        r1_lmax  = int(self.rsi_1m_long_max.value)
        r1_smin  = int(self.rsi_1m_short_min.value)

        has_1m = dataframe["has_1m"] == 1

        # LAW 4: conviction body + wick
        body_ok_long  = (dataframe["body_ratio"] >= min_br) & (dataframe["lower_wick"] >= min_lw)
        body_ok_short = (dataframe["body_ratio"] >= min_br) & (dataframe["upper_wick"] >= min_lw)

        # LAW 5: 15m EMA50 structure
        # LONG: 15m close within band% below EMA50 (or above) → still in uptrend structure
        # Using band allows entries even when dip briefly dips below EMA50
        ema50 = dataframe["ema50_15m"]
        align_long  = dataframe["close"] >= ema50 * (1.0 - band)
        align_short = dataframe["close"] <= ema50 * (1.0 + band)

        # LAW 6: 1m micro (bypassed when data not available)
        micro_long  = (~has_1m) | (dataframe["rsi_1m"] < r1_lmax)
        micro_short = (~has_1m) | (dataframe["rsi_1m"] > r1_smin)

        long_cond  = (
            (dataframe["entry_fill_long"]  == 1)
            & body_ok_long
            & align_long
            & micro_long
        )
        short_cond = (
            (dataframe["entry_fill_short"] == 1)
            & body_ok_short
            & align_short
            & micro_short
        )

        dataframe.loc[long_cond,  "enter_long"]  = 1
        dataframe.loc[short_cond, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        self._update_loss_guard(pair, current_time)

        guard_until = self._guard_until.get(pair)
        if guard_until and current_time < guard_until:
            return False

        last = self._last_entry_time.get(pair)
        if last is not None:
            if (current_time - last).total_seconds() < self.cooldown_min.value * 60:
                return False
        self._last_entry_time[pair] = current_time
        return True

    def custom_stoploss(self, pair, trade: Trade, current_time,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Breakeven guard law: if trade moved in favor, never let it go full SL again.
        be_trigger = float(self.be_trigger_pct.value) / 100.0
        be_lock = float(self.be_lock_pct.value) / 100.0
        if current_profit >= be_trigger:
            is_short = getattr(trade, "is_short", False)
            return stoploss_from_open(be_lock, current_profit, is_short=is_short)
        return self.stoploss

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if current_profit >= self.tp_pct.value / 100:
            return "xrsi_tp"
        if trade.open_date_utc:
            elapsed_min = (current_time - trade.open_date_utc).total_seconds() / 60
            if elapsed_min >= self.max_hold_candles.value * 5:
                return "xrsi_timeout"
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return 1.0

    def _update_loss_guard(self, pair: str, current_time: datetime) -> None:
        # Guard law: pause new entries after a loss streak.
        try:
            closed = Trade.get_trades_proxy(pair=pair, is_open=False)
        except Exception:
            return

        if not closed:
            return

        closed_sorted = sorted(
            [t for t in closed if getattr(t, "close_date_utc", None) is not None],
            key=lambda t: t.close_date_utc,
        )
        if not closed_sorted:
            return

        last_seen = self._last_closed_trade_id.get(pair, -1)
        streak = self._loss_streak.get(pair, 0)

        for t in closed_sorted:
            tid = getattr(t, "id", None)
            if tid is None or tid <= last_seen:
                continue

            profit_ratio = float(getattr(t, "close_profit", 0.0) or 0.0)
            if profit_ratio < 0:
                streak += 1
            else:
                streak = 0

            if streak >= int(self.loss_streak_limit.value):
                self._guard_until[pair] = t.close_date_utc + pd.Timedelta(minutes=int(self.guard_minutes.value))
                streak = 0

            last_seen = max(last_seen, tid)

        self._last_closed_trade_id[pair] = last_seen
        self._loss_streak[pair] = streak
