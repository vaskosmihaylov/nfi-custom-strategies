"""
OSIRIS DAY TRADE v7 — Deep Pullback + Pattern + Progressive Trail
================================================================
EVOLUÇÃO:
  v1-v3: stops apertados, WR 28%, PF 0.34 → falhou
  v4: stops 2 ATR, WR 33%, PF 0.43 → melhor mas ruim
  v5: stops 3 ATR + 1:1 R:R, WR 43%, PF 0.63 → progresso
  v6: trailing puro + pullback, WR 42%, PF 0.72, -40% → QUASE

v7 FIX:
  1. Pullback mais PROFUNDO (3+ candles OU RSI < 40)
  2. Reversal PATTERN (engulfing/hammer) → timing melhor
  3. ADX > 22 (só trends fortes) → menos falsas
  4. Shorts com filtro EXTRA (ADX > 28) → menos shorts ruins
  5. Max hold 48 candles (12h) → mais tempo pra winners
  6. Sem trend_flip exit (trailing faz o trabalho)
"""

import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional
from datetime import datetime, timedelta

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.persistence import Trade
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


class OsirisDayTradeV7(IStrategy):
    """
    OSIRIS v7 — Deep pullback + pattern + progressive trailing.
    Tighter entry criteria, wider time horizon, let winners run.
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "15m"

    minimal_roi = {"0": 0.50}  # disabled

    stoploss = -0.06

    trailing_stop = False
    use_custom_stoploss = True

    startup_candle_count = 200
    process_only_new_candles = True

    _daily_trades = {}
    _consecutive_losses = 0
    _last_loss_time = None

    # ═══════════════════════════════════════════════════════════════════
    # PARAMETERS
    # ═══════════════════════════════════════════════════════════════════

    # Trailing (ATR multiples)
    buy_sl_initial = DecimalParameter(3.0, 5.0, default=4.0, decimals=1, space="buy", optimize=True)
    buy_sl_trail_1 = DecimalParameter(2.5, 4.0, default=3.0, decimals=1, space="buy", optimize=True)
    buy_sl_trail_2 = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)
    buy_sl_trail_3 = DecimalParameter(1.0, 2.0, default=1.5, decimals=1, space="buy", optimize=True)

    # Trend: separate thresholds for long vs short
    buy_adx_long = IntParameter(15, 28, default=20, space="buy", optimize=True)
    buy_adx_short = IntParameter(20, 35, default=28, space="buy", optimize=True)

    # Pullback depth
    buy_pb_candles = IntParameter(2, 5, default=3, space="buy", optimize=True)
    buy_rsi_pb_long = IntParameter(30, 45, default=40, space="buy", optimize=True)
    buy_rsi_pb_short = IntParameter(55, 70, default=60, space="buy", optimize=True)

    # Volume
    buy_vol_min = DecimalParameter(0.4, 1.2, default=0.7, decimals=1, space="buy", optimize=True)

    # Max daily + hold
    buy_max_daily = IntParameter(6, 15, default=10, space="buy", optimize=True)
    buy_max_hold = IntParameter(24, 64, default=48, space="buy", optimize=True)

    # ═══════════════════════════════════════════════════════════════════
    # INFORMATIVE
    # ═══════════════════════════════════════════════════════════════════

    def informative_pairs(self):
        return [("BTC/USDT:USDT", "1h")]

    # ═══════════════════════════════════════════════════════════════════
    # INDICATORS
    # ═══════════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 15m core
        dataframe["ema9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        dataframe["vol_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["vol_sma"].replace(0, 1)

        # Candle anatomy
        c = dataframe["close"]
        o = dataframe["open"]
        h = dataframe["high"]
        l = dataframe["low"]

        dataframe["is_green"] = (c > o).astype(int)
        dataframe["is_red"] = (c < o).astype(int)
        dataframe["body"] = abs(c - o)
        dataframe["upper_wick"] = h - pd.concat([c, o], axis=1).max(axis=1)
        dataframe["lower_wick"] = pd.concat([c, o], axis=1).min(axis=1) - l
        dataframe["range"] = h - l

        # Previous candle
        dataframe["prev_body"] = dataframe["body"].shift(1)
        dataframe["prev_green"] = dataframe["is_green"].shift(1)
        dataframe["prev_red"] = dataframe["is_red"].shift(1)
        dataframe["prev_close"] = c.shift(1)
        dataframe["prev_open"] = o.shift(1)

        # ── Reversal patterns ──

        # Bullish engulfing: current green engulfs previous red body
        dataframe["bullish_engulf"] = (
            (dataframe["is_green"] == 1) &
            (dataframe["prev_red"] == 1) &
            (o <= dataframe["prev_close"]) &
            (c >= dataframe["prev_open"])
        ).astype(int)

        # Hammer: long lower wick, small body, green
        atr_safe = dataframe["atr"].replace(0, 1)
        dataframe["hammer"] = (
            (dataframe["is_green"] == 1) &
            (dataframe["lower_wick"] > 2 * dataframe["body"].replace(0, 0.01)) &
            (dataframe["lower_wick"] > 2 * dataframe["upper_wick"].replace(0, 0.01)) &
            (dataframe["range"] > 0.3 * atr_safe)
        ).astype(int)

        # Bearish engulfing
        dataframe["bearish_engulf"] = (
            (dataframe["is_red"] == 1) &
            (dataframe["prev_green"] == 1) &
            (o >= dataframe["prev_close"]) &
            (c <= dataframe["prev_open"])
        ).astype(int)

        # Shooting star
        dataframe["shooting_star"] = (
            (dataframe["is_red"] == 1) &
            (dataframe["upper_wick"] > 2 * dataframe["body"].replace(0, 0.01)) &
            (dataframe["upper_wick"] > 2 * dataframe["lower_wick"].replace(0, 0.01)) &
            (dataframe["range"] > 0.3 * atr_safe)
        ).astype(int)

        # Any bullish/bearish reversal pattern
        dataframe["pat_bull"] = ((dataframe["bullish_engulf"] == 1) | (dataframe["hammer"] == 1)).astype(int)
        dataframe["pat_bear"] = ((dataframe["bearish_engulf"] == 1) | (dataframe["shooting_star"] == 1)).astype(int)

        # ── Consecutive candle count ──
        red = dataframe["is_red"].values
        green = dataframe["is_green"].values
        n = len(dataframe)
        c_red = np.zeros(n, dtype=int)
        c_green = np.zeros(n, dtype=int)
        for i in range(1, n):
            c_red[i] = c_red[i - 1] + 1 if red[i - 1] else 0
            c_green[i] = c_green[i - 1] + 1 if green[i - 1] else 0
        dataframe["consec_red"] = c_red
        dataframe["consec_green"] = c_green

        # 1h merge
        if self.dp:
            pair = metadata["pair"]
            inf_1h = self.dp.get_pair_dataframe(pair=pair, timeframe="1h")
            if not inf_1h.empty:
                inf_1h["ema9"] = ta.EMA(inf_1h, timeperiod=9)
                inf_1h["ema21"] = ta.EMA(inf_1h, timeperiod=21)
                inf_1h["ema50"] = ta.EMA(inf_1h, timeperiod=50)
                inf_1h["adx"] = ta.ADX(inf_1h, timeperiod=14)
                dataframe = merge_informative_pair(
                    dataframe, inf_1h, self.timeframe, "1h", ffill=True
                )

        return dataframe

    # ═══════════════════════════════════════════════════════════════════
    # ENTRIES — Deep pullback + reversal pattern
    # ═══════════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        has_1h = "ema9_1h" in dataframe.columns
        pb = self.buy_pb_candles.value
        rsi_l = self.buy_rsi_pb_long.value
        rsi_s = self.buy_rsi_pb_short.value
        vol_min = self.buy_vol_min.value

        vol_ok = dataframe["vol_ratio"] > vol_min
        has_vol = dataframe["volume"] > 0
        atr_ok = dataframe["atr"] > 1

        # ── 1H TREND (separate thresholds) ──
        if has_1h:
            h1_up = (dataframe["ema9_1h"] > dataframe["ema21_1h"]) & (dataframe["adx_1h"] > self.buy_adx_long.value)
            h1_dn = (dataframe["ema9_1h"] < dataframe["ema21_1h"]) & (dataframe["adx_1h"] > self.buy_adx_short.value)
        else:
            h1_up = pd.Series(True, index=dataframe.index)
            h1_dn = pd.Series(True, index=dataframe.index)

        # ── PULLBACK (deep) ──
        deep_pullback_long = (
            (dataframe["consec_red"] >= pb) |
            (dataframe["rsi"] < rsi_l)
        )
        deep_pullback_short = (
            (dataframe["consec_green"] >= pb) |
            (dataframe["rsi"] > rsi_s)
        )

        # ── REVERSAL PATTERN (better timing) ──
        # Type A: candlestick pattern
        pat_long = dataframe["pat_bull"] == 1
        pat_short = dataframe["pat_bear"] == 1

        # Type B: simple green/red after pullback (less strict but more frequent)
        simple_long = (dataframe["is_green"] == 1) & (dataframe["close"] > dataframe["ema50"])
        simple_short = (dataframe["is_red"] == 1) & (dataframe["close"] < dataframe["ema50"])

        # ── EMA STRUCTURE (pullback within trend) ──
        ema_ok_long = dataframe["ema9"] > dataframe["ema21"]
        ema_ok_short = dataframe["ema9"] < dataframe["ema21"]

        # ── COMBINE ──
        # Pattern entries (strongest)
        go_long_pat = h1_up & deep_pullback_long & pat_long & vol_ok & has_vol & atr_ok
        go_short_pat = h1_dn & deep_pullback_short & pat_short & vol_ok & has_vol & atr_ok

        # Simple entries (still requires EMA structure)
        go_long_simple = h1_up & deep_pullback_long & simple_long & ema_ok_long & vol_ok & has_vol & atr_ok
        go_short_simple = h1_dn & deep_pullback_short & simple_short & ema_ok_short & vol_ok & has_vol & atr_ok

        go_long = go_long_pat | go_long_simple
        go_short = go_short_pat | go_short_simple

        dataframe.loc[go_long, "enter_long"] = 1
        dataframe.loc[go_short, "enter_short"] = 1

        # Tags
        dataframe.loc[go_long_pat, "enter_tag"] = "pattern_long"
        dataframe.loc[go_long_simple & ~go_long_pat, "enter_tag"] = "simple_long"
        dataframe.loc[go_short_pat, "enter_tag"] = "pattern_short"
        dataframe.loc[go_short_simple & ~go_short_pat, "enter_tag"] = "simple_short"

        return dataframe

    # ═══════════════════════════════════════════════════════════════════
    # EXIT — Let trailing handle it
    # ═══════════════════════════════════════════════════════════════════

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["rsi"] > 90, "exit_long"] = 1
        dataframe.loc[dataframe["rsi"] < 10, "exit_short"] = 1
        return dataframe

    # ═══════════════════════════════════════════════════════════════════
    # CUSTOM STOPLOSS — Progressive trailing
    # ═══════════════════════════════════════════════════════════════════

    def custom_stoploss(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return -0.04

        last = dataframe.iloc[-1]
        atr = last.get("atr", 0)
        if atr == 0 or trade.open_rate == 0:
            return -0.04

        # R = initial risk
        r_pct = self.buy_sl_initial.value * atr / trade.open_rate
        r_pct = max(0.02, min(r_pct, 0.05))

        if current_profit <= 0:
            return -r_pct

        r_mult = current_profit / r_pct

        # Progressive trailing
        if r_mult >= 3.0:
            trail = self.buy_sl_trail_3.value * atr / trade.open_rate
        elif r_mult >= 2.0:
            trail = self.buy_sl_trail_2.value * atr / trade.open_rate
        elif r_mult >= 1.0:
            trail = self.buy_sl_trail_1.value * atr / trade.open_rate
        else:
            return -r_pct

        trail = max(0.008, min(trail, 0.04))
        return -trail

    # ═══════════════════════════════════════════════════════════════════
    # CUSTOM EXIT — Time only (no trend flip)
    # ═══════════════════════════════════════════════════════════════════

    def custom_exit(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ) -> Optional[str]:
        minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        max_min = self.buy_max_hold.value * 15

        if minutes > max_min * 0.7 and current_profit > 0.005:
            return "v7_time_profit"

        if minutes > max_min:
            return "v7_time_force"

        return None

    # ═══════════════════════════════════════════════════════════════════
    # CONFIRM
    # ═══════════════════════════════════════════════════════════════════

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        today = current_time.strftime("%Y-%m-%d")
        if today not in self._daily_trades:
            self._daily_trades = {today: 0}

        if self._daily_trades.get(today, 0) >= self.buy_max_daily.value:
            return False

        if self._consecutive_losses >= 3 and self._last_loss_time:
            if current_time < self._last_loss_time + timedelta(minutes=45):
                return False
            self._consecutive_losses = 0

        self._daily_trades[today] = self._daily_trades.get(today, 0) + 1
        return True

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate,
                           time_in_force, exit_reason, current_time, **kwargs) -> bool:
        if trade.calc_profit_ratio(rate) < 0:
            self._consecutive_losses += 1
            self._last_loss_time = current_time
        else:
            self._consecutive_losses = 0
        return True