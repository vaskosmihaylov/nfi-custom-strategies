# HighFrequency_Hybrid_Fixed.py
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import numpy as np
import pandas as pd


class HighFrequency_Hybrid_Fixed(IStrategy):
    """
    混合型策略（修复版）
    - 盘整时做市（高频小利）
    - 趋势时低频趋势跟随（回调入场）
    - 使用 ROI（entry-based）做 TP，使用 custom_stoploss 做出场保护（基于 entry_price + ATR）
    - 使用 confirm_trade_entry 过滤大 spread / 突发 ATR
    - dynamic stake（波动大时降低仓位）
    """
    INTERFACE_VERSION = 3
    timeframe = "1m"
    startup_candle_count = 30

    # ROI (take-profit) 以 entry_price 为基准（避免用 mid）
    minimal_roi = {
        "0": 0.0015,   # 0.15% 立即可做目标（可按需调）
        "30": 0.001,
        "60": 0
    }

    # 基本 stoploss（作为最后后备）
    stoploss = -0.06  # 若 custom_stoploss 返回 None，将使用此值作为保险

    # 下单类型（完整字段）
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "force_entry": "limit",
        "force_exit": "limit",
    }

    # ============= 超参数（可 hyperopt） =============
    stake_unit = DecimalParameter(0.006, 0.015, default=0.01, decimals=3, space="buy")  # 基础 stake（1%）
    num_levels = IntParameter(1, 2, default=1, space="buy")
    price_offset_pct = DecimalParameter(0.08, 0.2, default=0.12, decimals=3, space="buy")  # 0.12% 默认
    take_profit_pct = DecimalParameter(0.12, 0.30, default=0.18, decimals=3, space="sell")  # ROI 主目标 0.18%
    atr_threshold_pct = DecimalParameter(0.10, 0.35, default=0.20, decimals=3, space="buy")  # ATR阈值（% of mid）
    atr_multiplier = DecimalParameter(1.0, 3.0, default=1.6, decimals=2, space="sell")  # custom stoploss基于ATR倍数
    max_spread = DecimalParameter(0.003, 0.02, default=0.006, decimals=4, space="buy")  # 最大允许spread（比例）
    ema_fast = IntParameter(5, 8, default=5, space="buy")
    ema_slow = IntParameter(12, 20, default=12, space="buy")
    trend_ema_period = IntParameter(34, 60, default=34, space="buy")
    trend_strength_threshold = DecimalParameter(0.005, 0.015, default=0.008, decimals=4, space="buy")

    # safety caps
    max_total_risk = 0.08  # 总仓位上限 8%

    # ================================================

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = df.copy()

        # mid price
        df["mid"] = (df["high"] + df["low"]) / 2.0

        # TR & ATR (短周期)
        df["tr"] = np.maximum(df["high"] - df["low"],
                              np.maximum(abs(df["high"] - df["close"].shift(1)),
                                         abs(df["low"] - df["close"].shift(1))))
        # 用 7 或 9 更短的 ATR 对 1m 更敏感（不使用过长会错判）
        df["atr"] = df["tr"].rolling(7).mean().bfill()

        # EMAs
        df["ema_fast"] = df["close"].ewm(span=self.ema_fast.value, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow.value, adjust=False).mean()
        df["ema_trend"] = df["close"].ewm(span=self.trend_ema_period.value, adjust=False).mean()

        # trend strength
        df["trend_strength"] = abs(df["close"] / df["ema_trend"] - 1)

        # spread proxy - if bid/ask available use it, else proxy with high-low but keep threshold loose
        if "bid" in df.columns and "ask" in df.columns:
            mid_askbid = (df["ask"] + df["bid"]) / 2.0
            df["spread"] = (df["ask"] - df["bid"]) / mid_askbid
        else:
            df["spread"] = (df["high"] - df["low"]) / df["mid"]  # proxy (candle range)

        # in_range (盘整判断)：ATR 相对较小并且趋势强度低
        df["in_range"] = (df["atr"] < (df["mid"] * (self.atr_threshold_pct.value / 100.0))) & \
                         (df["trend_strength"] < self.trend_strength_threshold.value)

        return df

    def is_trending(self, row) -> bool:
        # 快慢 EMA 明显分离并且趋势强度较高
        return ((row["ema_fast"] > row["ema_slow"] and (row["ema_fast"] / row["ema_slow"] - 1) > 0.005) or
                (row["ema_fast"] < row["ema_slow"] and (row["ema_slow"] / row["ema_fast"] - 1) > 0.005)) and \
               row["trend_strength"] > self.trend_strength_threshold.value

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        入场逻辑：
         - 盘整模式：当 close <= mid*(1-offset) 且 in_range 且 spread 小，标记 buy（limit）
         - 趋势模式：当正在趋势中且发生小幅回调到 ema_fast * (1 - pullback_pct) 时买入（低频）
        """
        df = df.copy()
        df["buy"] = 0
        levels = int(self.num_levels.value)
        base_offset = float(self.price_offset_pct.value) / 100.0
        pullback_pct = 0.005  # 趋势回调阈值 0.5%

        cond_common = (df["spread"] <= float(self.max_spread.value))

        for i in range(1, levels + 1):
            offset = base_offset * i
            buy_price = df["mid"] * (1 - offset)
            # 盘整模式买点
            df.loc[(df["in_range"]) & (df["close"] <= buy_price) & cond_common, "buy"] = 1

        # 趋势模式回调入场（低频）
        df.loc[(df.apply(self.is_trending, axis=1)) & (df["close"] < df["ema_fast"] * (1 - pullback_pct)) & cond_common, "buy"] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        exit 使用 ROI（entry based）与 custom_stoploss；populate_exit_trend 仍提供保底退出（例如极端情况）
        为兼容回测/可视化，保留 conservative mid-based exits as fallback but main logic uses ROI/custom_stoploss.
        """
        df = df.copy()
        df["sell"] = 0

        # fallback conservative exits to avoid infinite holds (not the main exit)
        tp_fallback = float(self.take_profit_pct.value) / 100.0
        sl_fallback = 0.05  # 5%极端回退保护（仅作最后保障）

        df.loc[df["close"] >= df["mid"] * (1 + tp_fallback), "sell"] = 1
        df.loc[df["close"] <= df["mid"] * (1 - sl_fallback), "sell"] = 1

        return df

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        动态止损基于 entry_rate 与实时 ATR：
         - stop_pct = max(min_stop, atr_pct * atr_multiplier)
         - 返回 stoploss as negative fraction (e.g. -0.003)
        如果返回 None 则使用默认 self.stoploss。
        """
        try:
            # 获取最新df
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe, limit=5)
            last = df.iloc[-1]
            entry_price = trade.open_rate if hasattr(trade, "open_rate") else trade.entry_price
            atr = float(last["atr"])
            atr_pct = atr / entry_price if entry_price and entry_price > 0 else 0.001
            mult = float(self.atr_multiplier.value)
            # dynamic stoploss: 根据 atr_pct 放大；至少 min_stop
            min_stop = 0.003  # 最小止损 0.3%
            computed = max(min_stop, atr_pct * mult)
            stoploss_val = -computed
            # 保证不超过全局 stoploss cap
            # self.stoploss 是 -0.06（-6%），不放开更大
            if stoploss_val < self.stoploss:
                # 不小于全局 stoploss（即不要更严）
                return self.stoploss
            return stoploss_val
        except Exception:
            return None

    def custom_entry(self, pair: str, current_time, current_rate, current_profit, **kwargs):
        """
        根据当前波动动态调整 stake：波动越大，stake越小。
        返回 fraction(0..1)
        """
        try:
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe, limit=5)
            last = df.iloc[-1]
            atr = float(last["atr"])
            atr_pct = atr / last["close"] if last["close"] and last["close"] > 0 else 0.001
            base = float(self.stake_unit.value)
            # simple scale: if atr_pct > 0.5% -> reduce stake by half; else keep base
            if atr_pct > 0.005:
                stake = max(0.002, base * 0.5)  # 最小 0.2%
            elif atr_pct > 0.003:
                stake = max(0.003, base * 0.7)
            else:
                stake = base
            # ensure not to exceed max_total_risk minus existing position
            position = self.dp.get_position(pair, lookback=1)
            current_amount = position.amount if position else 0
            # current_amount is in base asset units — we can't easily translate to fraction here reliably,
            # so we just return stake fraction as designed
            return float(stake)
        except Exception:
            return float(self.stake_unit.value)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        """
        下单前再次检查最新盘口/波动，避免在 spread/ATR 突增时下单
        """
        try:
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe, limit=2)
            last = df.iloc[-1]
            # 若实际spread列存在，优先使用
            spread = float(last.get("spread", 0.0))
            atr = float(last.get("atr", 0.0))
            mid = float(last.get("mid", last["close"]))
            # 如果 spread 超过阈值，不下单
            if spread > float(self.max_spread.value):
                return False
            # 如果 ATR 突增（> mid * 1%），不下单
            if atr > mid * 0.01:
                return False
            return True
        except Exception:
            return True
