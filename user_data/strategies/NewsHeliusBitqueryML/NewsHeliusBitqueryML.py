import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.parameters import CategoricalParameter
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes
from .indicators import calculate_all_indicators
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class NewsHeliusBitqueryML(IStrategy):
    timeframe = "5m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 240
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
    }
    
    # Параметры для гипероптимизации ROI
    minimal_roi = {
        "0": 0.01,
        "30": 0.005,
        "90": 0.0
    }
    
    stoploss = -0.02
    
    trailing_stop = True
    trailing_stop_positive = 0.004
    trailing_stop_positive_offset = 0.009
    trailing_only_offset_is_reached = True
    
    ignore_buying_expired_candle_after = 1
    
    # Параметры для гипероптимизации
    buy_params = {
        "vol_min": 0.002,
        "rsi_long_th": 45,
        "adx_min": 14,
        "ema_fast": 12,
        "ema_slow": 26,
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14
    }
    
    sell_params = {
        "rsi_exit_long": 70,
        "ema_exit_long": True
    }
    
    # Параметры для гипероптимизации ROI
    roi_params = {
        "roi_0": 0.01,
        "roi_30": 0.005,
        "roi_90": 0.0
    }
    
    # Параметры для гипероптимизации Stoploss
    stoploss_params = {
        "stoploss": -0.02
    }
    
    # Параметры для гипероптимизации Trailing Stop
    trailing_params = {
        "trailing_stop": True,
        "trailing_stop_positive": 0.004,
        "trailing_stop_positive_offset": 0.009,
        "trailing_only_offset_is_reached": True
    }

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = calculate_all_indicators(df)
        for c in ["macd", "macd_sig", "rsi", "atr", "ema50", "ema200"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["vol_ok"] = (df["atr"] / df["close"] > 0.0015)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        trend_long  = df["ema50"] > df["ema200"]
        trend_short = df["ema50"] < df["ema200"]

        macd_cross_up   = (df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1))
        macd_cross_down = (df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1))

        rsi_ok_long  = df["rsi"] > 45
        rsi_ok_short = df["rsi"] < 55

        df["enter_long"]  = (trend_long  & df["vol_ok"] & macd_cross_up   & rsi_ok_long).astype(int)
        df["enter_short"] = (trend_short & df["vol_ok"] & macd_cross_down & rsi_ok_short).astype(int)
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"]  = ((df["macd"] < df["macd_sig"]) | (df["rsi"] < 50)).astype(int)
        df["exit_short"] = ((df["macd"] > df["macd_sig"]) | (df["rsi"] > 50)).astype(int)
        return df
    
    # Методы для гипероптимизации ROI
    def custom_roi(self, dataframe: DataFrame, trade: Trade, current_time: datetime, **kwargs) -> float:
        # Динамический ROI на основе времени удержания позиции
        open_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        
        if open_minutes <= 30:
            return self.roi_params['roi_0']
        elif open_minutes <= 90:
            return self.roi_params['roi_30']
        else:
            return self.roi_params['roi_90']
    
    # Методы для гипероптимизации Stoploss
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        return self.stoploss_params['stoploss']
    
    # Методы для гипероптимизации Trailing Stop
    def custom_trailing_stop(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        if not self.trailing_params['trailing_stop']:
            return 0
        
        if current_profit > self.trailing_params['trailing_stop_positive_offset']:
            return self.trailing_params['trailing_stop_positive']
        
        return 0

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            {"method": "StoplossGuard", "lookback_period_candles": 288,
             "stop_duration_candles": 30, "only_per_pair": False, "trade_limit": 2},
            {"method": "MaxDrawdown", "lookback_period_candles": 288,
             "stop_duration_candles": 60, "max_allowed_drawdown": 8,
             "only_per_pair": False}
        ]