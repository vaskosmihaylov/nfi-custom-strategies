import talib.abstract as ta
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter
) 
# strategy_BandtasticFiboHyper_2025-07-07_11-44-49.fthypt 
# 79/134:    820 trades. 450/101/269 Wins/Draws/Losses. 
# Avg profit   0.21%. 
# Median profit   0.14%. 
# Total profit 616.80536480 USDT (  61.68%). 
# Long / Short                  │ 587 / 233                         │
# Total profit Long %           │ 48.69%                            │
# Total profit Short %          │ 12.99%                            │
# Absolute profit Long          │ 486.935 USDT                      │
# Absolute profit Short         │ 129.87 USDT                       │
# Avg duration 5:54:00 min. 
# Objective: -7.65106
# max_open_trade = 3
class BandtasticFiboHyper_opt79(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'

    # ROI table (from hyperspace)
    minimal_roi = {
        "0": 0.08,
        "36": 0.039,
        "79": 0.029,
        "176": 0
    }

    stoploss = -0.314
    startup_candle_count = 999

    trailing_stop = True
    trailing_stop_positive = 0.237
    trailing_stop_positive_offset = 0.278
    trailing_only_offset_is_reached = False

    # Max open trades
    max_open_trades = 3

    # ========= 杠杆参数 =========
    max_leverage = DecimalParameter(1.0, 5.0, default=1.357, space='protection', optimize=True)
    max_short_leverage = DecimalParameter(1.0, 3.0, default=2.532, space='protection', optimize=True)
    atr_threshold_low = DecimalParameter(0.005, 0.03, default=0.03, space='protection', optimize=True)
    atr_threshold_high = DecimalParameter(0.02, 0.08, default=0.065, space='protection', optimize=True)

    # ========= long 参数 =========
    buy_fastema = IntParameter(1, 236, default=94, space='buy', optimize=True)
    buy_slowema = IntParameter(1, 250, default=221, space='buy', optimize=True)
    buy_rsi = IntParameter(15, 70, default=52, space='buy', optimize=True)
    buy_mfi = IntParameter(15, 70, default=69, space='buy', optimize=True)
    buy_rsi_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_ema_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_trigger = CategoricalParameter(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'], default='fibonacci', space='buy', optimize=True)
    buy_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_fib_level = CategoricalParameter(['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'], default='fib_618', space='buy', optimize=True)

    # ====== Short 参数 ==========
    short_fastema = IntParameter(1, 250, default=44, space='sell', optimize=True)
    short_slowema = IntParameter(1, 250, default=79, space='sell', optimize=True)
    short_rsi = IntParameter(30, 100, default=49, space='sell', optimize=True)
    short_mfi = IntParameter(30, 100, default=90, space='sell', optimize=True)
    short_rsi_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    short_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    short_ema_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    short_trigger = CategoricalParameter(['bb_upper1', 'bb_upper2', 'bb_upper3', 'bb_upper4'], default='bb_upper1', space='sell', optimize=True)

    # ========= Sell 参数 =========
    sell_fastema = IntParameter(1, 365, default=23, space='sell', optimize=True)
    sell_slowema = IntParameter(1, 365, default=187, space='sell', optimize=True)
    sell_rsi = IntParameter(30, 100, default=55, space='sell', optimize=True)
    sell_mfi = IntParameter(30, 100, default=83, space='sell', optimize=True)
    sell_rsi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    sell_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    sell_ema_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    sell_trigger = CategoricalParameter(['sell-bb_upper1', 'sell-bb_upper2', 'sell-bb_upper3', 'sell-bb_upper4'], default='sell-bb_upper3', space='sell', optimize=True)
    
    cover_fastema = IntParameter(1, 250, default=245, space='buy', optimize=True)
    cover_slowema = IntParameter(1, 250, default=206, space='buy', optimize=True)
    cover_rsi = IntParameter(10, 70, default=34, space='buy', optimize=True)
    cover_mfi = IntParameter(10, 70, default=45, space='buy', optimize=True)
    cover_rsi_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
    cover_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_ema_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
    cover_trigger = CategoricalParameter(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'], default='bb_lower1', space='buy', optimize=True)
    cover_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_fib_level = CategoricalParameter(['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'], default='fib_382', space='buy', optimize=True)

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        使用 ATR/价格 标准化波动率动态调整杠杆。
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 20:
            return 2.0  # fallback 默认值

        close = dataframe['close'].iloc[-1]
        atr = ta.ATR(dataframe, timeperiod=14).iloc[-1]
        normalized_atr = atr / close if close > 0 else 0

        if normalized_atr < self.atr_threshold_low.value:
            lev = 4.0
        elif normalized_atr < self.atr_threshold_high.value:
            lev = 2.5
        else:
            lev = 1.5

        if side == 'short':
            lev = min(lev, self.max_short_leverage.value)

        return min(lev, float(self.max_leverage.value))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['normalized_atr'] = dataframe['atr'] / dataframe['close']

        for std in range(1, 5):
            bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=std)
            dataframe[f'bb_lowerband{std}'] = bb['lower']
            dataframe[f'bb_middleband{std}'] = bb['mid']
            dataframe[f'bb_upperband{std}'] = bb['upper']

        ema_periods = set([
            int(self.buy_fastema.value), int(self.buy_slowema.value),
            int(self.sell_fastema.value), int(self.sell_slowema.value),
            int(self.short_fastema.value), int(self.short_slowema.value),
            int(self.cover_fastema.value), int(self.cover_slowema.value)
        ])
        for period in ema_periods:
            if period > 0 and len(dataframe) >= period:
                dataframe[f'EMA_{period}'] = ta.EMA(dataframe, timeperiod=period)

        # Fibonacci Levels
        lookback = 50
        if len(dataframe) >= lookback:
            recent_max = dataframe['high'].rolling(lookback).max()
            recent_min = dataframe['low'].rolling(lookback).min()
            diff = recent_max - recent_min
            dataframe['fib_236'] = recent_max - diff * 0.236
            dataframe['fib_382'] = recent_max - diff * 0.382
            dataframe['fib_5'] = recent_max - diff * 0.5
            dataframe['fib_618'] = recent_max - diff * 0.618
            dataframe['fib_786'] = recent_max - diff * 0.786

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -------- 做多逻辑 ----------
        long_conditions = []

        if self.buy_rsi_enabled.value:
            long_conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        if self.buy_mfi_enabled.value:
            long_conditions.append(dataframe['mfi'] < self.buy_mfi.value)

        if self.buy_ema_enabled.value:
            fast_col = f'EMA_{self.buy_fastema.value}'
            slow_col = f'EMA_{self.buy_slowema.value}'
            if fast_col in dataframe and slow_col in dataframe:
                long_conditions.append(dataframe[fast_col] > dataframe[slow_col])

        if self.buy_trigger.value.startswith('bb_lower'):
            bb_col = f'bb_lowerband{self.buy_trigger.value[-1]}'
            long_conditions.append(dataframe['close'] < dataframe[bb_col])

        if self.buy_trigger.value == 'fibonacci' and self.buy_fib_enabled.value:
            fib_col = self.buy_fib_level.value
            if fib_col in dataframe.columns:
                long_conditions.append(dataframe['close'] < dataframe[fib_col])

        long_conditions.append(dataframe['volume'] > 0)

        if long_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1

        # -------- 做空逻辑 ----------
        short_conditions = []

        if self.short_rsi_enabled.value:
            short_conditions.append(dataframe['rsi'] > self.short_rsi.value)

        if self.short_mfi_enabled.value:
            short_conditions.append(dataframe['mfi'] > self.short_mfi.value)

        if self.short_ema_enabled.value:
            fast_col = f'EMA_{self.short_fastema.value}'
            slow_col = f'EMA_{self.short_slowema.value}'
            if fast_col in dataframe and slow_col in dataframe:
                short_conditions.append(dataframe[fast_col] < dataframe[slow_col])

        if self.short_trigger.value.startswith('bb_upper'):
            bb_col = f'bb_upperband{self.short_trigger.value[-1]}'
            short_conditions.append(dataframe['close'] > dataframe[bb_col])

        short_conditions.append(dataframe['volume'] > 0)

        if short_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_short'] = 1
        
        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -------- 做多平仓 ----------
        long_exit = []

        if self.sell_rsi_enabled.value:
            long_exit.append(dataframe['rsi'] > self.sell_rsi.value)

        if self.sell_mfi_enabled.value:
            long_exit.append(dataframe['mfi'] > self.sell_mfi.value)

        if self.sell_ema_enabled.value:
            fast_col = f'EMA_{self.sell_fastema.value}'
            slow_col = f'EMA_{self.sell_slowema.value}'
            if fast_col in dataframe and slow_col in dataframe:
                long_exit.append(dataframe[fast_col] < dataframe[slow_col])

        if self.sell_trigger.value.startswith('sell-bb_upper'):
            bb_col = f'bb_upperband{self.sell_trigger.value[-1]}'
            long_exit.append(dataframe['close'] > dataframe[bb_col])

        long_exit.append(dataframe['volume'] > 0)

        if long_exit:
            dataframe.loc[reduce(lambda x, y: x & y, long_exit), 'exit_long'] = 1

        # -------- 做空平仓逻辑 ----------
        short_exit = []

        if self.cover_rsi_enabled.value:
            short_exit.append(dataframe['rsi'] < self.cover_rsi.value)

        if self.cover_mfi_enabled.value:
            short_exit.append(dataframe['mfi'] < self.cover_mfi.value)

        if self.cover_ema_enabled.value:
            fast_col = f'EMA_{self.cover_fastema.value}'
            slow_col = f'EMA_{self.cover_slowema.value}'
            if fast_col in dataframe and slow_col in dataframe:
                short_exit.append(dataframe[fast_col] > dataframe[slow_col])

        if self.cover_trigger.value.startswith('bb_lower'):
            bb_col = f'bb_lowerband{self.cover_trigger.value[-1]}'
            short_exit.append(dataframe['close'] < dataframe[bb_col])

        if self.cover_trigger.value == 'fibonacci' and self.cover_fib_enabled.value:
            fib_col = self.cover_fib_level.value
            if fib_col in dataframe.columns:
                short_exit.append(dataframe['close'] < dataframe[fib_col])

        short_exit.append(dataframe['volume'] > 0)

        if short_exit:
            dataframe.loc[reduce(lambda x, y: x & y, short_exit), 'exit_short'] = 1

        return dataframe
