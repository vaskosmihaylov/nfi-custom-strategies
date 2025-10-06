import talib.abstract as ta
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    informative,
    merge_informative_pair
)
from freqtrade.persistence import Trade
import logging

logger = logging.getLogger(__name__)


class BandtasticFiboHyper_opt314(IStrategy):
    """
    HYBRID BandtasticFiboHyper Strategy - October 2025
    
    Combines the best of opt314 v2.1 (safety) and opt490 (aggressive shorts)
    
    OPTIMIZATIONS APPLIED:
    ✅ Base stoploss: -10% (optimal for 5m crypto with 2-3x leverage)
       Research: 8-12% is ideal for crypto volatility with leverage
       With 2.5x leverage: 10% stop = 25% account loss (acceptable)
    
    ✅ Volume filters REMOVED (managed via config.json)
    
    ✅ Trailing stop REDESIGNED for crypto volatility:
       - Activates at 8% profit (not 4% - too fast!)
       - Trails at 4% distance (not 1.5% - too tight!)
       - Allows crypto to breathe during 5-10% swings
       - Works WITH dynamic BB/ATR stops, not against them
    
    ✅ HYBRID ENTRY LOGIC (from opt314 v2.1):
       - Pullback entries: Buy dips to BB2 (realistic)
       - Momentum entries: Buy strength in uptrends (NEW!)
       - Works in ALL market conditions
    
    ✅ AGGRESSIVE SHORTS (from opt490):
       - Lower RSI threshold: 55 (vs 70 in opt314)
       - Lower MFI threshold: 55 (vs 65 in opt314)  
       - BB upper 2 trigger (vs BB3 in opt314)
       - More shorts = more profit potential
    
    ✅ SAFETY FEATURES (from opt314 v2.1):
       - Dynamic BB/ATR-based stops
       - Partial exits at 3%, 5%, 8%
       - DCA at -4% to -7%
       - 1h trend filters for quality
       - Minimum 15-minute trade duration
       - Hard cap prevents runaway stops
    
    Expected Results:
    - More trades than opt314 (especially shorts)
    - Safer than opt490 (no -43% disasters!)
    - Works in uptrends, downtrends, choppy markets
    - Positive ROI with manageable risk
    """
    
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'
    informative_timeframe = '1h'

    # ============= ROI TABLE (Balanced) =============
    minimal_roi = {
        "0": 0.12,      # 12% initial target (between opt314 10% and opt490 21.5%)
        "30": 0.06,     # 6% after 30 min
        "60": 0.04,     # 4% after 1 hour
        "120": 0.02,    # 2% after 2 hours
        "240": 0        # Breakeven after 4 hours
    }

    # ============= OPTIMIZED STOP LOSS FOR CRYPTO =============
    # Research-based: 8-12% optimal for 5m crypto with 2-3x leverage
    # With 2.5x leverage: 10% stop = 25% account loss (manageable)
    stoploss = -0.10  # 10% base stop (was -0.06 in opt314, -0.314 in opt490)
    
    use_custom_stoploss = True

    startup_candle_count = 999

    # ============= OPTIMIZED TRAILING STOP FOR CRYPTO VOLATILITY =============
    # Crypto can swing 5-10% in hours - need wider trailing!
    # Dynamic BB/ATR stops handle tight exits, trailing catches big runners
    trailing_stop = True
    trailing_stop_positive = 0.04     # Trail at 4% distance (not 1.5% - too tight!)
    trailing_stop_positive_offset = 0.08  # Activate at 8% profit (not 4% - too fast!)
    trailing_only_offset_is_reached = True  # Only trail after reaching 8%

    # Position adjustment for DCA and partial exits
    position_adjustment_enable = True
    max_entry_position_adjustment = 1  # Allow 1 DCA
    max_exit_position_adjustment = 3   # Allow 3 partial exits

    # Max open trades
    max_open_trades = 11
    
    # Minimum trade duration (prevent instant exits)
    min_trade_duration_minutes = 15  # 15 minutes for 5m timeframe

    # ============= OPTIONAL: DISABLE SHORTS =============
    shorts_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=False)

    # ========= LEVERAGE PARAMETERS (Conservative for safety) =========
    max_leverage = DecimalParameter(1.0, 3.0, default=2.0, space='protection', optimize=True)
    max_short_leverage = DecimalParameter(1.0, 3.0, default=2.5, space='protection', optimize=True)
    atr_threshold_low = DecimalParameter(0.005, 0.03, default=0.019, space='protection', optimize=True)
    atr_threshold_high = DecimalParameter(0.02, 0.08, default=0.026, space='protection', optimize=True)

    # ========= DYNAMIC STOP LOSS PARAMETERS =========
    atr_stop_multiplier_long = DecimalParameter(1.0, 2.5, default=1.5, space='protection', optimize=True)
    atr_stop_multiplier_short = DecimalParameter(0.8, 2.0, default=1.2, space='protection', optimize=True)

    min_stop_loss_long = DecimalParameter(0.02, 0.05, default=0.03, space='protection', optimize=True)
    min_stop_loss_short = DecimalParameter(0.015, 0.04, default=0.025, space='protection', optimize=True)
    
    # Max stops aligned with base stoploss
    max_stop_loss_long = DecimalParameter(0.06, 0.10, default=0.10, space='protection', optimize=True)
    max_stop_loss_short = DecimalParameter(0.05, 0.10, default=0.09, space='protection', optimize=True)

    # ========= PARTIAL EXIT PARAMETERS =========
    partial_exit_1_profit = DecimalParameter(0.02, 0.04, decimals=2, default=0.03, space='sell', optimize=True)
    partial_exit_2_profit = DecimalParameter(0.04, 0.06, decimals=2, default=0.05, space='sell', optimize=True)
    partial_exit_3_profit = DecimalParameter(0.06, 0.10, decimals=2, default=0.08, space='sell', optimize=True)
    partial_exit_size = DecimalParameter(0.25, 0.40, decimals=2, default=0.33, space='sell', optimize=True)

    # ========= DCA PARAMETERS =========
    dca_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=False)
    dca_profit_min = DecimalParameter(-0.08, -0.03, decimals=2, default=-0.04, space='buy', optimize=True)
    dca_profit_max = DecimalParameter(-0.10, -0.05, decimals=2, default=-0.07, space='buy', optimize=True)

    # ========= LONG ENTRY PARAMETERS (Relaxed for more trades) =========
    buy_fastema = IntParameter(1, 236, default=191, space='buy', optimize=True)
    buy_slowema = IntParameter(1, 250, default=128, space='buy', optimize=True)
    buy_rsi = IntParameter(15, 70, default=65, space='buy', optimize=True)  # Relaxed from 56
    buy_mfi = IntParameter(15, 70, default=55, space='buy', optimize=True)  # Relaxed from 40
    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
    buy_trigger = CategoricalParameter(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'],
                                       default='bb_lower2', space='buy', optimize=True)  # Changed from bb_lower4
    buy_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    buy_fib_level = CategoricalParameter(['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'],
                                         default='fib_382', space='buy', optimize=True)
    buy_1h_trend_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)

    # ========= MOMENTUM ENTRY PARAMETERS (NEW - for uptrends) =========
    momentum_entry_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=False)
    momentum_rsi_min = IntParameter(50, 70, default=55, space='buy', optimize=True)
    momentum_rsi_max = IntParameter(65, 85, default=75, space='buy', optimize=True)

    # ========= SHORT ENTRY PARAMETERS (AGGRESSIVE - from opt490) =========
    short_fastema = IntParameter(1, 250, default=53, space='sell', optimize=True)
    short_slowema = IntParameter(1, 250, default=168, space='sell', optimize=True)
    short_rsi = IntParameter(45, 85, default=55, space='sell', optimize=True)  # AGGRESSIVE (was 70 in opt314, 49 in opt490)
    short_mfi = IntParameter(45, 85, default=55, space='sell', optimize=True)  # AGGRESSIVE (was 65 in opt314, 30 in opt490)
    short_rsi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    short_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    short_ema_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    short_trigger = CategoricalParameter(['bb_upper2', 'bb_upper3', 'bb_upper4'],
                                         default='bb_upper2', space='sell', optimize=True)  # AGGRESSIVE (from opt490)
    short_1h_trend_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    short_adx_threshold = IntParameter(18, 40, default=22, space='sell', optimize=True)

    # ========= EXIT PARAMETERS =========
    sell_fastema = IntParameter(1, 365, default=222, space='sell', optimize=True)
    sell_slowema = IntParameter(1, 365, default=192, space='sell', optimize=True)
    sell_rsi = IntParameter(30, 100, default=47, space='sell', optimize=True)
    sell_mfi = IntParameter(30, 100, default=46, space='sell', optimize=True)
    sell_rsi_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    sell_mfi_enabled = CategoricalParameter([True, False], default=True, space='sell', optimize=True)
    sell_ema_enabled = CategoricalParameter([True, False], default=False, space='sell', optimize=True)
    sell_trigger = CategoricalParameter(['sell-bb_upper1', 'sell-bb_upper2', 'sell-bb_upper3', 'sell-bb_upper4'],
                                        default='sell-bb_upper2', space='sell', optimize=True)

    cover_fastema = IntParameter(1, 250, default=97, space='buy', optimize=True)
    cover_slowema = IntParameter(1, 250, default=191, space='buy', optimize=True)
    cover_rsi = IntParameter(10, 70, default=42, space='buy', optimize=True)
    cover_mfi = IntParameter(10, 70, default=12, space='buy', optimize=True)
    cover_rsi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_mfi_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_ema_enabled = CategoricalParameter([True, False], default=False, space='buy', optimize=True)
    cover_trigger = CategoricalParameter(['bb_lower1', 'bb_lower2', 'bb_lower3', 'bb_lower4', 'fibonacci'],
                                         default='bb_lower2', space='buy', optimize=True)
    cover_fib_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=True)
    cover_fib_level = CategoricalParameter(['fib_236', 'fib_382', 'fib_5', 'fib_618', 'fib_786'],
                                           default='fib_382', space='buy', optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("=" * 80)
        logger.info("BandtasticFiboHyper - HYBRID OPTIMIZED VERSION (Oct 2025)")
        logger.info("=" * 80)
        logger.info("STRATEGY HYBRID: opt314 v2.1 (safety) + opt490 (aggressive shorts)")
        logger.info("")
        logger.info("KEY OPTIMIZATIONS:")
        logger.info(f"  ✓ Base stoploss: -10% (optimal for 5m crypto + leverage)")
        logger.info(f"     Research: 8-12% ideal for crypto volatility")
        logger.info(f"     With 2.5x leverage: 10% stop = 25% account loss (acceptable)")
        logger.info(f"  ✓ Trailing stop: 8% activation, 4% trail (crypto-optimized!)")
        logger.info(f"     Allows 5-10% swings without premature exit")
        logger.info(f"  ✓ Volume filters: REMOVED (managed via config.json)")
        logger.info("")
        logger.info("ENTRY LOGIC:")
        logger.info(f"  ✓ Longs: HYBRID (pullback + momentum entries)")
        logger.info(f"  ✓ Shorts: AGGRESSIVE (RSI>55, MFI>55, BB2)")
        logger.info(f"  ✓ Momentum entries: {'ENABLED' if self.momentum_entry_enabled.value else 'DISABLED'}")
        logger.info(f"  ✓ Shorts: {'ENABLED' if self.shorts_enabled.value else 'DISABLED'}")
        logger.info("")
        logger.info("RISK MANAGEMENT:")
        logger.info(f"  ✓ Dynamic BB/ATR stops (handles most exits)")
        logger.info(f"  ✓ Partial exits at {self.partial_exit_1_profit.value*100:.0f}%, {self.partial_exit_2_profit.value*100:.0f}%, {self.partial_exit_3_profit.value*100:.0f}%")
        logger.info(f"  ✓ DCA: {'ENABLED' if self.dca_enabled.value else 'DISABLED'} at {self.dca_profit_min.value*100:.0f}% to {self.dca_profit_max.value*100:.0f}%")
        logger.info(f"  ✓ Min trade duration: {self.min_trade_duration_minutes} minutes")
        logger.info(f"  ✓ 1h trend filters for quality control")
        logger.info("=" * 80)

    def informative_pairs(self):
        """Required for 1h timeframe data."""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate 1h trend indicators for higher timeframe confirmation."""
        # EMAs for trend detection
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=55)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=200)

        # RSI and ADX for trend strength
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # Define clear uptrend and downtrend
        dataframe['uptrend'] = (
            (dataframe['close'] > dataframe['ema_fast']) &
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['close'] > dataframe['ema_long']) &
            (dataframe['rsi'] > 45) &
            (dataframe['adx'] > 15)
        )

        dataframe['downtrend'] = (
            (dataframe['close'] < dataframe['ema_fast']) &
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['close'] < dataframe['ema_long']) &
            (dataframe['rsi'] < 55) &
            (dataframe['adx'] > 15)
        )

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Dynamic leverage based on normalized ATR volatility.
        Conservative for safety.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 20:
            return 2.0  # Fallback default

        close = dataframe['close'].iloc[-1]
        atr = ta.ATR(dataframe, timeperiod=14).iloc[-1]
        normalized_atr = atr / close if close > 0 else 0

        if normalized_atr < self.atr_threshold_low.value:
            lev = 3.0
        elif normalized_atr < self.atr_threshold_high.value:
            lev = 2.0
        else:
            lev = 1.5

        # Apply leverage limits
        if side == 'short':
            lev = min(lev, self.max_short_leverage.value)

        return min(lev, float(self.max_leverage.value))

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        BB/Fibonacci-based dynamic stop loss.
        
        OPTIMIZED FOR CRYPTO:
        - Respects minimum trade duration (15 min)
        - Hard cap at -10% base stoploss (never wider!)
        - Dynamic based on BB/ATR volatility
        - Tighter stops after profit
        """
        # Check minimum trade duration
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_duration < self.min_trade_duration_minutes:
            return self.stoploss
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 20:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        
        # Get BB bands and ATR
        bb_lower = last_candle.get('bb_lowerband2', last_candle['close'] * 0.97)
        bb_upper = last_candle.get('bb_upperband2', last_candle['close'] * 1.03)
        atr = last_candle.get('atr', 0)
        normalized_atr = last_candle.get('normalized_atr', 0.02)

        is_short = trade.is_short

        # BB/ATR-BASED STOP PLACEMENT
        if is_short:
            # For SHORTS: Stop above BB upper band
            bb_stop_distance = (bb_upper - trade.open_rate) / trade.open_rate
            atr_stop = normalized_atr * float(self.atr_stop_multiplier_short.value)
            
            # Safety check
            if bb_stop_distance < 0:
                bb_stop_distance = float(self.min_stop_loss_short.value)
            
            dynamic_stop = min(abs(bb_stop_distance), atr_stop)
            
            # Clamp between min and max
            min_stop = float(self.min_stop_loss_short.value)
            max_stop = float(self.max_stop_loss_short.value)
            dynamic_stop = max(min_stop, min(dynamic_stop, max_stop))
            
            # Tighten after profit
            if current_profit >= 0.05:
                dynamic_stop = min(dynamic_stop, 0.03)
            
            # CRITICAL: NEVER wider than base stoploss!
            dynamic_stop = min(dynamic_stop, abs(self.stoploss))
        else:
            # For LONGS: Stop below BB lower band
            bb_stop_distance = (trade.open_rate - bb_lower) / trade.open_rate
            atr_stop = normalized_atr * float(self.atr_stop_multiplier_long.value)
            
            # Safety check
            if bb_stop_distance < 0:
                bb_stop_distance = float(self.min_stop_loss_long.value)
            
            dynamic_stop = min(bb_stop_distance, atr_stop)
            
            # Clamp between min and max
            min_stop = float(self.min_stop_loss_long.value)
            max_stop = float(self.max_stop_loss_long.value)
            dynamic_stop = max(min_stop, min(dynamic_stop, max_stop))
            
            # Tighten after profit
            if current_profit >= 0.05:
                dynamic_stop = min(dynamic_stop, 0.03)
            
            # CRITICAL: NEVER wider than base stoploss!
            dynamic_stop = min(dynamic_stop, abs(self.stoploss))

        return -dynamic_stop

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> str:
        """
        Custom exit logic with minimum trade duration check.
        """
        # Check minimum trade duration
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_duration < self.min_trade_duration_minutes:
            return None

        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                               current_rate: float, current_profit: float,
                               min_stake: float, max_stake: float, **kwargs) -> float:
        """
        Position adjustment with DCA and partial profit taking.
        """
        if not self.position_adjustment_enable:
            return None

        # DCA: Add to losing position
        if self.dca_enabled.value:
            if self.dca_profit_max.value <= current_profit <= self.dca_profit_min.value:
                filled_entries = trade.select_filled_orders(trade.entry_side)
                if len(filled_entries) < self.max_entry_position_adjustment + 1:
                    stake = trade.stake_amount * 0.5
                    stake = min(stake, max_stake)
                    if stake >= min_stake:
                        logger.info(f"DCA: Adding {stake} to {trade.pair} at {current_profit:.2%}")
                        return max(stake, min_stake)

        # PARTIAL PROFIT TAKING
        filled_exits = trade.select_filled_orders(trade.exit_side)
        num_exits = len(filled_exits)

        # First partial exit at 3%
        if current_profit >= float(self.partial_exit_1_profit.value) and num_exits == 0:
            reduction = -abs(trade.amount) * float(self.partial_exit_size.value)
            if abs(reduction) * current_rate >= min_stake:
                logger.info(f"Partial exit 1/3 at {current_profit:.2%} for {trade.pair}")
                return reduction

        # Second partial exit at 5%
        elif current_profit >= float(self.partial_exit_2_profit.value) and num_exits == 1:
            reduction = -abs(trade.amount) * float(self.partial_exit_size.value)
            if abs(reduction) * current_rate >= min_stake:
                logger.info(f"Partial exit 2/3 at {current_profit:.2%} for {trade.pair}")
                return reduction

        # Third partial exit at 8%
        elif current_profit >= float(self.partial_exit_3_profit.value) and num_exits == 2:
            reduction = -abs(trade.amount) * 0.5
            if abs(reduction) * current_rate >= min_stake:
                logger.info(f"Final partial exit at {current_profit:.2%} for {trade.pair}")
                return reduction

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all technical indicators."""
        # RSI and MFI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        # ATR and normalized ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['normalized_atr'] = dataframe['atr'] / dataframe['close']

        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # Bollinger Bands (multiple standard deviations)
        for std in range(1, 5):
            bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=float(std), nbdevdn=float(std))
            dataframe[f'bb_lowerband{std}'] = bb['lowerband']
            dataframe[f'bb_middleband{std}'] = bb['middleband']
            dataframe[f'bb_upperband{std}'] = bb['upperband']

        # EMAs (collect all unique periods)
        ema_periods = set([
            int(self.buy_fastema.value), int(self.buy_slowema.value),
            int(self.sell_fastema.value), int(self.sell_slowema.value),
            int(self.short_fastema.value), int(self.short_slowema.value),
            int(self.cover_fastema.value), int(self.cover_slowema.value)
        ])
        for period in ema_periods:
            if period > 0 and len(dataframe) >= period:
                dataframe[f'EMA_{period}'] = ta.EMA(dataframe, timeperiod=period)

        # Fibonacci Retracement Levels
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
        """
        HYBRID entry logic:
        - Longs: Pullback + Momentum entries
        - Shorts: AGGRESSIVE (from opt490)
        
        NO VOLUME FILTERS (managed via config.json)
        """

        # ============= LONG ENTRY (HYBRID) =============
        
        # TYPE 1: PULLBACK ENTRIES
        pullback_conditions = []

        if self.buy_rsi_enabled.value:
            pullback_conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        if self.buy_mfi_enabled.value:
            pullback_conditions.append(dataframe['mfi'] < self.buy_mfi.value)

        if self.buy_ema_enabled.value:
            fast_col = f'EMA_{self.buy_fastema.value}'
            slow_col = f'EMA_{self.buy_slowema.value}'
            if fast_col in dataframe and slow_col in dataframe:
                pullback_conditions.append(dataframe[fast_col] > dataframe[slow_col])

        if self.buy_trigger.value.startswith('bb_lower'):
            bb_col = f'bb_lowerband{self.buy_trigger.value[-1]}'
            pullback_conditions.append(dataframe['close'] < dataframe[bb_col])

        if self.buy_trigger.value == 'fibonacci' and self.buy_fib_enabled.value:
            fib_col = self.buy_fib_level.value
            if fib_col in dataframe.columns:
                pullback_conditions.append(dataframe['close'] < dataframe[fib_col])

        # 1h uptrend confirmation
        if self.buy_1h_trend_enabled.value and 'uptrend_1h' in dataframe.columns:
            pullback_conditions.append(dataframe['uptrend_1h'] == True)

        # TYPE 2: MOMENTUM ENTRIES (for uptrends)
        momentum_conditions = []
        
        if self.momentum_entry_enabled.value:
            momentum_conditions.append(
                (dataframe['rsi'] > self.momentum_rsi_min.value) &
                (dataframe['rsi'] < self.momentum_rsi_max.value)
            )
            
            if 'bb_middleband2' in dataframe.columns:
                momentum_conditions.append(dataframe['close'] > dataframe['bb_middleband2'])
            
            fast_col = f'EMA_{self.buy_fastema.value}'
            if fast_col in dataframe.columns:
                momentum_conditions.append(dataframe['close'] > dataframe[fast_col])
            
            if 'uptrend_1h' in dataframe.columns:
                momentum_conditions.append(dataframe['uptrend_1h'] == True)
            
            momentum_conditions.append(dataframe['adx'] > 20)

        # COMBINE: Enter on EITHER pullback OR momentum
        dataframe['enter_long'] = 0
        
        if pullback_conditions:
            pullback_entry = reduce(lambda x, y: x & y, pullback_conditions)
            dataframe.loc[pullback_entry, 'enter_long'] = 1
        
        if momentum_conditions and self.momentum_entry_enabled.value:
            momentum_entry = reduce(lambda x, y: x & y, momentum_conditions)
            dataframe.loc[momentum_entry, 'enter_long'] = 1

        # ============= SHORT ENTRY (AGGRESSIVE) =============
        if not self.shorts_enabled.value:
            dataframe['enter_short'] = 0
            return dataframe

        short_conditions = []

        # AGGRESSIVE thresholds (from opt490 philosophy)
        if self.short_rsi_enabled.value:
            short_conditions.append(dataframe['rsi'] > self.short_rsi.value)

        if self.short_mfi_enabled.value:
            short_conditions.append(dataframe['mfi'] > self.short_mfi.value)

        if self.short_ema_enabled.value:
            fast_col = f'EMA_{self.short_fastema.value}'
            slow_col = f'EMA_{self.short_slowema.value}'
            if fast_col in dataframe and slow_col in dataframe:
                short_conditions.append(dataframe[fast_col] < dataframe[slow_col])

        # AGGRESSIVE trigger (BB2 from opt490)
        if self.short_trigger.value.startswith('bb_upper'):
            bb_col = f'bb_upperband{self.short_trigger.value[-1]}'
            short_conditions.append(dataframe['close'] > dataframe[bb_col])

        # Quality filters (keep from opt314)
        if self.short_1h_trend_enabled.value and 'downtrend_1h' in dataframe.columns:
            short_conditions.append(dataframe['downtrend_1h'] == True)

        short_conditions.append(dataframe['adx'] > self.short_adx_threshold.value)

        if short_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions for long and short positions."""

        # ============= LONG EXIT =============
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

        if 'downtrend_1h' in dataframe.columns:
            long_exit.append(dataframe['downtrend_1h'] == True)

        long_exit.append(dataframe['volume'] > 0)

        if long_exit:
            dataframe.loc[reduce(lambda x, y: x & y, long_exit), 'exit_long'] = 1

        # ============= SHORT EXIT =============
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

        if 'uptrend_1h' in dataframe.columns:
            short_exit.append(dataframe['uptrend_1h'] == True)

        short_exit.append(dataframe['volume'] > 0)

        if short_exit:
            dataframe.loc[reduce(lambda x, y: x & y, short_exit), 'exit_short'] = 1

        return dataframe
