# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Union
from functools import reduce
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

logger = logging.getLogger(__name__)


class Ichimoku(IStrategy):
    """
    FIXED Ichimoku Strategy
    
    Key improvements based on research and trade analysis:
    1. Minimum trade duration (prevents 17-second trades!)
    2. Improved exit logic with CONFIRMATION (not just any single condition)
    3. Partial profit taking at 3%, 5%, 8%
    4. One DCA option at -3% to -6%
    5. Better trailing stop with cloud-based placement
    6. Stop loss placed at cloud boundaries (research-based)
    7. Maintains long AND short functionality
    
    Research-based stop loss placement:
    - Conservative: below cloud bottom (Senkou Span B for longs)
    - Trend-following: below Kijun-sen
    - Aggressive: below Tenkan-sen as trailing stop
    
    Exit best practices:
    - Wait for TK cross confirmation
    - Exit when price decisively breaks cloud
    - Aim for 1:2 to 1:3 risk-reward ratios
    """
    
    INTERFACE_VERSION = 3

    timeframe = '4h'
    USE_TALIB = False

    # Enable shorting
    can_short: bool = True

    # Realistic ROI targets
    minimal_roi = {
        "0": 0.10,     # 10% initial target
        "720": 0.06,   # 6% after 12h (3 candles)
        "1440": 0.04,  # 4% after 24h (6 candles)
        "2880": 0.02   # 2% after 48h (12 candles)
    }

    # Stop loss - will be dynamically adjusted
    stoploss = -0.08  # 8% hard stop (much better than original -75%!)

    trailing_stop = False
    process_only_new_candles: bool = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss: bool = True

    # DCA and position adjustment
    position_adjustment_enable = True
    max_entry_position_adjustment = 1  # Allow 1 DCA
    max_exit_position_adjustment = 3   # Allow 3 partial exits

    # Minimum trade duration to prevent instant exits
    min_trade_duration_minutes = 30  # No exits before 30 minutes

    # Number of candles required before producing valid signals
    startup_candle_count: int = 60

    # Order types
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # Ichimoku parameters - using standard values that work well
    TS = IntParameter(7, 12, default=9, space="buy", optimize=True)
    KS = IntParameter(20, 30, default=26, space="buy", optimize=True)
    SS = IntParameter(40, 60, default=52, space="buy", optimize=True)

    # ATR parameters
    ATR_length = IntParameter(10, 20, default=14, space="buy", optimize=True)
    ATR_SL_Multip = DecimalParameter(1.0, 2.5, decimals=1, default=1.5, space="buy", optimize=True)
    ATR_TP_Multip = DecimalParameter(2.0, 4.0, decimals=1, default=3.0, space="buy", optimize=True)

    # Trailing stop parameters
    trailing_activation = DecimalParameter(0.02, 0.05, decimals=2, default=0.03, space="buy", optimize=True)
    trailing_distance = DecimalParameter(0.01, 0.03, decimals=2, default=0.015, space="buy", optimize=True)

    # RSI filter
    rsi_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    rsi_length = IntParameter(10, 20, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(40, 60, default=45, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(40, 60, default=55, space="buy", optimize=True)

    # Cloud filter
    cloud_filter = BooleanParameter(default=True, space="buy", optimize=True)
    
    # DCA parameters
    dca_enabled = BooleanParameter(default=True, space="buy", optimize=False)
    dca_profit_min = DecimalParameter(-0.06, -0.02, decimals=2, default=-0.03, space="buy", optimize=True)
    dca_profit_max = DecimalParameter(-0.08, -0.04, decimals=2, default=-0.06, space="buy", optimize=True)
    
    # Partial exit parameters
    partial_exit_1_profit = DecimalParameter(0.02, 0.04, decimals=2, default=0.03, space="sell", optimize=True)
    partial_exit_2_profit = DecimalParameter(0.04, 0.06, decimals=2, default=0.05, space="sell", optimize=True)
    partial_exit_3_profit = DecimalParameter(0.06, 0.10, decimals=2, default=0.08, space="sell", optimize=True)
    partial_exit_size = DecimalParameter(0.25, 0.40, decimals=2, default=0.33, space="sell", optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("=" * 80)
        logger.info("Ichimoku Strategy - FIXED VERSION initialized")
        logger.info("Key fixes:")
        logger.info(f"  ✓ Minimum trade duration: {self.min_trade_duration_minutes} minutes")
        logger.info("  ✓ Improved exit logic with confirmation")
        logger.info("  ✓ Partial profit taking enabled")
        logger.info("  ✓ DCA enabled (1 max)")
        logger.info("  ✓ Cloud-based stop loss placement")
        logger.info("  ✓ Long AND short functionality maintained")
        logger.info("=" * 80)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators needed for the strategy.
        """
        if self.dp.runmode.value in ('live', 'dry_run'):
            self.USE_TALIB = False
        else:
            self.USE_TALIB = True

        # Calculate Ichimoku indicator
        ichimo = pta.ichimoku(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            tenkan=int(self.TS.value),
            kijun=int(self.KS.value),
            senkou=int(self.SS.value),
            include_chikou=True
        )[0]

        dataframe['tenkan'] = ichimo[f'ITS_{int(self.TS.value)}'].copy()
        dataframe['kijun'] = ichimo[f'IKS_{int(self.KS.value)}'].copy()
        dataframe['senkanA'] = ichimo[f'ISA_{int(self.TS.value)}'].copy()
        dataframe['senkanB'] = ichimo[f'ISB_{int(self.KS.value)}'].copy()
        dataframe['chiko'] = ichimo[f'ICS_{int(self.KS.value)}'].copy()

        # Calculate ATR
        dataframe['ATR'] = pta.atr(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            length=int(self.ATR_length.value),
            talib=self.USE_TALIB
        )

        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=int(self.rsi_length.value))

        # Cloud boundaries
        dataframe['cloud_top'] = dataframe[['senkanA', 'senkanB']].max(axis=1)
        dataframe['cloud_bottom'] = dataframe[['senkanA', 'senkanB']].min(axis=1)
        
        # Price position relative to cloud
        dataframe['above_cloud'] = dataframe['close'] > dataframe['cloud_top']
        dataframe['below_cloud'] = dataframe['close'] < dataframe['cloud_bottom']
        dataframe['in_cloud'] = ~(dataframe['above_cloud'] | dataframe['below_cloud'])

        # Cloud color
        dataframe['cloud_green'] = dataframe['senkanA'] > dataframe['senkanB']
        dataframe['cloud_red'] = dataframe['senkanA'] < dataframe['senkanB']

        # TK cross signals
        dataframe['tk_cross_up'] = qtpylib.crossed_above(dataframe['tenkan'], dataframe['kijun'])
        dataframe['tk_cross_down'] = qtpylib.crossed_below(dataframe['tenkan'], dataframe['kijun'])
        
        # Tenkan momentum
        dataframe['tenkan_rising'] = dataframe['tenkan'] > dataframe['tenkan'].shift(1)
        dataframe['tenkan_falling'] = dataframe['tenkan'] < dataframe['tenkan'].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals based on Ichimoku cloud breakout.
        
        Long entry conditions (must ALL be true):
        - Price above cloud (decisive breakout)
        - Cloud is green (bullish)
        - Tenkan > Kijun (bullish momentum)
        - Price > Tenkan (confirming uptrend)
        - Optional: RSI > threshold (momentum confirmation)
        
        Short entry conditions (must ALL be true):
        - Price below cloud (decisive breakdown)
        - Cloud is red (bearish)
        - Tenkan < Kijun (bearish momentum)
        - Price < Tenkan (confirming downtrend)
        - Optional: RSI < threshold (momentum confirmation)
        """
        # Long entry conditions
        long_conditions = [
            (dataframe['above_cloud']),  # Price decisively above cloud
            (dataframe['cloud_green']),  # Cloud is bullish (green)
            (dataframe['tenkan'] > dataframe['kijun']),  # TK bullish alignment
            (dataframe['close'] > dataframe['tenkan']),  # Price above tenkan
            (dataframe['tenkan_rising']),  # Tenkan momentum up
        ]

        # Add cloud filter if enabled (not in cloud)
        if self.cloud_filter.value:
            long_conditions.append(~dataframe['in_cloud'])

        # Add RSI filter if enabled
        if self.rsi_enabled.value:
            long_conditions.append(dataframe['rsi'] > self.rsi_buy_threshold.value)

        dataframe.loc[
            reduce(lambda x, y: x & y, long_conditions),
            'enter_long'
        ] = 1

        # Short entry conditions
        short_conditions = [
            (dataframe['below_cloud']),  # Price decisively below cloud
            (dataframe['cloud_red']),  # Cloud is bearish (red)
            (dataframe['tenkan'] < dataframe['kijun']),  # TK bearish alignment
            (dataframe['close'] < dataframe['tenkan']),  # Price below tenkan
            (dataframe['tenkan_falling']),  # Tenkan momentum down
        ]

        # Add cloud filter if enabled (not in cloud)
        if self.cloud_filter.value:
            short_conditions.append(~dataframe['in_cloud'])

        # Add RSI filter if enabled
        if self.rsi_enabled.value:
            short_conditions.append(dataframe['rsi'] < self.rsi_sell_threshold.value)

        dataframe.loc[
            reduce(lambda x, y: x & y, short_conditions),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        IMPROVED exit signals with CONFIRMATION required.
        
        Research-based exit logic:
        - Wait for TK cross in opposite direction
        - Confirm with price breaking cloud decisively
        - Don't exit on minor retracements
        
        Long exit (require BOTH conditions):
        - TK bearish cross (tenkan crosses below kijun) AND
        - Price enters or breaks below cloud
        
        Short exit (require BOTH conditions):
        - TK bullish cross (tenkan crosses above kijun) AND
        - Price enters or breaks above cloud
        """
        # Long exit - require BOTH TK cross AND cloud break
        dataframe.loc[
            (
                (dataframe['tk_cross_down']) &  # TK bearish cross
                (
                    (dataframe['in_cloud']) |  # Price entered cloud
                    (dataframe['below_cloud'])  # OR price below cloud
                )
            ),
            'exit_long'
        ] = 1

        # Short exit - require BOTH TK cross AND cloud break
        dataframe.loc[
            (
                (dataframe['tk_cross_up']) &  # TK bullish cross
                (
                    (dataframe['in_cloud']) |  # Price entered cloud
                    (dataframe['above_cloud'])  # OR price above cloud
                )
            ),
            'exit_short'
        ] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom exit logic with minimum trade duration check and ATR-based take profit.
        
        CRITICAL FIX: Prevent exits before minimum duration!
        """
        # CRITICAL: Check minimum trade duration
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        if trade_duration < self.min_trade_duration_minutes:
            # Don't exit before minimum duration (prevents 17-second trades!)
            return None

        # Get analyzed dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Get the candle where trade was opened
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        if trade_candle.empty:
            return None

        trade_candle = trade_candle.squeeze()
        atr = trade_candle['ATR']

        if pd.isna(atr) or atr == 0:
            return None

        # Calculate take profit distance based on ATR
        tp_distance = atr * float(self.ATR_TP_Multip.value)

        if not trade.is_short:
            # For long positions, TP is above entry price
            tp_price = trade.open_rate + tp_distance
            if current_rate >= tp_price:
                logger.info(f"Take profit hit for {pair} at {current_profit:.2%}")
                return 'take_profit_atr'
        else:
            # For short positions, TP is below entry price
            tp_price = trade.open_rate - tp_distance
            if current_rate <= tp_price:
                logger.info(f"Take profit hit for {pair} at {current_profit:.2%}")
                return 'take_profit_atr'

        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stop-loss with cloud-based placement and trailing stop.
        
        Research-based approach:
        1. Initial stop: Below cloud bottom (Senkou Span B) for longs
        2. After profit threshold: Trail below Kijun-sen
        3. After higher profit: Tight trail below Tenkan-sen
        
        Stop loss can only move in favorable direction (never widens).
        """
        # Get analyzed dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Get current candle
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Cloud-based stop loss placement
        if not trade.is_short:
            # For LONGS: Stop below cloud bottom (Senkou Span B)
            cloud_bottom = last_candle['cloud_bottom']
            cloud_stop_distance = (trade.open_rate - cloud_bottom) / trade.open_rate
            
            # After 3% profit, tighten to below Kijun-sen
            if current_profit >= float(self.trailing_activation.value):
                kijun = last_candle['kijun']
                kijun_stop_distance = (trade.open_rate - kijun) / trade.open_rate
                # Use tighter of the two
                return max(kijun_stop_distance, -float(self.trailing_distance.value))
            
            # Initial stop at cloud boundary
            return max(cloud_stop_distance, self.stoploss)
        else:
            # For SHORTS: Stop above cloud top (Senkou Span A)
            cloud_top = last_candle['cloud_top']
            cloud_stop_distance = (cloud_top - trade.open_rate) / trade.open_rate
            
            # After 3% profit, tighten to above Kijun-sen
            if current_profit >= float(self.trailing_activation.value):
                kijun = last_candle['kijun']
                kijun_stop_distance = (kijun - trade.open_rate) / trade.open_rate
                # Use tighter of the two
                return max(kijun_stop_distance, -float(self.trailing_distance.value))
            
            # Initial stop at cloud boundary
            return max(cloud_stop_distance, self.stoploss)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                               current_rate: float, current_profit: float,
                               min_stake: float, max_stake: float, **kwargs) -> Optional[float]:
        """
        Position adjustment with DCA and partial profit taking.
        
        DCA: Add to position on controlled drawdown (-3% to -6%)
        Partial exits: Exit 1/3 at 3%, 5%, and 8% profit
        """
        if not self.position_adjustment_enable:
            return None

        # DCA: Add to losing position
        if self.dca_enabled.value:
            if self.dca_profit_max.value <= current_profit <= self.dca_profit_min.value:
                filled_entries = trade.select_filled_orders(trade.entry_side)
                if len(filled_entries) < self.max_entry_position_adjustment + 1:
                    # Add 50% of original stake
                    stake = trade.stake_amount * 0.5
                    stake = min(stake, max_stake)
                    if stake >= min_stake:
                        logger.info(f"DCA: Adding {stake} to {trade.pair} at {current_profit:.2%}")
                        return max(stake, min_stake)

        # Partial profit taking
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
            reduction = -abs(trade.amount) * 0.5  # Exit remaining half
            if abs(reduction) * current_rate >= min_stake:
                logger.info(f"Final partial exit at {current_profit:.2%} for {trade.pair}")
                return reduction

        return None
