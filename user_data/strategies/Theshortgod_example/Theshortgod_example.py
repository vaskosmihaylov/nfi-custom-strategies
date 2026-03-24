from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)

logger = logging.getLogger(__name__)

INTERFACE_VERSION = 3

# self.dp.send_msg()

class Theshortgod(IStrategy):
    can_short = True

    # Execute on 1h candles, using 8h informative data as context
    timeframe = "1h"
    informative_timeframe = "8h"

    process_only_new_candles = True
    startup_candle_count = 400  # Need sufficient historical data for percentile calculations

    # Disable static stoploss; rely on custom_exit for exits
    # Original setting reference: -8% at 10x leverage (~0.8% price move)
    stoploss = -1000
    use_custom_stoploss = False  # This strategy uses custom_exit for drawdown-based profit taking, hard stop as backup

    # Enable position adjustment/DCA
    position_adjustment_enable = True
    
    # make by yourself

    # ===== Trailing drawdown exit =====
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Logic:
          - When profit >= trail_start, record max_profit
          - If (max_profit - current_profit) >= dynamic threshold → trigger exit
          - Dynamic threshold: base 0.03 + floor( (max_profit - trail_start)/0.05 ) * trail_step
            (i.e., for every 5% more profit, drawdown tolerance + trail_step)
        """
        # [Live Trading Consistency] Only check exit conditions at new candle start
        # Calculate current candle start time (aligned to hour)
        current_candle_start = current_time.replace(minute=0, second=0, microsecond=0)
        last_check_candle = trade.get_custom_data("last_exit_check_candle", None)


        # Skip if already checked within same candle（avoid price noise）
        if last_check_candle == current_candle_start.isoformat():
            return None

        # Record current candle check time
        trade.set_custom_data("last_exit_check_candle", current_candle_start.isoformat())

        # 1. Hard stoploss protection
        if current_profit <= float(self.max_single_loss.value):
            logger.critical(f"[Hard Stoploss]{pair} | {current_profit:.2%}")
            return "hard_stoploss"

        # 2. Cumulative loss protection after multiple DCA
        if trade.nr_of_successful_entries >= 3:
            max_allow = float(self.max_dca_loss.value)
            if current_profit <= max_allow:
                logger.critical(
                    f"[DCA Cumulative Stoploss]{pair} | {current_profit:.2%} | entries{trade.nr_of_successful_entries} times"
                )
                return "dca_max_loss"

        start = float(self.trail_start.value)
        step = float(self.trail_step.value)

        maxp = float(trade.get_custom_data("max_profit", -1.0))

        if current_profit >= start:
            if current_profit > maxp:
                trade.set_custom_data("max_profit", current_profit)
                logger.debug(
                    f"[Update Max Profit]{pair} | "
                    f"New max: {current_profit:.2%} | "
                    f"Price: {current_rate:.4f}"
                )
                return None

            # Derive allowed drawdown
            # Example: start=0.5, step=0.05
            # Example: max_profit=0.7 => over=0.2 => floor(...)=2 => allow=0.1
            over = max(0.0, maxp - start)
            buckets = int(np.floor(over / 0.10))
            allow_draw = step * (1 + buckets)
            drawdown = maxp - current_profit

            if drawdown >= allow_draw:
                logger.info(
                    f"[Drawdown Exit]{pair} | "
                    f"Max profit: {maxp:.2%} | "
                    f"Current profit: {current_profit:.2%} | "
                    f"Drawdown: {drawdown:.2%} | "
                    f"Allowed Drawdown: {allow_draw:.2%} | "
                    f"Price: {current_rate:.4f} | "
                    f"Position held: {(current_time - trade.open_date_utc).days}days"
                )
                return "trail_drawdown_exit"

        else:
            # Reset tracking when profit below activation threshold
            if maxp > 0:
                trade.set_custom_data("max_profit", -1.0)

        return None