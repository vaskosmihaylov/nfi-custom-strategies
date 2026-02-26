import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

class FenixTopProfit(IStrategy):

    ### Strategy parameters ###
    exit_profit_only = True
    use_custom_stoploss = False
    trailing_stop = False
    ignore_roi_if_entry_signal = True
    can_short = True
    use_exit_signal = True
    stoploss = -0.10
    startup_candle_count: int = 100
    timeframe = '1h'

    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 1  # Maximum DCA multiplier

    # ROI table:
    minimal_roi = {
        "0": 0.3,   # 3% de lucro após 0 minutos
        "30": 0.02,  # Reduz para 2% após 30 min
        "120": 0.01, # Reduz para 1% após 2 horas
        "240": 0     # Sem forçar saída após 4 horas
    }

    ### Hyperopt ###

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 120, default=72, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    ### Protections ###
    @property
    def protections(self):
        """
            Defines the protections to apply during trading operations.
        """
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot

    ### Dollar Cost Averaging (DCA) ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
            Calculates the stake amount to use for a trade, adjusted dynamically based on the DCA multiplier.
            - The proposed stake is divided by the maximum DCA multiplier (`self.max_dca_multiplier`)
              to determine the adjusted stake.
            - If the adjusted stake is lower than the allowed minimum (`min_stake`), it is automatically increased
              to meet the minimum stake requirement.
        """
        # Calculates the adjusted stake amount based on the DCA multiplier.
        adjusted_stake = proposed_stake / self.max_dca_multiplier

        # Automatically adjusts to the minimum stake if it is too low.
        if adjusted_stake < min_stake:
            adjusted_stake = min_stake

        return adjusted_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.
        """
        if trade.entry_side == "buy":  # For longs
            if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)
            if current_profit > -0.04 and trade.nr_of_successful_entries == 1:
                return None
            if current_profit > -0.06 and trade.nr_of_successful_entries == 2:
                return None
            if current_profit > -0.08 and trade.nr_of_successful_entries == 3:
                return None

        else:  # For shorts
            if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)
            if current_profit > -0.04 and trade.nr_of_successful_entries == 1:
                return None
            if current_profit > -0.06 and trade.nr_of_successful_entries == 2:
                return None
            if current_profit > -0.08 and trade.nr_of_successful_entries == 3:
                return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        try:
            stake_amount = filled_entries[0].cost

            if count_of_entries > 1:
                stake_amount = stake_amount * (1 + (count_of_entries - 1) * 0.5)  # 50% increase per additional entry

            return stake_amount

        except Exception as exception:
            logger.error(f"Error adjusting DCA position for the pair {trade.pair}: {exception}")
            return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
            Calculates technical indicators used to define entry and exit signals.
        """
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # ADX for trend strength
        dataframe['pdi'] = ta.PLUS_DI(dataframe, timeperiod=14)  # Positive directional index
        dataframe['mdi'] = ta.MINUS_DI(dataframe, timeperiod=14)  # Negative directional index
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)  # 200-period SMA for trend direction
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  # RSI for momentum

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
            Defines the conditions for long/short entries based on
            technical indicators such as ADX, PDI, MDI, and RSI.
        """
        # Long entry: PDI crosses above MDI and ADX > 25 (strong trend), RSI > 30 (not oversold)
        df.loc[
            (
                (qtpylib.crossed_above(df['pdi'], df['mdi'])) &
                (df['adx'] > 25) &
                (df['rsi'] > 30) &
                (df['close'] > df['sma200'])  # Check for uptrend using 200 SMA
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'FenixTopProfit Entry Long')

        # Short entry: MDI crosses above PDI and ADX > 25 (strong trend), RSI < 70 (not overbought)
        df.loc[
            (
                (qtpylib.crossed_above(df['mdi'], df['pdi'])) &
                (df['adx'] > 25) &
                (df['rsi'] < 70) &
                (df['close'] < df['sma200'])  # Check for downtrend using 200 SMA
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'FenixTopProfit Entry Short')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
            Defines exit conditions for trades based on the ADX, PDI, and MDI:
            - Exit Long: Triggered when 'adx' drops below 25 and RSI > 50 (indicating weakening trend).
            - Exit Short: Triggered when 'adx' drops below 25 and RSI < 50 (indicating weakening downtrend).
        """
        # Long exit: ADX < 25 and RSI > 50 (indicating trend weakness or reversal)
        df.loc[
            (
                (df['adx'] < 25) &
                (df['rsi'] > 50)  # RSI confirms weakening upward momentum
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'FenixTopProfit Exit Long')

        # Short exit: ADX < 25 and RSI < 50 (indicating trend weakness or reversal)
        df.loc[
            (
                (df['adx'] < 25) &
                (df['rsi'] < 50)  # RSI confirms weakening downward momentum
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'FenixTopProfit Exit Short')

        return df
