# --- Do not remove these libs ---
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from pathlib import Path
import logging
import time

current_time = time.time()
import datetime
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (
    IStrategy,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    merge_informative_pair,
    stoploss_from_open,
    stoploss_from_absolute,
    merge_informative_pair,
)
from freqtrade.persistence import Trade
from typing import List, Tuple, Optional, Dict, Any
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque
import optuna
from optuna.samplers import TPESampler
from optuna.exceptions import OptunaError
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


# FIX: Remove the problematic relative import and create a simple inline replacement
# from .optuna_manager import OptunaManager
class RealOptunaManager:
    """Real Optuna-based optimization manager"""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.studies = {}  # Store studies per pair
        self.best_params_cache = {}
        self.performance_history = {}
        self.optimization_trigger_count = {}
        self.pretraining_enabled = True
        self.pretraining_days = 14
        self.pretraining_done = {}

        logger.info(
            f"📊 [OPTUNA] Initializing RealOptunaManager for strategy: {strategy_name}"
        )
        logger.info(f"🎯 [OPTUNA] Real Optuna optimization enabled")

    def get_best_params(self, pair: str) -> Dict[str, Any]:
        """Get best parameters for a pair"""
        if pair in self.best_params_cache:
            return self.best_params_cache[pair]

        # Try to load from existing study
        try:
            study_name = (
                f"{self.strategy_name}_{pair.replace('/', '_').replace(':', '_')}"
            )
            study = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///user_data/strategies/optuna_studies/optuna_{study_name}.db",
            )
            if study.trials:
                self.studies[pair] = study
                self.best_params_cache[pair] = study.best_params
                return study.best_params
        except:
            pass

        return None

    def should_optimize(self, pair: str) -> bool:
        """Check if optimization should be triggered"""
        if pair not in self.best_params_cache:
            return True

        if pair in self.performance_history:
            recent_performance = self.performance_history[pair][-10:]
            if len(recent_performance) >= 5:
                avg_performance = sum(recent_performance) / len(recent_performance)
                if avg_performance < -0.02:
                    return True

        return False

    def should_optimize_based_on_performance(self, pair: str) -> bool:
        """Check if performance-based optimization is needed"""
        if pair not in self.performance_history:
            return False

        recent_trades = self.performance_history[pair][-10:]
        if len(recent_trades) >= 5:
            avg_performance = sum(recent_trades) / len(recent_trades)
            return avg_performance < -0.015

        return False

    def update_performance(self, pair: str, profit_ratio: float):
        """Update performance tracking"""
        if pair not in self.performance_history:
            self.performance_history[pair] = []

        self.performance_history[pair].append(profit_ratio)

        if len(self.performance_history[pair]) > 50:
            self.performance_history[pair] = self.performance_history[pair][-50:]

        logger.debug(f"📈 [OPTUNA] Updated performance for {pair}: {profit_ratio:.4f}")

    def optimize_coin(self, pair: str, objective_func, n_trials: int = 10):
        """Optimize parameters for a specific coin using real Optuna"""
        logger.info(
            f"🚀 [OPTUNA] Starting REAL optimization for {pair} with {n_trials} trials"
        )

        try:
            from pathlib import Path

            study_name = (
                f"{self.strategy_name}_{pair.replace('/', '_').replace(':', '_')}"
            )

            Path("user_data/strategies/optuna_studies").mkdir(
                parents=True, exist_ok=True
            )
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=TPESampler(),
                storage=f"sqlite:///user_data/strategies/optuna_studies/optuna_{study_name}.db",
                load_if_exists=True,
            )

            study.optimize(objective_func, n_trials=n_trials)

            self.studies[pair] = study
            self.best_params_cache[pair] = study.best_params

            logger.info(f"✅ [OPTUNA] REAL optimization completed for {pair}")
            logger.info(f"🏆 [OPTUNA] Best value: {study.best_value:.4f}")
            self._log_formatted_parameters(pair, study.best_params)

            return True

        except Exception as e:
            logger.error(f"❌ [OPTUNA] Real optimization failed for {pair}: {e}")
            return False

    def _log_formatted_parameters(self, pair: str, params: Dict[str, Any]):
        """Log parameters in a nicely formatted way - Enhanced with exit parameters"""
        logger.info(f"📈 [OPTUNA] Optimized parameters for {pair}:")
        logger.info(f"┌─ DIVERGENCE PARAMETERS")
        logger.info(
            f"│  • min_divergence_count: {params.get('min_divergence_count', 'N/A')}"
        )
        logger.info(
            f"│  • min_signal_strength: {params.get('min_signal_strength', 'N/A')}"
        )
        logger.info(
            f"│  • min_strong_divergence_count: {params.get('min_strong_divergence_count', 'N/A')}"
        )
        logger.info(
            f"│  • divergence_freshness_periods: {params.get('divergence_freshness_periods', 'N/A')}"
        )

        logger.info(f"├─ RSI TIMING PARAMETERS")
        logger.info(f"│  • rsi_overbought: {params.get('rsi_overbought', 'N/A')}")
        logger.info(f"│  • rsi_oversold: {params.get('rsi_oversold', 'N/A')}")
        logger.info(
            f"│  • rsi_long_upper_tight: {params.get('rsi_long_upper_tight', 'N/A')}"
        )
        logger.info(
            f"│  • rsi_short_lower_tight: {params.get('rsi_short_lower_tight', 'N/A')}"
        )
        logger.info(
            f"│  • rsi_recovery_periods: {params.get('rsi_recovery_periods', 'N/A')}"
        )

        logger.info(f"├─ VOLUME PARAMETERS")
        logger.info(f"│  • volume_threshold: {params.get('volume_threshold', 'N/A')}")
        logger.info(f"│  • volume_ratio_min: {params.get('volume_ratio_min', 'N/A')}")
        logger.info(
            f"│  • volume_trend_multiplier: {params.get('volume_trend_multiplier', 'N/A')}"
        )
        logger.info(
            f"│  • volume_momentum_multiplier: {params.get('volume_momentum_multiplier', 'N/A')}"
        )

        logger.info(f"├─ POSITION TIMING")
        logger.info(
            f"│  • bb_percent_long_max: {params.get('bb_percent_long_max', 'N/A')}"
        )
        logger.info(
            f"│  • bb_percent_short_min: {params.get('bb_percent_short_min', 'N/A')}"
        )
        logger.info(
            f"│  • price_position_long_max: {params.get('price_position_long_max', 'N/A')}"
        )
        logger.info(
            f"│  • price_position_short_min: {params.get('price_position_short_min', 'N/A')}"
        )

        logger.info(f"├─ MARKET STRUCTURE")
        logger.info(f"│  • adx_threshold: {params.get('adx_threshold', 'N/A')}")
        logger.info(f"│  • adx_trending_min: {params.get('adx_trending_min', 'N/A')}")
        logger.info(f"│  • chop_threshold: {params.get('chop_threshold', 'N/A')}")
        logger.info(f"│  • max_volatility: {params.get('max_volatility', 'N/A')}")
        logger.info(f"│  • min_volatility: {params.get('min_volatility', 'N/A')}")

        logger.info(f"├─ REVERSAL CONDITIONS")
        logger.info(
            f"│  • min_oversold_conditions: {params.get('min_oversold_conditions', 'N/A')}"
        )
        logger.info(
            f"│  • min_overbought_conditions: {params.get('min_overbought_conditions', 'N/A')}"
        )
        logger.info(
            f"│  • min_reversal_signals: {params.get('min_reversal_signals', 'N/A')}"
        )
        logger.info(f"│  • min_trend_filters: {params.get('min_trend_filters', 'N/A')}")

        logger.info(f"├─ VOLUME BREAKOUT PARAMETERS")
        logger.info(
            f"│  • volume_breakout_multiplier: {params.get('volume_breakout_multiplier', 'N/A')}"
        )
        logger.info(
            f"│  • volume_breakout_rsi_min: {params.get('volume_breakout_rsi_min', 'N/A')}"
        )
        logger.info(
            f"│  • volume_breakout_rsi_max: {params.get('volume_breakout_rsi_max', 'N/A')}"
        )

        logger.info(f"├─ MEAN REVERSION PARAMETERS")
        logger.info(
            f"│  • bb_oversold_threshold: {params.get('bb_oversold_threshold', 'N/A')}"
        )
        logger.info(
            f"│  • bb_overbought_threshold: {params.get('bb_overbought_threshold', 'N/A')}"
        )
        logger.info(
            f"│  • mean_reversion_rsi_oversold: {params.get('mean_reversion_rsi_oversold', 'N/A')}"
        )
        logger.info(
            f"│  • mean_reversion_rsi_overbought: {params.get('mean_reversion_rsi_overbought', 'N/A')}"
        )

        logger.info(f"├─ MOMENTUM CONTINUATION PARAMETERS")
        logger.info(
            f"│  • momentum_ema_periods: {params.get('momentum_ema_periods', 'N/A')}"
        )
        logger.info(
            f"│  • momentum_strength_min: {params.get('momentum_strength_min', 'N/A')}"
        )
        logger.info(
            f"│  • momentum_pullback_max: {params.get('momentum_pullback_max', 'N/A')}"
        )
        logger.info(
            f"│  • momentum_pullback_min: {params.get('momentum_pullback_min', 'N/A')}"
        )

        # === NEW EXIT PARAMETER LOGGING ===
        logger.info(f"├─ 🎯 GLOBAL EXIT SETTINGS")
        logger.info(
            f"│  • emergency_profit_limit: {params.get('emergency_profit_limit', 'N/A')}"
        )
        logger.info(
            f"│  • session_multiplier_overlap: {params.get('session_multiplier_overlap', 'N/A')}"
        )
        logger.info(
            f"│  • session_multiplier_major: {params.get('session_multiplier_major', 'N/A')}"
        )
        logger.info(
            f"│  • session_multiplier_quiet: {params.get('session_multiplier_quiet', 'N/A')}"
        )
        logger.info(
            f"│  • volatility_sensitivity: {params.get('volatility_sensitivity', 'N/A')}"
        )
        logger.info(
            f"│  • high_signal_multiplier: {params.get('high_signal_multiplier', 'N/A')}"
        )
        logger.info(
            f"│  • low_signal_multiplier: {params.get('low_signal_multiplier', 'N/A')}"
        )

        logger.info(f"├─ 🔄 MEAN REVERSION (MR1) EXITS")
        logger.info(
            f"│  • mr1_max_hold_minutes: {params.get('mr1_max_hold_minutes', 'N/A')} min"
        )
        logger.info(f"│  • mr1_rsi_exit_long: {params.get('mr1_rsi_exit_long', 'N/A')}")
        logger.info(
            f"│  • mr1_rsi_exit_short: {params.get('mr1_rsi_exit_short', 'N/A')}"
        )
        logger.info(
            f"│  • mr1_quick_profit_target: {params.get('mr1_quick_profit_target', 'N/A')}"
        )
        logger.info(
            f"│  • mr1_timeout_min_profit: {params.get('mr1_timeout_min_profit', 'N/A')}"
        )
        logger.info(f"│  • mr1_bb_exit_long: {params.get('mr1_bb_exit_long', 'N/A')}")
        logger.info(f"│  • mr1_bb_exit_short: {params.get('mr1_bb_exit_short', 'N/A')}")

        logger.info(f"├─ 🚀 MOMENTUM CONTINUATION (MC1) EXITS")
        logger.info(f"│  • mc1_profit_target: {params.get('mc1_profit_target', 'N/A')}")
        logger.info(
            f"│  • mc1_momentum_break_threshold: {params.get('mc1_momentum_break_threshold', 'N/A')}"
        )
        logger.info(
            f"│  • mc1_max_hold_minutes: {params.get('mc1_max_hold_minutes', 'N/A')} min"
        )
        logger.info(
            f"│  • mc1_adx_exit_threshold: {params.get('mc1_adx_exit_threshold', 'N/A')}"
        )
        logger.info(
            f"│  • mc1_timeout_min_profit: {params.get('mc1_timeout_min_profit', 'N/A')}"
        )

        logger.info(f"├─ 📊 VOLUME BREAKOUT (VB1) EXITS")
        logger.info(
            f"│  • vb1_volume_fade_threshold: {params.get('vb1_volume_fade_threshold', 'N/A')}"
        )
        logger.info(f"│  • vb1_profit_target: {params.get('vb1_profit_target', 'N/A')}")
        logger.info(f"│  • vb1_quick_profit: {params.get('vb1_quick_profit', 'N/A')}")
        logger.info(
            f"│  • vb1_min_volatility: {params.get('vb1_min_volatility', 'N/A')}"
        )

        logger.info(f"├─ 🔄 REVERSAL (RSV1) EXITS")
        logger.info(
            f"│  • rsv1_rsi_recovery_threshold: {params.get('rsv1_rsi_recovery_threshold', 'N/A')}"
        )
        logger.info(
            f"│  • rsv1_profit_target: {params.get('rsv1_profit_target', 'N/A')}"
        )
        logger.info(
            f"│  • rsv1_max_hold_minutes: {params.get('rsv1_max_hold_minutes', 'N/A')} min"
        )
        logger.info(
            f"│  • rsv1_min_timeout_profit: {params.get('rsv1_min_timeout_profit', 'N/A')}"
        )

        logger.info(f"├─ 📈 TREND FOLLOWING EXITS")
        logger.info(
            f"│  • trend_profit_target: {params.get('trend_profit_target', 'N/A')}"
        )
        logger.info(
            f"│  • trend_ema_break_periods: {params.get('trend_ema_break_periods', 'N/A')}"
        )

        logger.info(f"├─ ⏰ UNIVERSAL EXIT PARAMETERS")
        logger.info(
            f"│  • medium_term_base_target: {params.get('medium_term_base_target', 'N/A')}"
        )
        logger.info(
            f"│  • reversal_min_profit_long: {params.get('reversal_min_profit_long', 'N/A')}"
        )
        logger.info(
            f"│  • reversal_min_profit_short: {params.get('reversal_min_profit_short', 'N/A')}"
        )
        logger.info(
            f"│  • rsi_exit_overbought: {params.get('rsi_exit_overbought', 'N/A')}"
        )
        logger.info(f"│  • rsi_exit_oversold: {params.get('rsi_exit_oversold', 'N/A')}")
        logger.info(
            f"│  • momentum_fade_threshold: {params.get('momentum_fade_threshold', 'N/A')}"
        )

        logger.info(f"├─ 🕐 EXTENDED DURATION MANAGEMENT")
        logger.info(
            f"│  • extended_2hr_target: {params.get('extended_2hr_target', 'N/A')}"
        )
        logger.info(
            f"│  • extended_3hr_target: {params.get('extended_3hr_target', 'N/A')}"
        )
        logger.info(
            f"│  • extended_5hr_target: {params.get('extended_5hr_target', 'N/A')}"
        )

        logger.info(f"└─ 🛡️ MARKET PROTECTION")
        logger.info(
            f"   • friday_close_min_profit: {params.get('friday_close_min_profit', 'N/A')}"
        )
        logger.info(
            f"   • overnight_min_profit: {params.get('overnight_min_profit', 'N/A')}"
        )
        logger.info(
            f"   • low_liquidity_min_profit: {params.get('low_liquidity_min_profit', 'N/A')}"
        )

        logger.info(f"📊 [OPTUNA] Total parameters optimized: {len(params)}")

    def pretrain_with_historical_data(self, pair: str, strategy_instance) -> bool:
        """Stub for compatibility - real Optuna doesn't need pre-training"""
        return False


class PlotConfig:
    def __init__(self):
        self.config = {
            "main_plot": {
                # Try direct column names first to test
                "bollinger_upperband": {"color": "rgba(4,137,122,0.7)"},
                "kc_upperband": {"color": "rgba(4,146,250,0.7)"},
                "kc_middleband": {"color": "rgba(4,146,250,0.7)"},
                "kc_lowerband": {"color": "rgba(4,146,250,0.7)"},
                "bollinger_lowerband": {
                    "color": "rgba(4,137,122,0.7)",
                    "fill_to": "bollinger_upperband",
                    "fill_color": "rgba(4,137,122,0.07)",
                },
                "ema9": {"color": "purple"},
                "ema20": {"color": "yellow"},
                "ema50": {"color": "red"},
                "ema200": {"color": "white"},
                "trend_1h_1h": {"color": "orange"},
            },
            "subplots": {
                "RSI": {"rsi": {"color": "green"}},
                "ATR": {"atr": {"color": "firebrick"}},
                "Signal Strength": {"signal_strength": {"color": "blue"}},
            },
        }

    def add_total_divergences_in_config(self, dataframe):
        # Test if columns exist before adding them
        if "total_bullish_divergences" in dataframe.columns:
            self.config["main_plot"]["total_bullish_divergences"] = {
                "plotly": {
                    "mode": "markers",
                    "marker": {"symbol": "diamond", "size": 11, "color": "green"},
                }
            }

        if "total_bearish_divergences" in dataframe.columns:
            self.config["main_plot"]["total_bearish_divergences"] = {
                "plotly": {
                    "mode": "markers",
                    "marker": {"symbol": "diamond", "size": 11, "color": "crimson"},
                }
            }

        return self


class AlexBandSniperV58COptuna(
    IStrategy
):  # CHANGED: Klassenname von AlexBandSniperV51C zu AlexBandSniperV51COptuna
    """
    Alex BandSniper on 15m Timeframe - OPTIMIZED VERSION WITH OPTUNA INTEGRATION
    Version 58C-Optuna - Claude optimized Entry & Exit with Optuna Management  # CHANGED: Versionshinweis angepasst
    Key improvements:
    - Dynamic Trailing & New Custom Exits
    - Integrated OPTUNA Fully
    - Opimizing Daily Optimizing
    - Enhanced Parameters for Optuna
    - Optuna integration
    - Fixed ROI and Trailing adjusted Custom Exits
    - Fixed Entry Signals
    - Included 1h Informative Timeframe
    - Multi-timeframe analysis (1h trend confirmation)
    - Enhanced signal filtering with minimum divergence counts
    - Volume and volatility filters
    - Adaptive position sizing based on signal strength
    - Improved risk management
    - ADDED: Optuna-based parameter optimization per coin  # ADDED: Neuer Kommentar
    - ADDED: Historical pre-training for better startup performance  # ADDED: Neuer Kommentar
    - ADDED: Dynamic parameter adjustment based on performance  # ADDED: Neuer Kommentar
    """

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v58C-optuna"  # CHANGED: Version von "v51C-optimized" zu "v51C-optuna"

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.15, -0.03, decimals=2, name="stoploss")]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(3, 8, name="max_open_trades"),
            ]

        def trailing_space() -> List[Dimension]:
            return [
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.02, 0.3, decimals=2, name="trailing_stop_positive"),
                SKDecimal(
                    0.03, 0.1, decimals=2, name="trailing_stop_positive_offset_p1"
                ),
                Categorical([True, False], name="trailing_only_offset_is_reached"),
            ]

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 100  # Disables ROI completely - let custom_exit handle everything
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.20
    can_short = True
    use_custom_stoploss = False
    leverage_value = 10.0  # Reduced leverage for better risk management

    # trailing_stop = False
    # trailing_stop_positive = 0.40        # Only trail after 40% profit (very high)
    # trailing_stop_positive_offset = 0.45 # Start trailing at 45% profit
    # trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = "15m"
    timeframe_minutes = timeframe_to_minutes(timeframe)

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "exit_pricing" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_exits = True
    # In your hyperopt parameters, consider these more permissive defaults:
    min_divergence_count = IntParameter(
        1, 3, default=1, space="buy", optimize=True, load=True
    )  # Reduced from 2-5
    min_signal_strength = IntParameter(
        1, 5, default=1, space="buy", optimize=True, load=True
    )  # Reduced from 3-10
    volume_threshold = DecimalParameter(
        1.0, 1.5, default=1.0, decimals=1, space="buy", optimize=True, load=True
    )  # Reduced from 1.1-2.5

    # Make ADX less restrictive
    adx_threshold = IntParameter(
        15, 30, default=15, space="buy", optimize=True, load=True
    )  # Reduced from 25-45

    # Market Condition Filters
    rsi_overbought = DecimalParameter(
        65.0, 85.0, default=80.0, decimals=1, space="buy", optimize=True, load=True
    )
    rsi_oversold = DecimalParameter(
        15.0, 35.0, default=15.0, decimals=1, space="buy", optimize=True, load=True
    )

    # Volatility Filters
    max_volatility = DecimalParameter(
        0.015, 0.035, default=0.025, decimals=3, space="buy", optimize=True, load=True
    )
    min_volatility = DecimalParameter(
        0.003, 0.008, default=0.005, decimals=3, space="buy", optimize=True, load=True
    )

    # Exit Parameters
    rsi_exit_overbought = DecimalParameter(
        70.0, 90.0, default=80.0, decimals=1, space="sell", optimize=True, load=True
    )
    rsi_exit_oversold = DecimalParameter(
        10.0, 30.0, default=20.0, decimals=1, space="sell", optimize=True, load=True
    )
    adx_exit_threshold = IntParameter(
        15, 30, default=20, space="sell", optimize=True, load=True
    )

    # Trend Confirmation Parameters
    trend_strength_threshold = IntParameter(
        20, 40, default=25, space="buy", optimize=True, load=True
    )

    # Technical Parameters
    window = IntParameter(3, 6, default=4, space="buy", optimize=True, load=True)
    index_range = IntParameter(
        20, 50, default=30, space="buy", optimize=True, load=True
    )

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 10

    # Protection parameters
    cooldown_lookback = IntParameter(
        2, 48, default=5, space="protection", optimize=True
    )
    stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(
        default=True, space="protection", optimize=True
    )
    use_cooldown_protection = BooleanParameter(
        default=True, space="protection", optimize=True
    )

    # Enhanced protection parameters
    use_max_drawdown_protection = BooleanParameter(
        default=False, space="protection", optimize=True
    )
    max_drawdown_lookback = IntParameter(
        100, 300, default=200, space="protection", optimize=True
    )
    max_drawdown_trade_limit = IntParameter(
        5, 15, default=10, space="protection", optimize=True
    )
    max_drawdown_stop_duration = IntParameter(
        1, 5, default=1, space="protection", optimize=True
    )
    max_allowed_drawdown = DecimalParameter(
        0.08, 0.25, default=0.15, decimals=2, space="protection", optimize=True
    )

    stoploss_guard_lookback = IntParameter(
        30, 80, default=50, space="protection", optimize=True
    )
    stoploss_guard_trade_limit = IntParameter(
        2, 6, default=3, space="protection", optimize=True
    )
    stoploss_guard_only_per_pair = BooleanParameter(
        default=True, space="protection", optimize=True
    )
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    bb_period = IntParameter(15, 25, default=20, space="buy")
    bb_std = DecimalParameter(1.8, 2.5, default=2.0, space="buy")
    volume_factor = DecimalParameter(1.2, 2.0, default=1.5, space="buy")
    willr_period = IntParameter(10, 20, default=14, space="buy")
    willr_oversold = DecimalParameter(-90, -70, default=-80, space="buy")
    willr_overbought = DecimalParameter(-30, -10, default=-20, space="sell")
    cci_period = IntParameter(15, 25, default=20, space="buy")
    cci_oversold = DecimalParameter(-120, -80, default=-100, space="buy")
    cci_overbought = DecimalParameter(80, 120, default=100, space="sell")
    atr_multiplier = DecimalParameter(2.0, 4.0, default=3.0, space="sell")
    # Optional order type mapping.
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "limit",
        "stoploss_on_exchange": True,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    plot_config = None

    def __init__(self, config: dict):
        super().__init__(config)

        logger.info(
            f"🚀 [STRATEGY] Initializing {self.__class__.__name__}"
        )  # AUTO-UPDATED: Klassenname wird automatisch angepasst

        # Initialize simple Optuna manager replacement
        try:
            self.optuna_manager = RealOptunaManager(
                "AlexBandSniperV58CO"
            )  # CHANGED: Strategy name angepasst
            logger.info(f"✅ [OPTUNA] Successfully initialized RealOptunaManager")
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to initialize RealOptunaManager: {e}")
            self.optuna_manager = None

        # Per-coin optimized parameters
        self.coin_params: Dict[str, Dict] = (
            {}
        )  # ADDED: Dictionary für coin-spezifische Parameter

        # Track performance for optimization
        self.coin_performance: Dict[str, float] = (
            {}
        )  # ADDED: Performance-Tracking pro Coin

        # Optimization settings - CHANGED: Enable proactive optimization
        self.enable_dynamic_optimization = (
            True  # ADDED: Dynamische Optimierung aktiviert
        )
        self.optimization_min_trades = (
            0  # CHANGED: No minimum trades required for startup optimization
        )

        logger.info(
            f"⚙️  [STRATEGY] Dynamic optimization: {'ENABLED' if self.enable_dynamic_optimization else 'DISABLED'}"
        )
        logger.info(
            f"⚙️  [STRATEGY] Minimum trades for optimization: {self.optimization_min_trades}"
        )
        logger.info(f"✅ [STRATEGY] {self.__class__.__name__} initialization completed")

    # ...existing code...

    def _allow_dynamic_optimization(self) -> bool:
        """Disable live-style optimization loops during backtest/hyperopt."""
        if not self.enable_dynamic_optimization:
            return False

        runmode = getattr(getattr(self, "dp", None), "runmode", None)
        if runmode and getattr(runmode, "value", None) in ["backtest", "hyperopt"]:
            return False

        return True

    def get_coin_params(self, pair: str) -> Dict[str, Any]:
        """Get optimized parameters for specific coin using real Optuna"""
        if pair not in self.coin_params:
            logger.debug(f"🔍 [OPTUNA] Loading parameters for new pair: {pair}")

            # Real Optuna optimization - no pre-training needed
            if self._allow_dynamic_optimization() and self.optuna_manager:
                logger.info(
                    f"🚀 [OPTUNA] Real Optuna will optimize {pair} based on actual trade results"
                )

            if self.optuna_manager:
                best_params = self.optuna_manager.get_best_params(pair)

                if best_params:
                    self.coin_params[pair] = best_params
                    logger.info(
                        f"✅ [OPTUNA] Loaded optimized parameters for {pair}: {len(best_params)} params"
                    )
                    logger.debug(
                        f"📊 [OPTUNA] Parameters from study with {len(self.optuna_manager.studies.get(pair, {}).trials) if pair in self.optuna_manager.studies else 0} trials"
                    )
                else:
                    self.coin_params[pair] = self.get_default_params()
                    logger.info(
                        f"🔧 [OPTUNA] Using default parameters for {pair} (no study found - will optimize after trades)"
                    )
            else:
                self.coin_params[pair] = self.get_default_params()
                logger.warning(
                    f"⚠️  [OPTUNA] RealOptunaManager not available, using default parameters for {pair}"
                )

        # ENSURE ALL PARAMETERS ARE PRESENT
        default_params = self.get_default_params()
        current_params = self.coin_params.get(pair, {})

        # Merge defaults with current params (current params override defaults)
        complete_params = {**default_params, **current_params}
        self.coin_params[pair] = complete_params

        logger.debug(f"♻️  [OPTUNA] Using complete parameter set for {pair}")
        return self.coin_params[pair]

    def get_default_params(self) -> Dict[str, Any]:
        """Get enhanced default parameters including exit optimization"""
        return {
            # === EXISTING ENTRY PARAMETERS ===
            "min_divergence_count": self.min_divergence_count.value,
            "min_signal_strength": self.min_signal_strength.value,
            "volume_threshold": self.volume_threshold.value,
            "adx_threshold": self.adx_threshold.value,
            "rsi_overbought": self.rsi_overbought.value,
            "rsi_oversold": self.rsi_oversold.value,
            "max_volatility": self.max_volatility.value,
            "min_volatility": self.min_volatility.value,
            # === EXISTING ENTRY TIMING PARAMETERS ===
            "rsi_recovery_periods": 3,
            "rsi_long_upper_tight": 65.0,
            "rsi_short_lower_tight": 35.0,
            "volume_ratio_min": 1.1,
            "volume_trend_multiplier": 1.5,
            "volume_momentum_multiplier": 1.2,
            "bb_percent_long_max": 0.25,
            "bb_percent_short_min": 0.75,
            "price_position_long_max": 0.3,
            "price_position_short_min": 0.7,
            "divergence_freshness_periods": 5,
            "min_strong_divergence_count": 2,
            "min_oversold_conditions": 3,
            "min_overbought_conditions": 3,
            "min_reversal_signals": 2,
            "min_trend_filters": 1,
            "chop_threshold": 61.8,
            "adx_trending_min": 25,
            "volume_breakout_multiplier": 2.5,
            "volume_breakout_rsi_min": 45.0,
            "volume_breakout_rsi_max": 60.0,
            "bb_oversold_threshold": 0.15,
            "bb_overbought_threshold": 0.85,
            "mean_reversion_rsi_oversold": 28.0,
            "mean_reversion_rsi_overbought": 72.0,
            "momentum_ema_periods": 9,
            "momentum_strength_min": 0.025,
            "momentum_pullback_max": 0.35,
            "momentum_pullback_min": 0.65,
            # === NEW EXIT PARAMETERS ===
            # Global Exit Settings
            "emergency_profit_limit": 0.25,
            "session_multiplier_overlap": 1.3,
            "session_multiplier_major": 1.15,
            "session_multiplier_quiet": 0.85,
            "volatility_sensitivity": 40,
            "high_signal_multiplier": 1.2,
            "low_signal_multiplier": 0.8,
            # Mean Reversion (MR1) Exit Parameters
            "mr1_max_hold_minutes": 35,
            "mr1_rsi_exit_long": 68.0,
            "mr1_rsi_exit_short": 32.0,
            "mr1_quick_profit_target": 0.025,
            "mr1_timeout_min_profit": 0.008,
            "mr1_bb_exit_long": 0.7,
            "mr1_bb_exit_short": 0.3,
            # Momentum Continuation (MC1) Exit Parameters
            "mc1_profit_target": 0.04,
            "mc1_momentum_break_threshold": 0.015,
            "mc1_max_hold_minutes": 45,
            "mc1_adx_exit_threshold": 22,
            "mc1_timeout_min_profit": 0.01,
            # Volume Breakout (VB1) Exit Parameters
            "vb1_volume_fade_threshold": 1.15,
            "vb1_profit_target": 0.03,
            "vb1_quick_profit": 0.02,
            "vb1_min_volatility": 0.012,
            # Reversal (RSV1) Exit Parameters
            "rsv1_rsi_recovery_threshold": 58.0,
            "rsv1_profit_target": 0.035,
            "rsv1_max_hold_minutes": 60,
            "rsv1_min_timeout_profit": 0.01,
            # Trend Following Exit Parameters
            "trend_profit_target": 0.045,
            "trend_ema_break_periods": 3,
            # Universal Exit Parameters
            "medium_term_base_target": 0.05,
            "reversal_min_profit_long": 0.008,
            "reversal_min_profit_short": 0.008,
            "rsi_exit_overbought": 78,
            "rsi_exit_oversold": 22,
            "rsi_extreme_min_profit": 0.015,
            "momentum_fade_min_profit": 0.012,
            "momentum_fade_threshold": 0.02,
            # Extended Duration Management
            "extended_2hr_target": 0.025,
            "extended_3hr_target": 0.015,
            "extended_5hr_target": 0.008,
            # Market Protection
            "friday_close_min_profit": 0.012,
            "overnight_min_profit": 0.008,
            "low_liquidity_min_profit": 0.01,
        }

    def create_objective_function(self, pair: str):
        """Create enhanced objective function with exit parameter optimization"""

        def objective(trial):
            params = {
                # === EXISTING ENTRY PARAMETERS ===
                "min_divergence_count": trial.suggest_int("min_divergence_count", 1, 3),
                "min_signal_strength": trial.suggest_int("min_signal_strength", 1, 5),
                "volume_threshold": trial.suggest_float("volume_threshold", 1.0, 1.5),
                "adx_threshold": trial.suggest_int("adx_threshold", 15, 30),
                "rsi_overbought": trial.suggest_float("rsi_overbought", 65.0, 85.0),
                "rsi_oversold": trial.suggest_float("rsi_oversold", 15.0, 35.0),
                "max_volatility": trial.suggest_float("max_volatility", 0.015, 0.035),
                "min_volatility": trial.suggest_float("min_volatility", 0.003, 0.008),
                # === EXISTING ENTRY TIMING PARAMETERS ===
                "rsi_recovery_periods": trial.suggest_int("rsi_recovery_periods", 2, 5),
                "rsi_long_upper_tight": trial.suggest_float(
                    "rsi_long_upper_tight", 60.0, 70.0
                ),
                "rsi_short_lower_tight": trial.suggest_float(
                    "rsi_short_lower_tight", 30.0, 40.0
                ),
                "volume_ratio_min": trial.suggest_float("volume_ratio_min", 1.0, 1.4),
                "volume_trend_multiplier": trial.suggest_float(
                    "volume_trend_multiplier", 1.2, 2.0
                ),
                "volume_momentum_multiplier": trial.suggest_float(
                    "volume_momentum_multiplier", 1.1, 1.5
                ),
                "bb_percent_long_max": trial.suggest_float(
                    "bb_percent_long_max", 0.15, 0.35
                ),
                "bb_percent_short_min": trial.suggest_float(
                    "bb_percent_short_min", 0.65, 0.85
                ),
                "price_position_long_max": trial.suggest_float(
                    "price_position_long_max", 0.2, 0.4
                ),
                "price_position_short_min": trial.suggest_float(
                    "price_position_short_min", 0.6, 0.8
                ),
                "divergence_freshness_periods": trial.suggest_int(
                    "divergence_freshness_periods", 3, 8
                ),
                "min_strong_divergence_count": trial.suggest_int(
                    "min_strong_divergence_count", 1, 3
                ),
                "min_oversold_conditions": trial.suggest_int(
                    "min_oversold_conditions", 2, 4
                ),
                "min_overbought_conditions": trial.suggest_int(
                    "min_overbought_conditions", 2, 4
                ),
                "min_reversal_signals": trial.suggest_int("min_reversal_signals", 1, 3),
                "min_trend_filters": trial.suggest_int("min_trend_filters", 1, 2),
                "chop_threshold": trial.suggest_float("chop_threshold", 55.0, 65.0),
                "adx_trending_min": trial.suggest_int("adx_trending_min", 20, 30),
                "volume_breakout_multiplier": trial.suggest_float(
                    "volume_breakout_multiplier", 2.0, 4.0
                ),
                "volume_breakout_rsi_min": trial.suggest_float(
                    "volume_breakout_rsi_min", 35.0, 55.0
                ),
                "volume_breakout_rsi_max": trial.suggest_float(
                    "volume_breakout_rsi_max", 45.0, 65.0
                ),
                "bb_oversold_threshold": trial.suggest_float(
                    "bb_oversold_threshold", 0.05, 0.25
                ),
                "bb_overbought_threshold": trial.suggest_float(
                    "bb_overbought_threshold", 0.75, 0.95
                ),
                "mean_reversion_rsi_oversold": trial.suggest_float(
                    "mean_reversion_rsi_oversold", 20.0, 35.0
                ),
                "mean_reversion_rsi_overbought": trial.suggest_float(
                    "mean_reversion_rsi_overbought", 65.0, 80.0
                ),
                "momentum_ema_periods": trial.suggest_int(
                    "momentum_ema_periods", 8, 21
                ),
                "momentum_strength_min": trial.suggest_float(
                    "momentum_strength_min", 0.015, 0.035
                ),
                "momentum_pullback_max": trial.suggest_float(
                    "momentum_pullback_max", 0.25, 0.45
                ),
                "momentum_pullback_min": trial.suggest_float(
                    "momentum_pullback_min", 0.55, 0.75
                ),
                # === NEW EXIT PARAMETER OPTIMIZATION ===
                # Global Exit Settings
                "session_multiplier_overlap": trial.suggest_float(
                    "session_multiplier_overlap", 1.1, 1.5
                ),
                "session_multiplier_major": trial.suggest_float(
                    "session_multiplier_major", 1.0, 1.3
                ),
                "volatility_sensitivity": trial.suggest_int(
                    "volatility_sensitivity", 25, 60
                ),
                "high_signal_multiplier": trial.suggest_float(
                    "high_signal_multiplier", 1.0, 1.4
                ),
                "low_signal_multiplier": trial.suggest_float(
                    "low_signal_multiplier", 0.6, 1.0
                ),
                # Mean Reversion (MR1) Exit Optimization
                "mr1_max_hold_minutes": trial.suggest_int(
                    "mr1_max_hold_minutes", 25, 50
                ),
                "mr1_rsi_exit_long": trial.suggest_float(
                    "mr1_rsi_exit_long", 62.0, 75.0
                ),
                "mr1_rsi_exit_short": trial.suggest_float(
                    "mr1_rsi_exit_short", 25.0, 38.0
                ),
                "mr1_quick_profit_target": trial.suggest_float(
                    "mr1_quick_profit_target", 0.015, 0.035
                ),
                "mr1_timeout_min_profit": trial.suggest_float(
                    "mr1_timeout_min_profit", 0.005, 0.015
                ),
                "mr1_bb_exit_long": trial.suggest_float("mr1_bb_exit_long", 0.6, 0.8),
                "mr1_bb_exit_short": trial.suggest_float("mr1_bb_exit_short", 0.2, 0.4),
                # Momentum Continuation (MC1) Exit Optimization
                "mc1_profit_target": trial.suggest_float(
                    "mc1_profit_target", 0.025, 0.055
                ),
                "mc1_momentum_break_threshold": trial.suggest_float(
                    "mc1_momentum_break_threshold", 0.01, 0.025
                ),
                "mc1_max_hold_minutes": trial.suggest_int(
                    "mc1_max_hold_minutes", 30, 60
                ),
                "mc1_adx_exit_threshold": trial.suggest_int(
                    "mc1_adx_exit_threshold", 18, 28
                ),
                "mc1_timeout_min_profit": trial.suggest_float(
                    "mc1_timeout_min_profit", 0.005, 0.02
                ),
                # Volume Breakout (VB1) Exit Optimization
                "vb1_volume_fade_threshold": trial.suggest_float(
                    "vb1_volume_fade_threshold", 1.0, 1.3
                ),
                "vb1_profit_target": trial.suggest_float(
                    "vb1_profit_target", 0.02, 0.045
                ),
                "vb1_quick_profit": trial.suggest_float(
                    "vb1_quick_profit", 0.012, 0.03
                ),
                "vb1_min_volatility": trial.suggest_float(
                    "vb1_min_volatility", 0.008, 0.018
                ),
                # Reversal (RSV1) Exit Optimization
                "rsv1_rsi_recovery_threshold": trial.suggest_float(
                    "rsv1_rsi_recovery_threshold", 52.0, 65.0
                ),
                "rsv1_profit_target": trial.suggest_float(
                    "rsv1_profit_target", 0.025, 0.05
                ),
                "rsv1_max_hold_minutes": trial.suggest_int(
                    "rsv1_max_hold_minutes", 40, 80
                ),
                "rsv1_min_timeout_profit": trial.suggest_float(
                    "rsv1_min_timeout_profit", 0.005, 0.02
                ),
                # Trend Following Exit Optimization
                "trend_profit_target": trial.suggest_float(
                    "trend_profit_target", 0.03, 0.06
                ),
                "trend_ema_break_periods": trial.suggest_int(
                    "trend_ema_break_periods", 2, 5
                ),
                # Universal Exit Optimization
                "medium_term_base_target": trial.suggest_float(
                    "medium_term_base_target", 0.035, 0.065
                ),
                "reversal_min_profit_long": trial.suggest_float(
                    "reversal_min_profit_long", 0.005, 0.015
                ),
                "reversal_min_profit_short": trial.suggest_float(
                    "reversal_min_profit_short", 0.005, 0.015
                ),
                "rsi_exit_overbought": trial.suggest_float(
                    "rsi_exit_overbought", 72, 85
                ),
                "rsi_exit_oversold": trial.suggest_float("rsi_exit_oversold", 15, 28),
                "rsi_extreme_min_profit": trial.suggest_float(
                    "rsi_extreme_min_profit", 0.01, 0.025
                ),
                "momentum_fade_min_profit": trial.suggest_float(
                    "momentum_fade_min_profit", 0.008, 0.02
                ),
                "momentum_fade_threshold": trial.suggest_float(
                    "momentum_fade_threshold", 0.015, 0.03
                ),
                # Extended Duration Management
                "extended_2hr_target": trial.suggest_float(
                    "extended_2hr_target", 0.015, 0.035
                ),
                "extended_3hr_target": trial.suggest_float(
                    "extended_3hr_target", 0.008, 0.025
                ),
                "extended_5hr_target": trial.suggest_float(
                    "extended_5hr_target", 0.005, 0.015
                ),
                # Market Protection Optimization
                "friday_close_min_profit": trial.suggest_float(
                    "friday_close_min_profit", 0.008, 0.02
                ),
                "overnight_min_profit": trial.suggest_float(
                    "overnight_min_profit", 0.005, 0.015
                ),
                "low_liquidity_min_profit": trial.suggest_float(
                    "low_liquidity_min_profit", 0.006, 0.018
                ),
            }

            # Return performance for this coin
            recent_trades = self.get_recent_trade_performance(pair)
            if len(recent_trades) >= 3:  # Need at least 3 trades
                return sum(recent_trades) / len(recent_trades)
            else:
                return 0.0  # No optimization until we have trade data

        return objective

    def get_recent_trade_performance(self, pair: str) -> List[float]:
        """Get recent trade performance for optimization"""
        try:
            from freqtrade.persistence import Trade

            trades = Trade.get_trades_proxy(pair=pair)

            if not trades:
                return []

            # Get last 20 trades for this pair
            recent_trades = trades[-20:] if len(trades) >= 20 else trades

            # Calculate profit ratios
            performance = []
            for trade in recent_trades:
                if trade.close_date:  # Only closed trades
                    # FIX: Use the close_rate for calc_profit_ratio
                    performance.append(trade.calc_profit_ratio(trade.close_rate))

            logger.debug(
                f"📊 [OPTUNA] Found {len(performance)} completed trades for {pair}"
            )
            return performance

        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get trade performance for {pair}: {e}")
            return []

    def daily_optimization_check(self):
        """Enhanced optimization check with smarter scheduling"""
        try:
            if not self.optuna_manager or not self._allow_dynamic_optimization():
                return

            current_time = time.time()
            all_pairs = list(self.coin_params.keys())

            if not all_pairs:
                logger.info("No pairs available for optimization yet")
                return

            optimized_count = 0

            for pair in all_pairs:
                try:
                    if self.should_retrain_pair(pair, current_time):
                        logger.info(f"🔄 Optimizing {pair}")
                        self.optuna_manager.optimize_coin(
                            pair, self.create_objective_function(pair), n_trials=15
                        )

                        # Track last optimization time
                        setattr(
                            self,
                            f'last_optimization_{pair.replace("/", "_").replace(":", "_")}',
                            current_time,
                        )
                        optimized_count += 1

                except Exception as e:
                    logger.error(f"❌ Optimization failed for {pair}: {e}")

            logger.info(
                f"✅ Optimization check completed: {optimized_count}/{len(all_pairs)} pairs optimized"
            )

        except Exception as e:
            logger.error(f"❌ Daily optimization check failed: {e}")

    def should_retrain_pair(self, pair: str, current_time: float) -> bool:
        """Smarter retraining logic based on trading activity"""
        last_opt_time = getattr(
            self, f'last_optimization_{pair.replace("/", "_").replace(":", "_")}', 0
        )
        trades_since_last = self.get_trades_since_optimization(pair, last_opt_time)

        # Different schedules based on activity
        if trades_since_last >= 10:  # High activity pairs
            return current_time - last_opt_time > 43200  # 12 hours
        elif trades_since_last >= 5:  # Medium activity
            return current_time - last_opt_time > 86400  # 24 hours
        elif trades_since_last >= 2:  # Low activity
            return current_time - last_opt_time > 172800  # 48 hours
        else:
            return current_time - last_opt_time > 604800  # 7 days if no trades

    def get_trades_since_optimization(self, pair: str, last_opt_time: float) -> int:
        """Count trades since last optimization"""
        try:
            from freqtrade.persistence import Trade
            from datetime import datetime

            last_opt_datetime = (
                datetime.fromtimestamp(last_opt_time)
                if last_opt_time > 0
                else datetime.min
            )
            trades = Trade.get_trades_proxy(pair=pair)

            if not trades:
                return 0

            recent_trades = [t for t in trades if t.open_date_utc > last_opt_datetime]
            return len(recent_trades)

        except Exception as e:
            logger.debug(f"Failed to count recent trades for {pair}: {e}")
            return 0

    def informative_pairs(self):
        """Define additional timeframes to download"""
        pairs = self.dp.current_whitelist()
        return [(pair, "1h") for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced indicator population with multi-timeframe analysis and Optuna optimization
        ENHANCED: Now includes proactive Optuna optimization with parameterized indicators
        """

        pair = metadata["pair"]

        logger.debug(f"📊 [INDICATORS] Processing {pair} with {len(dataframe)} candles")

        # ===== PROACTIVE OPTUNA OPTIMIZATION FIRST =====
        # ENHANCED: Get optimized parameters EARLY (triggers proactive optimization)
        try:
            coin_params = self.get_coin_params(pair)  # Get coin-specific parameters
            logger.debug(
                f"📈 [OPTUNA] Applied parameters for {pair}: min_div={coin_params.get('min_divergence_count')}, min_signal={coin_params.get('min_signal_strength')}"
            )
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get coin parameters for {pair}: {e}")
            coin_params = self.get_default_params()

        # === MULTI-TIMEFRAME ANALYSIS ===
        # Get 1h timeframe for trend confirmation with improved error handling
        try:
            # Check if we're in backtesting mode or if pair supports 1h data
            if hasattr(self.dp, "runmode") and self.dp.runmode.value in [
                "backtest",
                "hyperopt",
            ]:
                # In backtesting, try to get 1h data but don't fail if unavailable
                informative_1h = self.dp.get_pair_dataframe(
                    pair=metadata["pair"], timeframe="1h"
                )
            else:
                # In live/dry run, be more cautious about data availability
                try:
                    informative_1h = self.dp.get_pair_dataframe(
                        pair=metadata["pair"], timeframe="1h"
                    )
                except Exception:
                    informative_1h = None

            # Enhanced data validation
            if (
                informative_1h is not None
                and len(informative_1h) > 50  # Reduced minimum requirement
                and not informative_1h.empty
                and "close" in informative_1h.columns
            ):

                try:
                    # 1h Trend indicators with additional error checking
                    informative_1h["ema50_1h"] = ta.EMA(informative_1h, timeperiod=50)
                    informative_1h["ema200_1h"] = ta.EMA(informative_1h, timeperiod=200)
                    informative_1h["trend_1h"] = ta.EMA(informative_1h, timeperiod=21)
                    informative_1h["trend_strength_1h"] = ta.ADX(informative_1h)
                    informative_1h["rsi_1h"] = ta.RSI(informative_1h)

                    # Fill NaN values before merging
                    informative_1h = informative_1h.bfill().ffill()

                    # Safe merge with additional error handling
                    dataframe = merge_informative_pair(
                        dataframe, informative_1h, self.timeframe, "1h", ffill=True
                    )

                except Exception as merge_error:
                    logger.warning(
                        f"Failed to merge 1h data for {metadata['pair']}: {merge_error}"
                    )
                    self._add_dummy_1h_columns(dataframe)
            else:
                logger.info(
                    f"Using fallback 1h indicators for {metadata['pair']} (insufficient data)"
                )
                self._add_dummy_1h_columns(dataframe)

        except Exception as e:
            logger.warning(f"Error accessing 1h data for {metadata['pair']}: {e}")
            self._add_dummy_1h_columns(dataframe)

        # === 15M TIMEFRAME INDICATORS ===
        informative = dataframe.copy()

        # === VOLUME ANALYSIS ===
        try:
            informative["volume_sma"] = ta.SMA(informative["volume"], timeperiod=20)
            informative["volume_ratio"] = (
                informative["volume"] / informative["volume_sma"]
            )
            informative["volume_ratio"] = informative["volume_ratio"].fillna(1.0)
        except:
            informative["volume_sma"] = informative["volume"]
            informative["volume_ratio"] = 1.0

        # === VOLATILITY ANALYSIS ===
        try:
            informative["atr"] = ta.ATR(informative, timeperiod=14)
            informative["volatility"] = informative["atr"] / informative["close"]
            informative["volatility"] = informative["volatility"].fillna(0.01)
        except:
            informative["atr"] = informative["close"] * 0.02
            informative["volatility"] = 0.01

        # === MOMENTUM INDICATORS WITH PARAMETERIZED VALUES ===
        try:
            # Parameterized indicators from first file
            informative["rsi"] = ta.RSI(informative, timeperiod=self.rsi_period.value)
            informative["willr"] = ta.WILLR(
                informative, timeperiod=self.willr_period.value
            )
            informative["cci"] = ta.CCI(informative, timeperiod=self.cci_period.value)
            informative["mom"] = ta.MOM(informative, timeperiod=10)

            # MACD with all components (parameterized)
            macd = ta.MACD(informative)
            informative["macd"] = macd["macd"]
            informative["macdsignal"] = macd["macdsignal"]
            informative["macdhist"] = macd["macdhist"]

            # Additional momentum indicators
            informative["stoch"] = ta.STOCH(informative)["slowk"]
            informative["roc"] = ta.ROC(informative)
            informative["uo"] = ta.ULTOSC(informative)
            informative["ao"] = qtpylib.awesome_oscillator(informative)
            informative["cmf"] = chaikin_money_flow(informative, 20)
            informative["obv"] = ta.OBV(informative)
            informative["mfi"] = ta.MFI(informative)
            informative["adx"] = ta.ADX(informative)

            # Fill NaN values for all indicators
            indicator_columns = [
                "rsi",
                "stoch",
                "roc",
                "uo",
                "ao",
                "macd",
                "macdsignal",
                "macdhist",
                "cci",
                "cmf",
                "obv",
                "mfi",
                "adx",
                "willr",
                "mom",
            ]
            for col in indicator_columns:
                if col in informative.columns:
                    informative[col] = (
                        informative[col]
                        .bfill()
                        .fillna(50 if col in ["rsi", "mfi"] else 0)
                    )

        except Exception as e:
            logger.warning(f"Error calculating momentum indicators: {e}")
            # Provide fallback values
            informative["rsi"] = 50
            informative["stoch"] = 50
            informative["roc"] = 0
            informative["uo"] = 50
            informative["ao"] = 0
            informative["macd"] = 0
            informative["macdsignal"] = 0
            informative["macdhist"] = 0
            informative["cci"] = 0
            informative["cmf"] = 0
            informative["obv"] = informative["volume"].cumsum()
            informative["mfi"] = 50
            informative["adx"] = 25
            informative["willr"] = -50
            informative["mom"] = 0

        # === KELTNER CHANNEL ===
        try:
            keltner = emaKeltner(informative)
            informative["kc_upperband"] = keltner["upper"]
            informative["kc_middleband"] = keltner["mid"]
            informative["kc_lowerband"] = keltner["lower"]
        except:
            informative["kc_upperband"] = informative["close"] * 1.02
            informative["kc_middleband"] = informative["close"]
            informative["kc_lowerband"] = informative["close"] * 0.98

        # === BOLLINGER BANDS WITH PARAMETERIZED VALUES ===
        try:
            # Parameterized Bollinger Bands from first file
            bollinger = qtpylib.bollinger_bands(
                informative["close"],
                window=self.bb_period.value,
                stds=self.bb_std.value,
            )
            informative["bb_lower"] = bollinger["lower"]
            informative["bb_middle"] = bollinger["mid"]
            informative["bb_upper"] = bollinger["upper"]
            informative["bb_percent"] = (
                informative["close"] - informative["bb_lower"]
            ) / (informative["bb_upper"] - informative["bb_lower"])

            # Keep the original naming for compatibility
            informative["bollinger_upperband"] = bollinger["upper"]
            informative["bollinger_lowerband"] = bollinger["lower"]
        except:
            informative["bb_lower"] = informative["close"] * 0.98
            informative["bb_middle"] = informative["close"]
            informative["bb_upper"] = informative["close"] * 1.02
            informative["bb_percent"] = 0.5
            informative["bollinger_upperband"] = informative["close"] * 1.02
            informative["bollinger_lowerband"] = informative["close"] * 0.98

        # === EMA WITH PARAMETERIZED VALUES ===
        try:
            # Parameterized EMAs from first file
            informative["ema_short"] = ta.EMA(informative, timeperiod=8)
            informative["ema_long"] = ta.EMA(informative, timeperiod=21)

            # Standard EMAs
            informative["ema9"] = ta.EMA(informative, timeperiod=9)
            informative["ema20"] = ta.EMA(informative, timeperiod=20)
            informative["ema50"] = ta.EMA(informative, timeperiod=50)
            informative["ema200"] = ta.EMA(informative, timeperiod=200)

            # Fill NaN values for EMAs
            ema_columns = ["ema_short", "ema_long", "ema9", "ema20", "ema50", "ema200"]
            for col in ema_columns:
                if col in informative.columns:
                    informative[col] = (
                        informative[col].bfill().fillna(informative["close"])
                    )
        except:
            informative["ema_short"] = informative["close"]
            informative["ema_long"] = informative["close"]
            informative["ema9"] = informative["close"]
            informative["ema20"] = informative["close"]
            informative["ema50"] = informative["close"]
            informative["ema200"] = informative["close"]

        # === PRICE POSITION ANALYSIS (FROM FIRST FILE) ===
        try:
            informative["high_20"] = informative["high"].rolling(window=20).max()
            informative["low_20"] = informative["low"].rolling(window=20).min()
            informative["price_position"] = (
                informative["close"] - informative["low_20"]
            ) / (informative["high_20"] - informative["low_20"])
        except:
            informative["high_20"] = informative["high"]
            informative["low_20"] = informative["low"]
            informative["price_position"] = 0.5

        # === PIVOT POINTS ===
        try:
            pivots = pivot_points(informative, self.window.value)
            informative["pivot_lows"] = pivots["pivot_lows"]
            informative["pivot_highs"] = pivots["pivot_highs"]
        except Exception as e:
            logger.warning(f"Error calculating pivot points: {e}")
            informative["pivot_lows"] = np.nan
            informative["pivot_highs"] = np.nan

        # === DIVERGENCE ANALYSIS ===
        try:
            self.initialize_divergences_lists(informative)
            (high_iterator, low_iterator) = self.get_iterators(informative)

            # Add divergences for multiple indicators (expanded list from first file)
            indicators = [
                "rsi",
                "stoch",
                "roc",
                "uo",
                "ao",
                "macd",
                "cci",
                "cmf",
                "obv",
                "mfi",
            ]
            for indicator in indicators:
                try:
                    if indicator in informative.columns:
                        self.add_divergences(
                            informative, indicator, high_iterator, low_iterator
                        )
                except Exception as e:
                    logger.warning(f"Error adding divergences for {indicator}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error in divergence analysis: {e}")
            # Initialize with empty divergence data
            informative["total_bullish_divergences"] = np.nan
            informative["total_bullish_divergences_count"] = 0
            informative["total_bullish_divergences_names"] = ""
            informative["total_bearish_divergences"] = np.nan
            informative["total_bearish_divergences_count"] = 0
            informative["total_bearish_divergences_names"] = ""

        # === SIGNAL STRENGTH CALCULATION ===
        try:
            informative["signal_strength"] = self.calculate_signal_strength(informative)
        except:
            informative["signal_strength"] = 0

        # === MERGE BACK TO DATAFRAME ===
        for col in informative.columns:
            if col not in dataframe.columns:
                dataframe[col] = informative[col]
            else:
                dataframe[col] = informative[col]

        # ===== APPLY COIN-SPECIFIC OPTUNA ADJUSTMENTS =====
        # Apply coin-specific adjustments AFTER all indicators are calculated
        dataframe["coin_min_divergence"] = coin_params.get(
            "min_divergence_count", self.min_divergence_count.value
        )
        dataframe["coin_min_signal"] = coin_params.get(
            "min_signal_strength", self.min_signal_strength.value
        )
        dataframe["coin_volume_threshold"] = coin_params.get(
            "volume_threshold", self.volume_threshold.value
        )
        dataframe["coin_adx_threshold"] = coin_params.get(
            "adx_threshold", self.adx_threshold.value
        )

        # === ADDITIONAL MARKET STRUCTURE ANALYSIS ===
        try:
            dataframe["chop"] = choppiness_index(
                dataframe["high"], dataframe["low"], dataframe["close"], window=14
            )
            dataframe["natr"] = ta.NATR(
                dataframe["high"], dataframe["low"], dataframe["close"], window=14
            )
            dataframe["natr_diff"] = dataframe["natr"] - dataframe["natr"].shift(1)
            dataframe["natr_direction_change"] = (
                dataframe["natr_diff"] * dataframe["natr_diff"].shift(1) < 0
            )
        except:
            dataframe["chop"] = 50
            dataframe["natr"] = 0.02
            dataframe["natr_diff"] = 0
            dataframe["natr_direction_change"] = False

        # === SUPPORT/RESISTANCE LEVELS ===
        try:
            dataframe["swing_high"] = (
                dataframe["high"].rolling(window=50, min_periods=1).max()
            )
            dataframe["swing_low"] = (
                dataframe["low"].rolling(window=50, min_periods=1).min()
            )
            dataframe["distance_to_resistance"] = (
                dataframe["swing_high"] - dataframe["close"]
            ) / dataframe["close"]
            dataframe["distance_to_support"] = (
                dataframe["close"] - dataframe["swing_low"]
            ) / dataframe["close"]
        except:
            dataframe["swing_high"] = dataframe["high"]
            dataframe["swing_low"] = dataframe["low"]
            dataframe["distance_to_resistance"] = 0.02
            dataframe["distance_to_support"] = 0.02

        # === PLOT CONFIGURATION ===
        try:
            self.plot_config = (
                PlotConfig().add_total_divergences_in_config(dataframe).config
            )
        except:
            self.plot_config = None

        logger.debug(f"✅ [INDICATORS] Completed processing for {pair}")
        # Run daily optimization check (only once per day)
        if not hasattr(self, "last_daily_check"):
            self.last_daily_check = 0

        current_time = time.time()
        if (
            current_time - self.last_daily_check > 3600
        ):  # Check every hour instead of daily
            self.daily_optimization_check()
            self.last_daily_check = current_time
        return dataframe

    def _add_dummy_1h_columns(self, dataframe):
        """Add dummy 1h columns when higher timeframe data is unavailable"""
        # Use current 15m data to simulate 1h trend
        try:
            dataframe["ema50_1h_1h"] = ta.EMA(
                dataframe, timeperiod=200
            )  # Use longer period on 15m
            dataframe["ema200_1h_1h"] = ta.EMA(
                dataframe, timeperiod=800
            )  # Use much longer period
            dataframe["trend_1h_1h"] = ta.EMA(
                dataframe, timeperiod=84
            )  # 21 * 4 (4x 15m = 1h)
            dataframe["trend_strength_1h_1h"] = ta.ADX(dataframe)
            dataframe["rsi_1h_1h"] = ta.RSI(
                dataframe, timeperiod=56
            )  # Adjusted for timeframe

            # Fill NaN values
            columns_1h = [
                "ema50_1h_1h",
                "ema200_1h_1h",
                "trend_1h_1h",
                "trend_strength_1h_1h",
                "rsi_1h_1h",
            ]
            for col in columns_1h:
                if col in dataframe.columns:
                    dataframe[col] = (
                        dataframe[col]
                        .bfill()
                        .fillna(
                            dataframe["close"]
                            if "ema" in col or "trend" in col
                            else 25 if "strength" in col else 50
                        )
                    )
        except Exception as e:
            logger.warning(f"Error creating dummy 1h columns: {e}")
            # Absolute fallback
            dataframe["ema50_1h_1h"] = dataframe["close"]
            dataframe["ema200_1h_1h"] = dataframe["close"]
            dataframe["trend_1h_1h"] = dataframe["close"]
            dataframe["trend_strength_1h_1h"] = 25
            dataframe["rsi_1h_1h"] = 50

    def calculate_signal_strength(self, dataframe: DataFrame) -> Series:
        """
        Calculate overall signal strength based on multiple factors
        """
        strength = pd.Series(0, index=dataframe.index)

        # Divergence strength
        strength += dataframe["total_bullish_divergences_count"] * 2
        strength += dataframe["total_bearish_divergences_count"] * 2

        # Volume strength
        volume_strength = np.where(
            dataframe["volume_ratio"] > 1.5,
            2,
            np.where(dataframe["volume_ratio"] > 1.2, 1, 0),
        )
        strength += volume_strength

        # Trend alignment strength
        ema_bullish = (dataframe["ema20"] > dataframe["ema50"]) & (
            dataframe["ema50"] > dataframe["ema200"]
        )
        ema_bearish = (dataframe["ema20"] < dataframe["ema50"]) & (
            dataframe["ema50"] < dataframe["ema200"]
        )
        strength += np.where(ema_bullish | ema_bearish, 1, 0)

        # ADX strength
        strength += np.where(dataframe["adx"] > 30, 1, 0)

        return strength

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        coin_params = self.get_coin_params(metadata["pair"])
        # === PRIMARY DIVERGENCE CONDITIONS ===
        bullish_divergence = (dataframe["total_bullish_divergences"].shift(1) > 0) & (
            dataframe["total_bullish_divergences_count"].shift(1)
            >= coin_params.get("min_divergence_count", self.min_divergence_count.value)
        )

        bearish_divergence = (dataframe["total_bearish_divergences"].shift(1) > 0) & (
            dataframe["total_bearish_divergences_count"].shift(1)
            >= coin_params.get("min_divergence_count", self.min_divergence_count.value)
        )

        # === FILTER CONDITIONS ===
        volatility_ok = (
            dataframe["volatility"].shift(1) >= self.min_volatility.value * 0.5
        ) & (dataframe["volatility"].shift(1) <= self.max_volatility.value * 2.0)

        bands_long = (
            dataframe["low"].shift(1) <= dataframe["kc_lowerband"].shift(1)
        ) | (dataframe["close"].shift(1) <= dataframe["kc_lowerband"].shift(1))

        bands_short = (
            dataframe["high"].shift(1) >= dataframe["kc_upperband"].shift(1)
        ) | (dataframe["close"].shift(1) >= dataframe["kc_upperband"].shift(1))

        rsi_long_ok = (dataframe["rsi"].shift(1) < self.rsi_overbought.value + 5) & (
            dataframe["rsi"].shift(1) > 25
        )

        rsi_short_ok = (dataframe["rsi"].shift(1) > self.rsi_oversold.value - 5) & (
            dataframe["rsi"].shift(1) < 75
        )

        has_volume = dataframe["volume"].shift(1) > 0

        volume_ok = (dataframe["volume"].shift(1) > 0) & (
            dataframe["volume_ratio"].shift(1)
            > coin_params.get("volume_ratio_min", 1.1)
        )
        market_trending = dataframe["chop"].shift(1) < coin_params.get(
            "chop_threshold", 61.8
        )

        # Price position filters for better entry timing
        good_long_position = (
            dataframe["bb_percent"].shift(1)
            < coin_params.get("bb_percent_long_max", 0.25)
        ) | (
            dataframe["price_position"].shift(1)
            < coin_params.get("price_position_long_max", 0.3)
        )

        good_short_position = (
            dataframe["bb_percent"].shift(1)
            > coin_params.get("bb_percent_short_min", 0.75)
        ) | (
            dataframe["price_position"].shift(1)
            > coin_params.get("price_position_short_min", 0.7)
        )

        # === OPTIMIZED RSI TIMING ===
        rsi_long_timing = (
            (dataframe["rsi"].shift(1) > coin_params.get("rsi_long_lower_bound", 25))
            & (
                dataframe["rsi"].shift(1)
                < coin_params.get("rsi_long_upper_tight", 65.0)
            )
            & (
                dataframe["rsi"].shift(1)
                > dataframe["rsi"].shift(coin_params.get("rsi_recovery_periods", 3))
            )
        )

        rsi_short_timing = (
            (dataframe["rsi"].shift(1) > coin_params.get("rsi_short_lower_tight", 35.0))
            & (dataframe["rsi"].shift(1) < coin_params.get("rsi_short_upper_bound", 75))
            & (
                dataframe["rsi"].shift(1)
                < dataframe["rsi"].shift(coin_params.get("rsi_decline_periods", 3))
            )
        )
        fresh_periods = coin_params.get("divergence_freshness_periods", 5)
        strong_bull_divergence = (
            (
                dataframe["total_bullish_divergences_count"].shift(1)
                >= coin_params.get("min_strong_divergence_count", 2)
            )
            & bullish_divergence
            & (
                dataframe["total_bullish_divergences_count"].shift(1)
                > dataframe["total_bullish_divergences_count"].shift(fresh_periods)
            )
        )

        strong_bear_divergence = (
            (
                dataframe["total_bearish_divergences_count"].shift(1)
                >= coin_params.get("min_strong_divergence_count", 2)
            )
            & bearish_divergence
            & (
                dataframe["total_bearish_divergences_count"].shift(1)
                > dataframe["total_bearish_divergences_count"].shift(fresh_periods)
            )
        )
        # === VOLUME BREAKOUT CONDITIONS ===
        volume_breakout_long = (
            (
                dataframe["volume"].shift(1)
                > dataframe["volume_sma"].shift(1)
                * coin_params.get("volume_breakout_multiplier", 2.5)
            )
            & (
                dataframe["rsi"].shift(1)
                > coin_params.get("volume_breakout_rsi_min", 45)
            )
            & (
                dataframe["rsi"].shift(1)
                < coin_params.get("volume_breakout_rsi_max", 60)
            )
            & (dataframe["close"] > dataframe["open"])  # Green candle
            & (dataframe["close"] > dataframe["ema20"].shift(1))
            & (dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25))
            & has_volume
        )

        volume_breakout_short = (
            (
                dataframe["volume"].shift(1)
                > dataframe["volume_sma"].shift(1)
                * coin_params.get("volume_breakout_multiplier", 2.5)
            )
            & (
                dataframe["rsi"].shift(1)
                > coin_params.get("volume_breakout_rsi_min", 45)
            )
            & (
                dataframe["rsi"].shift(1)
                < coin_params.get("volume_breakout_rsi_max", 60)
            )
            & (dataframe["close"] < dataframe["open"])  # Red candle
            & (dataframe["close"] < dataframe["ema20"].shift(1))
            & (dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25))
            & has_volume
        )

        # === MEAN REVERSION CONDITIONS ===
        mean_reversion_long = (
            (
                dataframe["bb_percent"].shift(1)
                < coin_params.get("bb_oversold_threshold", 0.15)
            )
            & (
                dataframe["rsi"].shift(1)
                < coin_params.get("mean_reversion_rsi_oversold", 28)
            )
            & (dataframe["close"] > dataframe["low"].shift(1))  # Not continuing down
            & (
                dataframe["volume_ratio"].shift(1)
                > coin_params.get("volume_ratio_min", 1.1)
            )
            & (dataframe["close"] > dataframe["open"])  # Recovery candle
            & has_volume
        )

        mean_reversion_short = (
            (
                dataframe["bb_percent"].shift(1)
                > coin_params.get("bb_overbought_threshold", 0.85)
            )
            & (
                dataframe["rsi"].shift(1)
                > coin_params.get("mean_reversion_rsi_overbought", 72)
            )
            & (dataframe["close"] < dataframe["high"].shift(1))  # Not continuing up
            & (
                dataframe["volume_ratio"].shift(1)
                > coin_params.get("volume_ratio_min", 1.1)
            )
            & (dataframe["close"] < dataframe["open"])  # Rejection candle
            & has_volume
        )

        # === MOMENTUM CONTINUATION CONDITIONS ===
        momentum_continuation_long = (
            (dataframe["ema9"] > dataframe["ema20"])
            & (dataframe["ema20"] > dataframe["ema50"])
            & (dataframe["close"].shift(1) < dataframe["ema9"].shift(1))  # Pullback
            & (dataframe["close"] > dataframe["ema9"])  # Back above
            & (
                dataframe["price_position"].shift(1)
                > coin_params.get("momentum_pullback_max", 0.35)
            )
            & (dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25))
            & (
                dataframe["volume_ratio"].shift(1)
                > coin_params.get("volume_ratio_min", 1.1)
            )
            & has_volume
        )

        momentum_continuation_short = (
            (dataframe["ema9"] < dataframe["ema20"])
            & (dataframe["ema20"] < dataframe["ema50"])
            & (dataframe["close"].shift(1) > dataframe["ema9"].shift(1))  # Pullback
            & (dataframe["close"] < dataframe["ema9"])  # Back below
            & (
                dataframe["price_position"].shift(1)
                < coin_params.get("momentum_pullback_min", 0.65)
            )
            & (dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25))
            & (
                dataframe["volume_ratio"].shift(1)
                > coin_params.get("volume_ratio_min", 1.1)
            )
            & has_volume
        )
        # === PRIMARY CONDITIONS (HIGHEST PRIORITY) ===
        long_condition_primary = (
            bullish_divergence & volatility_ok & bands_long & rsi_long_ok & has_volume
        )

        short_condition_primary = (
            bearish_divergence & volatility_ok & bands_short & rsi_short_ok & has_volume
        )

        # === YOUR NEW REVERSAL CONDITIONS ===
        # Long reversal conditions
        oversold_conditions = [
            (dataframe["rsi"] < self.rsi_oversold.value),
            (dataframe["willr"] < self.willr_oversold.value),
            (dataframe["cci"] < self.cci_oversold.value),
            (dataframe["close"] <= dataframe["bb_lower"]),
            (dataframe["bb_percent"] < 0.1),
        ]
        long_reversal_signals = [
            (dataframe["volume_ratio"] > self.volume_factor.value),
            (dataframe["close"] > dataframe["open"]),
            (dataframe["macdhist"] > dataframe["macdhist"].shift(1)),
            (dataframe["price_position"] < 0.2),
        ]
        long_trend_filter = [
            (dataframe["ema_short"] > dataframe["ema_short"].shift(3)),
            (dataframe["close"] > dataframe["close"].shift(2)),
        ]

        long_condition_reversal = (
            (sum(oversold_conditions) >= coin_params.get("min_oversold_conditions", 3))
            & (sum(long_reversal_signals) >= coin_params.get("min_reversal_signals", 2))
            & (sum(long_trend_filter) >= coin_params.get("min_trend_filters", 1))
            & (dataframe["volume"] > 0)
        )

        # Short reversal conditions
        overbought_conditions = [
            (dataframe["rsi"] > self.rsi_overbought.value),
            (dataframe["willr"] > self.willr_overbought.value),
            (dataframe["cci"] > self.cci_overbought.value),
            (dataframe["close"] >= dataframe["bb_upper"]),
            (dataframe["bb_percent"] > 0.9),
        ]
        short_reversal_signals = [
            (dataframe["volume_ratio"] > self.volume_factor.value),
            (dataframe["close"] < dataframe["open"]),
            (dataframe["macdhist"] < dataframe["macdhist"].shift(1)),
            (dataframe["price_position"] > 0.8),
        ]
        short_trend_filter = [
            (dataframe["ema_short"] < dataframe["ema_short"].shift(3)),
            (dataframe["close"] < dataframe["close"].shift(2)),
        ]

        short_condition_reversal = (
            (
                sum(overbought_conditions)
                >= coin_params.get("min_overbought_conditions", 3)
            )
            & (
                sum(short_reversal_signals)
                >= coin_params.get("min_reversal_signals", 2)
            )
            & (sum(short_trend_filter) >= coin_params.get("min_trend_filters", 1))
            & (dataframe["volume"] > 0)
        )

        # === TREND BREAKOUT CONDITIONS ===
        long_condition_trend = (
            (dataframe["close"].shift(1) > dataframe["ema20"].shift(1))
            & (dataframe["ema20"].shift(1) > dataframe["ema50"].shift(1))
            & (dataframe["ema50"].shift(1) > dataframe["ema200"].shift(1))
            & (dataframe["close"].shift(2) < dataframe["ema20"].shift(2))
            & (dataframe["close"].shift(1) > dataframe["ema20"].shift(1))
            & (
                ~(
                    (dataframe["close"].shift(3) > dataframe["ema20"].shift(3))
                    & (dataframe["close"].shift(4) < dataframe["ema20"].shift(4))
                )
            )
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(20).mean()
                * coin_params.get("volume_trend_multiplier", 1.5)
            )  # CHANGED
            & (dataframe["rsi"].shift(1) > 40)
            & (dataframe["rsi"].shift(1) < 60)
            & (dataframe["close"].shift(1) > dataframe["close"].shift(3))
            & (
                dataframe["atr"].shift(1)
                > dataframe["atr"].shift(1).rolling(20).mean() * 1.0
            )
            & (dataframe["close"].shift(1) > dataframe["kc_middleband"].shift(1))
            & (
                dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25)
            )  # CHANGED
            & has_volume
        )

        short_condition_trend = (
            (dataframe["close"].shift(1) < dataframe["ema20"].shift(1))
            & (dataframe["ema20"].shift(1) < dataframe["ema50"].shift(1))
            & (dataframe["ema50"].shift(1) < dataframe["ema200"].shift(1))
            & (dataframe["close"].shift(2) > dataframe["ema20"].shift(2))
            & (dataframe["close"].shift(1) < dataframe["ema20"].shift(1))
            & (
                ~(
                    (dataframe["close"].shift(3) < dataframe["ema20"].shift(3))
                    & (dataframe["close"].shift(4) > dataframe["ema20"].shift(4))
                )
            )
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(20).mean()
                * coin_params.get("volume_trend_multiplier", 1.5)
            )  # CHANGED
            & (dataframe["rsi"].shift(1) > 40)
            & (dataframe["rsi"].shift(1) < 60)
            & (dataframe["close"].shift(1) < dataframe["close"].shift(3))
            & (
                dataframe["atr"].shift(1)
                > dataframe["atr"].shift(1).rolling(20).mean() * 1.0
            )
            & (dataframe["close"].shift(1) < dataframe["kc_middleband"].shift(1))
            & (
                dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25)
            )  # CHANGED
            & has_volume
        )

        # === MOMENTUM CONTINUATION CONDITIONS ===
        long_condition_momentum = (
            (dataframe["close"].shift(1) > dataframe["ema50"].shift(1))
            & (dataframe["ema20"].shift(1) > dataframe["ema50"].shift(1))
            & (dataframe["ema50"].shift(1) > dataframe["ema200"].shift(1))
            & (dataframe["rsi"].shift(4) < 50)
            & (dataframe["rsi"].shift(2) > 55)
            & (dataframe["rsi"].shift(1) > dataframe["rsi"].shift(2))
            & (dataframe["low"].shift(2) > dataframe["low"].shift(3))
            & (dataframe["close"].shift(1) > dataframe["high"].shift(3))
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(10).mean()
                * coin_params.get("volume_momentum_multiplier", 1.2)
            )  # CHANGED
            & (dataframe["close"].shift(1) < dataframe["kc_upperband"].shift(1) * 0.995)
            & (
                dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25)
            )  # CHANGED
            & has_volume
        )

        short_condition_momentum = (
            (dataframe["close"].shift(1) < dataframe["ema50"].shift(1))
            & (dataframe["ema20"].shift(1) < dataframe["ema50"].shift(1))
            & (dataframe["ema50"].shift(1) < dataframe["ema200"].shift(1))
            & (dataframe["rsi"].shift(4) > 50)
            & (dataframe["rsi"].shift(2) < 45)
            & (dataframe["rsi"].shift(1) < dataframe["rsi"].shift(2))
            & (dataframe["high"].shift(2) < dataframe["high"].shift(3))
            & (dataframe["close"].shift(1) < dataframe["low"].shift(3))
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(10).mean()
                * coin_params.get("volume_momentum_multiplier", 1.2)
            )  # CHANGED
            & (dataframe["close"].shift(1) > dataframe["kc_lowerband"].shift(1) * 1.005)
            & (
                dataframe["adx"].shift(1) > coin_params.get("adx_trending_min", 25)
            )  # CHANGED
            & has_volume
        )

        # === SECONDARY CONDITIONS ===
        long_condition_secondary = (
            bullish_divergence
            & (dataframe["close"].shift(1) > dataframe["close"].shift(3))
            & (dataframe["rsi"].shift(1) > 35)
            & (dataframe["rsi"].shift(1) < 70)
            & has_volume
        )

        short_condition_secondary = (
            bearish_divergence
            & (dataframe["close"].shift(1) < dataframe["close"].shift(3))
            & (dataframe["rsi"].shift(1) > 30)
            & (dataframe["rsi"].shift(1) < 80)
            & has_volume
        )

        # === TERTIARY CONDITIONS ===
        long_condition_tertiary = (
            bullish_divergence
            & (dataframe["close"].shift(1) > dataframe["ema20"].shift(1))
            & (dataframe["rsi"].shift(1) > 40)
            & (dataframe["rsi"].shift(1) < 65)
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(10).mean()
            )
            & has_volume
        )

        short_condition_tertiary = (
            bearish_divergence
            & (dataframe["close"].shift(1) < dataframe["ema20"].shift(1))
            & (dataframe["rsi"].shift(1) > 35)
            & (dataframe["rsi"].shift(1) < 75)
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(10).mean()
            )
            & has_volume
        )

        # === QUATERNARY CONDITIONS ===
        long_condition_quaternary = (
            bullish_divergence
            & (dataframe["close"].shift(2) < dataframe["ema20"].shift(2))
            & (dataframe["close"].shift(1) > dataframe["ema20"].shift(1))
            & (dataframe["rsi"].shift(1) > 45)
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(5).mean()
                * coin_params.get("volume_momentum_multiplier", 1.2)
            )  # CHANGED
            & has_volume
        )

        short_condition_quaternary = (
            bearish_divergence
            & (dataframe["close"].shift(2) > dataframe["ema20"].shift(2))
            & (dataframe["close"].shift(1) < dataframe["ema20"].shift(1))
            & (dataframe["rsi"].shift(1) < 55)
            & (
                dataframe["volume"].shift(1)
                > dataframe["volume"].shift(1).rolling(5).mean()
                * coin_params.get("volume_momentum_multiplier", 1.2)
            )  # CHANGED
            & has_volume
        )

        # === FIFTH CONDITIONS ===
        long_condition_fifth = (
            bullish_divergence
            & (dataframe["close"].shift(1) > dataframe["kc_middleband"].shift(1))
            & (dataframe["close"].shift(1) < dataframe["kc_upperband"].shift(1))
            & (dataframe["close"].shift(1) > dataframe["close"].shift(2))
            & (dataframe["rsi"].shift(1) > 35)
            & (dataframe["rsi"].shift(1) < 70)
            & has_volume
        )

        short_condition_fifth = (
            bearish_divergence
            & (dataframe["close"].shift(1) < dataframe["kc_middleband"].shift(1))
            & (dataframe["close"].shift(1) > dataframe["kc_lowerband"].shift(1))
            & (dataframe["close"].shift(1) < dataframe["close"].shift(2))
            & (dataframe["rsi"].shift(1) > 30)
            & (dataframe["rsi"].shift(1) < 65)
            & has_volume
        )

        # === SIXTH CONDITIONS - STRONG DIVERGENCE COUNT ===
        long_condition_sixth = (
            strong_bull_divergence
            & rsi_long_timing
            & good_long_position
            & volume_ok
            & market_trending
        )

        short_condition_sixth = (
            strong_bear_divergence
            & rsi_short_timing
            & good_short_position
            & volume_ok
            & market_trending
        )

        # === SIGNAL ASSIGNMENTS (Priority Order) ===
        # 1. REVERSAL SIGNALS (Highest Priority - Your new conditions)
        reversal_long = long_condition_reversal & (dataframe["enter_tag"] == "")
        reversal_short = short_condition_reversal & (dataframe["enter_tag"] == "")
        dataframe.loc[reversal_long, "enter_long"] = 1
        dataframe.loc[reversal_long, "enter_tag"] = "Bull_RSV1"
        dataframe.loc[reversal_short, "enter_short"] = 1
        dataframe.loc[reversal_short, "enter_tag"] = "Bear_RSV1"

        # 2. PRIMARY CONDITIONS (Divergence-based)
        primary_long = long_condition_primary & (dataframe["enter_tag"] == "")
        primary_short = short_condition_primary & (dataframe["enter_tag"] == "")
        dataframe.loc[primary_long, "enter_long"] = 1
        dataframe.loc[primary_long, "enter_tag"] = "Bull_E1"
        dataframe.loc[primary_short, "enter_short"] = 1
        dataframe.loc[primary_short, "enter_tag"] = "Bear_E1"

        # 3. TREND CONDITIONS
        trend_long = long_condition_trend & (dataframe["enter_tag"] == "")
        trend_short = short_condition_trend & (dataframe["enter_tag"] == "")
        dataframe.loc[trend_long, "enter_long"] = 1
        dataframe.loc[trend_long, "enter_tag"] = "Bull_Trend"
        dataframe.loc[trend_short, "enter_short"] = 1
        dataframe.loc[trend_short, "enter_tag"] = "Bear_Trend"

        # 4. MOMENTUM CONDITIONS
        momentum_long = long_condition_momentum & (dataframe["enter_tag"] == "")
        momentum_short = short_condition_momentum & (dataframe["enter_tag"] == "")
        dataframe.loc[momentum_long, "enter_long"] = 1
        dataframe.loc[momentum_long, "enter_tag"] = "Bull_Momentum"
        dataframe.loc[momentum_short, "enter_short"] = 1
        dataframe.loc[momentum_short, "enter_tag"] = "Bear_Momentum"

        # 5. SIXTH CONDITIONS (Strong divergence count)
        sixth_long = long_condition_sixth & (dataframe["enter_tag"] == "")
        sixth_short = short_condition_sixth & (dataframe["enter_tag"] == "")
        dataframe.loc[sixth_long, "enter_long"] = 1
        dataframe.loc[sixth_long, "enter_tag"] = "Bull_E6"
        dataframe.loc[sixth_short, "enter_short"] = 1
        dataframe.loc[sixth_short, "enter_tag"] = "Bear_E6"

        # 6. SECONDARY CONDITIONS
        secondary_long = long_condition_secondary & (dataframe["enter_tag"] == "")
        secondary_short = short_condition_secondary & (dataframe["enter_tag"] == "")
        dataframe.loc[secondary_long, "enter_long"] = 1
        dataframe.loc[secondary_long, "enter_tag"] = "Bull_E2"
        dataframe.loc[secondary_short, "enter_short"] = 1
        dataframe.loc[secondary_short, "enter_tag"] = "Bear_E2"

        # 7. TERTIARY CONDITIONS
        tertiary_long = long_condition_tertiary & (dataframe["enter_tag"] == "")
        tertiary_short = short_condition_tertiary & (dataframe["enter_tag"] == "")
        dataframe.loc[tertiary_long, "enter_long"] = 1
        dataframe.loc[tertiary_long, "enter_tag"] = "Bull_E3"
        dataframe.loc[tertiary_short, "enter_short"] = 1
        dataframe.loc[tertiary_short, "enter_tag"] = "Bear_E3"

        # 8. QUATERNARY CONDITIONS
        quaternary_long = long_condition_quaternary & (dataframe["enter_tag"] == "")
        quaternary_short = short_condition_quaternary & (dataframe["enter_tag"] == "")
        dataframe.loc[quaternary_long, "enter_long"] = 1
        dataframe.loc[quaternary_long, "enter_tag"] = "Bull_E4"
        dataframe.loc[quaternary_short, "enter_short"] = 1
        dataframe.loc[quaternary_short, "enter_tag"] = "Bear_E4"

        # 9. FIFTH CONDITIONS
        fifth_long = long_condition_fifth & (dataframe["enter_tag"] == "")
        fifth_short = short_condition_fifth & (dataframe["enter_tag"] == "")
        dataframe.loc[fifth_long, "enter_long"] = 1
        dataframe.loc[fifth_long, "enter_tag"] = "Bull_E5"
        dataframe.loc[fifth_short, "enter_short"] = 1
        dataframe.loc[fifth_short, "enter_tag"] = "Bear_E5"
        # 10. VOLUME BREAKOUT CONDITIONS
        volume_breakout_long_signal = volume_breakout_long & (
            dataframe["enter_tag"] == ""
        )
        volume_breakout_short_signal = volume_breakout_short & (
            dataframe["enter_tag"] == ""
        )
        dataframe.loc[volume_breakout_long_signal, "enter_long"] = 1
        dataframe.loc[volume_breakout_long_signal, "enter_tag"] = "Bull_VB1"
        dataframe.loc[volume_breakout_short_signal, "enter_short"] = 1
        dataframe.loc[volume_breakout_short_signal, "enter_tag"] = "Bear_VB1"

        # 11. MEAN REVERSION CONDITIONS
        mean_reversion_long_signal = mean_reversion_long & (
            dataframe["enter_tag"] == ""
        )
        mean_reversion_short_signal = mean_reversion_short & (
            dataframe["enter_tag"] == ""
        )
        dataframe.loc[mean_reversion_long_signal, "enter_long"] = 1
        dataframe.loc[mean_reversion_long_signal, "enter_tag"] = "Bull_MR1"
        dataframe.loc[mean_reversion_short_signal, "enter_short"] = 1
        dataframe.loc[mean_reversion_short_signal, "enter_tag"] = "Bear_MR1"

        # 12. MOMENTUM CONTINUATION CONDITIONS
        momentum_continuation_long_signal = momentum_continuation_long & (
            dataframe["enter_tag"] == ""
        )
        momentum_continuation_short_signal = momentum_continuation_short & (
            dataframe["enter_tag"] == ""
        )
        dataframe.loc[momentum_continuation_long_signal, "enter_long"] = 1
        dataframe.loc[momentum_continuation_long_signal, "enter_tag"] = "Bull_MC1"
        dataframe.loc[momentum_continuation_short_signal, "enter_short"] = 1
        dataframe.loc[momentum_continuation_short_signal, "enter_tag"] = "Bear_MC1"

        # Logging

        # Only log when there's an actual entry signal on the current candle
        if len(dataframe) > 0:
            latest = dataframe.iloc[-1]
            if latest.get("enter_long", 0) == 1 or latest.get("enter_short", 0) == 1:
                logger.info(f"🚀 {metadata['pair']} ENTRY DETECTED!")
                logger.info(f"   🏷️ Tag: {latest['enter_tag']}")
                logger.info(f"   📊 RSI: {latest['rsi']:.1f}")
                logger.info(f"   💧 Volume Ratio: {latest['volume_ratio']:.2f}")
                logger.info(
                    f"   🎯 Bull Div Count: {latest.get('total_bullish_divergences_count', 0)}"
                )
                logger.info(
                    f"   🎯 Bear Div Count: {latest.get('total_bearish_divergences_count', 0)}"
                )
                logger.info(
                    f"   📈 Signal Strength: {latest.get('signal_strength', 0)}"
                )
                logger.info(f"   📊 ADX: {latest.get('adx', 0):.1f}")

        return dataframe

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
        """
        Adaptive leverage based on signal strength
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 0:
                current_signal_strength = dataframe["signal_strength"].iloc[-1]

                # Reduce leverage for weaker signals
                if current_signal_strength >= 8:
                    return self.leverage_value  # Full leverage for strong signals
                elif current_signal_strength >= 6:
                    return self.leverage_value * 0.8  # 80% leverage
                elif current_signal_strength >= 4:
                    return self.leverage_value * 0.6  # 60% leverage
                else:
                    return self.leverage_value * 0.4  # 40% leverage for weak signals
            else:
                return self.leverage_value * 0.5  # Default to 50% leverage if no data
        except:
            pass

        return self.leverage_value * 0.5  # Fallback to 50% leverage on error

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """Modified to not interfere with profit taking"""

        # Only apply stoploss for losses or very small profits
        if current_profit > 0.04:  # Let custom_exit handle profits > 4%
            return None

        # Your existing stoploss logic here for losses only
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current_candle = dataframe.iloc[-1]
                atr_value = current_candle.get("atr", 0.02)
                atr_multiplier = 3.0  # More conservative

                if trade.is_short:
                    stoploss_price = trade.open_rate + (atr_value * atr_multiplier)
                else:
                    stoploss_price = trade.open_rate - (atr_value * atr_multiplier)

                return stoploss_from_absolute(
                    stoploss_price,
                    current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
        except:
            pass

        return None  # Keep current stoploss

    def initialize_divergences_lists(self, dataframe: DataFrame):
        """Initialize divergence tracking columns"""
        # Bullish Divergences
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ""

        # Bearish Divergences
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ""

    def get_iterators(self, dataframe):
        """Get pivot point iterators for divergence detection"""
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name="Pandas")):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(
                    0 if len(high_iterator) == 0 else high_iterator[-1]
                )
            else:
                high_iterator.append(index)

        return high_iterator, low_iterator

    def add_divergences(
        self, dataframe: DataFrame, indicator: str, high_iterator, low_iterator
    ):
        """Add divergence detection for a specific indicator"""
        (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = (
            self.divergence_finder_dataframe(
                dataframe, indicator, high_iterator, low_iterator
            )
        )
        dataframe["bearish_divergence_" + indicator + "_occurence"] = (
            bearish_divergences
        )
        dataframe["bullish_divergence_" + indicator + "_occurence"] = (
            bullish_divergences
        )

    def divergence_finder_dataframe(
        self, dataframe: DataFrame, indicator_source: str, high_iterator, low_iterator
    ) -> Tuple[pd.Series, pd.Series]:
        """Enhanced divergence finder with improved logic"""
        bearish_lines = [np.empty(len(dataframe["close"])) * np.nan]
        bearish_divergences = np.empty(len(dataframe["close"])) * np.nan
        bullish_lines = [np.empty(len(dataframe["close"])) * np.nan]
        bullish_divergences = np.empty(len(dataframe["close"])) * np.nan

        for index, row in enumerate(dataframe.itertuples(index=True, name="Pandas")):

            # Bearish divergence detection
            bearish_occurence = self.bearish_divergence_finder(
                dataframe, dataframe[indicator_source], high_iterator, index
            )

            if bearish_occurence is not None:
                (prev_pivot, current_pivot) = bearish_occurence
                bearish_prev_pivot = dataframe["close"][prev_pivot]
                bearish_current_pivot = dataframe["close"][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]

                # Enhanced validation for bearish divergence
                price_diff = abs(bearish_current_pivot - bearish_prev_pivot)
                indicator_diff = abs(bearish_ind_current_pivot - bearish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot

                # Only accept divergences with sufficient magnitude and time separation
                if (
                    price_diff > dataframe["atr"][current_pivot] * 0.5
                    and indicator_diff > 5
                    and time_diff >= 5
                ):

                    bearish_divergences[index] = row.close
                    dataframe.loc[index, "total_bearish_divergences"] = row.close
                    dataframe.loc[index, "total_bearish_divergences_count"] += 1
                    dataframe.loc[index, "total_bearish_divergences_names"] += (
                        indicator_source.upper() + "<br>"
                    )

            # Bullish divergence detection
            bullish_occurence = self.bullish_divergence_finder(
                dataframe, dataframe[indicator_source], low_iterator, index
            )

            if bullish_occurence is not None:
                (prev_pivot, current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe["close"][prev_pivot]
                bullish_current_pivot = dataframe["close"][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]

                # Enhanced validation for bullish divergence
                price_diff = abs(bullish_current_pivot - bullish_prev_pivot)
                indicator_diff = abs(bullish_ind_current_pivot - bullish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot

                # Only accept divergences with sufficient magnitude and time separation
                if (
                    price_diff > dataframe["atr"][current_pivot] * 0.5
                    and indicator_diff > 5
                    and time_diff >= 5
                ):

                    bullish_divergences[index] = row.close
                    dataframe.loc[index, "total_bullish_divergences"] = row.close

                    # CORRECT - increment BULLISH counters for bullish divergence:
                    dataframe.loc[index, "total_bullish_divergences_count"] += 1
                    dataframe.loc[index, "total_bullish_divergences_names"] += (
                        indicator_source.upper() + "<br>"
                    )

        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index):
        """Enhanced bearish divergence detection"""
        try:
            if high_iterator[index] == index:
                current_pivot = high_iterator[index]
                occurences = list(dict.fromkeys(high_iterator))
                current_index = occurences.index(high_iterator[index])

                for i in range(
                    current_index - 1, current_index - self.window.value - 1, -1
                ):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue

                    # Enhanced divergence validation
                    price_higher = (
                        dataframe["pivot_highs"][current_pivot]
                        > dataframe["pivot_highs"][prev_pivot]
                    )
                    indicator_lower = indicator[current_pivot] < indicator[prev_pivot]

                    price_lower = (
                        dataframe["pivot_highs"][current_pivot]
                        < dataframe["pivot_highs"][prev_pivot]
                    )
                    indicator_higher = indicator[current_pivot] > indicator[prev_pivot]

                    # Check for classic or hidden divergence
                    if (price_higher and indicator_lower) or (
                        price_lower and indicator_higher
                    ):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(
                            dataframe, prev_pivot, current_pivot, "bearish"
                        ):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index):
        """Enhanced bullish divergence detection"""
        try:
            if low_iterator[index] == index:
                current_pivot = low_iterator[index]
                occurences = list(dict.fromkeys(low_iterator))
                current_index = occurences.index(low_iterator[index])

                for i in range(
                    current_index - 1, current_index - self.window.value - 1, -1
                ):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue

                    # Enhanced divergence validation
                    price_lower = (
                        dataframe["pivot_lows"][current_pivot]
                        < dataframe["pivot_lows"][prev_pivot]
                    )
                    indicator_higher = indicator[current_pivot] > indicator[prev_pivot]

                    price_higher = (
                        dataframe["pivot_lows"][current_pivot]
                        > dataframe["pivot_lows"][prev_pivot]
                    )
                    indicator_lower = indicator[current_pivot] < indicator[prev_pivot]

                    # Check for classic or hidden divergence
                    if (price_lower and indicator_higher) or (
                        price_higher and indicator_lower
                    ):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(
                            dataframe, prev_pivot, current_pivot, "bullish"
                        ):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def validate_divergence_trend(
        self, dataframe, prev_pivot, current_pivot, divergence_type
    ):
        """Validate divergence by checking intermediate trend"""
        try:
            # Check if there's a clear trend between pivots
            mid_point = (prev_pivot + current_pivot) // 2

            if divergence_type == "bearish":
                # For bearish divergence, expect uptrend in between
                return dataframe["ema20"][mid_point] > dataframe["ema20"][prev_pivot]
            else:
                # For bullish divergence, expect downtrend in between
                return dataframe["ema20"][mid_point] < dataframe["ema20"][prev_pivot]
        except:
            return True  # Default to accepting divergence if validation fails

    @property
    def protections(self):
        """Enhanced protection configuration"""
        prot = []

        if self.use_cooldown_protection.value:
            prot.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self.cooldown_lookback.value,
                }
            )

        if self.use_max_drawdown_protection.value:
            prot.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self.max_drawdown_lookback.value,
                    "trade_limit": self.max_drawdown_trade_limit.value,
                    "stop_duration_candles": self.max_drawdown_stop_duration.value,
                    "max_allowed_drawdown": self.max_allowed_drawdown.value,
                }
            )

        if self.use_stop_protection.value:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self.stoploss_guard_lookback.value,
                    "trade_limit": self.stoploss_guard_trade_limit.value,
                    "stop_duration_candles": self.stop_duration.value,
                    "only_per_pair": self.stoploss_guard_only_per_pair.value,
                }
            )

        return prot

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        # === YOUR OTHER EXIT CONDITIONS FIRST ===
        # (Add any other exit conditions you have here)
        # any_long_exit = (some_other_condition)
        # any_short_exit = (some_other_condition)

        # === EXIT ON OPPOSITE SIGNALS (your previous approach) ===
        if "enter_long" in dataframe.columns and "enter_short" in dataframe.columns:
            # Exit longs on short signals
            reversal_long_exit = dataframe["enter_short"] == 1
            dataframe.loc[reversal_long_exit, "exit_long"] = 1
            dataframe.loc[reversal_long_exit, "exit_tag"] = "Reversal_Short_Signal"

            # Exit shorts on long signals
            if self.can_short:
                reversal_short_exit = dataframe["enter_long"] == 1
                dataframe.loc[reversal_short_exit, "exit_short"] = 1
                dataframe.loc[reversal_short_exit, "exit_tag"] = "Reversal_Long_Signal"

        return dataframe

    def dynamic_trailing_stop(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ):
        """Dynamic trailing based on entry type and market conditions"""

        entry_tag = trade.enter_tag or ""
        coin_params = self.get_coin_params(pair)

        # Get current market data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_row = dataframe.iloc[-1]
        volatility = last_row.get("volatility", 0.02)

        # Entry-specific trailing logic
        if "MR1" in entry_tag:
            # Mean reversion - tighter trailing
            trailing_distance = 0.015 * coin_params.get("mr1_trailing_multiplier", 1.0)
        elif "MC1" in entry_tag:
            # Momentum continuation - wider trailing
            trailing_distance = 0.025 * coin_params.get("mc1_trailing_multiplier", 1.0)
        elif "VB1" in entry_tag:
            # Volume breakout - volatility-based
            trailing_distance = volatility * 2.0
        else:
            # Default trailing
            trailing_distance = 0.02

        # Activate trailing only after minimum profit
        min_profit_for_trailing = coin_params.get("min_trailing_profit", 0.015)

        if current_profit >= min_profit_for_trailing:
            return stoploss_from_open(
                trailing_distance, current_profit, is_short=trade.is_short
            )

        return None

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Enhanced exit strategy with Optuna-optimized parameters and entry-type specific handling
        """
        from logging import getLogger

        logger = getLogger(__name__)

        # Get optimized parameters for this pair
        coin_params = self.get_coin_params(pair)

        # Calculate trade duration
        trade_duration_minutes = (
            current_time - trade.open_date_utc
        ).total_seconds() / 60
        entry_tag = trade.enter_tag or ""

        # === DATA FETCH WITH ERROR HANDLING ===
        signal_strength = 5
        dataframe = None
        volatility = 0.02
        momentum_score = 0
        rsi = 50
        volume_ratio = 1.0
        adx = 25
        bb_percent = 0.5

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                last_row = dataframe.iloc[-1]

                signal_strength = last_row.get("signal_strength", 5)
                volatility = max(0.005, last_row.get("volatility", 0.02))
                rsi = last_row.get("rsi", 50)
                volume_ratio = last_row.get("volume_ratio", 1.0)
                adx = last_row.get("adx", 25)
                bb_percent = last_row.get("bb_percent", 0.5)

                if len(dataframe) >= 10:
                    momentum_score = (
                        last_row["close"] - dataframe.iloc[-10]["close"]
                    ) / dataframe.iloc[-10]["close"]
        except Exception:
            pass

        # === HELPER FUNCTION ===
        def log_and_exit(reason: str):
            logger.info(f"🔤 {pair} EXIT: {reason}")
            logger.info(f"   🕐 Time Held: {int(trade_duration_minutes)} min")
            logger.info(f"   💰 Profit: {current_profit * 100:.2f}%")
            logger.info(f"   📊 Signal Strength: {signal_strength}")
            logger.info(f"   📈 Momentum: {momentum_score * 100:.2f}%")
            logger.info(f"   📊 RSI: {rsi:.1f}")
            logger.info(f"   💧 Volume Ratio: {volume_ratio:.2f}")
            logger.info(f"   📊 Volatility: {volatility:.4f}")
            logger.info(f"   🔄 Exit Trigger: {reason}")
            return reason

        # === EMERGENCY EXITS (HIGHEST PRIORITY) ===
        emergency_profit_limit = coin_params.get("emergency_profit_limit", 0.25)
        if current_profit >= emergency_profit_limit:
            return log_and_exit(f"emergency_exit_{int(emergency_profit_limit*100)}pct")

        # === SESSION AND MARKET CONTEXT ===
        hour = current_time.hour
        is_major_session = (8 <= hour <= 11) or (13 <= hour <= 16)
        is_overlap_session = (8 <= hour <= 10) or (13 <= hour <= 15)

        # Dynamic multipliers based on market conditions
        session_multiplier = (
            coin_params.get("session_multiplier_overlap", 1.3)
            if is_overlap_session
            else (
                coin_params.get("session_multiplier_major", 1.15)
                if is_major_session
                else coin_params.get("session_multiplier_quiet", 0.85)
            )
        )

        volatility_multiplier = min(
            2.5, max(0.6, volatility * coin_params.get("volatility_sensitivity", 40))
        )

        # === ENTRY-TYPE SPECIFIC EXITS ===

        # === MEAN REVERSION (MR1) EXITS ===
        if "MR1" in entry_tag:
            mr1_max_hold = coin_params.get("mr1_max_hold_minutes", 35)
            mr1_rsi_exit_long = coin_params.get("mr1_rsi_exit_long", 68.0)
            mr1_rsi_exit_short = coin_params.get("mr1_rsi_exit_short", 32.0)
            mr1_quick_profit = coin_params.get("mr1_quick_profit_target", 0.025)
            mr1_timeout_profit = coin_params.get("mr1_timeout_min_profit", 0.008)

            # Quick profit exit for mean reversion
            if (
                trade_duration_minutes >= 8
                and current_profit >= mr1_quick_profit * session_multiplier
            ):
                return log_and_exit("mr1_quick_target")

            # RSI-based exit (reversion completed)
            if trade_duration_minutes >= 12 and current_profit >= 0.015:
                if not trade.is_short and rsi >= mr1_rsi_exit_long:
                    return log_and_exit("mr1_rsi_reversion_complete")
                elif trade.is_short and rsi <= mr1_rsi_exit_short:
                    return log_and_exit("mr1_rsi_reversion_complete")

            # BB percent exit (moved to expected range)
            if trade_duration_minutes >= 15 and current_profit >= 0.012:
                if not trade.is_short and bb_percent >= coin_params.get(
                    "mr1_bb_exit_long", 0.7
                ):
                    return log_and_exit("mr1_bb_reversion")
                elif trade.is_short and bb_percent <= coin_params.get(
                    "mr1_bb_exit_short", 0.3
                ):
                    return log_and_exit("mr1_bb_reversion")

            # Timeout exit with minimum profit
            if trade_duration_minutes >= mr1_max_hold:
                if current_profit >= mr1_timeout_profit:
                    return log_and_exit("mr1_optimized_timeout")

        # === MOMENTUM CONTINUATION (MC1) EXITS ===
        elif "MC1" in entry_tag:
            mc1_profit_target = (
                coin_params.get("mc1_profit_target", 0.04)
                * session_multiplier
                * volatility_multiplier
            )
            mc1_momentum_break = coin_params.get("mc1_momentum_break_threshold", 0.015)
            mc1_max_hold = coin_params.get("mc1_max_hold_minutes", 45)

            # Primary profit target
            if current_profit >= mc1_profit_target:
                return log_and_exit("mc1_optimized_target")

            # Momentum break exit
            if trade_duration_minutes >= 15 and current_profit >= 0.015:
                if not trade.is_short and momentum_score < -mc1_momentum_break:
                    return log_and_exit("mc1_momentum_break")
                elif trade.is_short and momentum_score > mc1_momentum_break:
                    return log_and_exit("mc1_momentum_break")

            # ADX weakening (trend losing strength)
            if trade_duration_minutes >= 20 and current_profit >= 0.02:
                if adx < coin_params.get("mc1_adx_exit_threshold", 22):
                    return log_and_exit("mc1_trend_weakening")

            # Timeout with scaled profit requirement
            if trade_duration_minutes >= mc1_max_hold:
                min_timeout_profit = coin_params.get("mc1_timeout_min_profit", 0.01)
                if current_profit >= min_timeout_profit:
                    return log_and_exit("mc1_timeout_exit")

        # === VOLUME BREAKOUT (VB1) EXITS ===
        elif "VB1" in entry_tag:
            vb1_volume_fade = coin_params.get("vb1_volume_fade_threshold", 1.15)
            vb1_profit_target = (
                coin_params.get("vb1_profit_target", 0.03) * session_multiplier
            )
            vb1_quick_profit = coin_params.get("vb1_quick_profit", 0.02)

            # Quick profit for volume breakouts
            if trade_duration_minutes >= 10 and current_profit >= vb1_quick_profit:
                return log_and_exit("vb1_quick_profit")

            # Volume fade exit
            if trade_duration_minutes >= 15 and current_profit >= 0.012:
                if volume_ratio < vb1_volume_fade:
                    return log_and_exit("vb1_volume_fade")

            # Volatility collapse after breakout
            if trade_duration_minutes >= 25 and current_profit >= 0.015:
                if volatility < coin_params.get("vb1_min_volatility", 0.012):
                    return log_and_exit("vb1_volatility_collapse")

            # Primary target
            if current_profit >= vb1_profit_target:
                return log_and_exit("vb1_target_reached")

        # === REVERSAL (RSV1) EXITS ===
        elif "RSV1" in entry_tag:
            rsv1_rsi_recovery = coin_params.get("rsv1_rsi_recovery_threshold", 58.0)
            rsv1_profit_target = coin_params.get("rsv1_profit_target", 0.035)
            rsv1_max_hold = coin_params.get("rsv1_max_hold_minutes", 60)

            # RSI recovery exit (reversal working)
            if trade_duration_minutes >= 15 and current_profit >= 0.02:
                if not trade.is_short and rsi >= rsv1_rsi_recovery:
                    return log_and_exit("rsv1_rsi_recovery")
                elif trade.is_short and rsi <= (100 - rsv1_rsi_recovery):
                    return log_and_exit("rsv1_rsi_recovery")

            # Primary reversal target
            if current_profit >= rsv1_profit_target * session_multiplier:
                return log_and_exit("rsv1_reversal_complete")

            # Extended hold exit
            if trade_duration_minutes >= rsv1_max_hold:
                if current_profit >= coin_params.get("rsv1_min_timeout_profit", 0.01):
                    return log_and_exit("rsv1_extended_hold")

        # === TREND FOLLOWING EXITS ===
        elif "Trend" in entry_tag or "E1" in entry_tag:
            trend_profit_target = (
                coin_params.get("trend_profit_target", 0.045) * session_multiplier
            )
            trend_ema_break = coin_params.get("trend_ema_break_periods", 3)

            # Primary trend target
            if current_profit >= trend_profit_target:
                return log_and_exit("trend_target_reached")

            # EMA break exit (trend reversal)
            try:
                if (
                    dataframe is not None
                    and len(dataframe) >= trend_ema_break
                    and trade_duration_minutes >= 20
                ):
                    current_ema20 = dataframe.iloc[-1]["ema20"]
                    past_ema20 = dataframe.iloc[-(trend_ema_break + 1)]["ema20"]

                    if (
                        not trade.is_short
                        and current_ema20 < past_ema20
                        and current_profit >= 0.015
                    ):
                        return log_and_exit("trend_ema_break_down")
                    elif (
                        trade.is_short
                        and current_ema20 > past_ema20
                        and current_profit >= 0.015
                    ):
                        return log_and_exit("trend_ema_break_up")
            except:
                pass

        # === UNIVERSAL TIME-BASED EXITS ===

        # Quick scalp exits (0-20 minutes)
        if trade_duration_minutes <= 20:
            quick_targets = [
                (0.035 * session_multiplier * volatility_multiplier, "quick_3_5pct"),
                (0.025 * session_multiplier, "quick_2_5pct"),
                (0.018 * session_multiplier, "quick_1_8pct"),
            ]

            for target, tag in quick_targets:
                if current_profit >= target:
                    return log_and_exit(tag)

        # Medium-term exits (20-60 minutes)
        elif 20 < trade_duration_minutes <= 60:
            medium_base = coin_params.get("medium_term_base_target", 0.05)
            medium_target = medium_base * session_multiplier * volatility_multiplier

            # Adjust based on signal strength
            if signal_strength >= 7:
                medium_target *= coin_params.get("high_signal_multiplier", 1.2)
            elif signal_strength <= 3:
                medium_target *= coin_params.get("low_signal_multiplier", 0.8)

            if current_profit >= medium_target:
                return log_and_exit("medium_term_target")

        # === REVERSAL SIGNAL EXITS ===
        if trade_duration_minutes >= 10:
            try:
                if dataframe is not None and not dataframe.empty:
                    last_row = dataframe.iloc[-1]

                    # Exit on opposite entry signals
                    if not trade.is_short and last_row.get("enter_short", 0) == 1:
                        if current_profit >= coin_params.get(
                            "reversal_min_profit_long", 0.008
                        ):
                            return log_and_exit("reversal_short_signal")

                    if trade.is_short and last_row.get("enter_long", 0) == 1:
                        if current_profit >= coin_params.get(
                            "reversal_min_profit_short", 0.008
                        ):
                            return log_and_exit("reversal_long_signal")
            except:
                pass

        # === RSI EXTREME EXITS ===
        if trade_duration_minutes >= 15 and current_profit >= coin_params.get(
            "rsi_extreme_min_profit", 0.015
        ):
            rsi_overbought = coin_params.get("rsi_exit_overbought", 78)
            rsi_oversold = coin_params.get("rsi_exit_oversold", 22)

            if not trade.is_short and rsi >= rsi_overbought:
                return log_and_exit("rsi_extreme_overbought")
            elif trade.is_short and rsi <= rsi_oversold:
                return log_and_exit("rsi_extreme_oversold")

        # === MOMENTUM FADE PROTECTION ===
        if trade_duration_minutes >= 15 and current_profit >= coin_params.get(
            "momentum_fade_min_profit", 0.012
        ):
            fade_threshold = (
                coin_params.get("momentum_fade_threshold", 0.02) * volatility_multiplier
            )

            if not trade.is_short and momentum_score < -fade_threshold:
                return log_and_exit("momentum_fade_long")
            elif trade.is_short and momentum_score > fade_threshold:
                return log_and_exit("momentum_fade_short")

        # === EXTENDED DURATION MANAGEMENT ===
        if trade_duration_minutes > 90:
            extended_targets = [
                (120, coin_params.get("extended_2hr_target", 0.025), "extended_2hr"),
                (180, coin_params.get("extended_3hr_target", 0.015), "extended_3hr"),
                (300, coin_params.get("extended_5hr_target", 0.008), "extended_5hr"),
            ]

            for duration, target, tag in extended_targets:
                if trade_duration_minutes >= duration and current_profit >= target:
                    return log_and_exit(tag)

        # === MARKET PROTECTION EXITS ===

        # Friday close protection
        if current_time.weekday() == 4 and current_time.hour >= 15:
            friday_min_profit = coin_params.get("friday_close_min_profit", 0.012)
            if current_profit >= friday_min_profit:
                return log_and_exit("friday_close_protection")

        # Overnight protection
        if current_time.hour >= 22 or current_time.hour <= 3:
            overnight_min_profit = coin_params.get("overnight_min_profit", 0.008)
            if current_profit >= overnight_min_profit:
                return log_and_exit("overnight_protection")

        # Low liquidity hours
        if 4 <= current_time.hour <= 7:
            low_liquidity_profit = coin_params.get("low_liquidity_min_profit", 0.01)
            if current_profit >= low_liquidity_profit and trade_duration_minutes >= 30:
                return log_and_exit("low_liquidity_exit")

        return None

    def maybe_optimize_coin(self, pair: str, force_startup: bool = False):
        """ENHANCED optimization trigger with better permanent optimization"""
        # ADDED: Komplette neue Methode für Optimierungslogik
        if not self.optuna_manager or not self._allow_dynamic_optimization():
            logger.debug(f"🚫 [OPTUNA] OptunaManager not available for {pair}")
            return

        # ENHANCED: Force startup optimization or intelligent check
        if force_startup:
            logger.info(f"🚀 [OPTUNA] FORCED STARTUP OPTIMIZATION for {pair}")
            should_optimize = True
        else:
            should_optimize = self.optuna_manager.should_optimize(pair)

        if not should_optimize:
            return

        logger.info(f"🔄 [OPTUNA] Checking optimization conditions for {pair}")

        try:
            trades_count = self.get_coin_trades_count(pair)
            optimization_count = self.optuna_manager.optimization_trigger_count.get(
                pair, 0
            )

            logger.info(
                f"📊 [OPTUNA] {pair} has {trades_count} trades, {optimization_count} optimizations done"
            )

            # ENHANCED: More flexible trade requirements
            min_trades_required = max(
                0, optimization_count * 5
            )  # Require more trades for subsequent optimizations

            if not force_startup and trades_count < min_trades_required:
                logger.info(
                    f"⏳ [OPTUNA] Not enough trades for {pair} optimization ({trades_count}/{min_trades_required})"
                )
                return

            # Track optimization type
            if force_startup:
                opt_type = "STARTUP"
            elif self.optuna_manager.should_optimize_based_on_performance(pair):
                opt_type = "PERFORMANCE-TRIGGERED"
            else:
                opt_type = "PERIODIC"

            logger.info(
                f"✨ [OPTUNA] {opt_type} OPTIMIZATION for {pair} (trades: {trades_count})"
            )

            objective_func = self.create_objective_function(pair)

            # ENHANCED: More trials for performance-triggered optimizations
            n_trials = 15 if opt_type == "PERFORMANCE-TRIGGERED" else 10

            # Start optimization
            self.optuna_manager.optimize_coin(pair, objective_func, n_trials=n_trials)

            # Track optimization count
            self.optuna_manager.optimization_trigger_count[pair] = (
                optimization_count + 1
            )

            logger.info(f"✅ [OPTUNA] Completed {opt_type} optimization for {pair}")

        except Exception as e:
            logger.error(f"❌ [OPTUNA] Optimization failed for {pair}: {e}")
            import traceback

            logger.debug(f"🔍 [OPTUNA] Full traceback: {traceback.format_exc()}")

    def get_coin_trades_count(self, pair: str) -> int:
        """Get number of trades for specific coin"""
        # ADDED: Hilfsmethode für Trade-Anzahl pro Coin
        try:
            from freqtrade.persistence import Trade

            trades = Trade.get_trades_proxy(pair=pair)
            count = len(trades) if trades else 0
            logger.debug(f"📊 [OPTUNA] Trade count for {pair}: {count}")
            return count
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get trade count for {pair}: {e}")
            return 0

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Track trade exits and update performance"""
        # ADDED: Performance-Tracking und Optuna-Updates

        try:
            profit_ratio = trade.calc_profit_ratio(rate)

            if self.optuna_manager:
                self.optuna_manager.update_performance(pair, profit_ratio)
                logger.debug(
                    f"📈 [OPTUNA] Updated performance for {pair}: {profit_ratio:.4f}"
                )

            if pair not in self.coin_performance:
                self.coin_performance[pair] = 0.0
            self.coin_performance[pair] += profit_ratio

            # Log significant performance updates
            if abs(profit_ratio) > 0.02:  # More than 2% gain/loss
                logger.info(
                    f"💰 [OPTUNA] Significant trade result for {pair}: {profit_ratio:.2%} (cumulative: {self.coin_performance[pair]:.2%})"
                )

        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to update performance for {pair}: {e}")

        return True


def choppiness_index(high, low, close, window=14):
    """Calculate Choppiness Index"""
    natr = pd.Series(ta.NATR(high, low, close, window=window))
    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()

    choppiness = (
        100
        * np.log10((natr.rolling(window=window).sum()) / (high_max - low_min))
        / np.log10(window)
    )
    return choppiness


def resample(indicator):
    """Resample function for compatibility"""
    return indicator


def two_bands_check_long(dataframe):
    """Allow long when price is near/at lower band (oversold area)"""
    return (dataframe["low"] <= dataframe["kc_lowerband"]) | (
        dataframe["close"] <= dataframe["kc_lowerband"]
    )


def two_bands_check_short(dataframe):
    """Allow short when price is near/at upper band (overbought area)"""
    return (dataframe["high"] >= dataframe["kc_upperband"]) | (
        dataframe["close"] >= dataframe["kc_upperband"]
    )


def green_candle(dataframe):
    """Check for green candle"""
    return dataframe[resample("open")] < dataframe[resample("close")]


def red_candle(dataframe):
    """Check for red candle"""
    return dataframe[resample("open")] > dataframe[resample("close")]


def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source=None) -> DataFrame:
    """Enhanced pivot point detection"""
    from enum import Enum

    class PivotSource(Enum):
        HighLow = 0
        Close = 1

    if pivot_source is None:
        pivot_source = PivotSource.Close

    high_source = "close" if pivot_source == PivotSource.Close else "high"
    low_source = "close" if pivot_source == PivotSource.Close else "low"

    pivot_points_lows = np.empty(len(dataframe["close"])) * np.nan
    pivot_points_highs = np.empty(len(dataframe["close"])) * np.nan
    last_values = deque()

    # Find pivot points with enhanced validation
    for index, row in enumerate(dataframe.itertuples(index=True, name="Pandas")):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True

            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(
                    current_value, high_source, low_source, left, right
                )
                is_greater &= local_is_greater
                is_less &= local_is_less

            # Additional validation: ensure pivot is significant
            if is_greater:
                current_high = getattr(current_value, high_source)
                # Stamp the pivot on the confirmation candle, not the pivot candle.
                # This avoids backdating signals that require future candles to confirm.
                if hasattr(current_value, "atr") and current_high > 0:
                    pivot_points_highs[index] = current_high

            if is_less:
                current_low = getattr(current_value, low_source)
                # Stamp the pivot on the confirmation candle, not the pivot candle.
                if hasattr(current_value, "atr") and current_low > 0:
                    pivot_points_lows[index] = current_low

            last_values.popleft()

    return pd.DataFrame(
        index=dataframe.index,
        data={"pivot_lows": pivot_points_lows, "pivot_highs": pivot_points_highs},
    )


def check_if_pivot_is_greater_or_less(
    current_value, high_source: str, low_source: str, left, right
) -> Tuple[bool, bool]:
    """Helper function for pivot point validation"""
    is_greater = True
    is_less = True

    if getattr(current_value, high_source) <= getattr(left, high_source) or getattr(
        current_value, high_source
    ) <= getattr(right, high_source):
        is_greater = False

    if getattr(current_value, low_source) >= getattr(left, low_source) or getattr(
        current_value, low_source
    ) >= getattr(right, low_source):
        is_less = False

    return (is_greater, is_less)


def emaKeltner(dataframe):
    """Calculate EMA-based Keltner Channels"""
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner["upper"] = ema20 + atr
    keltner["mid"] = ema20
    keltner["lower"] = ema20 - atr
    return keltner


def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Calculate Chaikin Money Flow indicator"""
    df = dataframe.copy()
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    mfv = mfv.fillna(0.0)
    mfv *= df["volume"]
    cmf = (
        mfv.rolling(n, min_periods=0).sum()
        / df["volume"].rolling(n, min_periods=0).sum()
    )
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name="cmf")
