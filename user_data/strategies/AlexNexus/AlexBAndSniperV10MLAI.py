# --- Do not remove these libs ---
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.persistence import Trade, Order  # Added Order here
from freqtrade.strategy import (IStrategy, BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, merge_informative_pair, stoploss_from_open,
                                stoploss_from_absolute)
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

from pathlib import Path
import logging
import time
current_time = time.time()
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import csv
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from typing import List, Tuple, Optional, Dict, Any
import json
import os
from freqtrade.persistence import Trade
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval, resampled_merge
from collections import deque
import optuna
from optuna.samplers import TPESampler
from optuna.exceptions import OptunaError
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)
# ============================================================================
# ML PREDICTIVE ENTRY SYSTEM
# ============================================================================
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

ASCII_ART = '''
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                       PURE OPTUNA OPTIMIZING STRATEGY                                                  ║
    ║                                         Enhanced Multi-Feature Optimazion                                              ║
    ║                                           Machine Learning Powered                                                     ║
    ║                                                                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
'''
def choppiness_index(high, low, close, window=14):
    """Calculate Choppiness Index"""
    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift(1))
    tr['l-pc'] = abs(low - close.shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = tr['tr'].rolling(window=window).sum()
    highh = high.rolling(window=window).max()
    lowl = low.rolling(window=window).min()
    chop = 100 * np.log10(atr / (highh - lowl)) / np.log10(window)
    return chop.fillna(50)

def chaikin_money_flow(dataframe, period=20):
    """Calculate Chaikin Money Flow"""
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)
    mfv *= dataframe['volume']
    cmf = mfv.rolling(window=period).sum() / dataframe['volume'].rolling(window=period).sum()
    return cmf.fillna(0)

def emaKeltner(dataframe, ema_length=20, atr_length=10, atr_multiplier=1):
    """Calculate Keltner Channel"""
    df = dataframe.copy()
    df['ema'] = ta.EMA(df['close'], timeperiod=ema_length)
    df['atr'] = ta.ATR(df, timeperiod=atr_length)
    df['upper'] = df['ema'] + (df['atr'] * atr_multiplier)
    df['lower'] = df['ema'] - (df['atr'] * atr_multiplier)
    return {'upper': df['upper'], 'mid': df['ema'], 'lower': df['lower']}

def pivot_points(dataframe, window=10):
    """Calculate pivot points"""
    df = dataframe.copy()
    df['pivot_lows'] = df['low'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] if x[len(x)//2] == min(x) else np.nan, raw=True
    )
    df['pivot_highs'] = df['high'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] if x[len(x)//2] == max(x) else np.nan, raw=True
    )
    return df[['pivot_lows', 'pivot_highs']]

def two_bands_check_long(df):
    """Check if conditions are met for long entry (not choppy)"""
    kc_ok = df['close'] <= df['kc_lowerband']
    bb_ok = df['close'] <= df['bollinger_lowerband']
    return kc_ok | bb_ok

def two_bands_check_short(df):
    """Check if conditions are met for short entry (not choppy)"""
    kc_ok = df['close'] >= df['kc_upperband']
    bb_ok = df['close'] >= df['bollinger_upperband']
    return kc_ok | bb_ok

def resample(column_name):
    """Helper function for resampling column names"""
    return column_name
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
        self.pretraining_days = 180
        self.pretraining_done = {}
        # NEW: Track optimization timestamps for 24-hour periodic re-optimization
        self.last_optimization_time: Dict[str, datetime] = {}
        self.optimization_interval_hours = 24  # Configure optimization frequency here
        logger.info(f"📊 [OPTUNA] Initializing RealOptunaManager for strategy: {strategy_name}")
        logger.info(f"🎯 [OPTUNA] Real Optuna optimization enabled")
    
    def get_best_params(self, pair: str) -> Dict[str, Any]:
        """Get best parameters for a pair"""
        if pair in self.best_params_cache:
            return self.best_params_cache[pair]
        
        # Try to load from existing study
        try:
            study_name = f"{self.strategy_name}_{pair.replace('/', '_').replace(':', '_')}"
            study = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///user_data/strategies/optuna_studies/optuna_{study_name}.db"
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
        from datetime import datetime
        
        # First: Check if we have no parameters yet (initial optimization)
        if pair not in self.best_params_cache:
            logger.info(f"🆕 [OPTUNA] {pair} has no cached parameters - triggering initial optimization")
            return True
        
        # Second: Check 8-hour periodic optimization trigger
        if pair in self.last_optimization_time:
            hours_since_last = (datetime.now() - self.last_optimization_time[pair]).total_seconds() / 3600
            if hours_since_last >= self.optimization_interval_hours:
                logger.info(f"⏰ [OPTUNA] {pair} due for periodic optimization ({hours_since_last:.1f}h since last)")
                return True
        else:
            # If we have cached params but no timestamp, this means params were loaded from DB
            # We should set the timestamp to now to start the 8-hour clock
            logger.info(f"🕐 [OPTUNA] {pair} has cached params but no timestamp - starting 8h timer")
            self.last_optimization_time[pair] = datetime.now()
        
        # Third: Performance-based trigger
        if pair in self.performance_history:
            recent_performance = self.performance_history[pair][-10:]
            if len(recent_performance) >= 5:
                avg_performance = sum(recent_performance) / len(recent_performance)
                if avg_performance < -0.02:
                    logger.info(f"📉 [OPTUNA] {pair} triggered by poor performance ({avg_performance:.2%})")
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
    
    def optimize_coin(self, pair: str, objective_func, n_trials: int = 60):
        """Optimize parameters for a specific coin using real Optuna"""
        logger.info(f"🚀 [OPTUNA] Starting REAL optimization for {pair} with {n_trials} trials")
        
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            from pathlib import Path
            study_name = f"{self.strategy_name}_{pair.replace('/', '_').replace(':', '_')}"
            
            Path("user_data/strategies/optuna_studies").mkdir(parents=True, exist_ok=True)
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=TPESampler(),
                storage=f"sqlite:///user_data/strategies/optuna_studies/optuna_{study_name}.db",
                load_if_exists=True
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
        """Log parameters in a nicely formatted way - Enhanced with all optimized parameters"""
        logger.info(f"📈 [OPTUNA] Optimized parameters for {pair}:")
        
        # === CORE SHARED INDICATORS ===
        logger.info(f"┌─ 📊 CORE SHARED INDICATORS")
        logger.info(f"│  • rsi_period: {params.get('rsi_period', 'N/A')}")
        logger.info(f"│  • atr_period: {params.get('atr_period', 'N/A')}")
        logger.info(f"│  • volume_sma_period: {params.get('volume_sma_period', 'N/A')}")
        logger.info(f"│  • ema_short_period: {params.get('ema_short_period', 'N/A')}")
        logger.info(f"│  • ema_long_period: {params.get('ema_long_period', 'N/A')}")
        logger.info(f"│  • adx_threshold: {params.get('adx_threshold', 'N/A')}")
        logger.info(f"│  • bb_period: {params.get('bb_period', 'N/A')} (std: {params.get('bb_std', 'N/A')})")
        
        # === GLOBAL FILTERS ===
        logger.info(f"├─ 🎯 GLOBAL FILTERS")
        logger.info(f"│  • min_signal_strength: {params.get('min_signal_strength', 'N/A')}")
        logger.info(f"│  • volume_threshold: {params.get('volume_threshold', 'N/A')}")
        logger.info(f"│  • rsi_overbought/oversold: {params.get('rsi_overbought', 'N/A')}/{params.get('rsi_oversold', 'N/A')}")
        logger.info(f"│  • volatility range: {params.get('min_volatility', 'N/A'):.4f} - {params.get('max_volatility', 'N/A'):.4f}")
        logger.info(f"│  • volume_factor: {params.get('volume_factor', 'N/A')}")
        
        # === DIVERGENCE OPTIMIZATION (ENHANCED) ===
        logger.info(f"├─ 💎 DIVERGENCE OPTIMIZATION (Enhanced)")
        logger.info(f"│")
        logger.info(f"│  ┌─ Core Detection")
        logger.info(f"│  │  • window: {params.get('window', 'N/A')}")
        logger.info(f"│  │  • index_range: {params.get('index_range', 'N/A')}")
        logger.info(f"│  │  • lookback_period: {params.get('divergence_lookback_period', 'N/A')}")
        logger.info(f"│  │  • min_rsi_delta: {params.get('divergence_min_rsi_delta', 'N/A')}")
        logger.info(f"│  │  • min_price_delta: {params.get('divergence_min_price_delta', 'N/A')}")
        logger.info(f"│  │  • RSI thresholds: {params.get('divergence_rsi_low_threshold', 'N/A')}/{params.get('divergence_rsi_high_threshold', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ 🌡️ Market Regime Filters (NEW)")
        logger.info(f"│  │  • ADX range: {params.get('divergence_min_adx', 'N/A')}-{params.get('divergence_max_adx', 'N/A')}")
        logger.info(f"│  │  • ATR multiplier range: {params.get('divergence_min_atr_mult', 'N/A'):.2f}-{params.get('divergence_max_atr_mult', 'N/A'):.2f}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ ✅ Confirmation Requirements (NEW)")
        logger.info(f"│  │  • volume_multiplier: {params.get('divergence_volume_multiplier', 'N/A')}")
        logger.info(f"│  │  • require_volume_increase: {params.get('divergence_require_volume_increase', 'N/A')}")
        logger.info(f"│  │  • require_trend_context: {params.get('divergence_require_trend_context', 'N/A')}")
        logger.info(f"│  │  • require_1h_alignment: {params.get('divergence_require_1h_alignment', 'N/A')}")
        logger.info(f"│  │  • require_bands: {params.get('divergence_require_bands', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ 🎖️ Quality Scoring (NEW)")
        logger.info(f"│  │  • price_strength_weight: {params.get('divergence_price_strength_weight', 'N/A'):.2f}")
        logger.info(f"│  │  • rsi_strength_weight: {params.get('divergence_rsi_strength_weight', 'N/A'):.2f}")
        logger.info(f"│  │  • timespan_weight: {params.get('divergence_timespan_weight', 'N/A'):.2f}")
        logger.info(f"│  │  • min_quality_score: {params.get('divergence_min_quality_score', 'N/A'):.2f}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ ⏱️ Entry Timing (NEW)")
        logger.info(f"│  │  • entry_wait_candles: {params.get('divergence_entry_wait_candles', 'N/A')}")
        logger.info(f"│  │  • require_pullback: {params.get('divergence_require_pullback', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ 🛡️ HQ Divergence Filters")
        logger.info(f"│  │  LONG: adx_max={params.get('hq_adx_max', 'N/A')}, momentum_lb={params.get('hq_momentum_lookback', 'N/A')}")
        logger.info(f"│  │        green_candle={params.get('hq_require_green_candle', 'N/A')}, ema_trend={params.get('hq_ema_trend_filter', 'N/A')}")
        logger.info(f"│  │  SHORT: adx_max={params.get('hq_adx_max_short', 'N/A')}, momentum_lb={params.get('hq_momentum_lookback_short', 'N/A')}")
        logger.info(f"│  │         red_candle={params.get('hq_require_red_candle', 'N/A')}, ema_trend={params.get('hq_ema_trend_filter_short', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ 🎯 Quality-Based Risk Management (NEW)")
        logger.info(f"│  │  • SUPER stop ATR mult: {params.get('divergence_stop_atr_mult_super', 'N/A'):.2f}")
        logger.info(f"│  │  • HQ stop ATR mult: {params.get('divergence_stop_atr_mult_hq', 'N/A'):.2f}")
        logger.info(f"│  │  • REGULAR stop ATR mult: {params.get('divergence_stop_atr_mult_regular', 'N/A'):.2f}")
        logger.info(f"│  │")
        logger.info(f"│  └─ 🎯 Quality-Based Profit Targets (NEW)")
        logger.info(f"│  │  • SUPER target mult: {params.get('divergence_target_mult_super', 'N/A'):.2f}")
        logger.info(f"│  │  • HQ target mult: {params.get('divergence_target_mult_hq', 'N/A'):.2f}")
        logger.info(f"│  └─ • REGULAR target mult: {params.get('divergence_target_mult_regular', 'N/A'):.2f}")
        
        # === FISHER TRANSFORM OPTIMIZATION (ENHANCED) ===
        logger.info(f"├─ 🎣 FISHER TRANSFORM OPTIMIZATION (Enhanced)")
        logger.info(f"│")
        logger.info(f"│  ┌─ Core Parameters")
        logger.info(f"│  │  • fisher_period: {params.get('fisher_period', 'N/A')}")
        logger.info(f"│  │  • fisher_oversold: {params.get('fisher_oversold', 'N/A')}")
        logger.info(f"│  │  • fisher_overbought: {params.get('fisher_overbought', 'N/A')}")
        logger.info(f"│  │  • fisher_rsi_max_long: {params.get('fisher_rsi_max_long', 'N/A')}")
        logger.info(f"│  │  • fisher_rsi_min_short: {params.get('fisher_rsi_min_short', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ 🌡️ Market Regime Filters (NEW)")
        logger.info(f"│  │  • ADX range: {params.get('fisher_min_adx', 'N/A')}-{params.get('fisher_max_adx', 'N/A')}")
        logger.info(f"│  │  • Volatility range: {params.get('fisher_min_volatility', 'N/A'):.4f}-{params.get('fisher_max_volatility', 'N/A'):.4f}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ ✅ Confirmation Requirements (NEW)")
        logger.info(f"│  │  • require_volume: {params.get('fisher_require_volume', 'N/A')}")
        logger.info(f"│  │  • volume_mult: {params.get('fisher_volume_mult', 'N/A')}")
        logger.info(f"│  │  • require_1h_alignment: {params.get('fisher_require_1h_alignment', 'N/A')}")
        logger.info(f"│  │  • require_price_action: {params.get('fisher_require_price_action', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ 💪 Signal Strength (NEW)")
        logger.info(f"│  │  • min_signal_strength: {params.get('fisher_min_signal_strength', 'N/A'):.2f}")
        logger.info(f"│  │  • require_momentum: {params.get('fisher_require_momentum', 'N/A')}")
        logger.info(f"│  │  • momentum_lookback: {params.get('fisher_momentum_lookback', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ ⏱️ Entry Timing (NEW)")
        logger.info(f"│  │  • entry_wait_candles: {params.get('fisher_entry_wait_candles', 'N/A')}")
        logger.info(f"│  │  • require_reversal_candle: {params.get('fisher_require_reversal_candle', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  └─ 🎯 Risk Management (NEW)")
        logger.info(f"│  │  • stop_atr_mult: {params.get('fisher_stop_atr_mult', 'N/A'):.2f}")
        logger.info(f"│  └─ • profit_target_mult: {params.get('fisher_profit_target_mult', 'N/A'):.2f}")
        
        # === OTHER SIGNAL INDICATORS ===
        logger.info(f"├─ ┌─ 📉 OTHER SIGNAL INDICATORS")
        logger.info(f"│  │  • MACD: fast={params.get('macd_fast', 'N/A')}, slow={params.get('macd_slow', 'N/A')}, signal={params.get('macd_signal', 'N/A')}")
        logger.info(f"│  │  • Stochastic: k={params.get('stoch_k', 'N/A')}, d={params.get('stoch_d', 'N/A')}")
        logger.info(f"│  │  • Williams %R: {params.get('willr_period', 'N/A')}")
        logger.info(f"│  │  • CCI: {params.get('cci_period', 'N/A')}")
        logger.info(f"│  │  • Momentum: {params.get('mom_period', 'N/A')}")
        logger.info(f"│  │  • CMF: {params.get('cmf_period', 'N/A')}")
        logger.info(f"│  │  • NATR: {params.get('natr_period', 'N/A')}")
        logger.info(f"│  └─ • Choppiness: {params.get('chop_period', 'N/A')}")
        
        # === GLOBAL EXIT OPTIMIZATION ===
        logger.info(f"├─ 🚪 GLOBAL EXIT OPTIMIZATION")
        logger.info(f"│")
        logger.info(f"│  ┌─ Session Multipliers")
        logger.info(f"│  │  • overlap: {params.get('session_multiplier_overlap', 'N/A'):.2f}")
        logger.info(f"│  │  • major: {params.get('session_multiplier_major', 'N/A'):.2f}")
        logger.info(f"│  │  • volatility_sensitivity: {params.get('volatility_sensitivity', 'N/A')}")
        logger.info(f"│  │  • high_signal_mult: {params.get('high_signal_multiplier', 'N/A'):.2f}")
        logger.info(f"│  │  • low_signal_mult: {params.get('low_signal_multiplier', 'N/A'):.2f}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ Base Targets")
        logger.info(f"│  │  • medium_term_base: {params.get('medium_term_base_target', 'N/A'):.3f} ({params.get('medium_term_base_target', 0)*100:.1f}%)")
        logger.info(f"│  │  • trend_profit: {params.get('trend_profit_target', 'N/A'):.3f} ({params.get('trend_profit_target', 0)*100:.1f}%)")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ RSI Exit Conditions")
        logger.info(f"│  │  • overbought: {params.get('rsi_exit_overbought', 'N/A')}")
        logger.info(f"│  │  • oversold: {params.get('rsi_exit_oversold', 'N/A')}")
        logger.info(f"│  │  • extreme_min_profit: {params.get('rsi_extreme_min_profit', 'N/A'):.3f} ({params.get('rsi_extreme_min_profit', 0)*100:.1f}%)")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ Momentum Exit")
        logger.info(f"│  │  • fade_min_profit: {params.get('momentum_fade_min_profit', 'N/A'):.3f} ({params.get('momentum_fade_min_profit', 0)*100:.1f}%)")
        logger.info(f"│  │  • fade_threshold: {params.get('momentum_fade_threshold', 'N/A'):.3f} ({params.get('momentum_fade_threshold', 0)*100:.1f}%)")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ Extended Duration Management")
        logger.info(f"│  │  • 2hr target: {params.get('extended_2hr_target', 'N/A'):.3f} ({params.get('extended_2hr_target', 0)*100:.1f}%)")
        logger.info(f"│  │  • 3hr target: {params.get('extended_3hr_target', 'N/A'):.3f} ({params.get('extended_3hr_target', 0)*100:.1f}%)")
        logger.info(f"│  │  • 5hr target: {params.get('extended_5hr_target', 'N/A'):.3f} ({params.get('extended_5hr_target', 0)*100:.1f}%)")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ Market Protection")
        logger.info(f"│  │  • friday_close: {params.get('friday_close_min_profit', 'N/A'):.3f} ({params.get('friday_close_min_profit', 0)*100:.1f}%)")
        logger.info(f"│  │  • overnight: {params.get('overnight_min_profit', 'N/A'):.3f} ({params.get('overnight_min_profit', 0)*100:.1f}%)")
        logger.info(f"│  │  • low_liquidity: {params.get('low_liquidity_min_profit', 'N/A'):.3f} ({params.get('low_liquidity_min_profit', 0)*100:.1f}%)")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ Hybrid Exit Parameters")
        logger.info(f"│  │  • band_exit_tolerance: {params.get('band_exit_tolerance', 'N/A'):.4f}")
        logger.info(f"│  │  • atr_trailing_mult: {params.get('atr_trailing_multiplier', 'N/A'):.2f}")
        logger.info(f"│  │  • atr_trail_min_profit: {params.get('atr_trail_min_profit', 'N/A'):.3f} ({params.get('atr_trail_min_profit', 0)*100:.1f}%)")
        logger.info(f"│  │  • emergency_exit_pct: {params.get('emergency_exit_pct', 'N/A'):.2f} ({params.get('emergency_exit_pct', 0)*100:.0f}%)")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ Sideways Market Detection")
        logger.info(f"│  │  • adx_max: {params.get('sideways_adx_max', 'N/A')}")
        logger.info(f"│  │  • kc_width_max: {params.get('sideways_kc_width_max', 'N/A'):.4f}")
        logger.info(f"│  │  • ema_flat_max: {params.get('sideways_ema_flat_max', 'N/A'):.4f}")
        logger.info(f"│  │  • range_max: {params.get('sideways_range_max', 'N/A'):.3f}")
        logger.info(f"│  │  • min_conditions: {params.get('sideways_min_conditions', 'N/A')}")
        logger.info(f"│  │")
        logger.info(f"│  ┌─ Reversal Exits")
        logger.info(f"│  │  • min_profit_long: {params.get('reversal_min_profit_long', 'N/A'):.3f} ({params.get('reversal_min_profit_long', 0)*100:.1f}%)")
        logger.info(f"│  │  • min_profit_short: {params.get('reversal_min_profit_short', 'N/A'):.3f} ({params.get('reversal_min_profit_short', 0)*100:.1f}%)")
        logger.info(f"│  │")
        logger.info(f"│  └─ Trend Following Exit")
        logger.info(f"│  │  • ema_break_periods: {params.get('trend_ema_break_periods', 'N/A')}")
        logger.info(f"│  └─ • atr_multiplier: {params.get('atr_multiplier', 'N/A'):.2f}")
        
        # === OPTIMIZATION SUMMARY ===
        logger.info(f"├─ 📋 OPTIMIZATION SUMMARY")
        
        # Count parameters by category
        divergence_params = len([k for k in params.keys() if 'divergence_' in k or k in ['window', 'index_range']])
        fisher_params = len([k for k in params.keys() if 'fisher_' in k])
        exit_params = len([k for k in params.keys() if any(x in k for x in [
            'session_', 'extended_', 'friday_', 'overnight_', 'low_liquidity_',
            'band_exit', 'atr_trailing', 'emergency_exit', 'sideways_',
            'reversal_', 'trend_ema', 'momentum_fade', 'rsi_exit', 'medium_term'
        ])])
        indicator_params = len([k for k in params.keys() if any(x in k for x in [
            'rsi_period', 'atr_period', 'bb_period', 'bb_std', 'macd_', 'ema_',
            'stoch_', 'willr_', 'cci_', 'mom_', 'cmf_', 'natr_', 'chop_', 'volume_sma'
        ])])
        filter_params = len([k for k in params.keys() if any(x in k for x in [
            'min_signal_strength', 'volume_threshold', 'volume_factor',
            'adx_threshold', 'min_volatility', 'max_volatility',
            'rsi_overbought', 'rsi_oversold'
        ])])
        hq_params = len([k for k in params.keys() if 'hq_' in k])
        
        logger.info(f"│  • Divergence parameters: {divergence_params} (ENHANCED with regime filters)")
        logger.info(f"│  • Fisher Transform parameters: {fisher_params} (ENHANCED with regime filters)")
        logger.info(f"│  • HQ Divergence filters: {hq_params}")
        logger.info(f"│  • Exit parameters: {exit_params}")
        logger.info(f"│  • Indicator parameters: {indicator_params}")
        logger.info(f"│  • Global filter parameters: {filter_params}")
        logger.info(f"│  • Total parameters: {len(params)}")
        logger.info(f"└─ 🎯 Live optimization on {len(params)} parameters for {pair}")
        logger.info(f"✅ [OPTUNA] Parameter logging complete for {pair}")

    def pretrain_with_historical_data(self, pair: str, strategy_instance) -> bool:
        """Stub for compatibility - real Optuna doesn't need pre-training"""
        return False

class PlotConfig():
    def __init__(self):
        self.config = {
            'main_plot': {
                # Try direct column names first to test
                'bollinger_upperband': {'color': 'rgba(4,137,122,0.7)'},
                'kc_upperband': {'color': 'rgba(4,146,250,0.7)'},
                'kc_middleband': {'color': 'rgba(4,146,250,0.7)'},
                'kc_lowerband': {'color': 'rgba(4,146,250,0.7)'},
                'bollinger_lowerband': {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': 'bollinger_upperband',
                    'fill_color': 'rgba(4,137,122,0.07)'
                },
                'ema9': {'color': 'purple'},
                'ema20': {'color': 'yellow'},
                'ema50': {'color': 'red'},
                'ema200': {'color': 'white'},
                'trend_1h_1h': {'color': 'orange'},
            },
            'subplots': {
                "RSI": {
                    'rsi': {'color': 'green'}
                },
                "ATR": {
                    'atr': {'color': 'firebrick'}
                },
                "Signal Strength": {
                    'signal_strength': {'color': 'blue'}
                }
            }
        }
    
    def add_total_divergences_in_config(self, dataframe):
        # Test if columns exist before adding them
        if 'total_bullish_divergences' in dataframe.columns:
            self.config['main_plot']['total_bullish_divergences'] = {
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond',
                        'size': 11,
                        'color': 'green'
                    }
                }
            }
        
        if 'total_bearish_divergences' in dataframe.columns:
            self.config['main_plot']['total_bearish_divergences'] = {
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond',
                        'size': 11,
                        'color': 'crimson'
                    }
                }
            }
        
        return self

class AlexBandSniperV10AI(IStrategy):
    """
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                            Advanced RL + Optuna Trading Strategy                                       ║
    ║                                               Enhanced Multi-Pair Optimization                                         ║
    ║                                                 Machine Learning Powered                                               ║
    ║                                                                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    Alex BandSniper on 15m Timeframe - OPTIMIZED VERSION WITH OPTUNA INTEGRATION
    Version 9-Optuna - Claude optimized Entry & Exit with Optuna Management  # CHANGED: Versionshinweis angepasst
    Key improvements:
    --Divergence Optuna Full Integration
    --Fisher Optuna Full Integration
    --Full Optuna Integration
    --Fixed Repainting
    - Fixed Entry Tags Assignment
    - Added better Custom_Exit
    - Enabled ROI and TRAILING
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
        return "V97-Optuna"

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.15, -0.03, decimals=2, name='stoploss')]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(3, 8, name='max_open_trades'),
            ]
        
        def trailing_space() -> List[Dimension]:
            return [
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.02, 0.3, decimals=2, name='trailing_stop_positive'),
                SKDecimal(0.03, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),
                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]
    
    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.99  # Disabled - let custom_exit handle everything intelligently
    }
    
    # Optimal stoploss designed for the strategy.
    stoploss = -0.25
    can_short = True
    use_custom_stoploss = False
    leverage_value = 7.0  # Reduced leverage for better risk management
    trailing_stop = False
    #trailing_stop_positive = 0.04  # Activate trailing at 4% profit
    #trailing_stop_positive_offset = 0.041
    #trailing_only_offset_is_reached = True  # Only trail after 4% reached
    # Optimal timeframe for the strategy.
    timeframe = '15m'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "exit_pricing" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_exits = True
    use_entry_signal = True
    # In your hyperopt parameters, consider these more permissive defaults:
    # === HYBRID EXIT PARAMETERS ===
    # Band exit settings
    use_band_exits = BooleanParameter(default=True, space='sell', optimize=False)
    band_exit_tolerance = DecimalParameter(0.995, 1.005, default=0.998, decimals=3, space='sell', optimize=True, load=True)

    # ATR trailing settings
    use_atr_trailing = BooleanParameter(default=True, space='sell', optimize=False)
    atr_trailing_multiplier = DecimalParameter(1.5, 3.5, default=2.0, decimals=1, space='sell', optimize=True, load=True)
    atr_trail_min_profit = DecimalParameter(0.01, 0.03, default=0.015, decimals=3, space='sell', optimize=True, load=True)

    # RSI momentum exit settings
    use_rsi_exits = BooleanParameter(default=True, space='sell', optimize=False)
    rsi_exit_extreme = IntParameter(70, 85, default=75, space='sell', optimize=True, load=True)
    rsi_exit_extreme_profit = DecimalParameter(0.02, 0.05, default=0.03, decimals=3, space='sell', optimize=True, load=True)

    # Emergency profit taking
    emergency_exit_pct = DecimalParameter(0.12, 0.20, default=0.15, decimals=2, space='sell', optimize=True, load=True)
    min_signal_strength = IntParameter(1, 5, default=1, space='buy', optimize=False, load=True)   # Reduced from 3-10
    volume_threshold = DecimalParameter(1.0, 1.5, default=1.0, decimals=1, space='buy', optimize=True, load=True)  # Reduced from 1.1-2.5

    # Make ADX less restrictive
    adx_threshold = IntParameter(15, 30, default=15, space='buy', optimize=True, load=True)  # Reduced from 25-45
  
    # Market Condition Filters
    rsi_overbought = DecimalParameter(65.0, 85.0, default=80.0, decimals=1, space='buy', optimize=True, load=True)
    rsi_oversold = DecimalParameter(15.0, 35.0, default=15.0, decimals=1, space='buy', optimize=True, load=True)
    
    # Volatility Filters
    max_volatility = DecimalParameter(0.015, 0.035, default=0.025, decimals=3, space='buy', optimize=True, load=True)
    min_volatility = DecimalParameter(0.003, 0.008, default=0.005, decimals=3, space='buy', optimize=True, load=True)
    
    # Exit Parameters
    rsi_exit_overbought = DecimalParameter(70.0, 90.0, default=80.0, decimals=1, space='sell', optimize=True, load=True)
    rsi_exit_oversold = DecimalParameter(10.0, 30.0, default=20.0, decimals=1, space='sell', optimize=True, load=True)
    adx_exit_threshold = IntParameter(15, 30, default=20, space='sell', optimize=True, load=True)
    
    # Trend Confirmation Parameters
    trend_strength_threshold = IntParameter(20, 40, default=25, space='buy', optimize=True, load=True)
    
    # Technical Parameters
    window = IntParameter(3, 6, default=4, space="buy", optimize=True, load=True)
    index_range = IntParameter(20, 50, default=30, space='buy', optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 10

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    use_cooldown_protection = BooleanParameter(default=False, space="protection", optimize=True)

    # Enhanced protection parameters
    use_max_drawdown_protection = BooleanParameter(default=False, space="protection", optimize=True)
    max_drawdown_lookback = IntParameter(100, 300, default=200, space="protection", optimize=True)
    max_drawdown_trade_limit = IntParameter(5, 15, default=10, space="protection", optimize=True)
    max_drawdown_stop_duration = IntParameter(1, 5, default=1, space="protection", optimize=True)
    max_allowed_drawdown = DecimalParameter(0.08, 0.25, default=0.15, decimals=2, space="protection", optimize=True)

    stoploss_guard_lookback = IntParameter(30, 80, default=50, space="protection", optimize=True)
    stoploss_guard_trade_limit = IntParameter(2, 6, default=3, space="protection", optimize=True)
    stoploss_guard_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True)
    # Fisher Transform parameters


    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    def __init__(self, config: dict):
        # Signal Performance Tracking
        self.signal_performance_file = Path("user_data/strategies/AlexBandSniper_Signal_Performance.csv")
        self.signal_performance = self.load_signal_performance()
        self.signal_locks = {}
        
        # Configurable parameters - add these to your get_default_params()
        self.performance_window = 20        # More data points (was 15)
        self.min_trades_for_eval = 12      # More trades before evaluation (was 8)  
        self.min_win_rate = 0.25           # 25% win rate (was 35%)
        self.min_avg_profit = -0.025       # -2.5% average (was -1.5%)
        self.lock_duration_hours = 8       # Shorter lock period (was 12)
        
        # ============================================================================
        # ML PREDICTIVE ENTRY SYSTEM INITIALIZATION - MUST BE BEFORE super().__init__
        # ============================================================================
        self.ml_enabled = True  # Set False to disable ML predictions
        self.ml_model = None
        self.ml_trained = False
        self.ml_model_trained = False  # Flag for backtest training
        self.ml_training_data = []
        self.ml_feature_names = [
            'rsi', 'adx', 'atr_norm', 'volume_ratio', 'volatility',
            'ema_alignment', 'bb_position', 'signal_strength',
            'close_vs_ema21', 'close_vs_ema50', 'trend_strength',
            'momentum', 'divergence_present', 'recent_profit_avg'
        ]
        super().__init__(config)
        print(ASCII_ART, flush=True)
        logger.info(f"🚀 [STRATEGY] Initializing {self.__class__.__name__}")  # AUTO-UPDATED: Klassenname wird automatisch angepasst
        
        # Initialize simple Optuna manager replacement
        try:
            self.optuna_manager = RealOptunaManager("AlexBandSniperV10AI")  # CHANGED: Strategy name angepasst
            logger.info(f"✅ [OPTUNA] Successfully initialized RealOptunaManager")
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to initialize RealOptunaManager: {e}")
            self.optuna_manager = None
        
        # Per-coin optimized parameters
        self.coin_params: Dict[str, Dict] = {}  # ADDED: Dictionary für coin-spezifische Parameter
        
        # Track performance for optimization
        self.coin_performance: Dict[str, float] = {}  # ADDED: Performance-Tracking pro Coin
        
        # Optimization settings - CHANGED: Enable proactive optimization
        self.enable_dynamic_optimization = True  # ADDED: Dynamische Optimierung aktiviert (deaktivieren für Backtest)
        self.optimization_min_trades = 0  # CHANGED: No minimum trades required for startup optimization
        
        logger.info(f"⚙️  [STRATEGY] Dynamic optimization: {'ENABLED' if self.enable_dynamic_optimization else 'DISABLED'}")
        logger.info(f"⚙️  [STRATEGY] Minimum trades for optimization: {self.optimization_min_trades}")
        logger.info(f"✅ [STRATEGY] {self.__class__.__name__} initialization completed")
        
        # ML will be initialized in bot_start() where dp is available
        if ML_AVAILABLE and self.ml_enabled:
            logger.info("🔍 [INIT] ML system enabled - training will happen in bot_start()")
        else:
            logger.info(f"🧠 [INIT] ML predictions disabled")
    def bot_start(self, **kwargs) -> None:
        """
        Called once when the bot starts (after everything is initialized)
        Perfect place to do ML training since DataProvider is ready
        """
        if ML_AVAILABLE and self.ml_enabled:
            logger.info("🔍 [BOT_START] Checking ML model status...")
            
            # Try to load existing model first
            self.validate_and_load_ml_model()
            self.load_ml_training_data()
            logger.info(f"📊 [BOT_START] After loading: ml_trained={self.ml_trained}, samples={len(self.ml_training_data)}")
            
            if not self.ml_trained:
                logger.info("🎯 [BOT_START] No ML model found - starting backtest training...")
                success = self.train_ml_from_backtest()
                
                if success:
                    logger.info("✅ [BOT_START] ML training successful!")
                    self.ml_model_trained = True  # Set flag so trades can begin
                    self.ml_trained = True
                else:
                    logger.warning("⚠️ [BOT_START] Backtest training failed - ML disabled")
                    self.ml_enabled = False
                    self.ml_model_trained = False
            else:
                logger.info(f"✅ [BOT_START] ML model already trained with {len(self.ml_training_data)} samples")
                self.ml_model_trained = True  # Set flag since model is loaded

# ============================================================================
    # CUSTOM INDICATOR CALCULATION METHODS (From AlexSmartMoneyAdaptive)
    # ============================================================================
    
    def JMA(self, src, length, phase, power):
        """
        Jurik Moving Average (JMA)
        A smoothed moving average that reduces lag
        
        Args:
            src: Price series (typically close)
            length: Period for JMA
            phase: Phase parameter (typically 0)
            power: Power parameter (typically 2)
        
        Returns:
            pandas.Series: JMA values
        """
        beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
        alpha = beta ** power
        e0, e1, e2, jma = [0.0] * len(src), [0.0] * len(src), [0.0] * len(src), [0.0] * len(src)
        
        for i in range(1, len(src)):
            e0[i] = (1 - alpha) * src[i] + alpha * e0[i - 1]
            e1[i] = (src[i] - e0[i]) * (1 - beta) + beta * e1[i - 1]
            e2[i] = (e0[i] + phase * e1[i] - jma[i - 1]) * (1 - alpha) ** 2 + alpha ** 2 * e2[i - 1]
            jma[i] = e2[i] + jma[i - 1]
        
        return pd.Series(jma, index=src.index)
    
    def calculate_adx(self, dataframe, length):
        """
        Calculate ADX (Average Directional Index) with DI+ and DI-
        
        Args:
            dataframe: DataFrame with high, low, close columns
            length: Period for ADX calculation
        
        Returns:
            tuple: (di_plus, di_minus, adx) as pandas Series
        """
        high, low, close = dataframe['high'], dataframe['low'], dataframe['close']
        
        # Calculate True Range
        tr = pd.concat([
            high - low, 
            abs(high - close.shift()), 
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        tr_sma = tr.rolling(window=length).mean()
        
        # Calculate Directional Movement
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        
        # Only keep the larger directional movement
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        
        # Smooth the directional movements
        plus_dm_sma = plus_dm.rolling(window=length).mean()
        minus_dm_sma = minus_dm.rolling(window=length).mean()
        
        # Calculate Directional Indicators
        di_plus = 100 * (plus_dm_sma / tr_sma)
        di_minus = 100 * (minus_dm_sma / tr_sma)
        
        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=length).mean()
        
        return di_plus, di_minus, adx
    
    # ============================================================================
    # END OF CUSTOM INDICATOR METHODS
    # ============================================================================
    def get_coin_params(self, pair: str) -> Dict[str, Any]:
        """Get optimized parameters for specific coin using real Optuna"""
        if pair not in self.coin_params:
            logger.debug(f"🔍 [OPTUNA] Loading parameters for new pair: {pair}")
            
            # Check if optimization is disabled first (for backtesting)
            if not self.enable_dynamic_optimization:
                logger.info(f"⏸️ [OPTUNA] Dynamic optimization DISABLED for {pair} - using default parameters")
                self.coin_params[pair] = self.get_default_params()
                return self.coin_params[pair]
            
            # Real Optuna optimization - only if enabled
            if self.enable_dynamic_optimization and self.optuna_manager:
                logger.info(f"🚀 [OPTUNA] Real Optuna will optimize {pair} based on actual trade results")
            
            if self.optuna_manager:
                best_params = self.optuna_manager.get_best_params(pair)
                
                if best_params:
                    self.coin_params[pair] = best_params
                    logger.info(f"✅ [OPTUNA] Loaded optimized parameters for {pair}: {len(best_params)} params")
                    logger.debug(f"📊 [OPTUNA] Parameters from study with {len(self.optuna_manager.studies.get(pair, {}).trials) if pair in self.optuna_manager.studies else 0} trials")
                else:
                    self.coin_params[pair] = self.get_default_params()
                    logger.info(f"🔧 [OPTUNA] Using default parameters for {pair} (no study found - will optimize after trades)")
                    
                    # ADD THIS SECTION: Force startup optimization for new pairs
                    if self.enable_dynamic_optimization and self.optuna_manager:
                        logger.info(f"🎯 [OPTUNA] Triggering STARTUP OPTIMIZATION for new pair: {pair}")
                        try:
                            # Call maybe_optimize_coin with force_startup=True
                            self.maybe_optimize_coin(pair, force_startup=True)
                            
                            # After optimization, try to reload the optimized parameters
                            optimized_params = self.optuna_manager.get_best_params(pair)
                            if optimized_params:
                                self.coin_params[pair] = optimized_params
                                logger.info(f"✅ [OPTUNA] Successfully loaded STARTUP optimized parameters for {pair}")
                            else:
                                logger.warning(f"⚠️ [OPTUNA] Startup optimization completed but no parameters returned for {pair}")
                        except Exception as e:
                            logger.error(f"❌ [OPTUNA] Startup optimization failed for {pair}: {e}")
                            # Keep using default parameters if optimization fails
            else:
                self.coin_params[pair] = self.get_default_params()
                logger.warning(f"⚠️ [OPTUNA] RealOptunaManager not available, using default parameters for {pair}")
        
        # ENSURE ALL PARAMETERS ARE PRESENT
        default_params = self.get_default_params()
        current_params = self.coin_params.get(pair, {})
        
        # Merge defaults with current params (current params override defaults)
        complete_params = {**default_params, **current_params}
        self.coin_params[pair] = complete_params
        
        logger.debug(f"♻️ [OPTUNA] Using complete parameter set for {pair}")
        return self.coin_params[pair]
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get enhanced default parameters - aligned with Optuna optimization"""
        return {
            # ============================================
            # CORE SHARED INDICATORS
            # ============================================
            'rsi_period': 14,
            'atr_period': 14,
            'volume_sma_period': 20,
            'ema_short_period': 8,
            'ema_long_period': 21,
            'adx_threshold': 20,
            'bb_period': 20,
            'bb_std': 2.0,
            
            # ============================================
            # GLOBAL FILTERS
            # ============================================
            'min_signal_strength': 3,
            'volume_threshold': 1.2,
            'rsi_overbought': 75.0,
            'rsi_oversold': 25.0,
            'max_volatility': 0.025,
            'min_volatility': 0.005,
            'volume_factor': 1.5,
            
            # ============================================
            # DIVERGENCE PARAMETERS (Enhanced)
            # ============================================
            
            # Core Divergence Detection
            'window': 4,
            'index_range': 30,
            'divergence_lookback_period': 5,
            'divergence_min_rsi_delta': 4.0,
            'divergence_min_price_delta': 1.0,
            
            # Divergence Thresholds
            'divergence_rsi_low_threshold': 30,
            'divergence_rsi_high_threshold': 70,
            
            # === NEW: Market Regime Filters for Divergence ===
            'divergence_min_adx': 20,              # Avoid choppy markets
            'divergence_max_adx': 40,              # Avoid overly trending
            'divergence_min_atr_mult': 0.8,        # Min volatility
            'divergence_max_atr_mult': 3.0,        # Max volatility
            
            # === NEW: Divergence Confirmation Requirements ===
            'divergence_volume_multiplier': 1.5,
            'divergence_require_volume_increase': False,
            'divergence_require_trend_context': True,
            'divergence_require_1h_alignment': False,
            'divergence_require_bands': False,
            
            # === NEW: Divergence Quality Scoring ===
            'divergence_price_strength_weight': 1.0,
            'divergence_rsi_strength_weight': 1.0,
            'divergence_timespan_weight': 1.0,
            'divergence_min_quality_score': 5.0,
            
            # === NEW: Entry Timing for Divergence ===
            'divergence_entry_wait_candles': 1,
            'divergence_require_pullback': False,
            
            # HQ Divergence Filters (LONG)
            'hq_adx_max': 30,
            'hq_momentum_lookback': 2,
            'hq_require_green_candle': True,
            'hq_ema_trend_filter': True,
            
            # HQ Divergence Filters (SHORT)
            'hq_adx_max_short': 30,
            'hq_momentum_lookback_short': 2,
            'hq_require_red_candle': True,
            'hq_ema_trend_filter_short': True,
            
            # === NEW: Quality-Based Risk Management ===
            'divergence_stop_atr_mult_super': 3.0,
            'divergence_stop_atr_mult_hq': 3.5,
            'divergence_stop_atr_mult_regular': 4.5,
            
            # === NEW: Quality-Based Profit Targets ===
            'divergence_target_mult_super': 2.0,
            'divergence_target_mult_hq': 1.5,
            'divergence_target_mult_regular': 1.3,
            
            # ============================================
            # FISHER TRANSFORM PARAMETERS (Enhanced)
            # ============================================
            
            # Core Fisher Parameters
            'fisher_period': 9,
            'fisher_oversold': -2.0,
            'fisher_overbought': 2.0,
            'fisher_rsi_max_long': 40,
            'fisher_rsi_min_short': 60,
            
            # === NEW: Fisher Market Regime Filters ===
            'fisher_min_adx': 20,
            'fisher_max_adx': 40,
            'fisher_min_volatility': 0.005,
            'fisher_max_volatility': 0.025,
            
            # === NEW: Fisher Confirmation Requirements ===
            'fisher_require_volume': False,
            'fisher_volume_mult': 1.5,
            'fisher_require_1h_alignment': False,
            'fisher_require_price_action': False,
            
            # === NEW: Fisher Signal Strength ===
            'fisher_min_signal_strength': 2.0,
            'fisher_require_momentum': False,
            'fisher_momentum_lookback': 2,
            
            # === NEW: Fisher Entry Timing ===
            'fisher_entry_wait_candles': 0,
            'fisher_require_reversal_candle': False,
            
            # === NEW: Fisher Risk Management ===
            'fisher_stop_atr_mult': 3.5,
            'fisher_profit_target_mult': 1.5,
            
            # ============================================
            # OTHER SIGNAL INDICATORS
            # ============================================
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'stoch_k': 14,
            'stoch_d': 3,
            'willr_period': 14,
            'cci_period': 20,
            'mom_period': 10,
            'cmf_period': 20,
            'natr_period': 14,
            'chop_period': 14,
            
            # ============================================
            # GLOBAL EXIT OPTIMIZATION
            # ============================================
            
            # Session Multipliers
            'session_multiplier_overlap': 1.3,
            'session_multiplier_major': 1.15,
            'volatility_sensitivity': 40,
            'high_signal_multiplier': 1.2,
            'low_signal_multiplier': 0.8,
            
            # Base Targets
            'medium_term_base_target': 0.05,
            'trend_profit_target': 0.045,
            
            # RSI Exit Conditions
            'rsi_exit_overbought': 78,
            'rsi_exit_oversold': 22,
            'rsi_extreme_min_profit': 0.015,
            
            # Momentum Exit
            'momentum_fade_min_profit': 0.012,
            'momentum_fade_threshold': 0.02,
            
            # Extended Duration Management
            'extended_2hr_target': 0.025,
            'extended_3hr_target': 0.015,
            'extended_5hr_target': 0.008,
            
            # Market Protection
            'friday_close_min_profit': 0.012,
            'overnight_min_profit': 0.008,
            'low_liquidity_min_profit': 0.01,
            
            # Hybrid Exit Parameters
            'band_exit_tolerance': 0.998,
            'atr_trailing_multiplier': 2.0,
            'atr_trail_min_profit': 0.015,
            'emergency_exit_pct': 0.15,
            
            # Sideways Market Detection
            'sideways_adx_max': 25,
            'sideways_kc_width_max': 0.020,
            'sideways_ema_flat_max': 0.015,
            'sideways_range_max': 0.06,
            'sideways_min_conditions': 2,
            
            # Reversal Exits
            'reversal_min_profit_long': 0.008,
            'reversal_min_profit_short': 0.008,
            
            # Trend Following Exit
            'trend_ema_break_periods': 3,
            'atr_multiplier': 3.0,
            
            # ============================================
            # SIGNAL LOCKING (Keep your existing logic)
            # ============================================
            'signal_lock_enabled': True,
            'signal_performance_window': 15,
            'signal_lock_duration_hours': 12,
            'signal_min_win_rate': 0.35,
            'signal_min_avg_profit': -0.015,
            'signal_min_trades_for_eval': 3,
        }

    def create_objective_function(self, pair: str):
        """Create objective function that optimizes on LIVE trade performance"""
        def objective(trial):
            params = {
                # ============================================
                # CORE SHARED INDICATORS (Used by multiple signals)
                # ============================================
                'rsi_period': trial.suggest_int('rsi_period', 10, 20),
                'atr_period': trial.suggest_int('atr_period', 10, 20),
                'volume_sma_period': trial.suggest_int('volume_sma_period', 15, 25),
                'ema_short_period': trial.suggest_int('ema_short_period', 6, 12),
                'ema_long_period': trial.suggest_int('ema_long_period', 18, 25),
                'adx_threshold': trial.suggest_int('adx_threshold', 15, 30),
                'bb_period': trial.suggest_int('bb_period', 15, 25),
                'bb_std': trial.suggest_float('bb_std', 1.8, 2.5),
                
                # ============================================
                # GLOBAL FILTERS (Apply to all signals)
                # ============================================
                'min_signal_strength': trial.suggest_int('min_signal_strength', 1, 5),
                'volume_threshold': trial.suggest_float('volume_threshold', 1.0, 1.5),
                'rsi_overbought': trial.suggest_float('rsi_overbought', 65.0, 85.0),
                'rsi_oversold': trial.suggest_float('rsi_oversold', 15.0, 35.0),
                'max_volatility': trial.suggest_float('max_volatility', 0.015, 0.035),
                'min_volatility': trial.suggest_float('min_volatility', 0.003, 0.008),
                'volume_factor': trial.suggest_float('volume_factor', 1.2, 2.0),
                
                # ============================================
                # DIVERGENCE OPTIMIZATION (Enhanced)
                # ============================================
                
                # Core Divergence Detection
                'window': trial.suggest_int('window', 3, 6),
                'index_range': trial.suggest_int('index_range', 20, 50),
                'divergence_lookback_period': trial.suggest_int('divergence_lookback_period', 3, 10),
                'divergence_min_rsi_delta': trial.suggest_float('divergence_min_rsi_delta', 2.0, 8.0, step=0.5),
                'divergence_min_price_delta': trial.suggest_float('divergence_min_price_delta', 0.5, 3.0, step=0.25),
                
                # Divergence Thresholds
                'divergence_rsi_low_threshold': trial.suggest_int('divergence_rsi_low_threshold', 20, 35),
                'divergence_rsi_high_threshold': trial.suggest_int('divergence_rsi_high_threshold', 65, 80),
                
                # === NEW: Market Regime Filters for Divergence ===
                'divergence_min_adx': trial.suggest_int('divergence_min_adx', 15, 30),
                'divergence_max_adx': trial.suggest_int('divergence_max_adx', 35, 50),
                'divergence_min_atr_mult': trial.suggest_float('divergence_min_atr_mult', 0.5, 1.5),
                'divergence_max_atr_mult': trial.suggest_float('divergence_max_atr_mult', 2.0, 4.0),
                
                # === NEW: Divergence Confirmation Requirements ===
                'divergence_volume_multiplier': trial.suggest_float('divergence_volume_multiplier', 1.0, 2.5, step=0.1),
                'divergence_require_volume_increase': trial.suggest_categorical('divergence_require_volume_increase', [True, False]),
                'divergence_require_trend_context': trial.suggest_categorical('divergence_require_trend_context', [True, False]),
                'divergence_require_1h_alignment': trial.suggest_categorical('divergence_require_1h_alignment', [True, False]),
                'divergence_require_bands': trial.suggest_categorical('divergence_require_bands', [True, False]),
                
                # === NEW: Divergence Quality Scoring ===
                'divergence_price_strength_weight': trial.suggest_float('divergence_price_strength_weight', 0.5, 2.0),
                'divergence_rsi_strength_weight': trial.suggest_float('divergence_rsi_strength_weight', 0.5, 2.0),
                'divergence_timespan_weight': trial.suggest_float('divergence_timespan_weight', 0.5, 2.0),
                'divergence_min_quality_score': trial.suggest_float('divergence_min_quality_score', 3.0, 8.0),
                
                # === NEW: Entry Timing for Divergence ===
                'divergence_entry_wait_candles': trial.suggest_int('divergence_entry_wait_candles', 0, 3),
                'divergence_require_pullback': trial.suggest_categorical('divergence_require_pullback', [True, False]),
                
                # HQ Divergence Filters (LONG)
                'hq_adx_max': trial.suggest_int('hq_adx_max', 25, 35),
                'hq_momentum_lookback': trial.suggest_int('hq_momentum_lookback', 1, 3),
                'hq_require_green_candle': trial.suggest_categorical('hq_require_green_candle', [True, False]),
                'hq_ema_trend_filter': trial.suggest_categorical('hq_ema_trend_filter', [True, False]),
                
                # HQ Divergence Filters (SHORT)
                'hq_adx_max_short': trial.suggest_int('hq_adx_max_short', 25, 35),
                'hq_momentum_lookback_short': trial.suggest_int('hq_momentum_lookback_short', 1, 3),
                'hq_require_red_candle': trial.suggest_categorical('hq_require_red_candle', [True, False]),
                'hq_ema_trend_filter_short': trial.suggest_categorical('hq_ema_trend_filter_short', [True, False]),
                
                # === NEW: Quality-Based Risk Management ===
                'divergence_stop_atr_mult_super': trial.suggest_float('divergence_stop_atr_mult_super', 2.5, 4.0),
                'divergence_stop_atr_mult_hq': trial.suggest_float('divergence_stop_atr_mult_hq', 3.0, 5.0),
                'divergence_stop_atr_mult_regular': trial.suggest_float('divergence_stop_atr_mult_regular', 3.5, 6.0),
                
                # === NEW: Quality-Based Profit Targets ===
                'divergence_target_mult_super': trial.suggest_float('divergence_target_mult_super', 1.5, 3.0),
                'divergence_target_mult_hq': trial.suggest_float('divergence_target_mult_hq', 1.2, 2.5),
                'divergence_target_mult_regular': trial.suggest_float('divergence_target_mult_regular', 1.0, 2.0),
                
                # ============================================
                # FISHER TRANSFORM OPTIMIZATION (Enhanced)
                # ============================================
                
                # Core Fisher Parameters
                'fisher_period': trial.suggest_int('fisher_period', 5, 20),
                'fisher_oversold': trial.suggest_float('fisher_oversold', -4.0, -1.0, step=0.5),
                'fisher_overbought': trial.suggest_float('fisher_overbought', 1.0, 4.0, step=0.5),
                'fisher_rsi_max_long': trial.suggest_int('fisher_rsi_max_long', 35, 50),
                'fisher_rsi_min_short': trial.suggest_int('fisher_rsi_min_short', 50, 65),
                
                # === NEW: Fisher Market Regime Filters ===
                'fisher_min_adx': trial.suggest_int('fisher_min_adx', 15, 30),
                'fisher_max_adx': trial.suggest_int('fisher_max_adx', 35, 50),
                'fisher_min_volatility': trial.suggest_float('fisher_min_volatility', 0.003, 0.008),
                'fisher_max_volatility': trial.suggest_float('fisher_max_volatility', 0.015, 0.035),
                
                # === NEW: Fisher Confirmation Requirements ===
                'fisher_require_volume': trial.suggest_categorical('fisher_require_volume', [True, False]),
                'fisher_volume_mult': trial.suggest_float('fisher_volume_mult', 1.0, 2.5),
                'fisher_require_1h_alignment': trial.suggest_categorical('fisher_require_1h_alignment', [True, False]),
                'fisher_require_price_action': trial.suggest_categorical('fisher_require_price_action', [True, False]),
                
                # === NEW: Fisher Signal Strength ===
                'fisher_min_signal_strength': trial.suggest_float('fisher_min_signal_strength', 1.5, 3.5),
                'fisher_require_momentum': trial.suggest_categorical('fisher_require_momentum', [True, False]),
                'fisher_momentum_lookback': trial.suggest_int('fisher_momentum_lookback', 1, 3),
                
                # === NEW: Fisher Entry Timing ===
                'fisher_entry_wait_candles': trial.suggest_int('fisher_entry_wait_candles', 0, 2),
                'fisher_require_reversal_candle': trial.suggest_categorical('fisher_require_reversal_candle', [True, False]),
                
                # === NEW: Fisher Risk Management ===
                'fisher_stop_atr_mult': trial.suggest_float('fisher_stop_atr_mult', 2.5, 5.0),
                'fisher_profit_target_mult': trial.suggest_float('fisher_profit_target_mult', 1.2, 2.5),
                
                # ============================================
                # OTHER SIGNAL INDICATORS (Keep minimal)
                # ============================================
                'macd_fast': trial.suggest_int('macd_fast', 8, 16),
                'macd_slow': trial.suggest_int('macd_slow', 22, 30),
                'macd_signal': trial.suggest_int('macd_signal', 6, 12),
                'stoch_k': trial.suggest_int('stoch_k', 10, 18),
                'stoch_d': trial.suggest_int('stoch_d', 2, 5),
                'willr_period': trial.suggest_int('willr_period', 10, 20),
                'cci_period': trial.suggest_int('cci_period', 15, 25),
                'mom_period': trial.suggest_int('mom_period', 8, 15),
                'cmf_period': trial.suggest_int('cmf_period', 15, 25),
                'natr_period': trial.suggest_int('natr_period', 10, 18),
                'chop_period': trial.suggest_int('chop_period', 10, 18),
                
                # ============================================
                # GLOBAL EXIT OPTIMIZATION
                # ============================================
                
                # Session Multipliers
                'session_multiplier_overlap': trial.suggest_float('session_multiplier_overlap', 1.1, 1.5),
                'session_multiplier_major': trial.suggest_float('session_multiplier_major', 1.0, 1.3),
                'volatility_sensitivity': trial.suggest_int('volatility_sensitivity', 25, 60),
                'high_signal_multiplier': trial.suggest_float('high_signal_multiplier', 1.0, 1.4),
                'low_signal_multiplier': trial.suggest_float('low_signal_multiplier', 0.6, 1.0),
                
                # Base Targets
                'medium_term_base_target': trial.suggest_float('medium_term_base_target', 0.035, 0.065),
                'trend_profit_target': trial.suggest_float('trend_profit_target', 0.03, 0.06),
                
                # RSI Exit Conditions
                'rsi_exit_overbought': trial.suggest_float('rsi_exit_overbought', 72, 85),
                'rsi_exit_oversold': trial.suggest_float('rsi_exit_oversold', 15, 28),
                'rsi_extreme_min_profit': trial.suggest_float('rsi_extreme_min_profit', 0.01, 0.025),
                
                # Momentum Exit
                'momentum_fade_min_profit': trial.suggest_float('momentum_fade_min_profit', 0.008, 0.02),
                'momentum_fade_threshold': trial.suggest_float('momentum_fade_threshold', 0.015, 0.03),
                
                # Extended Duration Management
                'extended_2hr_target': trial.suggest_float('extended_2hr_target', 0.015, 0.035),
                'extended_3hr_target': trial.suggest_float('extended_3hr_target', 0.008, 0.025),
                'extended_5hr_target': trial.suggest_float('extended_5hr_target', 0.005, 0.015),
                
                # Market Protection
                'friday_close_min_profit': trial.suggest_float('friday_close_min_profit', 0.008, 0.02),
                'overnight_min_profit': trial.suggest_float('overnight_min_profit', 0.005, 0.015),
                'low_liquidity_min_profit': trial.suggest_float('low_liquidity_min_profit', 0.006, 0.018),
                
                # Hybrid Exit Parameters
                'band_exit_tolerance': trial.suggest_float('band_exit_tolerance', 0.995, 1.005, step=0.001),
                'atr_trailing_multiplier': trial.suggest_float('atr_trailing_multiplier', 1.5, 3.5, step=0.1),
                'atr_trail_min_profit': trial.suggest_float('atr_trail_min_profit', 0.01, 0.03, step=0.005),
                'emergency_exit_pct': trial.suggest_float('emergency_exit_pct', 0.12, 0.20, step=0.01),
                
                # Sideways Market Detection
                'sideways_adx_max': trial.suggest_int('sideways_adx_max', 20, 30),
                'sideways_kc_width_max': trial.suggest_float('sideways_kc_width_max', 0.015, 0.029, step=0.002),
                'sideways_ema_flat_max': trial.suggest_float('sideways_ema_flat_max', 0.010, 0.024, step=0.002),
                'sideways_range_max': trial.suggest_float('sideways_range_max', 0.04, 0.08, step=0.005),
                'sideways_min_conditions': trial.suggest_int('sideways_min_conditions', 2, 3),
                
                # Reversal Exits
                'reversal_min_profit_long': trial.suggest_float('reversal_min_profit_long', 0.005, 0.015),
                'reversal_min_profit_short': trial.suggest_float('reversal_min_profit_short', 0.005, 0.015),
                
                # Trend Following Exit
                'trend_ema_break_periods': trial.suggest_int('trend_ema_break_periods', 2, 5),
                'atr_multiplier': trial.suggest_float('atr_multiplier', 2.0, 4.0),
            }
            
            # ============================================
            # LIVE TRADE PERFORMANCE EVALUATION
            # ============================================
            
            recent_trades = self.get_recent_trade_performance(pair)
            
            # === FIX: Allow startup optimization with 0 trades ===
            MIN_TRADES_FOR_OPTIMIZATION = 2
            
            if len(recent_trades) < MIN_TRADES_FOR_OPTIMIZATION:
                # Not enough trades yet - return neutral score to allow parameter exploration
                logger.debug(f"⏳ [OPTUNA] Insufficient trades for {pair}: {len(recent_trades)}/{MIN_TRADES_FOR_OPTIMIZATION} - using exploration mode")
                return 0.0  # ← THIS IS THE KEY! Returns 0.0 instead of raising TrialPruned
            
            # Extract profit percentages from trades
            profits = [trade['profit_pct'] for trade in recent_trades]
            
            # Calculate score (average profit)
            score = sum(profits) / len(profits)
            
            return score
        
        return objective

    def get_recent_trade_performance(self, pair: str) -> List[float]:
        """Get recent trade performance for optimization"""
        try:
            from freqtrade.persistence import Trade
            trades = Trade.get_trades_proxy(pair=pair)
            
            if not trades:
                return []
            
            # Get last 5 trades for this pair
            recent_trades = trades[-5:] if len(trades) >= 5 else trades
            
            # Calculate profit ratios
            # Extract performance data
            performance = []
            for trade in recent_trades:
                if trade.close_date:  # Only closed trades
                    performance.append({
                        'profit_pct': trade.close_profit * 100,
                        'trade_id': trade.id,
                        'open_date': trade.open_date,
                        'close_date': trade.close_date
                    })
            
            logger.debug(f"📊 [OPTUNA] Found {len(performance)} completed trades for {pair}")
            return performance
            
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get trade performance for {pair}: {e}")
            return []
    def daily_optimization_check(self):
        """Enhanced optimization check with smarter scheduling"""
        try:
            if not self.optuna_manager:
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
                        self.optuna_manager.optimize_coin(pair, self.create_objective_function(pair), n_trials=60)
                        
                        # Track last optimization time
                        setattr(self, f'last_optimization_{pair.replace("/", "_").replace(":", "_")}', current_time)
                        optimized_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Optimization failed for {pair}: {e}")
            
            logger.info(f"✅ Optimization check completed: {optimized_count}/{len(all_pairs)} pairs optimized")
            
        except Exception as e:
            logger.error(f"❌ Daily optimization check failed: {e}")

    def should_retrain_pair(self, pair: str, current_time: float) -> bool:
        """Smart retraining logic with multiple safeguards"""
        from datetime import datetime
        
        last_opt_time = getattr(self, f'last_optimization_{pair.replace("/", "_").replace(":", "_")}', 0)
        trades_since_last = self.get_trades_since_optimization(pair, last_opt_time)
        
        # === TIMING RULES ===
        
        # 1. Get current hour (UTC)
        current_hour = datetime.utcnow().hour
        
        # 2. Only optimize during QUIET hours (2 AM - 6 AM UTC)
        is_quiet_time = 2 <= current_hour <= 6
        
        if not is_quiet_time:
            return False  # Don't optimize during active trading hours
        
        # 3. Minimum time between optimizations: 48 hours (2 days)
        time_since_last = current_time - last_opt_time
        min_time_between_opts = 172800  # 48 hours in seconds
        
        if time_since_last < min_time_between_opts:
            return False  # Too soon since last optimization
        
        # === DATA REQUIREMENTS ===
        
        # 4. Need at least 10 completed trades for meaningful optimization
        if trades_since_last < 10:
            return False  # Not enough data yet
        
        # === PERFORMANCE CHECK ===
        
        # 5. Check if performance is degrading
        recent_performance = self.get_recent_trade_performance(pair)
        
        if len(recent_performance) >= 5:
            avg_profit = sum(recent_performance) / len(recent_performance)
            win_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
            
            # Only optimize if performance is poor OR it's been a long time
            is_performing_poorly = (avg_profit < -0.005) or (win_rate < 0.4)
            been_very_long_time = time_since_last > 604800  # 7 days
            
            if is_performing_poorly:
                logger.info(f"🔴 {pair} performing poorly - triggering optimization (avg: {avg_profit:.4f}, WR: {win_rate:.2%})")
                return True
            elif been_very_long_time:
                logger.info(f"⏰ {pair} hasn't been optimized in 7+ days - triggering maintenance optimization")
                return True
            else:
                logger.debug(f"✅ {pair} performing well - skipping optimization (avg: {avg_profit:.4f}, WR: {win_rate:.2%})")
                return False
        
        # 6. If we don't have enough performance data but have trades, optimize
        if trades_since_last >= 10:
            logger.info(f"📊 {pair} has {trades_since_last} trades - triggering optimization")
            return True
        
        return False

    def get_trades_since_optimization(self, pair: str, last_opt_time: float) -> int:
        """Count trades since last optimization"""
        try:
            from freqtrade.persistence import Trade
            from datetime import datetime
            
            last_opt_datetime = datetime.fromtimestamp(last_opt_time) if last_opt_time > 0 else datetime.min
            trades = Trade.get_trades_proxy(pair=pair)
            
            if not trades:
                return 0
            
            recent_trades = [t for t in trades if t.open_date_utc > last_opt_datetime]
            return len(recent_trades)
            
        except Exception as e:
            logger.debug(f"Failed to count recent trades for {pair}: {e}")
            return 0
    def load_signal_performance(self):
        """Load signal performance history from CSV file"""
        if os.path.exists(self.signal_performance_file):
            try:
                all_trades = []
                
                with open(self.signal_performance_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    
                    for row in reader:
                        # Convert profit to float and win to boolean
                        trade_record = {
                            'pair': row['pair'],
                            'entry_tag': row['entry_tag'],
                            'profit': float(row['profit']),
                            'timestamp': row['timestamp'],
                            'win': row['win'].lower() == 'true'
                        }
                        all_trades.append(trade_record)
                
                # Store loaded trades
                self.all_signal_trades = all_trades
                
                # Rebuild the nested format for signal locking
                signal_perf = {}
                for trade in all_trades:
                    perf_key = f"{trade['pair']}_{trade['entry_tag']}"
                    if perf_key not in signal_perf:
                        signal_perf[perf_key] = []
                    
                    signal_perf[perf_key].append({
                        'profit': trade['profit'],
                        'timestamp': trade['timestamp'],
                        'win': trade['win']
                    })
                
                print(f"✅ Loaded {len(all_trades)} trades from CSV")
                return signal_perf
                
            except Exception as e:
                print(f"Failed to load CSV: {e}")
                return {}
        
        return {}
    def save_signal_performance(self):
        """Save signal performance to CSV file - Excel friendly"""
        try:
            os.makedirs(os.path.dirname(self.signal_performance_file), exist_ok=True)
            
            # Get all trades
            all_trades = getattr(self, 'all_signal_trades', [])
            
            if not all_trades:
                print("No trades to save")
                return
            
            # Write to CSV
            with open(self.signal_performance_file, 'w', newline='', encoding='utf-8') as f:
                # Define headers with semicolon separator
                fieldnames = ['pair', 'entry_tag', 'profit', 'timestamp', 'win']
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                
                # Write header
                writer.writeheader()
                
                # Write all trades
                for trade in all_trades:
                    writer.writerow(trade)
                    
            print(f"✅ Saved {len(all_trades)} trades to CSV")
            
        except Exception as e:
            print(f"Failed to save signal performance: {e}")
    def is_signal_locked(self, entry_tag: str, pair: str) -> bool:
        """Check if a signal is locked due to poor performance"""
        if not hasattr(self, 'signal_locks'):
            return False
        
        lock_key = f"{pair}_{entry_tag}"
        if lock_key not in self.signal_locks:
            return False
        
        # Check if lock has expired (12 hours)
        from datetime import timezone
        if self.signal_locks[lock_key] > datetime.now(timezone.utc):
            return True
    def update_signal_performance(self, entry_tag, pair, profit_ratio):
        """Update performance tracking for a signal - Excel-friendly format + locking support"""
        
        # Create trade record with all info in one line (Excel-friendly)
        trade_record = {
            'pair': pair,
            'entry_tag': entry_tag, 
            'profit': profit_ratio,
            'timestamp': datetime.now().isoformat(),
            'win': profit_ratio > 0
        }
        
        # Store in flat list for Excel analysis
        if not hasattr(self, 'all_signal_trades'):
            self.all_signal_trades = []
        
        self.all_signal_trades.append(trade_record)
        
        # Keep only recent window
        if len(self.all_signal_trades) > 500:
            self.all_signal_trades = self.all_signal_trades[-500:]
        
        # Keep the nested format for the locking logic to work
        perf_key = f"{pair}_{entry_tag}"
        if perf_key not in self.signal_performance:
            self.signal_performance[perf_key] = []
        
        self.signal_performance[perf_key].append({
            'profit': profit_ratio,
            'timestamp': datetime.now().isoformat(),
            'win': profit_ratio > 0
        })
        
        if len(self.signal_performance[perf_key]) > self.performance_window:
            self.signal_performance[perf_key] = self.signal_performance[perf_key][-self.performance_window:]
        
        # Evaluate if signal should be locked
        self.evaluate_signal_lock(entry_tag, pair)
        
        # Save to file
        self.save_signal_performance()
    
    def evaluate_signal_lock(self, entry_tag, pair):
        """Evaluate if signal should be locked based on recent performance"""
        perf_key = f"{pair}_{entry_tag}"
        recent_trades = self.signal_performance.get(perf_key, [])
        
        # Need minimum trades for evaluation
        if len(recent_trades) < self.min_trades_for_eval:
            return
        
        # Calculate performance metrics
        profits = [trade['profit'] for trade in recent_trades]
        wins = sum(1 for profit in profits if profit > 0)
        win_rate = wins / len(profits)
        avg_profit = sum(profits) / len(profits)
        
        # Check if signal should be locked
        should_lock = (
            win_rate < self.min_win_rate or 
            avg_profit < self.min_avg_profit
        )
        
        if should_lock:
            lock_until = datetime.now() + timedelta(hours=self.lock_duration_hours)
            lock_key = f"{pair}_{entry_tag}"
            self.signal_locks[lock_key] = lock_until
            
            print(f"LOCKED SIGNAL: {entry_tag} for {pair}")
            print(f"  Win Rate: {win_rate:.1%} (min: {self.min_win_rate:.1%})")
            print(f"  Avg Profit: {avg_profit:.2%} (min: {self.min_avg_profit:.2%})")
            print(f"  Locked until: {lock_until.strftime('%Y-%m-%d %H:%M')}")
    
    def get_signal_status(self, pair):
        """Get status of all signals for a pair - for debugging"""
        status = {}
        
        # Complete list of entry tags from your trading script
        entry_tags = [
            'Bull_E1', 'Bear_E1',
            'Bull_E2', 'Bear_E2',
            'Bull_Div_AGG', 'Bear_Div_AGG',
            'Bull_Div_CONS', 'Bear_Div_CONS',
            'Bull_Div_PB', 'Bear_Div_PB',
            'Bull_Div_1H', 'Bear_Div_1H',
            'Bull_HDiv_AGG', 'Bear_HDiv_AGG',
            'Bull_HDiv_CONS', 'Bear_HDiv_CONS',
            'Bull_HDiv_PB', 'Bear_HDiv_PB',
            'Bull_HDiv_1H', 'Bear_HDiv_1H',
            'Bull_RSV1', 'Bear_RSV1',
            'Bull_Scalp_EMA', 'Bear_Scalp_EMA',
            'Bull_Scalp_Retest', 'Bear_Scalp_Retest',
        ]
        
        for entry_tag in entry_tags:
            perf_key = f"{pair}_{entry_tag}"
            recent_trades = self.signal_performance.get(perf_key, [])
            
            is_locked = self.is_signal_locked(entry_tag, pair)
            
            if recent_trades:
                profits = [trade['profit'] for trade in recent_trades]
                wins = sum(1 for profit in profits if profit > 0)
                win_rate = wins / len(profits) if profits else 0
                avg_profit = sum(profits) / len(profits) if profits else 0
                
                status[entry_tag] = {
                    'locked': is_locked,
                    'trades_count': len(recent_trades),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'last_profit': profits[-1] if profits else None
                }
            else:
                status[entry_tag] = {
                    'locked': is_locked,
                    'trades_count': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'last_profit': None
                }
        
        return status

    def informative_pairs(self):
        """Define additional timeframes to download"""
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enhanced indicator population with Optuna optimization and live adaptation"""
        
        pair = metadata['pair']
        logger.debug(f"📊 [INDICATORS] Processing {pair} with {len(dataframe)} candles")

        # ===== GET OPTIMIZED PARAMETERS =====
        try:
            coin_params = self.get_coin_params(pair)
            logger.debug(f"📈 [OPTUNA] Applied optimized parameters for {pair}")
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get coin parameters for {pair}: {e}")
            coin_params = self.get_default_params()
        
        # ===== PERIODIC OPTIMIZATION CHECK (24h + Performance) =====
        if hasattr(self, 'optuna_manager') and self.optuna_manager:
            if self.optuna_manager.should_optimize(pair):
                from datetime import datetime
                
                # Check if this is a time-based trigger (24-hour periodic)
                if pair in self.optuna_manager.last_optimization_time:
                    hours_since_last = (datetime.now() - self.optuna_manager.last_optimization_time[pair]).total_seconds() / 3600
                    
                    if hours_since_last >= self.optuna_manager.optimization_interval_hours:
                        logger.info(f"⏰ [OPTUNA] {pair} is due for {self.optuna_manager.optimization_interval_hours}h periodic optimization!")
                        logger.info(f"📊 [OPTUNA] Last optimization: {hours_since_last:.1f}h ago")
                        
                        # Trigger optimization in background (non-blocking)
                        try:
                            self.maybe_optimize_coin(pair, force_startup=False)
                            
                            # Reload parameters after optimization
                            updated_params = self.optuna_manager.get_best_params(pair)
                            if updated_params:
                                coin_params = updated_params  # Update local variable
                                logger.info(f"✅ [OPTUNA] Reloaded fresh parameters after periodic optimization for {pair}")
                        except Exception as e:
                            logger.error(f"❌ [OPTUNA] Periodic optimization failed for {pair}: {e}")
        
        # ===== EXTRACT ALL PARAMETERS =====
        
        # ===== EXTRACT ALL PARAMETERS =====

        # ... rest of your code continues

        # ===== EXTRACT ALL PARAMETERS =====
        # Core indicators
        rsi_period = coin_params.get('rsi_period', 14)
        atr_period = coin_params.get('atr_period', 14)
        volume_sma_period = coin_params.get('volume_sma_period', 20)
        ema_short_period = coin_params.get('ema_short_period', 8)
        ema_long_period = coin_params.get('ema_long_period', 21)
        bb_period = coin_params.get('bb_period', 20)
        bb_std = coin_params.get('bb_std', 2.0)
        
        # MACD
        macd_fast = coin_params.get('macd_fast', 12)
        macd_slow = coin_params.get('macd_slow', 26) 
        macd_signal = coin_params.get('macd_signal', 9)
        
        # Other oscillators
        willr_period = coin_params.get('willr_period', 14) 
        cci_period = coin_params.get('cci_period', 20)
        mom_period = coin_params.get('mom_period', 10)
        stoch_k = coin_params.get('stoch_k', 14)
        stoch_d = coin_params.get('stoch_d', 3)
        cmf_period = coin_params.get('cmf_period', 20)
        natr_period = coin_params.get('natr_period', 14)
        chop_period = coin_params.get('chop_period', 14)
        
        # Divergence parameters
        window = coin_params.get('window', 4)
        index_range = coin_params.get('index_range', 30)
        divergence_lookback = coin_params.get('divergence_lookback_period', 5)
        divergence_min_adx = coin_params.get('divergence_min_adx', 20)
        divergence_max_adx = coin_params.get('divergence_max_adx', 40)
        divergence_min_atr_mult = coin_params.get('divergence_min_atr_mult', 0.8)
        divergence_max_atr_mult = coin_params.get('divergence_max_atr_mult', 3.0)
        
        # Fisher parameters
        fisher_period = coin_params.get('fisher_period', 9)
        fisher_min_adx = coin_params.get('fisher_min_adx', 20)
        fisher_max_adx = coin_params.get('fisher_max_adx', 40)
        fisher_min_volatility = coin_params.get('fisher_min_volatility', 0.005)
        fisher_max_volatility = coin_params.get('fisher_max_volatility', 0.025)
        
        # ===== MULTI-TIMEFRAME (1H) =====
        try:
            # Try to get 1h data from DataProvider (normal live/dry-run mode)
            if hasattr(self, 'dp') and self.dp is not None:
                informative_1h = self.dp.get_pair_dataframe(pair=pair, timeframe='1h')
            else:
                # During ML training, load from disk
                from freqtrade.data.history import load_pair_history
                from pathlib import Path
                exchange_name = self.config.get('exchange', {}).get('name', 'bybit')
                data_dir = Path(f"user_data/data/{exchange_name}")
                
                informative_1h = load_pair_history(
                    datadir=data_dir,
                    timeframe='1h',
                    pair=pair,
                    data_format='json',
                    candle_type=self.config.get('candle_type_def', 'spot')
                )
            
            if informative_1h is not None and len(informative_1h) > 50 and not informative_1h.empty:
                informative_1h['ema50_1h'] = ta.EMA(informative_1h, timeperiod=50)
                informative_1h['ema200_1h'] = ta.EMA(informative_1h, timeperiod=200)
                informative_1h['trend_1h'] = ta.EMA(informative_1h, timeperiod=21)
                informative_1h['trend_strength_1h'] = ta.ADX(informative_1h)
                informative_1h['rsi_1h'] = ta.RSI(informative_1h)
                informative_1h = informative_1h.bfill().ffill()
                
                dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
            else:
                self._add_dummy_1h_columns(dataframe)
        except Exception as e:
            logger.warning(f"1h data unavailable for {pair}: {e}")
            self._add_dummy_1h_columns(dataframe)
        
        # ===== VOLUME ANALYSIS =====
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=volume_sma_period)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        dataframe['volume_ratio'] = dataframe['volume_ratio'].fillna(1.0)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=volume_sma_period).mean().fillna(dataframe['volume'])

        # ===== VOLATILITY ANALYSIS =====
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=atr_period)
        dataframe['volatility'] = dataframe['atr'] / dataframe['close']
        dataframe['volatility'] = dataframe['volatility'].fillna(0.01)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=natr_period)

        # ===== MOMENTUM INDICATORS =====
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=rsi_period)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=willr_period)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=cci_period)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=mom_period)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        
        # MACD
        macd = ta.MACD(dataframe, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=stoch_k, slowk_period=stoch_d, slowd_period=stoch_d)
        dataframe['stoch'] = stoch['slowk']
        
        # Other oscillators
        dataframe['roc'] = ta.ROC(dataframe)
        dataframe['uo'] = ta.ULTOSC(dataframe)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        dataframe['cmf'] = chaikin_money_flow(dataframe, cmf_period)
        dataframe['obv'] = ta.OBV(dataframe)
        
        # Fill NaN values
        oscillator_cols = ['rsi', 'stoch', 'mfi', 'roc', 'uo', 'ao', 'macd', 'macdsignal', 'macdhist', 
                           'cci', 'cmf', 'obv', 'adx', 'willr', 'mom']
        for col in oscillator_cols:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].bfill().fillna(50 if col in ['rsi', 'mfi', 'stoch'] else 0)

        # ===== BOLLINGER BANDS =====
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=bb_period, stds=bb_std)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        dataframe['bollinger_lowerband'] = bollinger['lower']
        dataframe['bollinger_upperband'] = bollinger['upper']

        # ===== KELTNER CHANNEL =====
        keltner = emaKeltner(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_lowerband"] = keltner["lower"]

        # ===== EMA =====
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=ema_short_period)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=ema_long_period)
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        
        # Price position vs EMAs
        dataframe['price_vs_ema21'] = (dataframe['close'] - dataframe['ema21']) / dataframe['ema21'] * 100
        dataframe['price_vs_ema50'] = (dataframe['close'] - dataframe['ema50']) / dataframe['ema50'] * 100
        dataframe['price_vs_ema200'] = (dataframe['close'] - dataframe['ema200']) / dataframe['ema200'] * 100
        
        # Fill NaN
        ema_cols = ['ema_short', 'ema_long', 'ema9', 'ema20', 'ema21', 'ema50', 'ema200']
        for col in ema_cols:
            dataframe[col] = dataframe[col].bfill().fillna(dataframe['close'])
        
        price_pos_cols = ['price_vs_ema21', 'price_vs_ema50', 'price_vs_ema200']
        for col in price_pos_cols:
            dataframe[col] = dataframe[col].bfill().fillna(0)

        # ===== FISHER TRANSFORM (ENHANCED) =====
        try:
            high_low_range = dataframe['high'].rolling(window=fisher_period).max() - dataframe['low'].rolling(window=fisher_period).min()
            high_low_range = high_low_range.replace(0, 0.001)
            
            value = 2 * ((dataframe['close'] - dataframe['low'].rolling(window=fisher_period).min()) / high_low_range) - 1
            value = value.clip(-0.999, 0.999)
            
            dataframe['fisher'] = 0.5 * np.log((1 + value) / (1 - value))
            dataframe['fisher_signal'] = dataframe['fisher'].shift(1)
            dataframe['fisher_rising'] = dataframe['fisher'] > dataframe['fisher_signal']
            dataframe['fisher_falling'] = dataframe['fisher'] < dataframe['fisher_signal']
            
            # === NEW: Fisher regime filters ===
            dataframe['fisher_adx_ok'] = (dataframe['adx'] >= fisher_min_adx) & (dataframe['adx'] <= fisher_max_adx)
            dataframe['fisher_volatility_ok'] = (dataframe['volatility'] >= fisher_min_volatility) & (dataframe['volatility'] <= fisher_max_volatility)
            
        except Exception as e:
            logger.warning(f"Fisher Transform error: {e}")
            dataframe['fisher'] = 0
            dataframe['fisher_signal'] = 0
            dataframe['fisher_rising'] = False
            dataframe['fisher_falling'] = False
            dataframe['fisher_adx_ok'] = True
            dataframe['fisher_volatility_ok'] = True

        # ===== PIVOT POINTS =====
        try:
            pivots = pivot_points(dataframe, window)
            dataframe['pivot_lows'] = pivots['pivot_lows']
            dataframe['pivot_highs'] = pivots['pivot_highs']
        except Exception as e:
            logger.warning(f"Pivot points error: {e}")
            dataframe['pivot_lows'] = np.nan
            dataframe['pivot_highs'] = np.nan

        # ===== DIVERGENCE ANALYSIS (ENHANCED) =====
        try:
            self.initialize_divergences_lists(dataframe)
            (high_iterator, low_iterator) = self.get_iterators(dataframe)
            
            #indicators = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'cci', 'cmf', 'obv', 'mfi']
            indicators = ['rsi', 'stoch', 'mfi', 'cci', 'macd', 'willr', 'fisher']
            for indicator in indicators:
                if indicator in dataframe.columns:
                    try:
                        # === PASS coin_params to divergence detection ===
                        (bearish_divs, bearish_lines, bullish_divs, bullish_lines) = self.divergence_finder_dataframe(
                            dataframe, indicator, high_iterator, low_iterator, coin_params)
                        dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divs
                        dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divs
                    except:
                        continue
            
            # Fill NaN values
            dataframe['total_bullish_divergences'] = dataframe['total_bullish_divergences'].fillna(0)
            dataframe['total_bearish_divergences'] = dataframe['total_bearish_divergences'].fillna(0)
            dataframe['total_bullish_divergences_count'] = dataframe['total_bullish_divergences_count'].fillna(0)
            dataframe['total_bearish_divergences_count'] = dataframe['total_bearish_divergences_count'].fillna(0)
            
            # === NEW: Divergence regime filters ===
            dataframe['divergence_adx_ok'] = (dataframe['adx'] >= divergence_min_adx) & (dataframe['adx'] <= divergence_max_adx)
            
            # Calculate ATR multiplier (current ATR / average ATR)
            atr_sma = dataframe['atr'].rolling(window=20).mean()
            dataframe['atr_multiplier'] = dataframe['atr'] / atr_sma
            dataframe['atr_multiplier'] = dataframe['atr_multiplier'].fillna(1.0)
            
            dataframe['divergence_atr_ok'] = (dataframe['atr_multiplier'] >= divergence_min_atr_mult) & (dataframe['atr_multiplier'] <= divergence_max_atr_mult)
            
        except Exception as e:
            logger.warning(f"Divergence analysis error: {e}")
            dataframe['total_bullish_divergences'] = 0
            dataframe['total_bearish_divergences'] = 0
            dataframe['total_bullish_divergences_count'] = 0
            dataframe['total_bearish_divergences_count'] = 0
            dataframe['divergence_adx_ok'] = True
            dataframe['divergence_atr_ok'] = True

        # ===== MARKET STRUCTURE =====
        dataframe['chop'] = choppiness_index(dataframe['high'], dataframe['low'], dataframe['close'], window=chop_period)
        
        # Support/Resistance
        dataframe['swing_high'] = dataframe['high'].rolling(window=50).max()
        dataframe['swing_low'] = dataframe['low'].rolling(window=50).min()
        dataframe['distance_to_resistance'] = (dataframe['swing_high'] - dataframe['close']) / dataframe['close']
        dataframe['distance_to_support'] = (dataframe['close'] - dataframe['swing_low']) / dataframe['close']
        
        # Recent highs/lows
        dataframe['recent_high_5'] = dataframe['high'].rolling(window=5).max()
        dataframe['recent_low_5'] = dataframe['low'].rolling(window=5).min()
        dataframe['recent_high_10'] = dataframe['high'].rolling(window=10).max()
        dataframe['recent_low_10'] = dataframe['low'].rolling(window=10).min()
        
        # Price position
        dataframe['high_20'] = dataframe['high'].rolling(window=20).max()
        dataframe['low_20'] = dataframe['low'].rolling(window=20).min()
        dataframe['price_position'] = (dataframe['close'] - dataframe['low_20']) / (dataframe['high_20'] - dataframe['low_20'])
        dataframe['price_position'] = dataframe['price_position'].fillna(0.5)

        # ===== PATTERN DETECTION =====
        # MA alignment
        dataframe['ma_alignment_bull'] = (dataframe['ema21'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])
        dataframe['ma_alignment_bear'] = (dataframe['ema21'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])
        
        # Rejection patterns
        dataframe['rejected_at_ema21'] = (dataframe['high'].shift(1) > dataframe['ema21'].shift(1)) & (dataframe['close'].shift(1) < dataframe['ema21'].shift(1))
        dataframe['rejected_at_ema50'] = (dataframe['high'].shift(1) > dataframe['ema50'].shift(1)) & (dataframe['close'].shift(1) < dataframe['ema50'].shift(1))
        dataframe['bounced_from_ema21'] = (dataframe['low'].shift(1) < dataframe['ema21'].shift(1)) & (dataframe['close'].shift(1) > dataframe['ema21'].shift(1))
        dataframe['bounced_from_ema50'] = (dataframe['low'].shift(1) < dataframe['ema50'].shift(1)) & (dataframe['close'].shift(1) > dataframe['ema50'].shift(1))
        
        # Candle patterns
        dataframe['red_candle'] = dataframe['close'] < dataframe['close'].shift(1)
        dataframe['green_candle'] = dataframe['close'] > dataframe['close'].shift(1)
        dataframe['strong_red'] = (dataframe['close'] < dataframe['close'].shift(1)) & (dataframe['volume_ratio'] > 1.2)
        dataframe['strong_green'] = (dataframe['close'] > dataframe['close'].shift(1)) & (dataframe['volume_ratio'] > 1.2)

        # ===== SIGNAL STRENGTH =====
        try:
            dataframe['signal_strength'] = self.calculate_signal_strength(dataframe)
        except:
            dataframe['signal_strength'] = 0

        # ===== APPLY COIN-SPECIFIC FILTERS =====
        dataframe['coin_min_signal'] = coin_params.get('min_signal_strength', 3)
        dataframe['coin_volume_threshold'] = coin_params.get('volume_threshold', 1.2)
        dataframe['coin_adx_threshold'] = coin_params.get('adx_threshold', 20)

        # ===== PLOT CONFIG =====
        try:
            self.plot_config = PlotConfig().add_total_divergences_in_config(dataframe).config
        except:
            self.plot_config = None

        logger.debug(f"✅ [INDICATORS] Completed {pair}")
        
        # Daily optimization check
        if not hasattr(self, 'last_daily_check'):
            self.last_daily_check = 0
        
        current_time = time.time()
        if current_time - self.last_daily_check > 86400:  # 24 hours
            self.daily_optimization_check()
            self.last_daily_check = current_time

        # Daily optimization check
        if not hasattr(self, 'last_daily_check'):
            self.last_daily_check = 0
        
        current_time = time.time()
        if current_time - self.last_daily_check > 86400:  # 24 hours
            self.daily_optimization_check()
            self.last_daily_check = current_time
        
        # Generate ML predictions if model is trained
        if self.ml_enabled and self.ml_model_trained:
            try:
                # Get ML confidence for both directions
                should_enter_long, conf_long = self.ml_predict_entry(dataframe, pair, 'long')
                should_enter_short, conf_short = self.ml_predict_entry(dataframe, pair, 'short')
                
                # Store predictions (use long confidence as default)
                dataframe['ml_confidence_long'] = conf_long
                dataframe['ml_confidence_short'] = conf_short
                dataframe['ml_prediction'] = conf_long  # Default to long confidence
                
            except Exception as e:
                logger.error(f"❌ [ML] Prediction error: {str(e)}")
                dataframe['ml_prediction'] = 0.5
                dataframe['ml_confidence_long'] = 0.5
                dataframe['ml_confidence_short'] = 0.5
        else:
            dataframe['ml_prediction'] = 0.5
            dataframe['ml_confidence_long'] = 0.5
            dataframe['ml_confidence_short'] = 0.5
        
        return dataframe


    def _add_dummy_1h_columns(self, dataframe):
        """Add dummy 1h columns when higher timeframe data is unavailable"""
        # Use current 15m data to simulate 1h trend
        try:
            dataframe['ema50_1h_1h'] = ta.EMA(dataframe, timeperiod=200)  # Use longer period on 15m
            dataframe['ema200_1h_1h'] = ta.EMA(dataframe, timeperiod=800)  # Use much longer period
            dataframe['trend_1h_1h'] = ta.EMA(dataframe, timeperiod=84)   # 21 * 4 (4x 15m = 1h)
            dataframe['trend_strength_1h_1h'] = ta.ADX(dataframe)
            dataframe['rsi_1h_1h'] = ta.RSI(dataframe, timeperiod=56)     # Adjusted for timeframe
            
            # Fill NaN values
            columns_1h = ['ema50_1h_1h', 'ema200_1h_1h', 'trend_1h_1h', 'trend_strength_1h_1h', 'rsi_1h_1h']
            for col in columns_1h:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].bfill().fillna(
                        dataframe['close'] if 'ema' in col or 'trend' in col else 
                        25 if 'strength' in col else 50
                    )
        except Exception as e:
            logger.warning(f"Error creating dummy 1h columns: {e}")
            # Absolute fallback
            dataframe['ema50_1h_1h'] = dataframe['close']
            dataframe['ema200_1h_1h'] = dataframe['close']
            dataframe['trend_1h_1h'] = dataframe['close']
            dataframe['trend_strength_1h_1h'] = 25
            dataframe['rsi_1h_1h'] = 50

    def calculate_signal_strength(self, dataframe: DataFrame) -> Series:
        """
        Calculate overall signal strength based on multiple factors
        """
        strength = pd.Series(0, index=dataframe.index)
        
        # Divergence strength
        strength += dataframe['total_bullish_divergences_count'] * 2
        strength += dataframe['total_bearish_divergences_count'] * 2
        
        # Volume strength
        volume_strength = np.where(dataframe['volume_ratio'] > 1.5, 2, 
                                 np.where(dataframe['volume_ratio'] > 1.2, 1, 0))
        strength += volume_strength
        
        # Trend alignment strength
        ema_bullish = (dataframe['ema20'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])
        ema_bearish = (dataframe['ema20'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])
        strength += np.where(ema_bullish | ema_bearish, 1, 0)
        
        # ADX strength
        strength += np.where(dataframe['adx'] > 30, 1, 0)
        
        return strength

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced Entry Strategy with Optuna-Optimized Divergence & Fisher Signals
        Non-repainting: all triggers use closed candle data via df_prev
        
        Entry Priority Order:
        1. Bull/Bear_Super (3+ divergences)
        2. Bull/Bear_Extreme (high-quality divergence with strict filters)
        3. Bull/Bear_HQ (2-bar divergence confirmation)
        4. Bull/Bear_Div (basic divergence with bands)
        5. Bull/Bear_Early (single divergence signal)
        6. Bull/Bear_Weak (2 divergences)
        7. Bull/Bear_Fisher (Fisher Transform crossover)
        """
        
        pair = metadata['pair']
        if self.ml_enabled and not self.ml_model_trained:
            logger.info(f"⏳ [ML] Waiting for model training before generating entries for {pair}")
            # Still need to initialize columns
            dataframe.loc[:, 'enter_long'] = 0
            dataframe.loc[:, 'enter_short'] = 0
            dataframe.loc[:, 'enter_tag'] = ''
            dataframe.loc[:, 'entry_type'] = 0
            return dataframe
        
        logger.info(f"🎯 [ENTRY] Processing entry signals for {pair}")
        
        # Initialize
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'entry_type'] = 0

        # Non-repainting: use closed candle
        df_prev = dataframe.shift(1)
        
        # Get optimized parameters
        coin_params = self.get_coin_params(pair)
        signal_lock_enabled = coin_params.get('signal_lock_enabled', False)
        
        
        # ========================================================================
        # BASIC FILTERS
        # ========================================================================
        has_volume = (df_prev['volume'] > 0)
        
        # ========================================================================
        # TREND DETECTION
        # ========================================================================
        in_uptrend = (
            (df_prev['ema20'] > df_prev['ema50']) &
            (df_prev['ema50'] > df_prev['ema200'])
        )
        
        in_downtrend = (
            (df_prev['ema20'] < df_prev['ema50']) &
            (df_prev['ema50'] < df_prev['ema200'])
        )
        
        # Trend blocking (prevents counter-trend trades)
        block_long_in_downtrend = (
            in_downtrend &
            (df_prev['adx'] > coin_params.get('adx_threshold', 20))
        )
        
        block_short_in_uptrend = (
            in_uptrend &
            (df_prev['adx'] > coin_params.get('adx_threshold', 20))
        )
        
        # ========================================================================
        # DIVERGENCE DETECTION (ENHANCED WITH REGIME FILTERS)
        # ========================================================================
        
        # Get divergence-specific parameters
        divergence_require_volume_increase = coin_params.get('divergence_require_volume_increase', False)
        divergence_volume_multiplier = coin_params.get('divergence_volume_multiplier', 1.5)
        divergence_require_1h_alignment = coin_params.get('divergence_require_1h_alignment', False)
        divergence_require_bands = coin_params.get('divergence_require_bands', False)
        
        # === NEW: Market Regime Filters for Divergence ===
        divergence_regime_ok = (
            df_prev['divergence_adx_ok'] &  # ADX in acceptable range
            df_prev['divergence_atr_ok']    # ATR multiplier in acceptable range
        )
        
        # === NEW: Volume Confirmation (Optional) ===
        if divergence_require_volume_increase:
            divergence_volume_ok = (
                df_prev['volume'] > df_prev['volume_sma'] * divergence_volume_multiplier
            )
        else:
            divergence_volume_ok = pd.Series([True] * len(dataframe), index=dataframe.index)
        
        # === NEW: 1H Alignment (Optional) ===

        if divergence_require_1h_alignment:
            divergence_1h_ok_long = (df_prev['trend_1h_1h'] > df_prev['ema200_1h_1h'])
            divergence_1h_ok_short = (df_prev['trend_1h_1h'] < df_prev['ema200_1h_1h'])
        else:
            divergence_1h_ok_long = pd.Series([True] * len(dataframe), index=dataframe.index)
            divergence_1h_ok_short = pd.Series([True] * len(dataframe), index=dataframe.index)
        
        # === NEW: Bands Confirmation (Optional) ===
        if divergence_require_bands:
            at_lower_band = (
                (df_prev['low'] <= df_prev['kc_lowerband']) |
                (df_prev['close'] <= df_prev['kc_lowerband']) |
                (df_prev['low'] < df_prev['bollinger_lowerband'])
            )
            at_upper_band = (
                (df_prev['high'] >= df_prev['kc_upperband']) |
                (df_prev['close'] >= df_prev['kc_upperband']) |
                (df_prev['high'] > df_prev['bollinger_upperband'])
            )
        else:
            at_lower_band = two_bands_check_long(df_prev)
            at_upper_band = two_bands_check_short(df_prev)
        
        # Base divergence signals
        bullish_divergence = (df_prev['total_bullish_divergences'] > 0)
        bearish_divergence = (df_prev['total_bearish_divergences'] > 0)
        
        dataframe['bull_div_strength'] = df_prev['total_bullish_divergences_count'].fillna(0)
        dataframe['bear_div_strength'] = df_prev['total_bearish_divergences_count'].fillna(0)
        dataframe['bull_div_signal'] = df_prev['total_bullish_divergences']
        dataframe['bear_div_signal'] = df_prev['total_bearish_divergences']
        
        # ========================================================================
        # FISHER TRANSFORM (ENHANCED WITH REGIME FILTERS)
        # ========================================================================
        
        # Get Fisher-specific parameters
        fisher_oversold = coin_params.get('fisher_oversold', -2.0)
        fisher_overbought = coin_params.get('fisher_overbought', 2.0)
        fisher_rsi_max_long = coin_params.get('fisher_rsi_max_long', 40)
        fisher_rsi_min_short = coin_params.get('fisher_rsi_min_short', 60)
        fisher_require_volume = coin_params.get('fisher_require_volume', False)
        fisher_volume_mult = coin_params.get('fisher_volume_mult', 1.5)
        fisher_require_1h_alignment = coin_params.get('fisher_require_1h_alignment', False)
        fisher_require_momentum = coin_params.get('fisher_require_momentum', False)
        fisher_momentum_lookback = coin_params.get('fisher_momentum_lookback', 2)
        
        # === NEW: Market Regime Filters for Fisher ===
        fisher_regime_ok = (
            df_prev['fisher_adx_ok'] &      # ADX in acceptable range
            df_prev['fisher_volatility_ok']  # Volatility in acceptable range
        )
        
        # === NEW: Volume Confirmation for Fisher (Optional) ===
        if fisher_require_volume:
            fisher_volume_ok = (
                df_prev['volume'] > df_prev['volume_sma'] * fisher_volume_mult
            )
        else:
            fisher_volume_ok = pd.Series([True] * len(dataframe), index=dataframe.index)
        

        # === NEW: 1H Alignment for Fisher (Optional) ===
        if fisher_require_1h_alignment:
            fisher_1h_ok_long = (df_prev['trend_1h_1h'] > df_prev['ema200_1h_1h'])
            fisher_1h_ok_short = (df_prev['trend_1h_1h'] < df_prev['ema200_1h_1h'])
        else:
            fisher_1h_ok_long = pd.Series([True] * len(dataframe), index=dataframe.index)
            fisher_1h_ok_short = pd.Series([True] * len(dataframe), index=dataframe.index)
        
        # === NEW: Momentum Confirmation for Fisher (Optional) ===
        if fisher_require_momentum:
            fisher_momentum_ok_long = (
                df_prev['close'] > df_prev['close'].shift(fisher_momentum_lookback)
            )
            fisher_momentum_ok_short = (
                df_prev['close'] < df_prev['close'].shift(fisher_momentum_lookback)
            )
        else:
            fisher_momentum_ok_long = pd.Series([True] * len(dataframe), index=dataframe.index)
            fisher_momentum_ok_short = pd.Series([True] * len(dataframe), index=dataframe.index)
        
        # ========================================================================
        # ENTRY CONDITIONS (Priority Order)
        # ========================================================================
        
        # 1. SUPER - Highest Priority (3+ divergences)
        super_long = (
            (dataframe['bull_div_strength'] >= 3) &
            (df_prev['rsi'] < 45) &
            divergence_regime_ok &
            divergence_volume_ok &
            ~block_long_in_downtrend &
            has_volume
        )
        
        super_short = (
            (dataframe['bear_div_strength'] >= 3) &
            (df_prev['rsi'] > 55) &
            divergence_regime_ok &
            divergence_volume_ok &
            ~block_short_in_uptrend &
            has_volume
        )
        
        # 2. EXTREME - Enhanced Divergence with Strict Filters
        long_condition_extreme = (
            bullish_divergence &
            (df_prev['rsi'] < coin_params.get('divergence_rsi_low_threshold', 30)) &
            divergence_regime_ok &
            divergence_volume_ok &
            divergence_1h_ok_long &
            at_lower_band &
            ~block_long_in_downtrend &
            has_volume
        )
        
        short_condition_extreme = (
            bearish_divergence &
            (df_prev['rsi'] > coin_params.get('divergence_rsi_high_threshold', 70)) &
            divergence_regime_ok &
            divergence_volume_ok &
            divergence_1h_ok_short &
            at_upper_band &
            ~block_short_in_uptrend &
            has_volume
        )
        
        # 3. HQ - High Quality (2-bar confirmation)
        recent_bull_div_hq = (
            (df_prev['total_bullish_divergences'].fillna(0) > 0) |
            (df_prev['total_bullish_divergences'].shift(1).fillna(0) > 0)
        )
        recent_bear_div_hq = (
            (df_prev['total_bearish_divergences'].fillna(0) > 0) |
            (df_prev['total_bearish_divergences'].shift(1).fillna(0) > 0)
        )
        
        long_condition_hq = (
            recent_bull_div_hq &
            divergence_regime_ok &
            divergence_volume_ok &
            two_bands_check_long(df_prev) &
            ~block_long_in_downtrend &
            has_volume
        )
        
        short_condition_hq = (
            recent_bear_div_hq &
            divergence_regime_ok &
            divergence_volume_ok &
            two_bands_check_short(df_prev) &
            ~block_short_in_uptrend &
            has_volume
        )
        
        # 4. DIV - Basic Divergence with Bands
        long_condition = (
            bullish_divergence &
            divergence_regime_ok &
            two_bands_check_long(df_prev) &
            ~block_long_in_downtrend &
            has_volume
        )
        
        short_condition = (
            bearish_divergence &
            divergence_regime_ok &
            two_bands_check_short(df_prev) &
            ~block_short_in_uptrend &
            has_volume
        )
        
        # 5. EARLY - Single Divergence Signal
        early_long = (
            (dataframe['bull_div_signal'].fillna(0) > 0) &
            (df_prev['rsi'] < 50) &
            divergence_regime_ok &
            has_volume
        )
        
        early_short = (
            (dataframe['bear_div_signal'].fillna(0) > 0) &
            (df_prev['rsi'] > 50) &
            divergence_regime_ok &
            has_volume
        )
        
        # 6. WEAK - Two Divergences
        weak_long = (
            (dataframe['bull_div_strength'] == 2) &
            (df_prev['rsi'] < 45) &
            divergence_regime_ok &
            has_volume
        )
        
        weak_short = (
            (dataframe['bear_div_strength'] == 2) &
            (df_prev['rsi'] > 55) &
            divergence_regime_ok &
            has_volume
        )
        
        # 7. FISHER - Lowest Priority (Enhanced with Regime Filters)
        long_condition_fisher = (
            # Crossover: current above, previous below
            (df_prev['fisher'] > df_prev['fisher_signal']) &
            (df_prev['fisher'].shift(1) <= df_prev['fisher_signal'].shift(1)) &
            # Was oversold recently (check last 3 candles)
            (
                (df_prev['fisher'].shift(1) < fisher_oversold) |
                (df_prev['fisher'].shift(2) < fisher_oversold) |
                (df_prev['fisher'].shift(3) < fisher_oversold)
            ) &
            # === NEW: Regime filters ===
            fisher_regime_ok &
            fisher_volume_ok &
            fisher_1h_ok_long &
            fisher_momentum_ok_long &
            # RSI filter
            (df_prev['rsi'] < fisher_rsi_max_long) &
            ~block_long_in_downtrend &
            has_volume
        )
        
        long_condition2 = (
            (
                (dataframe['total_bullish_divergences'].shift(1).fillna(0) > 0) |
                (dataframe['total_bullish_divergences'].shift(2).fillna(0) > 0) |
                (dataframe['total_bullish_divergences'].shift(3).fillna(0) > 0)  # NEW!
            ) &
            (df_prev['rsi'] < 45) &  # Was 40, now 45 (more entries)
            two_bands_check_long(df_prev) &
            ~block_long_in_downtrend &
            has_volume
        )

        short_condition2 = (
            (
                (dataframe['total_bearish_divergences'].shift(1).fillna(0) > 0) |
                (dataframe['total_bearish_divergences'].shift(2).fillna(0) > 0) |
                (dataframe['total_bearish_divergences'].shift(3).fillna(0) > 0)  # NEW!
            ) &
            (df_prev['rsi'] > 55) &  # Was 60, now 55 (more entries)
            two_bands_check_short(df_prev) &
            ~block_short_in_uptrend &
            has_volume
        )
        # ========================================================================
        # APPLY ENTRIES IN PRIORITY ORDER
        # ========================================================================
        # === DEBUG: Check why no divergence signals ===
        # 1. Super
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Super', pair):
            dataframe.loc[super_long, 'enter_long'] = 1
            dataframe.loc[super_long, 'enter_tag'] = 'Bull_Super'
        else:
            print(f"Signal LOCKED: Bull_Super for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Super', pair):
            dataframe.loc[super_short, 'enter_short'] = 1
            dataframe.loc[super_short, 'enter_tag'] = 'Bear_Super'
        else:
            print(f"Signal LOCKED: Bear_Super for {pair}")
        
        # 2. Extreme
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Extreme', pair):
            long_condition_extreme_filtered = long_condition_extreme & (dataframe['enter_tag'] == "")
            dataframe.loc[long_condition_extreme_filtered, 'enter_long'] = 1
            dataframe.loc[long_condition_extreme_filtered, 'enter_tag'] = 'Bull_Extreme'
        else:
            print(f"Signal LOCKED: Bull_Extreme for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Extreme', pair):
            short_condition_extreme_filtered = short_condition_extreme & (dataframe['enter_tag'] == "")
            dataframe.loc[short_condition_extreme_filtered, 'enter_short'] = 1
            dataframe.loc[short_condition_extreme_filtered, 'enter_tag'] = 'Bear_Extreme'
        else:
            print(f"Signal LOCKED: Bear_Extreme for {pair}")
        
        # 3. HQ
        if not signal_lock_enabled or not self.is_signal_locked('Bull_HQ', pair):
            long_condition_hq_filtered = long_condition_hq & (dataframe['enter_tag'] == "")
            dataframe.loc[long_condition_hq_filtered, 'enter_long'] = 1
            dataframe.loc[long_condition_hq_filtered, 'enter_tag'] = 'Bull_HQ'
        else:
            print(f"Signal LOCKED: Bull_HQ for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_HQ', pair):
            short_condition_hq_filtered = short_condition_hq & (dataframe['enter_tag'] == "")
            dataframe.loc[short_condition_hq_filtered, 'enter_short'] = 1
            dataframe.loc[short_condition_hq_filtered, 'enter_tag'] = 'Bear_HQ'
        else:
            print(f"Signal LOCKED: Bear_HQ for {pair}")
        
        # 4. Div
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Div', pair):
            long_condition_filtered = long_condition & (dataframe['enter_tag'] == "")
            dataframe.loc[long_condition_filtered, 'enter_long'] = 1
            dataframe.loc[long_condition_filtered, 'enter_tag'] = 'Bull_Div'
        else:
            print(f"Signal LOCKED: Bull_Div for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Div', pair):
            short_condition_filtered = short_condition & (dataframe['enter_tag'] == "")
            dataframe.loc[short_condition_filtered, 'enter_short'] = 1
            dataframe.loc[short_condition_filtered, 'enter_tag'] = 'Bear_Div'
        else:
            print(f"Signal LOCKED: Bear_Div for {pair}")
        
        # 5. Early
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Early', pair):
            early_long_filtered = early_long & (dataframe['enter_tag'] == "")
            dataframe.loc[early_long_filtered, 'enter_long'] = 1
            dataframe.loc[early_long_filtered, 'enter_tag'] = 'Bull_Early'
        else:
            print(f"Signal LOCKED: Bull_Early for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Early', pair):
            early_short_filtered = early_short & (dataframe['enter_tag'] == "")
            dataframe.loc[early_short_filtered, 'enter_short'] = 1
            dataframe.loc[early_short_filtered, 'enter_tag'] = 'Bear_Early'
        else:
            print(f"Signal LOCKED: Bear_Early for {pair}")
        
        # 6. Weak
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Weak', pair):
            weak_long_filtered = weak_long & (dataframe['enter_tag'] == "")
            dataframe.loc[weak_long_filtered, 'enter_long'] = 1
            dataframe.loc[weak_long_filtered, 'enter_tag'] = 'Bull_Weak'
        else:
            print(f"Signal LOCKED: Bull_Weak for {pair}")
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Confirmed', pair):
            confirmed_long_filtered = long_condition2 & (dataframe['enter_tag'] == "")
            dataframe.loc[confirmed_long_filtered, 'enter_long'] = 1
            dataframe.loc[confirmed_long_filtered, 'enter_tag'] = 'Bull_Confirmed'
        else:
            print(f"Signal LOCKED: Bull_Confirmed for {pair}")
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Confirmed', pair):
            confirmed_short_filtered = short_condition2 & (dataframe['enter_tag'] == "")
            dataframe.loc[confirmed_short_filtered, 'enter_long'] = 1
            dataframe.loc[confirmed_short_filtered, 'enter_tag'] = 'Bear_Confirmed'
        else:
            print(f"Signal LOCKED: Bull_Confirmed for {pair}")
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Weak', pair):
            weak_short_filtered = weak_short & (dataframe['enter_tag'] == "")
            dataframe.loc[weak_short_filtered, 'enter_short'] = 1
            dataframe.loc[weak_short_filtered, 'enter_tag'] = 'Bear_Weak'
        else:
            print(f"Signal LOCKED: Bear_Weak for {pair}")
        
        # 7. Fisher
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Fisher', pair):
            fisher_long_filtered = long_condition_fisher & (dataframe['enter_tag'] == "")
            dataframe.loc[fisher_long_filtered, 'enter_long'] = 1
            dataframe.loc[fisher_long_filtered, 'enter_tag'] = 'Bull_Fisher'
        else:
            print(f"Signal LOCKED: Bull_Fisher for {pair}")
        

        
        # ========================================================================
        # LOGGING
        # ========================================================================
        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]
            if last_row.get('enter_long', 0) == 1 or last_row.get('enter_short', 0) == 1:
                direction = "LONG 🟢" if last_row.get('enter_long', 0) == 1 else "SHORT 🔴"
                logger.info(f"🚀 {pair} ENTRY DETECTED - {direction}")
                logger.info(f"   Tag: {last_row.get('enter_tag', 'N/A')}")
                logger.info(f"   RSI: {last_row.get('rsi', 0):.1f} | ADX: {last_row.get('adx', 0):.1f}")
                logger.info(f"   Volume Ratio: {last_row.get('volume_ratio', 1):.2f}")
                logger.info(f"   BullDivs: {last_row.get('bull_div_strength', 0)} | BearDivs: {last_row.get('bear_div_strength', 0)}")
                
                ema20, ema50, ema200 = last_row.get('ema20', 0), last_row.get('ema50', 0), last_row.get('ema200', 0)
                
                if ema20 > ema50 > ema200:
                    trend_icon = "⬆️ UPTREND"
                elif ema20 < ema50 < ema200:
                    trend_icon = "⬇️ DOWNTREND"
                else:
                    trend_icon = "↔️ SIDEWAYS"
                
                logger.info(f"   Market: {trend_icon} | EMA20: {ema20:.2f} | EMA50: {ema50:.2f}")
        # ========================================================================
        # HYBRID ML PREDICTIONS - Add ML-based entries
        # ========================================================================
        # ========================================================================
        # HYBRID ML PREDICTIONS - Add ML-based entries
        # ========================================================================
        if self.ml_enabled and ML_AVAILABLE and self.ml_trained:
            try:
                # Check last candle only (real-time prediction)
                if len(dataframe) > 0:
                    ml_should_enter_long, ml_confidence_long = self.ml_predict_entry(dataframe, pair, 'long')
                    ml_should_enter_short, ml_confidence_short = self.ml_predict_entry(dataframe, pair, 'short')
                    
                    # ML predicts LONG entry
                    if ml_should_enter_long and dataframe['enter_tag'].iloc[-1] == '':
                        # Determine which signal would have triggered
                        last_row = dataframe.iloc[-1]
                        base_signal = "Unknown"
                        
                        # Check each signal condition to find what ML is predicting on
                        if last_row.get('total_bullish_divergences', 0) >= 2:
                            base_signal = "Bull_Super"
                        elif last_row.get('total_bullish_divergences', 0) == 1 and last_row.get('rsi', 50) < 35:
                            base_signal = "Bull_Extreme"
                        elif last_row.get('total_bullish_divergences', 0) == 1:
                            base_signal = "Bull_HQ"
                        elif last_row.get('bull_div_strength', 0) > 0:
                            base_signal = "Bull_Div"
                        elif last_row.get('rsi', 50) < 25:
                            base_signal = "Bull_Early"
                        elif last_row.get('rsi', 50) < 30:
                            base_signal = "Bull_Weak"

                        
                        # SKIP ML ENTRY IF NO KNOWN SIGNAL MATCHED
                        if base_signal == "Unknown":
                            logger.debug(f"⚠️ [ML] Skipping unknown pattern for {pair} - no matching signal")
                        else:
                            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
                            dataframe.loc[dataframe.index[-1], 'enter_tag'] = f'ML_{base_signal}_{ml_confidence_long:.0%}'
                            
                            logger.info(f"🤖 [ML ENTRY] {pair} - LONG predicted")
                            logger.info(f"   Signal: {base_signal} | Confidence: {ml_confidence_long:.1%}")
                            logger.info(f"   RSI: {dataframe['rsi'].iloc[-1]:.1f} | ADX: {dataframe['adx'].iloc[-1]:.1f}")
                        
                    # ML predicts SHORT entry
                    if ml_should_enter_short and dataframe['enter_tag'].iloc[-1] == '':
                        # Determine which signal would have triggered
                        last_row = dataframe.iloc[-1]
                        base_signal = "Unknown"
                        
                        # Check each signal condition to find what ML is predicting on
                        if last_row.get('total_bearish_divergences', 0) >= 2:
                            base_signal = "Bear_Super"
                        elif last_row.get('total_bearish_divergences', 0) == 1 and last_row.get('rsi', 50) > 65:
                            base_signal = "Bear_Extreme"
                        elif last_row.get('total_bearish_divergences', 0) == 1:
                            base_signal = "Bear_HQ"
                        elif last_row.get('bear_div_strength', 0) > 0:
                            base_signal = "Bear_Div"
                        elif last_row.get('rsi', 50) > 75:
                            base_signal = "Bear_Early"
                        elif last_row.get('rsi', 50) > 70:
                            base_signal = "Bear_Weak"

                        
                        # SKIP ML ENTRY IF NO KNOWN SIGNAL MATCHED
                        if base_signal == "Unknown":
                            logger.debug(f"⚠️ [ML] Skipping unknown pattern for {pair} - no matching signal")
                        else:
                            dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
                            dataframe.loc[dataframe.index[-1], 'enter_tag'] = f'ML_{base_signal}_{ml_confidence_short:.0%}'
                            
                            logger.info(f"🤖 [ML ENTRY] {pair} - SHORT predicted")
                            logger.info(f"   Signal: {base_signal} | Confidence: {ml_confidence_short:.1%}")
                            logger.info(f"   RSI: {dataframe['rsi'].iloc[-1]:.1f} | ADX: {dataframe['adx'].iloc[-1]:.1f}")
                            
            except Exception as e:
                logger.error(f"❌ [ML] Entry prediction error for {pair}: {e}")
        
        # ========================================================================
        # FINAL RETURN
        # ========================================================================
        return dataframe


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Adaptive leverage based on signal strength
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 0:
                current_signal_strength = dataframe['signal_strength'].iloc[-1]
                
                # Calculate leverage based on signal strength
                if current_signal_strength >= 8:
                    calculated_leverage = self.leverage_value  # Full leverage for strong signals
                elif current_signal_strength >= 6:
                    calculated_leverage = self.leverage_value * 0.8  # 80% leverage
                elif current_signal_strength >= 4:
                    calculated_leverage = self.leverage_value * 0.6  # 60% leverage
                else:
                    calculated_leverage = self.leverage_value * 0.4  # 40% leverage for weak signals
                
                # Round to 1 decimal place (or use 0 for whole numbers)
                return round(calculated_leverage, 1)
        except:
            pass
        
        # Conservative fallback - also rounded
        return round(self.leverage_value * 0.5, 1)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> float:
        """
        FIXED DYNAMIC STOPLOSS - Now properly widens in volatile markets
        
        Features:
        - ATR-based dynamic stops adapted to volatility
        - Signal-type specific stop distances  
        - Breakeven protection at realistic levels
        - Trailing stops that adapt to signal quality
        - Time-based stop tightening for stagnant trades
        
        Entry Tags:
        - Bull_Super / Bear_Super (highest quality) → Widest stops
        - Bull_Extreme / Bear_Extreme → Wide stops
        - Bull_HQ / Bear_HQ → Medium stops
        - Bull_Div / Bear_Div → Medium stops
        - Bull_Early / Bear_Early → Tighter stops
        - Bull_Weak / Bear_Weak (lowest quality) → Tightest stops
        - Bull_Fisher / Bear_Fisher → Medium stops
        """
        
        try:
            # Get market data
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < 1:
                return self.stoploss
            
            last_row = dataframe.iloc[-1]
            atr = last_row.get('atr', 0)
            current_price = last_row['close']
            volatility = last_row.get('volatility', 0.02)
            adx = last_row.get('adx', 20)
            
            # Get entry tag
            entry_tag = trade.enter_tag or ""
            trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
            
            # =======================================================================
            # STEP 1: BREAKEVEN LOGIC (Signal-Type Specific)
            # =======================================================================
            
            # Determine breakeven trigger based on signal quality
            if 'Super' in entry_tag:
                breakeven_trigger = 0.020  # 2% for Super signals
            elif 'Extreme' in entry_tag or 'HQ' in entry_tag:
                breakeven_trigger = 0.015  # 1.5% for Extreme/HQ
            elif 'Div' in entry_tag:
                breakeven_trigger = 0.012  # 1.2% for divergence
            elif 'Fisher' in entry_tag:
                breakeven_trigger = 0.015  # 1.5% for Fisher
            elif 'Early' in entry_tag:
                breakeven_trigger = 0.010  # 1% for Early
            elif 'Weak' in entry_tag:
                breakeven_trigger = 0.008  # 0.8% for Weak
            else:
                breakeven_trigger = 0.015  # Default 1.5%
            
            # Move to breakeven once threshold hit
            if current_profit >= breakeven_trigger:
                breakeven_stop = 0.003  # 0.3% positive (covers fees)
                return stoploss_from_open(breakeven_stop, current_profit, is_short=trade.is_short)
            
            # =======================================================================
            # STEP 2: TRAILING LOGIC (Starts at 3.5% profit)
            # =======================================================================
            
            trailing_trigger = 0.035  # Start trailing at 3.5% profit
            
            if current_profit >= trailing_trigger:
                # Base trailing distance by signal quality
                if 'Super' in entry_tag:
                    trailing_distance = 0.025  # 2.5% - wide trail
                elif 'Extreme' in entry_tag or 'HQ' in entry_tag:
                    trailing_distance = 0.020  # 2.0% - normal trail
                elif 'Div' in entry_tag or 'Fisher' in entry_tag:
                    trailing_distance = 0.015  # 1.5% - tighter trail
                elif 'Early' in entry_tag or 'Weak' in entry_tag:
                    trailing_distance = 0.012  # 1.2% - very tight trail
                else:
                    trailing_distance = 0.018  # Default 1.8%
                
                # Adjust for volatility
                volatility_adj = min(1.5, max(0.7, volatility / 0.02))
                trailing_distance *= volatility_adj
                
                # Adjust for ADX (trend strength)
                if adx > 40:
                    trailing_distance *= 1.2  # Strong trend - wider
                elif adx < 20:
                    trailing_distance *= 0.85  # Weak trend - tighter
                
                return stoploss_from_open(trailing_distance, current_profit, is_short=trade.is_short)
            
            # =======================================================================
            # STEP 3: INITIAL DYNAMIC STOPLOSS (ATR-Based) 🔧 FIXED!
            # =======================================================================
            
            # Base ATR multiplier by signal quality
            if 'Super' in entry_tag:
                atr_multiplier = 3.5  # Widest stop (was 3.0)
            elif 'Extreme' in entry_tag or 'HQ' in entry_tag:
                atr_multiplier = 3.0  # Wide (was 2.5)
            elif 'Div' in entry_tag or 'Fisher' in entry_tag:
                atr_multiplier = 2.5  # Medium (was 2.0)
            elif 'Early' in entry_tag:
                atr_multiplier = 2.0  # Tighter (was 1.8)
            elif 'Weak' in entry_tag:
                atr_multiplier = 1.5  # Tightest
            else:
                atr_multiplier = 2.5  # Default (was 2.0)
            
            # Adjust for volatility (wider in volatile markets)
            volatility_adjustment = min(1.5, max(0.8, volatility / 0.02))  # Changed min from 0.7 to 0.8
            atr_multiplier *= volatility_adjustment
            
            # Calculate stop distance
            if atr > 0:
                stop_distance = (atr * atr_multiplier) / current_price
            else:
                # Fallback if ATR not available
                stop_distance = 0.06 + (volatility * 2)
            
            # Signal-specific fine-tuning
            if 'Weak' in entry_tag or 'Early' in entry_tag:
                stop_distance *= 0.85  # Tighter for lower quality
            elif 'Super' in entry_tag or 'Extreme' in entry_tag:
                stop_distance *= 1.15  # Wider for highest quality
            
            # Safety limits (adjusted for better range)
            min_stop = 0.035  # 3.5% minimum
            max_stop = 0.15   # 15% maximum (was 0.12) - Allow wider stops
            stop_distance = max(min_stop, min(max_stop, stop_distance))
            
            dynamic_stoploss = -stop_distance
            
            # =======================================================================
            # STEP 4: TIME-BASED TIGHTENING (For Stagnant Trades)
            # =======================================================================
            
            # Tighten stop progressively for trades that aren't moving
            if trade_duration_minutes > 90:  # After 1.5 hours
                if current_profit < 0.005:  # Less than 0.5% profit
                    # Tighten by 10% every 30 minutes
                    time_factor = min(0.3, (trade_duration_minutes - 90) / 180)
                    dynamic_stoploss *= (1 - time_factor)
            
            # 🔧 CRITICAL FIX: Only enforce minimum stop, don't prevent widening!
            # Never tighter than 3.5%, but ALLOW wider than -7% in volatile markets
            if dynamic_stoploss > -0.035:  # If calculated stop is tighter than 3.5%
                dynamic_stoploss = -0.035  # Enforce minimum 3.5%
            
            # =======================================================================
            # STEP 5: LOGGING (Every 15 checks for better visibility)
            # =======================================================================
            
            if not hasattr(self, '_stoploss_log_counter'):
                self._stoploss_log_counter = {}
            
            if pair not in self._stoploss_log_counter:
                self._stoploss_log_counter[pair] = 0
            
            self._stoploss_log_counter[pair] += 1
            
            if self._stoploss_log_counter[pair] % 15 == 0:  # More frequent logging (was 20)
                if current_profit >= trailing_trigger:
                    logger.info(f"🔄 Trailing {pair}: Profit={current_profit:.3f}, Stop={dynamic_stoploss:.3f}, Tag={entry_tag}")
                elif current_profit >= breakeven_trigger:
                    logger.info(f"⚖️ Breakeven {pair}: Profit={current_profit:.3f}, Tag={entry_tag}")
                else:
                    logger.info(f"🛡️ Dynamic SL {pair}: ATR={atr:.4f}, Mult={atr_multiplier:.2f}x, Stop={dynamic_stoploss:.3f}, Tag={entry_tag}")
            
            return dynamic_stoploss
            
        except Exception as e:
            logger.error(f"❌ Error in custom_stoploss for {pair}: {e}")
            return self.stoploss

    def initialize_divergences_lists(self, dataframe: DataFrame):
        """Initialize divergence tracking columns"""
        # Bullish Divergences
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ''

        # Bearish Divergences
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ''

    def get_iterators(self, dataframe):
        """Get pivot point iterators for divergence detection"""
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
            else:
                high_iterator.append(index)
        
        return high_iterator, low_iterator

    def add_divergences(self, dataframe: DataFrame, indicator: str, high_iterator, low_iterator):
        """Add divergence detection for a specific indicator"""
        (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = self.divergence_finder_dataframe(
            dataframe, indicator, high_iterator, low_iterator)
        dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
        dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences

    def divergence_finder_dataframe(self, dataframe: DataFrame, indicator_source: str, high_iterator, low_iterator, coin_params: dict = None) -> Tuple[pd.Series, pd.Series]:
        """Enhanced divergence finder with Optuna-optimized parameters"""
        
        # Get optimized parameters (pass from populate_indicators)
        if coin_params is None:
            coin_params = {}
        
        # === NEW: Use optimized thresholds ===
        min_price_delta_pct = coin_params.get('divergence_min_price_delta', 1.0) / 100  # Convert to decimal
        min_rsi_delta = coin_params.get('divergence_min_rsi_delta', 4.0)
        min_time_separation = coin_params.get('divergence_lookback_period', 5)
        
        bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
        bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bullish_divergences = np.empty(len(dataframe['close'])) * np.nan

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

            # Bearish divergence detection
            bearish_occurence = self.bearish_divergence_finder(
                dataframe, dataframe[indicator_source], high_iterator, index, coin_params)

            if bearish_occurence is not None:
                (prev_pivot, current_pivot) = bearish_occurence
                bearish_prev_pivot = dataframe['close'][prev_pivot]
                bearish_current_pivot = dataframe['close'][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # === ENHANCED: Use Optuna-optimized validation ===
                price_diff_pct = abs(bearish_current_pivot - bearish_prev_pivot) / bearish_prev_pivot
                indicator_diff = abs(bearish_ind_current_pivot - bearish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Validate with optimized parameters
                if (price_diff_pct >= min_price_delta_pct and 
                    indicator_diff >= min_rsi_delta and 
                    time_diff >= min_time_separation):
                    
                    bearish_divergences[index] = row.close
                    dataframe.loc[index, "total_bearish_divergences"] = row.close
                    dataframe.loc[index, "total_bearish_divergences_count"] += 1
                    dataframe.loc[index, "total_bearish_divergences_names"] += indicator_source.upper() + '<br>'

            # Bullish divergence detection
            bullish_occurence = self.bullish_divergence_finder(
                dataframe, dataframe[indicator_source], low_iterator, index, coin_params)

            if bullish_occurence is not None:
                (prev_pivot, current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe['close'][prev_pivot]
                bullish_current_pivot = dataframe['close'][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # === ENHANCED: Use Optuna-optimized validation ===
                price_diff_pct = abs(bullish_current_pivot - bullish_prev_pivot) / bullish_prev_pivot
                indicator_diff = abs(bullish_ind_current_pivot - bullish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Validate with optimized parameters
                if (price_diff_pct >= min_price_delta_pct and 
                    indicator_diff >= min_rsi_delta and 
                    time_diff >= min_time_separation):
                    
                    bullish_divergences[index] = row.close
                    dataframe.loc[index, "total_bullish_divergences"] = row.close
                    dataframe.loc[index, "total_bullish_divergences_count"] += 1
                    dataframe.loc[index, "total_bullish_divergences_names"] += indicator_source.upper() + '<br>'

        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index, coin_params: dict = None):
        """Enhanced bullish divergence detection - ONLY CLASSIC DIVERGENCE with Optuna parameters"""
        
        # Get optimized window parameter
        if coin_params is None:
            window = getattr(self, 'window', type('obj', (), {'value': 4})).value
        else:
            window = coin_params.get('window', 4)
        
        try:
            if low_iterator[index] == index:
                current_pivot = low_iterator[index]
                occurences = list(dict.fromkeys(low_iterator))
                current_index = occurences.index(low_iterator[index])
                
                for i in range(current_index-1, current_index - window - 1, -1):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Classic bullish divergence
                    price_lower = dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot]
                    indicator_higher = indicator[current_pivot] > indicator[prev_pivot]
                    
                    if price_lower and indicator_higher:
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bullish', coin_params):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index, coin_params: dict = None):
        """Enhanced bearish divergence detection - ONLY CLASSIC DIVERGENCES with Optuna parameters"""
        
        # Get optimized window parameter
        if coin_params is None:
            window = getattr(self, 'window', type('obj', (), {'value': 4})).value
        else:
            window = coin_params.get('window', 4)
        
        try:
            if high_iterator[index] == index:
                current_pivot = high_iterator[index]
                occurences = list(dict.fromkeys(high_iterator))
                current_index = occurences.index(high_iterator[index])
                
                for i in range(current_index-1, current_index - window - 1, -1):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Classic bearish divergence
                    price_higher = dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot]
                    indicator_lower = indicator[current_pivot] < indicator[prev_pivot]
                    
                    if price_higher and indicator_lower:
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bearish', coin_params):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def validate_divergence_trend(self, dataframe, prev_pivot, current_pivot, divergence_type, coin_params: dict = None):
        """Validate divergence by checking intermediate trend with optional Optuna filter"""
        
        # === NEW: Optional trend context requirement ===
        if coin_params and not coin_params.get('divergence_require_trend_context', True):
            return True  # Skip trend validation if not required
        
        try:
            # Check if there's a clear trend between pivots
            mid_point = (prev_pivot + current_pivot) // 2
            
            if divergence_type == 'bearish':
                # For bearish divergence, expect uptrend in between
                return dataframe['ema20'][mid_point] > dataframe['ema20'][prev_pivot]
            else:
                # For bullish divergence, expect downtrend in between
                return dataframe['ema20'][mid_point] < dataframe['ema20'][prev_pivot]
        except:
            return True  # Default to accepting divergence if validation fails

    @property
    def protections(self):
        """Enhanced protection configuration"""
        prot = []
       
        if self.use_cooldown_protection.value:
            prot.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            })
       
        if self.use_max_drawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            })
       
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stoploss_guard_only_per_pair.value,
            })
       
        return prot
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0  
        dataframe['exit_tag'] = ''
        
        # === YOUR OTHER EXIT CONDITIONS FIRST ===
        # (Add any other exit conditions you have here)
        # any_long_exit = (some_other_condition)
        # any_short_exit = (some_other_condition)
        
        # === EXIT ON OPPOSITE SIGNALS (your previous approach) ===
        if 'enter_long' in dataframe.columns and 'enter_short' in dataframe.columns:
            # Exit longs on short signals
            reversal_long_exit = (dataframe['enter_short'] == 1)
            dataframe.loc[reversal_long_exit, 'exit_long'] = 1
            dataframe.loc[reversal_long_exit, 'exit_tag'] = 'Reversal_Short_Signal'
            
            # Exit shorts on long signals  
            if self.can_short:
                reversal_short_exit = (dataframe['enter_long'] == 1)
                dataframe.loc[reversal_short_exit, 'exit_short'] = 1
                dataframe.loc[reversal_short_exit, 'exit_tag'] = 'Reversal_Long_Signal'
        
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):
        """
        ENHANCED EXIT WITH 5 TIERS:
        - Under 1.0%: Minimal interference
        - 1.0%-1.5%: Light protection (0.8% pullback)
        - 1.5%-3%: Medium protection (1.2% pullback)
        - 3%-5%: Strong protection (1.0% pullback)
        - 5%+: Tight protection (1.2% pullback)
        """
        from logging import getLogger
        from datetime import timezone
        logger = getLogger(__name__)

        # Get actual profit
        actual_profit = trade.calc_profit_ratio(current_rate)
        current_profit = actual_profit

        # Basic info
        entry_tag = getattr(trade, 'enter_tag', 'Unknown')
        trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60

        # Profit peak tracking
        if not hasattr(self, 'profit_peaks'):
            self.profit_peaks = {}
        
        trade_id = str(trade.id)
        if trade_id not in self.profit_peaks:
            self.profit_peaks[trade_id] = current_profit
        else:
            self.profit_peaks[trade_id] = max(self.profit_peaks[trade_id], current_profit)
        
        profit_peak = self.profit_peaks[trade_id]
        
        # Status logging (every check when profitable)
        if current_profit >= 0.01:
            logger.info(f"📊 {pair} Status Check:")
            logger.info(f"   💰 Current: {current_profit*100:.2f}% | Peak: {profit_peak*100:.2f}%")
            logger.info(f"   ⚖️  Leverage: {trade.leverage}x | Duration: {int(trade_duration_minutes)}min")
            logger.info(f"   🏷️  Entry: {entry_tag}")

        # Get market data
        rsi = 50
        prev_rsi = 50
        atr = 0
        momentum_score = 0
        has_opposite_div = False
        last_row = None
        adx = 20
        ema20 = 0
        ema50 = 0
        close_price = current_rate

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty:
                last_rows = dataframe.tail(5)
                last_row = last_rows.iloc[-1]
                prev_row = last_rows.iloc[-2] if len(last_rows) > 1 else last_row

                rsi = last_row.get('rsi', 50)
                prev_rsi = prev_row.get('rsi', 50)
                atr = last_row.get('atr', 0)
                adx = last_row.get('adx', 20)
                ema20 = last_row.get('ema20', last_row.get('close', current_rate))
                ema50 = last_row.get('ema50', last_row.get('close', current_rate))
                close_price = last_row.get('close', current_rate)

                # Divergence check
                if not trade.is_short:
                    has_opposite_div = last_rows['total_bearish_divergences'].fillna(0).gt(0).any()
                else:
                    has_opposite_div = last_rows['total_bullish_divergences'].fillna(0).gt(0).any()

                # Momentum
                if len(dataframe) >= 5:
                    momentum_score = (close_price - dataframe.iloc[-5]['close']) / dataframe.iloc[-5]['close']
                
                if current_profit >= 0.01:
                    logger.info(f"   📊 RSI: {rsi:.1f} | Momentum: {momentum_score:.3f} | ADX: {adx:.1f}")
                    logger.info(f"   📈 EMA20: {ema20:.2f} | EMA50: {ema50:.2f}")
                    logger.info(f"   🔄 Opposite Div: {'⚠️ YES' if has_opposite_div else 'No'}")
                    
        except Exception as e:
            logger.debug(f"[{pair}] custom_exit data fetch error: {e}")

        # Logging helper
        def log_and_exit(reason: str):
            logger.info(f"🚪 {pair} EXIT [{reason}]")
            logger.info(f"   💰 Profit: {current_profit * 100:.2f}%")
            logger.info(f"   🕐 Duration: {int(trade_duration_minutes)} min")
            logger.info(f"   📊 RSI: {rsi:.1f} | ADX: {adx:.1f}")
            logger.info(f"   🏷️  Tag: {entry_tag}")
            logger.info(f"   ⚖️  Leverage: {trade.leverage}x")
            logger.info(f"   📈 Peak: {profit_peak * 100:.2f}%")
            if trade_id in self.profit_peaks:
                del self.profit_peaks[trade_id]
            return reason

        # =======================================================================
        # 2️⃣ EXTREME TSUNAMI PROTECTION
        # =======================================================================
        if last_row is not None and current_profit < -0.02:
            try:
                if not trade.is_short:
                    if adx > 60 and close_price < ema50 * 0.97:
                        if trade_duration_minutes >= 120:
                            return log_and_exit(f"Extreme_Tsunami_{current_profit*100:.1f}%")
                else:
                    if adx > 60 and close_price > ema50 * 1.03:
                        if trade_duration_minutes >= 120:
                            return log_and_exit(f"Extreme_Tsunami_Short_{current_profit*100:.1f}%")
            except Exception as e:
                logger.debug(f"[{pair}] Tsunami check error: {e}")

        # =======================================================================
        # 3️⃣ UNDER 1.0% ZONE - MINIMAL INTERFERENCE
        # =======================================================================
        if current_profit < 0.010:
            # Only extreme crashes
            if not trade.is_short and momentum_score < -0.045:
                if current_profit >= 0.005:
                    return log_and_exit(f"Extreme_Crash_{current_profit*100:.1f}%")
            elif trade.is_short and momentum_score > 0.045:
                if current_profit >= 0.005:
                    return log_and_exit(f"Extreme_Crash_Short_{current_profit*100:.1f}%")
            
            # Very long timeout - 12 hours
            if trade_duration_minutes >= 1440:
                if current_profit < 0.000:
                    return log_and_exit(f"Extended_Timeout_12HR_{current_profit*100:.1f}%")
            
            # Hold - let it develop
            return None

        # =======================================================================
        # 4️⃣ NEW: 1.0% - 1.5% ZONE - LIGHT PROTECTION
        # =======================================================================
        if current_profit >= 0.010 and current_profit < 0.015:
            very_light_trail_distance = 0.008  # 0.8% pullback
            very_light_trail_stop = profit_peak - very_light_trail_distance
            
            logger.info(f"   🎯 VERY LIGHT Trail: Peak {profit_peak*100:.1f}% | Stop at {very_light_trail_stop*100:.2f}%")
            logger.info(f"   📉 Pullback: {(profit_peak - current_profit)*100:.2f}% (exit if >= 0.8%)")
            
            # Only exit if peak was meaningful (1.2%+) and we've pulled back
            if current_profit < very_light_trail_stop and profit_peak >= 0.012:
                return log_and_exit(f"VeryLight_Trail_{profit_peak*100:.1f}%_to_{current_profit*100:.1f}%")
            
            return None

        # =======================================================================
        # 5️⃣ TIER 1: LIGHT TRAILING (1.5% - 3%)
        # =======================================================================
        if current_profit >= 0.015 and current_profit < 0.03:
            light_trail_distance = 0.012  # 1.2% pullback
            light_trail_stop = profit_peak - light_trail_distance
            
            logger.info(f"   🎯 LIGHT Trail: Peak {profit_peak*100:.1f}% | Stop at {light_trail_stop*100:.2f}%")
            logger.info(f"   📉 Pullback: {(profit_peak - current_profit)*100:.2f}% (exit if >= 1.2%)")
            
            if current_profit < light_trail_stop:
                return log_and_exit(f"Light_Trail_{profit_peak*100:.1f}%_to_{current_profit*100:.1f}%")
        
        # =======================================================================
        # 6️⃣ TIER 2: MEDIUM TRAILING (3% - 5%)
        # =======================================================================
        if current_profit >= 0.03 and current_profit < 0.05:
            medium_trail_distance = 0.010  # 1.0% pullback
            medium_trail_stop = profit_peak - medium_trail_distance
            
            logger.info(f"   🎯 MEDIUM Trail: Peak {profit_peak*100:.1f}% | Stop at {medium_trail_stop*100:.2f}%")
            logger.info(f"   📉 Pullback: {(profit_peak - current_profit)*100:.2f}% (exit if >= 1.0%)")
            
            if current_profit < medium_trail_stop:
                return log_and_exit(f"Medium_Trail_{profit_peak*100:.1f}%_to_{current_profit*100:.1f}%")
        
        # =======================================================================
        # 7️⃣ TIER 3: TIGHT TRAILING (5%+ profit)
        # =======================================================================
        if current_profit >= 0.05:
            # Adaptive trailing based on peak
            if profit_peak >= 0.10:
                trail_distance = 0.020  # 2.0% at 10%+
            elif profit_peak >= 0.08:
                trail_distance = 0.015  # 1.5% at 8%+
            else:
                trail_distance = 0.012  # 1.2% at 5-8%
            
            trailing_stop = profit_peak - trail_distance
            pullback_from_peak = profit_peak - current_profit
            
            logger.info(f"   🎯 TIGHT Trail: Distance {trail_distance*100:.1f}% | Stop at {trailing_stop*100:.2f}%")
            logger.info(f"   📉 Pullback: {pullback_from_peak*100:.2f}% (exit if >= {trail_distance*100:.1f}%)")
            
            if current_profit < trailing_stop:
                pullback_pct = pullback_from_peak * 100
                return log_and_exit(f"TRAIL_from_{profit_peak*100:.1f}%_pulled_{pullback_pct:.1f}%")

        # =======================================================================
        # 8️⃣ RSI EXTREME EXITS
        # =======================================================================
        if not trade.is_short:
            if rsi >= 85 and current_profit >= 0.05:
                return log_and_exit(f"RSI_85_{current_profit*100:.1f}%")
            if rsi >= 80 and current_profit >= 0.08:
                return log_and_exit(f"RSI_80_{current_profit*100:.1f}%")
        else:
            if rsi <= 15 and current_profit >= 0.05:
                return log_and_exit(f"RSI_15_SHORT_{current_profit*100:.1f}%")
            if rsi <= 20 and current_profit >= 0.08:
                return log_and_exit(f"RSI_20_SHORT_{current_profit*100:.1f}%")

        # =======================================================================
        # 9️⃣ STAGNANT TRADE EXITS
        # =======================================================================
        if 'Weak' in entry_tag or 'Early' in entry_tag:
            timeout = 360  # 6 hours
        elif 'Super' in entry_tag or 'Extreme' in entry_tag:
            timeout = 720  # 12 hours
        else:
            timeout = 540  # 9 hours
            
        if trade_duration_minutes >= timeout:
            if current_profit < 0.005:
                return log_and_exit(f"Stagnant_{int(timeout/60)}HR_{current_profit*100:.1f}%")

        # Force exit deeply losing trades after 16 hours
        if trade_duration_minutes >= 960 and current_profit < -0.04:
            return log_and_exit(f"Cut_Loss_16HR_{current_profit*100:.1f}%")

        # =======================================================================
        # 🔟 FRIDAY CLOSE
        # =======================================================================
        if current_time.weekday() == 4 and current_time.hour >= 22:
            if current_profit >= 0.015:
                return log_and_exit(f"Friday_Close_{current_profit*100:.1f}%")

        return None
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Capture market state when trade opens for ML training
        Called when a trade is about to be opened
        """
        
        if self.ml_enabled and ML_AVAILABLE:
            try:
                # Get current dataframe state
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                
                if len(dataframe) > 0:
                    # Extract features AT ENTRY TIME
                    features = self.extract_ml_features(dataframe, pair)
                    
                    # Initialize storage if needed
                    if not hasattr(self, 'trade_entry_features'):
                        self.trade_entry_features = {}
                    
                    # Generate temporary ID (use pair + timestamp as key)
                    temp_id = f"{pair}_{current_time.timestamp()}"
                    
                    # Store features for this trade
                    self.trade_entry_features[temp_id] = {
                        'features': features.tolist(),
                        'entry_time': current_time.isoformat(),
                        'entry_tag': entry_tag or 'Unknown',
                        'pair': pair,
                        'entry_rate': rate,
                        'timestamp': current_time.timestamp()
                    }
                    
                    logger.debug(f"📸 [ML] Captured entry features for {pair} at {entry_tag}")
                    
            except Exception as e:
                logger.error(f"❌ [ML] Error capturing entry features: {e}")
        
        return True
    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Called right after an order fills. 
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        
        LEARNS FROM ALL TRADES - Records performance for signal locking and Optuna optimization.
        
        :param pair: Pair for trade
        :param trade: trade object
        :param order: Order object
        :param current_time: datetime object, containing the current datetime
        :param kwargs: Ensure to keep this here so updates to this won't break your strategy
        """
        from logging import getLogger
        logger = getLogger(__name__)
        
        # Only process when trade is completely closed (exit order filled)
        if not trade.is_open and order.ft_order_side == 'sell':
            try:
                # Get trade details
                entry_tag = getattr(trade, 'enter_tag', 'Unknown')
                exit_reason = getattr(trade, 'exit_reason', 'Unknown')
                profit_ratio = trade.close_profit or 0.0
                profit_abs = trade.close_profit_abs or 0.0
                
                # Calculate trade duration
                duration = (trade.close_date_utc - trade.open_date_utc).total_seconds() / 60
                
                # Log trade close
                logger.info(f"🔒 TRADE CLOSED: {pair}")
                logger.info(f"   🏷️  Entry: {entry_tag} | Exit: {exit_reason}")
                logger.info(f"   💰 Profit: {profit_ratio*100:.2f}% ({profit_abs:.4f} USDT)")
                logger.info(f"   ⏱️  Duration: {duration:.0f} minutes")
                logger.info(f"   ⚖️  Leverage: {trade.leverage}x")
                
                # Record trade for signal performance tracking
                if not hasattr(self, 'trade_history'):
                    self.trade_history = {}
                
                if entry_tag not in self.trade_history:
                    self.trade_history[entry_tag] = []
                
                trade_record = {
                    'pair': pair,
                    'entry_tag': entry_tag,
                    'exit_reason': exit_reason,
                    'profit': profit_ratio,
                    'profit_abs': profit_abs,
                    'duration': duration,
                    'open_date': trade.open_date_utc,
                    'close_date': trade.close_date_utc,
                    'leverage': trade.leverage
                }
                
                self.trade_history[entry_tag].append(trade_record)
                
                # Keep only last 50 trades per signal (memory management)
                if len(self.trade_history[entry_tag]) > 50:
                    self.trade_history[entry_tag] = self.trade_history[entry_tag][-50:]
                
                # Trigger Optuna retraining if enabled
                if self.optuna_manager and hasattr(self, 'maybe_optimize_coin'):
                    # Check if we should retrain based on trade count
                    total_trades_for_pair = len([t for tag_trades in self.trade_history.values() 
                                                   for t in tag_trades if t['pair'] == pair])
                    
                    # Retrain every 20 trades
                    if total_trades_for_pair > 0 and total_trades_for_pair % 20 == 0:
                        logger.info(f"🔄 Triggering Optuna retraining for {pair} (trade #{total_trades_for_pair})")
                        self.maybe_optimize_coin(pair, force_startup=False)
                
                # Check signal performance for locking
                coin_params = self.coin_params.get(pair, {})
                signal_lock_enabled = coin_params.get('signal_lock_enabled', True)
                
                if signal_lock_enabled and entry_tag != 'Unknown':
                    recent_trades = self.trade_history[entry_tag][-15:]  # Last 15 trades
                    
                    if len(recent_trades) >= 8:  # Need at least 8 trades to evaluate
                        profits = [t['profit'] for t in recent_trades]
                        wins = sum(1 for p in profits if p > 0)
                        win_rate = wins / len(profits)
                        avg_profit = sum(profits) / len(profits)
                        
                        # Lock signal if performance is poor
                        if win_rate < 0.35 or avg_profit < -0.015:
                            logger.warning(f"⚠️ SIGNAL PERFORMANCE DEGRADED: {entry_tag}")
                            logger.warning(f"   Win Rate: {win_rate*100:.1f}% | Avg Profit: {avg_profit*100:.2f}%")
                            logger.warning(f"   🔒 Locking signal for 12 hours")
                            
                            if not hasattr(self, 'signal_locks'):
                                self.signal_locks = {}
                            
                            from datetime import timezone
                            self.signal_locks[lock_key] = datetime.now(timezone.utc)
                
                logger.info(f"✅ Trade close processing complete for {pair}")
                
            except Exception as e:
                logger.error(f"❌ Error in order_filled for {pair}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def maybe_optimize_coin(self, pair: str, force_startup: bool = False):
        """ENHANCED optimization trigger with profit protection before optimization"""
        if not self.optuna_manager:
            logger.debug(f"🚫 [OPTUNA] OptunaManager not available for {pair}")
            return
        
        # =======================================================================
        # 🆕 STEP 1: EXIT ALL PROFITABLE TRADES BEFORE OPTIMIZATION
        # =======================================================================
        try:
            open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
            profitable_trades = []
            
            for trade in open_trades:
                try:
                    # Calculate current profit
                    current_rate = trade.close_rate_requested if trade.close_rate_requested else trade.open_rate
                    current_profit = trade.calc_profit_ratio(current_rate)
                    
                    if current_profit > 0:
                        profitable_trades.append((trade, current_profit))
                except Exception as e:
                    logger.debug(f"[PRE-OPT] Error checking trade profit: {e}")
            
            if profitable_trades:
                logger.info(f"💰 [PRE-OPTIMIZATION] Found {len(profitable_trades)} profitable trades for {pair}")
                
                for trade, profit in profitable_trades:
                    logger.info(f"   🚪 Marking {pair} for exit at {profit*100:.2f}% profit before optimization")
                    trade.exit_reason = f"Pre_Opt_Profit_{profit*100:.1f}%"
                
                logger.info(f"✅ [PRE-OPTIMIZATION] Marked {len(profitable_trades)} trades for exit")
                
                # Give time for exits to process
                import time
                time.sleep(10)  # 10 seconds for orders to fill
            else:
                logger.info(f"ℹ️ [PRE-OPTIMIZATION] No profitable trades to exit for {pair}")
                
        except Exception as e:
            logger.error(f"❌ [PRE-OPTIMIZATION] Error exiting trades for {pair}: {e}")
            import traceback
            logger.debug(f"🔍 [PRE-OPT] Full traceback: {traceback.format_exc()}")
        
        # =======================================================================
        # STEP 2: PROCEED WITH OPTIMIZATION
        # =======================================================================
        
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
            optimization_count = self.optuna_manager.optimization_trigger_count.get(pair, 0)
            
            logger.info(f"📊 [OPTUNA] {pair} has {trades_count} trades, {optimization_count} optimizations done")
            
            # ENHANCED: More flexible trade requirements
            min_trades_required = max(0, optimization_count * 5)  # Require more trades for subsequent optimizations
            
            if not force_startup and trades_count < min_trades_required:
                logger.info(f"⏳ [OPTUNA] Not enough trades for {pair} optimization ({trades_count}/{min_trades_required})")
                return
            
            # Track optimization type
            if force_startup:
                opt_type = "STARTUP"
            elif self.optuna_manager.should_optimize_based_on_performance(pair):
                opt_type = "PERFORMANCE-TRIGGERED"
            else:
                opt_type = "PERIODIC"
            
            logger.info(f"✨ [OPTUNA] {opt_type} OPTIMIZATION for {pair} (trades: {trades_count})")
            
            objective_func = self.create_objective_function(pair)
            
            # ENHANCED: More trials for performance-triggered optimizations
            n_trials = 100 if opt_type == "PERFORMANCE-TRIGGERED" else 60
            
            # Start optimization
            self.optuna_manager.optimize_coin(pair, objective_func, n_trials=n_trials)
            
            # Track optimization count
            self.optuna_manager.optimization_trigger_count[pair] = optimization_count + 1
            
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
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """Track trade exits and update performance - works in backtesting"""
        
        try:
            profit_ratio = trade.calc_profit_ratio(rate)
            
            # Update Optuna performance if available
            if self.optuna_manager:
                self.optuna_manager.update_performance(pair, profit_ratio)
                logger.debug(f"📈 [OPTUNA] Updated performance for {pair}: {profit_ratio:.4f}")
            
            if pair not in self.coin_performance:
                self.coin_performance[pair] = 0.0
            self.coin_performance[pair] += profit_ratio
            
            # Update signal performance tracking for backtesting
            enter_tag = getattr(trade, 'enter_tag', None)
            if enter_tag:
                self.update_signal_performance(enter_tag, pair, profit_ratio)
                
                # Debug output
                result_type = "WIN" if profit_ratio > 0 else "LOSS"
                print(f"📊 Signal Performance Tracked: {pair} {enter_tag} - {result_type}")
                print(f"   Profit: {profit_ratio:.4f} ({profit_ratio*100:.2f}%)")
                print(f"   Exit reason: {exit_reason}")
            
            # Log significant performance updates
            if abs(profit_ratio) > 0.02:
                logger.info(f"💰 [OPTUNA] Significant trade result for {pair}: {profit_ratio:.2%} (cumulative: {self.coin_performance[pair]:.2%})")
            
        except Exception as e:
            logger.error(f"⛔ [OPTUNA] Failed to update performance for {pair}: {e}")

        # ML: Collect training data when trade closes - FIXED VERSION
        if self.ml_enabled and ML_AVAILABLE:
            try:
                profit_ratio = trade.calc_profit_ratio(rate)
                
                # Try to find stored entry features by pair
                if hasattr(self, 'trade_entry_features'):
                    # Find matching entry by pair (most recent for this pair)
                    matching_entries = {k: v for k, v in self.trade_entry_features.items() 
                                       if v['pair'] == pair}
                    
                    if matching_entries:
                        # Get most recent entry for this pair
                        latest_key = max(matching_entries.keys(), 
                                        key=lambda k: matching_entries[k]['timestamp'])
                        entry_data = self.trade_entry_features[latest_key]
                        
                        # Store complete trade data
                        trade_data = {
                            'pair': pair,
                            'features': entry_data['features'],
                            'signal_type': entry_data['entry_tag'],
                            'profit': profit_ratio,
                            'win': profit_ratio > 0,
                            'entry_time': entry_data['entry_time'],
                            'exit_time': current_time.isoformat(),
                            'exit_reason': exit_reason,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        self.ml_training_data.append(trade_data)
                        
                        # Cleanup
                        del self.trade_entry_features[latest_key]
                        
                        if len(self.ml_training_data) % 10 == 0:
                            self.save_ml_training_data()
                        
                        logger.info(f"✅ [ML] Collected data: {pair} {entry_data['entry_tag']} → {profit_ratio*100:.2f}%")
                    else:
                        logger.warning(f"⚠️ [ML] No entry features found for {pair}")
                        
            except Exception as e:
                logger.error(f"❌ [ML] Error collecting data: {e}")
        
        return True

    def on_trade_close(self, trade: Trade, **kwargs) -> None:
        """Called when a trade is closed - captures ALL exits including stoploss"""
        try:
            entry_tag = getattr(trade, 'enter_tag', '')
            if entry_tag:  # Only track if we have an entry tag
                pair = trade.pair
                profit_ratio = trade.calc_profit_ratio()
                
                # Update signal performance for ALL exits (wins and losses)
                self.update_signal_performance(entry_tag, pair, profit_ratio)
                
                # Log the trade result
                result_type = "WIN" if profit_ratio > 0 else "LOSS"
                exit_reason = getattr(trade, 'exit_reason', 'unknown')
                
                print(f"Trade closed: {pair} {entry_tag} - {result_type}")
                print(f"  Profit: {profit_ratio:.4f} ({profit_ratio*100:.2f}%)")
                print(f"  Exit reason: {exit_reason}")
                
        except Exception as e:
            print(f"Error in on_trade_close: {e}")

# ============================================================================
    # ML PREDICTIVE ENTRY METHODS
    # ============================================================================
    
    
    def extract_ml_features(self, dataframe: pd.DataFrame, pair: str) -> np.ndarray:
        """Extract features for ML prediction - ENHANCED VERSION"""
        try:
            last = dataframe.iloc[-1]
            prev = dataframe.iloc[-2] if len(dataframe) > 1 else last
            
            # Calculate recent profit average for this pair
            recent_profits = []
            if hasattr(self, 'ml_training_data'):
                pair_trades = [t for t in self.ml_training_data if t.get('pair') == pair]
                recent_profits = [t['profit'] for t in pair_trades[-10:]]
            recent_profit_avg = np.mean(recent_profits) if recent_profits else 0.0
            
            # NEW: Get divergence count
            div_count = last.get('total_bullish_divergences_count', 0) + last.get('total_bearish_divergences_count', 0)
            
            # NEW: Determine trend direction
            ema20 = last.get('ema20', last['close'])
            ema50 = last.get('ema50', last['close'])
            ema200 = last.get('ema200', last['close'])
            
            if ema20 > ema50 > ema200:
                trend_direction = 1.0  # Uptrend
            elif ema20 < ema50 < ema200:
                trend_direction = -1.0  # Downtrend
            else:
                trend_direction = 0.0  # Sideways
            
            features = np.array([
                last.get('rsi', 50),
                last.get('adx', 20),
                last.get('atr', 0) / last['close'] if last['close'] > 0 else 0,
                last.get('volume_ratio', 1.0),
                last.get('volatility', 0.01),
                1 if last.get('ema21', 0) > last.get('ema50', 0) else 0,
                (last['close'] - last.get('bollinger_lowerband', last['close'])) / 
                    (last.get('bollinger_upperband', last['close']) - last.get('bollinger_lowerband', last['close']) + 0.0001),
                last.get('signal_strength', 0),
                (last['close'] - last.get('ema21', last['close'])) / last['close'],
                (last['close'] - last.get('ema50', last['close'])) / last['close'],
                last.get('adx', 20) / 100.0,
                last.get('rsi', 50) / 100.0,
                1 if (last.get('total_bullish_divergences', 0) > 0 or last.get('total_bearish_divergences', 0) > 0) else 0,
                recent_profit_avg,
                div_count,  # NEW
                trend_direction  # NEW
            ])
            
            return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
        except Exception as e:
            logger.error(f"❌ [ML] Feature extraction error: {e}")
            return np.zeros(len(self.ml_feature_names))
    
    def ml_predict_entry(self, dataframe: pd.DataFrame, pair: str, direction: str) -> tuple:
        """ML predicts if we should enter now (long or short)"""
        if not self.ml_trained or self.ml_model is None:
            return False, 0.5
        
        try:
            features = self.extract_ml_features(dataframe, pair).reshape(1, -1)
            proba = self.ml_model.predict_proba(features)[0]
            confidence = proba[1]
            
            last_candle = dataframe.iloc[-1]
            rsi = last_candle.get('rsi', 50)
            
            if direction == 'long':
                direction_ok = rsi < 55
                threshold = 0.70
            else:
                direction_ok = rsi > 45
                threshold = 0.70
            
            should_enter = (confidence >= threshold) and direction_ok
            
            return should_enter, confidence
            
        except Exception as e:
            logger.error(f"❌ [ML] Prediction error: {e}")
            return False, 0.5
    
    def collect_ml_training_data(self, pair: str, trade: Trade, dataframe: pd.DataFrame, profit: float):
        """Collect data when trade closes"""
        try:
            features = self.extract_ml_features(dataframe, pair)
            
            trade_data = {
                'pair': pair,
                'features': features.tolist(),
                'profit': profit,
                'win': profit > 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.ml_training_data.append(trade_data)
            
            if len(self.ml_training_data) % 10 == 0:
                self.save_ml_training_data()
            
            if len(self.ml_training_data) >= 100 and len(self.ml_training_data) % 50 == 0:
                logger.info(f"🎯 [ML] Auto-retraining triggered ({len(self.ml_training_data)} trades)")
                self.train_ml_model()
                
        except Exception as e:
            logger.error(f"❌ [ML] Data collection error: {e}")
    
    def train_ml_model(self):
        """Train the ML model"""
        try:
            if len(self.ml_training_data) < 100:
                logger.warning(f"⚠️ [ML] Need 100+ trades, have {len(self.ml_training_data)}")
                return False
            
            logger.info(f"🎯 [ML] Training model with {len(self.ml_training_data)} trades...")
            
            X = np.array([t['features'] for t in self.ml_training_data])
            y = np.array([1 if t['profit'] > 0 else 0 for t in self.ml_training_data])
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.ml_model.fit(X_train, y_train)
            
            train_acc = self.ml_model.score(X_train, y_train)
            val_acc = self.ml_model.score(X_val, y_val)
            
            logger.info(f"✅ [ML] Training complete!")
            logger.info(f"   📊 Train accuracy: {train_acc:.1%}")
            logger.info(f"   📊 Val accuracy: {val_acc:.1%}")
            
            importances = self.ml_model.feature_importances_
            top_features = sorted(zip(self.ml_feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"   🔍 Top 5 features:")
            for feat, imp in top_features:
                logger.info(f"      • {feat}: {imp:.3f}")
            
            self.ml_trained = True
            self.save_ml_model()
            return True
            
        except Exception as e:
            logger.error(f"❌ [ML] Training failed: {e}")
            return False
    
    def save_ml_model(self):
        """Save ML model to disk"""
        try:
            from pathlib import Path
            model_dir = Path("user_data/strategies/ml_models")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "ml_entry_predictor.pkl"
            joblib.dump(self.ml_model, model_path)
            logger.info(f"💾 [ML] Model saved to {model_path}")
        except Exception as e:
            logger.error(f"❌ [ML] Failed to save model: {e}")
    
    def validate_and_load_ml_model(self):
        """Load ML model and validate feature compatibility"""
        try:
            from pathlib import Path
            model_path = Path("user_data/strategies/ml_models/ml_entry_predictor.pkl")
            
            if model_path.exists():
                temp_model = joblib.load(model_path)
                
                # Check if model expects correct number of features
                if hasattr(temp_model, 'n_features_in_'):
                    expected_features = temp_model.n_features_in_
                    current_features = len(self.ml_feature_names)
                    
                    if expected_features != current_features:
                        logger.warning(f"⚠️ [ML] Model mismatch: expects {expected_features}, code has {current_features}")
                        logger.warning(f"⚠️ [ML] Deleting old model - will retrain with new features")
                        model_path.unlink()
                        self.ml_model = None
                        self.ml_trained = False
                        return
                    else:
                        self.ml_model = temp_model
                        self.ml_trained = True
                        logger.info(f"📂 [ML] Model loaded ({expected_features} features)")
                else:
                    # Old sklearn version - just load it
                    self.ml_model = temp_model
                    self.ml_trained = True
                    logger.info(f"📂 [ML] Model loaded")
            else:
                logger.info(f"📂 [ML] No saved model found - will train from backtest")
                self.ml_model = None
                self.ml_trained = False
                
        except Exception as e:
            logger.error(f"❌ [ML] Failed to load model: {e}")
            self.ml_model = None
            self.ml_trained = False
    
    def save_ml_training_data(self):
        """Save training data to disk"""
        try:
            from pathlib import Path
            import json
            import numpy as np
            
            data_dir = Path("user_data/strategies/ml_models")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays and bools to native Python types
            cleaned_data = []
            for sample in self.ml_training_data:
                cleaned_sample = {}
                for key, value in sample.items():
                    if isinstance(value, np.ndarray):
                        cleaned_sample[key] = value.tolist()
                    elif isinstance(value, (np.bool_, bool)):
                        cleaned_sample[key] = bool(value)
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned_sample[key] = float(value)
                    else:
                        cleaned_sample[key] = value
                cleaned_data.append(cleaned_sample)
            
            data_path = data_dir / "ml_training_data.json"
            with open(data_path, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            
            logger.debug(f"💾 [ML] Training data saved ({len(self.ml_training_data)} trades)")
        except Exception as e:
            logger.error(f"❌ [ML] Failed to save training data: {e}")
    
    def load_ml_training_data(self):
        """Load training data from disk"""
        try:
            from pathlib import Path
            import json
            
            data_path = Path("user_data/strategies/ml_models/ml_training_data.json")
            
            if data_path.exists():
                with open(data_path, 'r') as f:
                    self.ml_training_data = json.load(f)
                
                logger.info(f"📂 [ML] Loaded {len(self.ml_training_data)} training samples")
                
                if not self.ml_trained and len(self.ml_training_data) >= 100:
                    logger.info(f"🎯 [ML] Starting initial training...")
                    self.train_ml_model()
            else:
                logger.info(f"📂 [ML] No training data found - starting fresh")
                self.ml_training_data = []
                
        except Exception as e:
            logger.error(f"❌ [ML] Failed to load training data: {e}")
            self.ml_training_data = []
    # ============================================================================
    # ML BACKTEST TRAINING SYSTEM
    # ============================================================================

    def train_ml_from_backtest(self) -> bool:
        """
        Generate ML training data from backtesting historical data
        Runs on startup if no trained model exists
        """
        try:
            logger.info("=" * 80)
            logger.info("🎯 [ML BACKTEST] Starting backtest-based training from disk files")
            logger.info("=" * 80)
            
            from pathlib import Path
            import time
            from freqtrade.data.history import load_pair_history
            
            start_time = time.time()
            training_samples = []
            
            # Get exchange name from config
            exchange_name = self.config.get('exchange', {}).get('name', 'bybit')
            data_dir = Path(f"user_data/data/{exchange_name}")
            
            logger.info(f"📂 [ML BACKTEST] Loading data from: {data_dir}")
            
            # Get list of pairs - try whitelist first, then scan directory
            try:
                pairs = self.dp.current_whitelist() if hasattr(self, 'dp') and self.dp else []
            except:
                pairs = []
            
            # If no pairs from whitelist, scan data directory
            if not pairs:
                logger.info("📂 [ML BACKTEST] Scanning data directory for available pairs...")
                timeframe_str = self.timeframe.replace('m', '')
                json_files = list(data_dir.glob(f"*-{self.timeframe}.json"))
                pairs = [f.stem.replace(f"-{self.timeframe}", "").replace("_", "/") for f in json_files]
                logger.info(f"📂 [ML BACKTEST] Found {len(pairs)} pairs with data files")
            
            if not pairs:
                logger.error("❌ [ML BACKTEST] No pairs found - check your data directory!")
                return False
                
            logger.info(f"📊 [ML BACKTEST] Training on {len(pairs)} pairs")
            
            # For each pair, generate training data
            for idx, pair in enumerate(pairs):
                logger.info(f"🔄 [ML BACKTEST] Processing {pair} ({idx+1}/{len(pairs)})")
                
                try:
                    # Load data from disk
                    dataframe = load_pair_history(
                        datadir=data_dir,
                        timeframe=self.timeframe,
                        pair=pair,
                        data_format='json',
                        candle_type=self.config.get('candle_type_def', 'spot')
                    )
                    
                    if dataframe is None or len(dataframe) < 500:
                        logger.warning(f"⚠️ [ML BACKTEST] Not enough data for {pair} ({len(dataframe) if dataframe is not None else 0} candles), skipping")
                        continue
                    
                    # Process this pair's data with indicators
                    logger.debug(f"📊 [ML BACKTEST] Populating indicators for {pair}...")
                    dataframe = self.populate_indicators(dataframe, {'pair': pair})
                    
                    # Extract training samples
                    pair_samples = self.extract_backtest_training_data(dataframe, pair)
                    training_samples.extend(pair_samples)
                    
                    logger.info(f"✅ [ML BACKTEST] {pair}: {len(pair_samples)} samples")
                    
                except Exception as e:
                    logger.error(f"❌ [ML BACKTEST] Error processing {pair}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
            
            # Check if we have enough data
            if len(training_samples) < 100:
                logger.error(f"❌ [ML BACKTEST] Only {len(training_samples)} samples - need 100+")
                return False
            
            logger.info(f"🎯 [ML BACKTEST] Collected {len(training_samples)} total samples")
            
            # Save training data
            self.ml_training_data = training_samples
            self.save_ml_training_data()
            
            # Train the model
            success = self.train_ml_model()
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ [ML BACKTEST] Training completed in {elapsed:.1f} seconds")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ [ML BACKTEST] Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def extract_backtest_training_data(self, dataframe: pd.DataFrame, pair: str) -> list:
        """
        Extract training samples from historical data for one pair
        
        For each candle where a signal would have triggered:
        - Capture market state (features)
        - Look ahead 15-50 candles
        - Label: Did it profit >1%?
        """
        samples = []
        
        try:
            # Need enough candles to look ahead
            lookback_window = 50  # Check next 50 candles (12.5 hours on 15m)
            profit_threshold = 0.01  # 1% profit target
            
            # Process each candle (except last 50, can't look ahead)
            for i in range(100, len(dataframe) - lookback_window):
                current_candle = dataframe.iloc[i]
                
                # Check if ANY signal would trigger at this candle
                signal_type = self.detect_signal_at_candle(dataframe, i)
                
                if signal_type:  # Signal detected!
                    # Extract features at THIS moment
                    features = self.extract_features_at_index(dataframe, i, pair)
                    
                    # Look ahead: would this trade be profitable?
                    is_profitable = self.check_future_profitability(
                        dataframe, i, lookback_window, profit_threshold, signal_type
                    )
                    
                    # Store sample
                    sample = {
                        'pair': pair,
                        'features': features.tolist(),
                        'signal_type': signal_type,
                        'profit': 0.015 if is_profitable else -0.01,  # Approximate
                        'win': is_profitable,
                        'timestamp': str(current_candle.name),
                        'candle_index': i
                    }
                    
                    samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.error(f"❌ [ML BACKTEST] Error extracting data: {e}")
            return []

    def detect_signal_at_candle(self, dataframe: pd.DataFrame, index: int) -> str:
        """
        Check if any signal would trigger at this candle
        Returns signal name or None
        """
        try:
            row = dataframe.iloc[index]
            prev_row = dataframe.iloc[index-1] if index > 0 else row
            
            # Check for divergence signals (simplified)
            if row.get('total_bullish_divergences', 0) > 0:
                div_count = row.get('total_bullish_divergences_count', 0)
                
                if div_count >= 3:
                    return 'Bull_Super'
                elif div_count == 2:
                    return 'Bull_HQ'
                else:
                    return 'Bull_Div'
            
            if row.get('total_bearish_divergences', 0) > 0:
                div_count = row.get('total_bearish_divergences_count', 0)
                
                if div_count >= 3:
                    return 'Bear_Super'
                elif div_count == 2:
                    return 'Bear_HQ'
                else:
                    return 'Bear_Div'
            
            # Check Fisher crossover
            fisher = row.get('fisher', 0)
            fisher_signal = row.get('fisher_signal', 0)
            prev_fisher = prev_row.get('fisher', 0)
            prev_fisher_signal = prev_row.get('fisher_signal', 0)
            
            if (fisher > fisher_signal and prev_fisher <= prev_fisher_signal):
                if row.get('rsi', 50) < 40:
                    return 'Bull_Fisher'
            
            if (fisher < fisher_signal and prev_fisher >= prev_fisher_signal):
                if row.get('rsi', 50) > 60:
                    return 'Bear_Fisher'
            
            # Check Early signals
            if row.get('total_bullish_divergences', 0) > 0 and row.get('rsi', 50) < 50:
                return 'Bull_Early'
            
            if row.get('total_bearish_divergences', 0) > 0 and row.get('rsi', 50) > 50:
                return 'Bear_Early'
            
            return None  # No signal
            
        except Exception as e:
            return None

    def check_future_profitability(self, dataframe: pd.DataFrame, entry_index: int, 
                                    lookback: int, threshold: float, signal_type: str) -> bool:
        """
        Look ahead and check if trade would have been profitable
        """
        try:
            entry_price = dataframe.iloc[entry_index]['close']
            is_long = 'Bull' in signal_type
            
            # Look at next N candles
            future_candles = dataframe.iloc[entry_index+1:entry_index+lookback+1]
            
            if len(future_candles) < 10:
                return False
            
            if is_long:
                # For longs: did price go up >threshold?
                max_profit = (future_candles['high'].max() - entry_price) / entry_price
                return max_profit >= threshold
            else:
                # For shorts: did price go down >threshold?
                max_profit = (entry_price - future_candles['low'].min()) / entry_price
                return max_profit >= threshold
                
        except Exception as e:
            return False

    def extract_features_at_index(self, dataframe: pd.DataFrame, index: int, pair: str) -> np.ndarray:
        """Extract features at a specific candle index (for backtesting)"""
        try:
            row = dataframe.iloc[index]
            prev_row = dataframe.iloc[index-1] if index > 0 else row
            
            # Get divergence count
            div_count = (row.get('total_bullish_divergences_count', 0) + 
                         row.get('total_bearish_divergences_count', 0))
            
            # Determine trend
            ema20 = row.get('ema20', row['close'])
            ema50 = row.get('ema50', row['close'])
            ema200 = row.get('ema200', row['close'])
            
            if ema20 > ema50 > ema200:
                trend_direction = 1.0
            elif ema20 < ema50 < ema200:
                trend_direction = -1.0
            else:
                trend_direction = 0.0
            
            features = np.array([
                row.get('rsi', 50),
                row.get('adx', 20),
                row.get('atr', 0) / row['close'] if row['close'] > 0 else 0,
                row.get('volume_ratio', 1.0),
                row.get('volatility', 0.01),
                1 if row.get('ema21', 0) > row.get('ema50', 0) else 0,
                (row['close'] - row.get('bollinger_lowerband', row['close'])) / 
                    (row.get('bollinger_upperband', row['close']) - row.get('bollinger_lowerband', row['close']) + 0.0001),
                row.get('signal_strength', 0),
                (row['close'] - row.get('ema21', row['close'])) / row['close'],
                (row['close'] - row.get('ema50', row['close'])) / row['close'],
                row.get('adx', 20) / 100.0,
                row.get('rsi', 50) / 100.0,
                1 if (row.get('total_bullish_divergences', 0) > 0 or 
                      row.get('total_bearish_divergences', 0) > 0) else 0,
                0.0,  # recent_profit_avg (not available in backtest)
                div_count,
                trend_direction
            ])
            
            return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
        except Exception as e:
            logger.error(f"❌ [ML] Feature extraction error at index {index}: {e}")
            return np.zeros(len(self.ml_feature_names))
def ema_cross_check(dataframe):
    dataframe['ema20_50_cross'] = qtpylib.crossed_below(dataframe[resample('ema20')],dataframe[resample('ema50')])
    dataframe['ema20_200_cross'] = qtpylib.crossed_below(dataframe[resample('ema20')],dataframe[resample('ema200')])
    dataframe['ema50_200_cross'] = qtpylib.crossed_below(dataframe[resample('ema50')],dataframe[resample('ema200')])
    return ~(
            dataframe['ema20_50_cross']
            | dataframe['ema20_200_cross']
            | dataframe['ema50_200_cross']
    )

def green_candle(dataframe):
    return dataframe[resample('open')] < dataframe[resample('close')]
def red_candle(dataframe):
    return dataframe[resample('open')] > dataframe[resample('close')]

def keltner_middleband_check(dataframe):
    return (dataframe[resample('low')] < dataframe[resample('kc_middleband')]) & (dataframe[resample('high')] > dataframe[resample('kc_middleband')])

def keltner_lowerband_check(dataframe):
    return (dataframe[resample('low')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('kc_lowerband')])

def bollinger_lowerband_check(dataframe):
    return (dataframe[resample('low')] < dataframe[resample('bollinger_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('bollinger_lowerband')])

def bollinger_keltner_check(dataframe):
    return (dataframe[resample('bollinger_lowerband')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('bollinger_upperband')] > dataframe[resample('kc_upperband')])

def keltner_upperband_check(dataframe):
    return (dataframe[resample('high')] > dataframe[resample('kc_upperband')]) & (dataframe[resample('low')] < dataframe[resample('kc_upperband')])

def bollinger_upperband_check(dataframe):
    return (dataframe[resample('high')] > dataframe[resample('bollinger_upperband')]) & (dataframe[resample('low')] < dataframe[resample('bollinger_upperband')])




def ema_check(dataframe):
    check = (
            (dataframe[resample('ema9')] < dataframe[resample('ema20')])
            & (dataframe[resample('ema20')] < dataframe[resample('ema50')])
            & (dataframe[resample('ema50')] < dataframe[resample('ema200')]))
    return ~check

from enum import Enum
class PivotSource(Enum):
    HighLow = 0
    Close = 1

def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source: PivotSource = PivotSource.Close) -> DataFrame:
    high_source = None
    low_source = None

    if pivot_source == PivotSource.Close:
        high_source = 'close'
        low_source = 'close'
    elif pivot_source == PivotSource.HighLow:
        high_source = 'high'
        low_source = 'low'

    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()

    # find pivot points
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index - window] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index - window] = getattr(current_value, low_source)
            last_values.popleft()

    # find last one
    if len(last_values) >= window + 2:
        current_value = last_values[-2]
        is_greater = True
        is_less = True
        for window_index in range(0, window):
            left = last_values[-2 - window_index - 1]
            right = last_values[-1]
            local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
            is_greater &= local_is_greater
            is_less &= local_is_less
        if is_greater:
            pivot_points_highs[index - 1] = getattr(current_value, high_source)
        if is_less:
            pivot_points_lows[index - 1] = getattr(current_value, low_source)

    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    is_greater = True
    is_less = True
    if (getattr(current_value, high_source) < getattr(left, high_source) or
            getattr(current_value, high_source) < getattr(right, high_source)):
        is_greater = False

    if (getattr(current_value, low_source) > getattr(left, low_source) or
            getattr(current_value, low_source) > getattr(right, low_source)):
        is_less = False
    return (is_greater, is_less)

def emaKeltner(dataframe):
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def is_sideways(dataframe, coin_params=None):
    """
    Comprehensive sideways market detection - OPTUNA OPTIMIZED
    Uses coin-specific thresholds for best performance per pair
    """
    if coin_params is None:
        coin_params = {}
    
    # Method 1: ADX (trend strength)
    adx_threshold = coin_params.get('sideways_adx_max', 25)
    weak_trend = (dataframe['adx'] < adx_threshold)
    
    # Method 2: Band Width (volatility)
    kc_width = (dataframe['kc_upperband'] - dataframe['kc_lowerband']) / dataframe['kc_middleband']
    kc_threshold = coin_params.get('sideways_kc_width_max', 0.02)
    narrow_bands = (kc_width < kc_threshold)
    
    # Method 3: EMA Flatness (direction)
    ema_change = abs(dataframe['ema_long'] - dataframe['ema_long'].shift(10)) / dataframe['ema_long'].shift(10)
    ema_threshold = coin_params.get('sideways_ema_flat_max', 0.015)
    flat_ema = (ema_change < ema_threshold)
    
    # Method 4: Price Range (consolidation)
    recent_high = dataframe['high'].rolling(20).max()
    recent_low = dataframe['low'].rolling(20).min()
    price_range = (recent_high - recent_low) / dataframe['close']
    range_threshold = coin_params.get('sideways_range_max', 0.06)
    tight_range = (price_range < range_threshold)
    
    # Combine: Count how many conditions are met
    sideways_score = (
        weak_trend.astype(int) +
        narrow_bands.astype(int) +
        flat_ema.astype(int) +
        tight_range.astype(int)
    )
    
    # Require minimum number of conditions (2 or 3)
    min_conditions = coin_params.get('sideways_min_conditions', 2)
    
    return (sideways_score >= min_conditions)