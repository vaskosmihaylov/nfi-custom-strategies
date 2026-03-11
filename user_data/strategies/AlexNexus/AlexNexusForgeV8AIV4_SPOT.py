import logging
import numpy as np
import pandas as pd
import pickle
import warnings
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from importlib import metadata
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
import talib.abstract as ta
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
)
from freqtrade.persistence import Trade

# Suppress deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import TimeSeriesSplit

    # V4: Purged TimeSeriesSplit with gap for preventing leakage
    class GapTimeSeriesSplit(TimeSeriesSplit):
        """V4: TimeSeriesSplit with purge gap between train and test"""

        def __init__(self, n_splits=3, gap=5):
            super().__init__(n_splits=n_splits)
            self.gap = gap

        def split(self, X, y=None, groups=None):
            for train_idx, test_idx in super().split(X):
                if len(train_idx) == 0:
                    continue
                # Purge last 'gap' observations from train set
                if len(train_idx) > self.gap:
                    train_idx = train_idx[: -self.gap]
                yield train_idx, test_idx

    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        AdaBoostClassifier,
        VotingClassifier,
        HistGradientBoostingClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.calibration import CalibratedClassifierCV

    SKLEARN_AVAILABLE = True

    # Check sklearn version using modern approach
    try:
        sklearn_version = metadata.version("scikit-learn")
        logger.info(f"Using scikit-learn version: {sklearn_version}")
    except Exception as e:
        logger.debug(f"Could not get sklearn version: {e}")

except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# Modern PyWavelets import
try:
    import pywt

    WAVELETS_AVAILABLE = True
    try:
        pywt_version = metadata.version("PyWavelets")
        logger.info(f"Using PyWavelets version: {pywt_version}")
    except Exception as e:
        logger.debug(f"Could not get PyWavelets version: {e}")
except ImportError as e:
    logger.warning(f"PyWavelets not available: {e}")
    WAVELETS_AVAILABLE = False

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P",
    "[-2/8]P",
    "[-1/8]P",
    "[0/8]P",
    "[1/8]P",
    "[2/8]P",
    "[3/8]P",
    "[4/8]P",
    "[5/8]P",
    "[6/8]P",
    "[7/8]P",
    "[8/8]P",
    "[+1/8]P",
    "[+2/8]P",
    "[+3/8]P",
]


def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):
        window_data = df["ha_close"].iloc[i - window : i + 1]
        if (
            df["ha_close"].iloc[i] == window_data.min()
            and (window_data == df["ha_close"].iloc[i]).sum() == 1
        ):
            minima[i] = -window
        if (
            df["ha_close"].iloc[i] == window_data.max()
            and (window_data == df["ha_close"].iloc[i]).sum() == 1
        ):
            maxima[i] = window

    return minima, maxima


def calc_slope_advanced(series, period):
    """
    Enhanced linear regression slope calculation with Wavelet Transform and FFT analysis
    for superior trend detection and noise filtering
    """
    if len(series) < period:
        return 0

    # Use only the last 'period' values for consistency
    y = series.values[-period:]

    # Enhanced data validation
    if np.isnan(y).any() or np.isinf(y).any():
        return 0

    # Check for constant values (no trend)
    if np.all(y == y[0]):
        return 0

    try:
        # === 1. WAVELET DENOISING ===
        # V4: Skip wavelets if data is too long (optimization)
        if WAVELETS_AVAILABLE and len(y) >= 8 and len(y) <= 500:
            wavelet = "db4"
            try:
                w = pywt.Wavelet(wavelet)
                max_level = pywt.dwt_max_level(len(y), w.dec_len)
                use_level = min(3, max_level)  # cap at 3 but adapt if shorter series
            except Exception:
                use_level = 1
            if use_level >= 1:
                coeffs = pywt.wavedec(y, wavelet, level=use_level, mode="periodization")
                threshold = 0.1 * np.std(coeffs[-1]) if len(coeffs) > 1 else 0.0
                coeffs_thresh = list(coeffs)
                for i in range(1, len(coeffs_thresh)):
                    coeffs_thresh[i] = pywt.threshold(
                        coeffs_thresh[i], threshold, mode="soft"
                    )
                y_denoised = pywt.waverec(coeffs_thresh, wavelet, mode="periodization")
                if len(y_denoised) != len(y):
                    y_denoised = y_denoised[: len(y)]
            else:
                y_denoised = y
        else:
            y_denoised = y

        # === 2. FFT FREQUENCY ANALYSIS ===
        # Analyze dominant frequencies to identify trend components
        if len(y_denoised) >= 4:
            # Apply FFT
            fft_values = fft(y_denoised)
            freqs = fftfreq(len(y_denoised))

            # Get magnitude spectrum
            magnitude = np.abs(fft_values)

            # Find dominant frequency (excluding DC component)
            non_dc_indices = np.where(freqs != 0)[0]
            if len(non_dc_indices) > 0:
                dominant_freq_idx = non_dc_indices[np.argmax(magnitude[non_dc_indices])]
                dominant_freq = freqs[dominant_freq_idx]

                # Calculate trend strength based on frequency content
                trend_frequency_weight = 1.0 / (1.0 + abs(dominant_freq) * 10)
            else:
                trend_frequency_weight = 1.0
        else:
            trend_frequency_weight = 1.0

        # === 3. MULTI-SCALE SLOPE CALCULATION ===
        x = np.linspace(0, period - 1, period)

        # Original slope calculation
        slope_original = np.polyfit(x, y, 1)[0]

        # Wavelet-denoised slope calculation
        slope_denoised = np.polyfit(x, y_denoised, 1)[0]

        # === 4. WAVELET-BASED TREND DECOMPOSITION ===
        # V4: Skip wavelets if data is too long (optimization)
        if WAVELETS_AVAILABLE and len(y) >= 8 and len(y) <= 500:
            # Extract trend component using wavelet approximation
            approx_coeffs = coeffs[0]  # Approximation coefficients (trend)

            # Reconstruct trend component
            trend_component = pywt.upcoef(
                "a", approx_coeffs, wavelet, level=3, take=len(y)
            )
            if len(trend_component) > len(y):
                trend_component = trend_component[: len(y)]
            elif len(trend_component) < len(y):
                # Pad with last value if needed
                pad_length = len(y) - len(trend_component)
                trend_component = np.pad(trend_component, (0, pad_length), mode="edge")

            # Calculate slope of trend component
            slope_trend = np.polyfit(x, trend_component, 1)[0]
        else:
            slope_trend = slope_denoised

        # === 5. FREQUENCY-WEIGHTED SLOPE COMBINATION ===
        # Weight slopes based on signal characteristics
        weights = {"original": 0.3, "denoised": 0.4, "trend": 0.3}

        # Adjust weights based on noise level
        noise_level = np.std(y - y_denoised) / np.std(y) if np.std(y) > 0 else 0
        if noise_level > 0.1:  # High noise
            weights = {"original": 0.2, "denoised": 0.5, "trend": 0.3}
        elif noise_level < 0.05:  # Low noise
            weights = {"original": 0.4, "denoised": 0.3, "trend": 0.3}

        # Combined slope calculation
        slope_combined = (
            slope_original * weights["original"]
            + slope_denoised * weights["denoised"]
            + slope_trend * weights["trend"]
        )

        # Apply frequency weighting
        final_slope = slope_combined * trend_frequency_weight

        # === 6. ENHANCED VALIDATION ===
        if np.isnan(final_slope) or np.isinf(final_slope):
            return (
                slope_original
                if not (np.isnan(slope_original) or np.isinf(slope_original))
                else 0
            )

        # Normalize extreme slopes
        max_reasonable_slope = np.std(y) / period
        if abs(final_slope) > max_reasonable_slope * 15:
            return np.sign(final_slope) * max_reasonable_slope * 15

        return final_slope

    except Exception:
        # Fallback to enhanced simple method if advanced processing fails
        try:
            # Apply simple moving average smoothing as fallback
            if len(y) >= 3:
                y_smooth = (
                    pd.Series(y)
                    .rolling(window=3, center=True)
                    .mean()
                    .bfill()
                    .ffill()
                    .values
                )
                x = np.linspace(0, period - 1, period)
                slope = np.polyfit(x, y_smooth, 1)[0]

                if not (np.isnan(slope) or np.isinf(slope)):
                    return slope

            # Ultimate fallback: simple difference
            simple_slope = (y[-1] - y[0]) / (period - 1)
            return (
                simple_slope
                if not (np.isnan(simple_slope) or np.isinf(simple_slope))
                else 0
            )

        except Exception:
            return 0


def calculate_advanced_trend_strength_with_wavelets(
    dataframe: pd.DataFrame,
    strong_threshold: float = 0.02,
    pair: str = None,
    feature_cache: dict = None,
    last_cache_update: dict = None,
) -> pd.DataFrame:
    """
    Enhanced trend strength calculation using Wavelet Transform and FFT analysis
    V4 FIX: Made cache parameters optional to work as standalone function
    """
    try:
        # === WAVELET-ENHANCED SLOPE CALCULATION ===
        dataframe["slope_5_advanced"] = (
            dataframe["close"]
            .rolling(5)
            .apply(lambda x: calc_slope_advanced(x, 5), raw=False)
        )
        dataframe["slope_10_advanced"] = (
            dataframe["close"]
            .rolling(10)
            .apply(lambda x: calc_slope_advanced(x, 10), raw=False)
        )
        dataframe["slope_20_advanced"] = (
            dataframe["close"]
            .rolling(20)
            .apply(lambda x: calc_slope_advanced(x, 20), raw=False)
        )

        # === WAVELET TREND DECOMPOSITION ===
        def wavelet_trend_analysis(series, window=20):
            """Analyze trend using adaptive wavelet (haar/db4), safe levels, symmetric mode, robust threshold."""
            if not WAVELETS_AVAILABLE or len(series) < window:
                return pd.Series([0.0] * len(series), index=series.index)
            results: list[float] = []
            for i in range(len(series)):
                if i < window:
                    results.append(0.0)
                    continue
                window_data = series.iloc[i - window + 1 : i + 1].values
                n = len(window_data)
                if n < 12:
                    results.append(0.0)
                    continue
                wavelet_name = "haar" if n < 24 else "db4"
                try:
                    w = pywt.Wavelet(wavelet_name)
                    max_level = pywt.dwt_max_level(n, w.dec_len)
                except Exception:
                    max_level = 1
                if n < 48:
                    max_level = min(max_level, 2)
                use_level = max(1, min(3, max_level))
                try:
                    coeffs = pywt.wavedec(
                        window_data, wavelet_name, level=use_level, mode="symmetric"
                    )
                    # Estimate sigma from finest detail
                    if len(coeffs) > 1 and len(coeffs[-1]):
                        detail = coeffs[-1]
                        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
                        thr = sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
                    else:
                        thr = 0.0
                    for j in range(1, len(coeffs)):
                        coeffs[j] = pywt.threshold(coeffs[j], thr, mode="soft")
                    approx = coeffs[0]
                    trend_strength = np.std(approx) / (np.std(window_data) + 1e-9)
                    direction = 0
                    if len(approx) >= 2:
                        direction = 1 if approx[-1] > approx[0] else -1
                    score = trend_strength * direction
                    if not np.isfinite(score):
                        score = 0.0
                    # Clamp extreme outliers
                    results.append(float(np.clip(score, -5, 5)))
                except Exception:
                    results.append(0.0)
            return pd.Series(results, index=series.index)

        # Apply wavelet trend analysis
        dataframe["wavelet_trend_strength"] = wavelet_trend_analysis(dataframe["close"])

        # === FFT-BASED CYCLE DETECTION ===
        def fft_cycle_analysis(series, window=50):
            """Detect market cycles using FFT"""
            if len(series) < window:
                return (
                    pd.Series([0] * len(series), index=series.index),
                    pd.Series([0] * len(series), index=series.index),
                )

            cycle_strength = []
            dominant_period = []

            for i in range(len(series)):
                if i < window:
                    cycle_strength.append(0)
                    dominant_period.append(0)
                    continue

                # Get window data
                window_data = series.iloc[i - window + 1 : i + 1].values

                try:
                    # Remove linear trend
                    x = np.arange(len(window_data))
                    slope, intercept = np.polyfit(x, window_data, 1)
                    detrended = window_data - (slope * x + intercept)

                    # Apply FFT
                    fft_values = fft(detrended)
                    freqs = fftfreq(len(detrended))
                    magnitude = np.abs(fft_values)

                    # Find dominant cycle (excluding DC component)
                    positive_freqs = freqs[1 : len(freqs) // 2]
                    positive_magnitude = magnitude[1 : len(magnitude) // 2]

                    if len(positive_magnitude) > 0:
                        max_idx = np.argmax(positive_magnitude)
                        dominant_freq = positive_freqs[max_idx]
                        dominant_per = 1.0 / (abs(dominant_freq) + 1e-8)

                        # Cycle strength (normalized)
                        cycle_str = positive_magnitude[max_idx] / (
                            np.sum(positive_magnitude) + 1e-8
                        )
                    else:
                        dominant_per = 0
                        cycle_str = 0

                    cycle_strength.append(cycle_str)
                    dominant_period.append(dominant_per)

                except Exception:
                    cycle_strength.append(0)
                    dominant_period.append(0)

            return (
                pd.Series(cycle_strength, index=series.index),
                pd.Series(dominant_period, index=series.index),
            )

        # V4: Optimized FFT with caching
        # Fix: Made cache optional - function works with or without it
        cache_key = (
            f"{pair}_fft_{len(dataframe)}" if pair else f"default_fft_{len(dataframe)}"
        )
        current_time = datetime.now()

        # Check if cache is valid (less than 5 candles old)
        use_cache = feature_cache is not None and last_cache_update is not None

        if (
            use_cache
            and cache_key in feature_cache
            and cache_key in last_cache_update
            and (current_time - last_cache_update[cache_key]).seconds < 300
        ):  # 5 min for 1h candles
            # Use cached FFT results
            cached_results = feature_cache[cache_key]
            dataframe["cycle_strength"] = cached_results["cycle_strength"]
            dataframe["dominant_cycle_period"] = cached_results["dominant_period"]
        else:
            # Calculate FFT (only last 500 candles for efficiency)
            if len(dataframe) > 500:
                recent_data = dataframe["close"].tail(500)
                (cycle_str, dominant_per) = fft_cycle_analysis(recent_data)
                # Pad with zeros for older data
                padding_length = len(dataframe) - 500
                cycle_strength_full = pd.concat(
                    [
                        pd.Series(
                            [0] * padding_length, index=dataframe.index[:padding_length]
                        ),
                        cycle_str,
                    ]
                )
                dominant_period_full = pd.concat(
                    [
                        pd.Series(
                            [0] * padding_length, index=dataframe.index[:padding_length]
                        ),
                        dominant_per,
                    ]
                )
                dataframe["cycle_strength"] = cycle_strength_full
                dataframe["dominant_cycle_period"] = dominant_period_full
            else:
                (dataframe["cycle_strength"], dataframe["dominant_cycle_period"]) = (
                    fft_cycle_analysis(dataframe["close"])
                )

            # Cache results if cache is available
            if use_cache:
                feature_cache[cache_key] = {
                    "cycle_strength": dataframe["cycle_strength"],
                    "dominant_period": dataframe["dominant_cycle_period"],
                }
                last_cache_update[cache_key] = current_time

        # === ENHANCED TREND STRENGTH CALCULATION ===
        # Normalize advanced slopes by price
        dataframe["trend_strength_5_advanced"] = (
            dataframe["slope_5_advanced"] / dataframe["close"] * 100
        )
        dataframe["trend_strength_10_advanced"] = (
            dataframe["slope_10_advanced"] / dataframe["close"] * 100
        )
        dataframe["trend_strength_20_advanced"] = (
            dataframe["slope_20_advanced"] / dataframe["close"] * 100
        )

        # Wavelet-weighted combined trend strength
        dataframe["trend_strength_wavelet"] = (
            dataframe["trend_strength_5_advanced"] * 0.4
            + dataframe["trend_strength_10_advanced"] * 0.35
            + dataframe["trend_strength_20_advanced"] * 0.25
        )

        # Incorporate wavelet trend analysis
        dataframe["trend_strength_combined"] = (
            dataframe["trend_strength_wavelet"] * 0.7
            + dataframe["wavelet_trend_strength"] * 0.3
        )

        # === CYCLE-ADJUSTED TREND STRENGTH ===
        # Adjust trend strength based on cycle analysis
        dataframe["trend_strength_cycle_adjusted"] = dataframe[
            "trend_strength_combined"
        ].copy()

        # Boost trend strength when aligned with dominant cycle
        strong_cycle_mask = dataframe["cycle_strength"] > 0.3
        dataframe.loc[strong_cycle_mask, "trend_strength_cycle_adjusted"] *= (
            1 + dataframe.loc[strong_cycle_mask, "cycle_strength"]
        )

        # === FINAL TREND CLASSIFICATION WITH ADVANCED FEATURES ===
        # strong_threshold is now passed as parameter

        # Enhanced trend classification
        dataframe["strong_uptrend_advanced"] = (
            (dataframe["trend_strength_cycle_adjusted"] > strong_threshold)
            & (dataframe["wavelet_trend_strength"] > 0)
            & (dataframe["cycle_strength"] > 0.1)
        )

        dataframe["strong_downtrend_advanced"] = (
            (dataframe["trend_strength_cycle_adjusted"] < -strong_threshold)
            & (dataframe["wavelet_trend_strength"] < 0)
            & (dataframe["cycle_strength"] > 0.1)
        )

        dataframe["ranging_advanced"] = (
            dataframe["trend_strength_cycle_adjusted"].abs() < strong_threshold * 0.5
        ) | (
            dataframe["cycle_strength"] < 0.05
        )  # Very weak cycles indicate ranging

        # === TREND CONFIDENCE SCORE ===
        # Calculate confidence based on agreement between methods
        methods_agreement = (
            (
                np.sign(dataframe["trend_strength_5_advanced"])
                == np.sign(dataframe["trend_strength_10_advanced"])
            ).astype(int)
            + (
                np.sign(dataframe["trend_strength_10_advanced"])
                == np.sign(dataframe["trend_strength_20_advanced"])
            ).astype(int)
            + (
                np.sign(dataframe["trend_strength_wavelet"])
                == np.sign(dataframe["wavelet_trend_strength"])
            ).astype(int)
        )

        dataframe["trend_confidence"] = methods_agreement / 3.0

        # High confidence trends
        dataframe["high_confidence_trend"] = (
            (dataframe["trend_confidence"] >= 0.67)
            & (dataframe["cycle_strength"] > 0.2)
            & (
                dataframe["trend_strength_cycle_adjusted"].abs()
                > strong_threshold * 0.8
            )
        )

        return dataframe

    except Exception as e:
        logger.warning(f"Advanced trend analysis failed: {e}. Using fallback method.")
        # Return dataframe with fallback values
        fallback_columns = [
            "slope_5_advanced",
            "slope_10_advanced",
            "slope_20_advanced",
            "wavelet_trend_strength",
            "cycle_strength",
            "dominant_cycle_period",
            "trend_strength_5_advanced",
            "trend_strength_10_advanced",
            "trend_strength_20_advanced",
            "trend_strength_wavelet",
            "trend_strength_combined",
            "trend_strength_cycle_adjusted",
            "strong_uptrend_advanced",
            "strong_downtrend_advanced",
            "ranging_advanced",
            "trend_confidence",
            "high_confidence_trend",
        ]

        for col in fallback_columns:
            if "strength" in col:
                dataframe[col] = 0.0
            else:
                dataframe[col] = False

        return dataframe


# === ADVANCED PREDICTIVE ANALYSIS SYSTEM ===


class AdvancedPredictiveEngine:
    """
    Advanced machine learning engine for high-precision trade entry prediction
    """

    def __init__(self):
        # Model containers
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = {}
        self.is_trained = {}
        self.model_performance = {}  # Initialize model_performance attribute
        self.prediction_horizon = 5  # Default prediction horizon
        self.adaptive_targets = {}  # Adaptive targets toggle per pair

        # Cached training dataframe per pair for incremental extension
        self.training_cache: dict[str, pd.DataFrame] = {}

        # V4: Cache for expensive features (FFT, wavelets, etc)
        self.feature_cache: dict = {}
        self.cache_expiry_candles = 5  # Cache valid for 5 candles
        self.last_cache_update: dict = {}

        # Retraining control
        self.last_train_time: dict[str, datetime] = {}
        self.last_train_index: dict[str, int] = {}
        # Periodic retraining interval (changed from 24h to 48h per latest requirement)
        self.retrain_interval_hours: int = 48
        self.initial_train_candles: int = 1000  # initial window size
        self.min_new_candles_for_retrain: int = 50  # skip tiny updates

        # Strategy startup tracking for 48h retrain rule
        self.strategy_start_time: datetime = datetime.utcnow()
        self.retrain_after_startup_hours: int = 48

        # Enable periodic retrain after startup period
        self.enable_startup_retrain: bool = True  # V4: Re-enabled with async training

        # Model persistence settings
        # V4: Check both paths for backward compatibility
        self.models_dir = Path("ml_models")
        self.legacy_models_dir = Path("user_data/strategies/ml_models")

        # Use legacy dir if it exists and has models, otherwise use new dir
        if self.legacy_models_dir.exists() and list(
            self.legacy_models_dir.glob("*.pkl")
        ):
            self.models_dir = self.legacy_models_dir
            logger.info(
                "[ML-V4] Using legacy models directory: user_data/strategies/ml_models"
            )
        else:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[ML-V4] Using new models directory: ml_models")

        # V4: Async training infrastructure
        self.training_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="MLTrain"
        )
        self.training_in_progress: dict[str, bool] = {}
        self.model_versions: dict[str, int] = {}
        self.training_start_times: dict[str, float] = {}

        # Load existing models if available
        self._load_models_from_disk()

    # ----------------- ASSET EXISTENCE HELPERS -----------------
    def _required_asset_paths(self, pair: str) -> list[Path]:
        """V4: Return minimum required assets (flexible for dynamic ensemble)"""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        # Only require scaler and metadata as minimum
        # Models are checked dynamically
        return [
            self.models_dir / f"{safe_pair}_scaler.pkl",
            self.models_dir / f"{safe_pair}_metadata.pkl",
        ]

    def _assets_exist(self, pair: str) -> bool:
        """V4: Check if minimum assets exist + at least 2 models"""
        # Check core assets (scaler, metadata)
        core_assets_exist = all(p.exists() for p in self._required_asset_paths(pair))

        # Check for at least 2 models (dynamic ensemble)
        if core_assets_exist:
            safe_pair = pair.replace("/", "_").replace(":", "_")
            model_files = list(self.models_dir.glob(f"{safe_pair}_model_*.pkl"))
            return len(model_files) >= 2

        return False

    def mark_trained_if_assets(self, pair: str):
        """Mark pair as trained if asset files exist (called at startup)."""
        if self._assets_exist(pair):
            self.is_trained[pair] = True

    def _get_model_filepath(self, pair: str, model_type: str) -> Path:
        """Get the filepath for saving/loading models"""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        return self.models_dir / f"{safe_pair}_{model_type}.pkl"

    def _save_models_to_disk(self, pair: str, output_dir: Optional[Path] = None):
        """Save trained models to disk for persistence
        V4: Support custom output directory for atomic swaps
        """
        try:
            if pair not in self.models:
                return

            # Use custom output dir if provided (for async training)
            save_dir = output_dir if output_dir else self.models_dir

            # Save models
            safe_pair = pair.replace("/", "_").replace(":", "_")
            for model_name, model in self.models[pair].items():
                # V4: Always use save_dir for atomic operations
                filepath = save_dir / f"{safe_pair}_model_{model_name}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)

            # Save scaler - V4: Use save_dir for atomic swap
            if pair in self.scalers:
                scaler_filepath = save_dir / f"{safe_pair}_scaler.pkl"
                with open(scaler_filepath, "wb") as f:
                    pickle.dump(self.scalers[pair], f)

            # Save metadata - V4: Use save_dir for atomic swap
            if pair in self.feature_importance:
                metadata_filepath = save_dir / f"{safe_pair}_metadata.pkl"
                metadata = {
                    "feature_importance": self.feature_importance[pair],
                    "is_trained": self.is_trained.get(pair, False),
                    "timestamp": datetime.now().isoformat(),
                }
                # V4: Include extended metadata if available
                if hasattr(self, "model_metadata") and pair in self.model_metadata:
                    metadata.update(self.model_metadata[pair])
                with open(metadata_filepath, "wb") as f:
                    pickle.dump(metadata, f)

            logger.info(f"[ML-V4] Models saved to {save_dir} for {pair} (atomic-ready)")

        except Exception as e:
            logger.warning(f"Failed to save models for {pair}: {e}")

    def _load_models_from_disk(self):
        """Load existing models from disk"""
        try:
            if not self.models_dir.exists():
                return

            # Find all model files
            model_files = list(self.models_dir.glob("*_model_*.pkl"))

            pairs_found = set()
            for model_file in model_files:
                # Extract pair name from filename
                filename = model_file.stem
                parts = filename.split("_model_")
                if len(parts) == 2:
                    pair_safe = parts[0]  # e.g. "BTC_USDC"
                    # V4 FIX: Deterministic pair reconstruction without ":"
                    tokens = pair_safe.split("_")
                    if len(tokens) >= 2:
                        base, quote = tokens[0], tokens[1]
                        pair = f"{base}/{quote}"
                        pairs_found.add(pair)

            # Load models for each pair
            for pair in pairs_found:
                try:
                    self._load_pair_models(pair)
                except Exception as e:
                    logger.warning(f"Failed to load models for {pair}: {e}")

            if pairs_found:
                logger.info(
                    f"Loaded ML models from disk for {len(pairs_found)} pairs: {list(pairs_found)}"
                )

        except Exception as e:
            logger.warning(f"Failed to load models from disk: {e}")

    def _load_pair_models(self, pair: str):
        """Load ALL models for a specific pair (V4: dynamic loader for full ensemble)"""
        safe_pair = pair.replace("/", "_").replace(":", "_")

        # Load ALL available models dynamically
        models = {}
        model_files = self.models_dir.glob(f"{safe_pair}_model_*.pkl")

        for model_file in model_files:
            # Extract model type from filename
            model_name = model_file.stem.replace(f"{safe_pair}_model_", "")
            try:
                with open(model_file, "rb") as f:
                    models[model_name] = pickle.load(f)
                    logger.info(f"[ML-V4] Loaded {model_name} for {pair}")
            except Exception as e:
                logger.warning(f"[ML-V4] Failed to load {model_name} for {pair}: {e}")

        if models:
            self.models[pair] = models
            logger.info(
                f"[ML-V4] Loaded {len(models)} models for {pair} (full ensemble ready)"
            )

        # Load scaler
        scaler_filepath = self._get_model_filepath(pair, "scaler")
        if scaler_filepath.exists():
            with open(scaler_filepath, "rb") as f:
                self.scalers[pair] = pickle.load(f)

        # Load metadata
        metadata_filepath = self._get_model_filepath(pair, "metadata")
        if metadata_filepath.exists():
            with open(metadata_filepath, "rb") as f:
                metadata = pickle.load(f)
                self.feature_importance[pair] = metadata.get("feature_importance", {})
                self.is_trained[pair] = metadata.get("is_trained", False)
                # V4: Load extended metadata
                if not hasattr(self, "model_metadata"):
                    self.model_metadata = {}
                if "feature_columns" in metadata:
                    self.model_metadata[pair] = {
                        "feature_columns": metadata.get("feature_columns"),
                        "calibration_method": metadata.get("calibration_method"),
                        "n_features": metadata.get("n_features"),
                        "n_samples_train": metadata.get("n_samples_train"),
                        "wfv_summary": metadata.get("wfv_summary"),
                    }

    def _cleanup_old_models(self, max_age_days: int = 7):
        """V4: Smart cleanup - remove only old temporary files, keep active models"""
        try:
            cutoff_time = datetime.now() - pd.Timedelta(days=max_age_days)

            # Track active models (latest version per pair)
            active_models = set()
            for pair in self.models.keys():
                safe_pair = pair.replace("/", "_").replace(":", "_")
                # Keep all files for active pairs
                for pattern in [
                    f"{safe_pair}_model_*.pkl",
                    f"{safe_pair}_scaler.pkl",
                    f"{safe_pair}_metadata.pkl",
                ]:
                    for file in self.models_dir.glob(pattern):
                        active_models.add(file.name)

            cleaned_count = 0
            for model_file in self.models_dir.glob("*.pkl"):
                # Only remove if:
                # 1. It's a temporary file (contains '_tmp' or timestamp pattern)
                # 2. It's old AND not an active model
                is_temp = "_tmp" in model_file.name or "_v1" in model_file.name
                is_old = model_file.stat().st_mtime < cutoff_time.timestamp()
                is_active = model_file.name in active_models

                if is_temp and is_old:
                    model_file.unlink()
                    logger.info(f"[ML-V4] Removed temporary file: {model_file.name}")
                    cleaned_count += 1
                elif is_old and not is_active:
                    # Only remove old non-active models
                    model_file.unlink()
                    logger.info(
                        f"[ML-V4] Removed old inactive model: {model_file.name}"
                    )
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"[ML-V4] Cleanup complete: removed {cleaned_count} files")

        except Exception as e:
            logger.warning(f"[ML-V4] Failed to cleanup old models: {e}")

    def extract_advanced_features_v3(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """V3 Enhanced feature extraction with market microstructure and advanced ML features"""
        df = self.extract_advanced_features(df, pair=pair)

        # === V3 ENHANCED FEATURES ===

        # 1. Cross-asset correlations (if BTC data available)
        if "btc_close" in df.columns:
            df["btc_correlation"] = df["close"].rolling(50).corr(df["btc_close"])
            df["btc_beta"] = self._calculate_beta(df["close"], df["btc_close"], 50)
            df["btc_divergence"] = (
                (df["close"].pct_change() - df["btc_close"].pct_change())
                .rolling(20)
                .mean()
            )

        # 2. Advanced volatility metrics
        returns = df["close"].pct_change()
        df["garch_volatility"] = self._estimate_garch_volatility(returns)
        df["realized_volatility"] = np.sqrt(
            (returns**2).rolling(20).mean()
        ) * np.sqrt(24)
        df["volatility_of_volatility"] = df["realized_volatility"].rolling(20).std()

        # 3. Market microstructure metrics
        df["kyle_lambda"] = self._calculate_kyle_lambda(df)
        df["amihud_illiquidity"] = abs(returns) / (df["volume"] + 1e-10)
        df["roll_spread"] = self._calculate_roll_spread(df)

        # 4. Information theory metrics
        df["mutual_information"] = self._calculate_mutual_information(df)
        df["transfer_entropy"] = self._calculate_transfer_entropy(df)

        # 5. Fractal and chaos metrics
        df["hurst_exponent"] = self._calculate_hurst_exponent(df["close"], 50)
        df["lyapunov_exponent"] = self._calculate_lyapunov_exponent(df["close"])

        # 6. Machine learning meta-features
        df["feature_importance_score"] = self._calculate_feature_importance(df, pair)
        df["prediction_confidence"] = self._calculate_prediction_confidence(df, pair)

        # === ENHANCED FEATURES V3+ ===

        # A. Kaufman Efficiency Ratio (trend quality)
        for period in [10, 20, 50]:
            df[f"ker_{period}"] = self._calculate_kaufman_efficiency_ratio(
                df["close"], period
            )

        # B. Bollinger Bandwidth and Squeeze
        bb_period = 20
        bb_std = 2
        bb_upper = (
            df["close"].rolling(bb_period).mean()
            + df["close"].rolling(bb_period).std() * bb_std
        )
        bb_lower = (
            df["close"].rolling(bb_period).mean()
            - df["close"].rolling(bb_period).std() * bb_std
        )
        df["bb_bandwidth"] = (bb_upper - bb_lower) / df["close"]
        df["bb_bandwidth_z"] = (
            df["bb_bandwidth"] - df["bb_bandwidth"].rolling(50).mean()
        ) / (df["bb_bandwidth"].rolling(50).std() + 1e-10)

        # Keltner channels for squeeze detection
        kc_period = 20
        kc_mult = 1.5
        atr = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        kc_upper = df["close"].rolling(kc_period).mean() + atr * kc_mult
        kc_lower = df["close"].rolling(kc_period).mean() - atr * kc_mult
        df["bb_squeeze"] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)

        # C. Pullback quality signals
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        df["distance_ema20"] = (df["close"] - ema20) / df["close"]
        df["distance_ema50"] = (df["close"] - ema50) / df["close"]
        df["ema20_slope"] = ema20.pct_change(5)  # 5-period slope
        df["ema50_slope"] = ema50.pct_change(10)  # 10-period slope

        # Pullback ratio from recent swing
        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()
        df["pullback_ratio"] = (high_20 - df["close"]) / (high_20 - low_20 + 1e-10)

        # D. Autocorrelation and sign persistence
        returns = df["close"].pct_change()
        for lag in [1, 2, 3]:
            df[f"autocorr_ret_lag{lag}"] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )

        # Sign persistence (proportion of green candles)
        df["sign_persistence_10"] = (df["close"] > df["open"]).rolling(10).mean()
        df["sign_persistence_20"] = (df["close"] > df["open"]).rolling(20).mean()

        # E. Divergence counters
        # Count divergences in rolling window (assuming divergence columns exist)
        if "rsi_divergence_bull" in df.columns:
            df["div_bull_count_50"] = df["rsi_divergence_bull"].rolling(50).sum()
            df["div_bear_count_50"] = df.get("rsi_divergence_bear", 0).rolling(50).sum()
            # Time since last divergence
            df["bars_since_bull_div"] = (
                (df["rsi_divergence_bull"] == 1)
                .astype(int)
                .groupby((df["rsi_divergence_bull"] == 1).cumsum())
                .cumcount()
            )

        # F. Breakout pressure
        df["closes_above_high20"] = (
            (df["close"] > df["high"].shift(1).rolling(20).max()).rolling(20).mean()
        )
        df["green_candle_ratio_20"] = (df["close"] > df["open"]).rolling(20).mean()

        # G. Cross-asset with ETH (if available)
        if "eth_close" in df.columns:
            df["eth_correlation"] = df["close"].rolling(50).corr(df["eth_close"])
            df["eth_beta"] = self._calculate_beta(df["close"], df["eth_close"], 50)
            df["eth_divergence"] = (
                (df["close"].pct_change() - df["eth_close"].pct_change())
                .rolling(20)
                .mean()
            )

        # H. Underwater metrics (drawdown)
        rolling_max = df["close"].expanding().max()
        df["drawdown_pct"] = (df["close"] - rolling_max) / rolling_max
        df["bars_in_drawdown"] = (
            (df["close"] < rolling_max)
            .astype(int)
            .groupby((df["close"] >= rolling_max).cumsum())
            .cumcount()
        )

        # I. Risk-aware metrics (Downside deviation and Sortino)
        returns = df["close"].pct_change()

        # Downside deviation (only negative returns)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        df["downside_deviation_20"] = downside_returns.rolling(20).std()
        df["downside_deviation_50"] = downside_returns.rolling(50).std()

        # Sortino ratio (better than Sharpe for crypto)
        # Using 0% as MAR (Minimum Acceptable Return)
        for window in [20, 50]:
            mean_return = returns.rolling(window).mean()
            downside_dev = downside_returns.rolling(window).std()
            df[f"sortino_ratio_{window}"] = mean_return / (downside_dev + 1e-10)

        # J. Relative Strength vs BTC/ETH
        if "btc_close" in df.columns:
            # RSI of pair/BTC ratio
            pair_btc_ratio = df["close"] / (df["btc_close"] + 1e-10)
            df["rsi_vs_btc"] = ta.RSI(pair_btc_ratio, timeperiod=14)

            # ROC (Rate of Change) vs BTC
            df["roc_vs_btc"] = pair_btc_ratio.pct_change(10) * 100

            # ADX on pair/BTC ratio (trend strength of relative performance)
            ratio_high = pair_btc_ratio.rolling(14).max()
            ratio_low = pair_btc_ratio.rolling(14).min()
            adx_result = ta.ADX(ratio_high, ratio_low, pair_btc_ratio, timeperiod=14)
            if isinstance(adx_result, pd.DataFrame):
                df["adx_vs_btc"] = adx_result["ADX_14"]
            else:
                df["adx_vs_btc"] = adx_result

        if "eth_close" in df.columns:
            # RSI of pair/ETH ratio
            pair_eth_ratio = df["close"] / (df["eth_close"] + 1e-10)
            df["rsi_vs_eth"] = ta.RSI(pair_eth_ratio, timeperiod=14)

            # ROC vs ETH
            df["roc_vs_eth"] = pair_eth_ratio.pct_change(10) * 100

            # ADX on pair/ETH ratio
            ratio_high = pair_eth_ratio.rolling(14).max()
            ratio_low = pair_eth_ratio.rolling(14).min()
            adx_result = ta.ADX(ratio_high, ratio_low, pair_eth_ratio, timeperiod=14)
            if isinstance(adx_result, pd.DataFrame):
                df["adx_vs_eth"] = adx_result["ADX_14"]
            else:
                df["adx_vs_eth"] = adx_result

        # K. Multi-timeframe features (if informative data available)
        # These should be populated from informative_pairs in populate_indicators
        # Check for 4h data
        if "close_4h" in df.columns:
            df["ret_4h"] = df["close_4h"].pct_change()
            df["vol_realized_4h"] = np.sqrt(
                (df["ret_4h"] ** 2).rolling(20).mean()
            ) * np.sqrt(
                6
            )  # 6 = 24h/4h
            # Add lag to prevent lookahead
            df["ret_4h"] = df["ret_4h"].shift(1)
            df["vol_realized_4h"] = df["vol_realized_4h"].shift(1)

            # Trend strength in higher timeframe
            if "ker_20" in df.columns:
                df["ker_4h"] = self._calculate_kaufman_efficiency_ratio(
                    df["close_4h"], 20
                ).shift(1)

        # Check for 15m data
        if "close_15m" in df.columns:
            df["ret_15m"] = df["close_15m"].pct_change()
            df["momentum_15m"] = df["close_15m"].pct_change(
                4
            )  # 1h momentum in 15m bars
            # Add lag to prevent lookahead
            df["ret_15m"] = df["ret_15m"].shift(1)
            df["momentum_15m"] = df["momentum_15m"].shift(1)

        return df

    def extract_advanced_features(
        self, dataframe: pd.DataFrame, lookback: int = 100, pair: str = None
    ) -> pd.DataFrame:
        """Extract sophisticated features for ML prediction with variance validation"""
        df = dataframe.copy()

        # === 1. ENHANCED PRICE ACTION FEATURES ===
        # Multi-period price patterns with variance
        for period in [1, 2, 3, 5]:
            df[f"price_velocity_{period}"] = df["close"].pct_change(period)
            df[f"price_acceleration_{period}"] = df[f"price_velocity_{period}"].diff(1)

            # Add rolling statistics for variance
            df[f"price_velocity_std_{period}"] = (
                df[f"price_velocity_{period}"].rolling(20).std()
            )
            df[f"price_velocity_skew_{period}"] = (
                df[f"price_velocity_{period}"].rolling(20).skew()
            )

        # Volatility-adjusted momentum
        returns = df["close"].pct_change(1)
        vol_20 = returns.rolling(20).std()
        df["vol_adjusted_momentum"] = returns / (vol_20 + 1e-10)

        # Price position within recent range
        for window in [10, 20, 50]:
            high_window = df["high"].rolling(window).max()
            low_window = df["low"].rolling(window).min()
            range_size = high_window - low_window
            df[f"price_position_{window}"] = (df["close"] - low_window) / (
                range_size + 1e-10
            )

        # Support/Resistance with dynamic thresholds
        if "minima_sort_threshold" in df.columns:
            support_distance = abs(df["low"] - df["minima_sort_threshold"]) / (
                df["close"] + 1e-10
            )
            df["support_strength"] = (
                (support_distance < 0.02).astype(int).rolling(20).mean()
            )
            df["support_distance_norm"] = support_distance
        else:
            df["support_strength"] = 0.5  # Neutral value when thresholds not available
            df["support_distance_norm"] = 0.05  # Neutral distance

        if "maxima_sort_threshold" in df.columns:
            resistance_distance = abs(df["high"] - df["maxima_sort_threshold"]) / (
                df["close"] + 1e-10
            )
            df["resistance_strength"] = (
                (resistance_distance < 0.02).astype(int).rolling(20).mean()
            )
            df["resistance_distance_norm"] = resistance_distance
        else:
            df["resistance_strength"] = (
                0.5  # Neutral value when thresholds not available
            )
            df["resistance_distance_norm"] = 0.05  # Neutral distance

        # === 2. VOLUME DYNAMICS ===
        # Volume profile analysis
        df["volume_profile_score"] = self._calculate_volume_profile_score(df)
        df["volume_imbalance"] = self._calculate_volume_imbalance(df)
        df["smart_money_index"] = self._calculate_smart_money_index(df)

        # Volume-price correlation
        df["volume_price_correlation"] = df["volume"].rolling(20).corr(df["close"])
        df["volume_breakout_strength"] = self._calculate_volume_breakout_strength(df)

        # === 3. VOLATILITY CLUSTERING ===
        df["volatility_regime"] = self._calculate_volatility_regime(df)
        df["volatility_persistence"] = self._calculate_volatility_persistence(df)
        df["volatility_mean_reversion"] = self._calculate_volatility_mean_reversion(df)

        # === 4. MOMENTUM DECOMPOSITION ===
        for period in [3, 5, 8, 13, 21]:
            df[f"momentum_{period}"] = df["close"].pct_change(period)
            df[f"momentum_strength_{period}"] = abs(df[f"momentum_{period}"])
            df[f"momentum_consistency_{period}"] = (
                np.sign(df[f"momentum_{period}"]).rolling(5).mean()
            )

        # Momentum regime detection
        df["momentum_regime"] = self._classify_momentum_regime(df)
        df["momentum_divergence_strength"] = self._calculate_momentum_divergence(df)

        # === 5. MICROSTRUCTURE FEATURES ===
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        df["market_impact"] = df["volume"] * df["spread_proxy"]
        df["order_flow_imbalance"] = self._calculate_order_flow_imbalance(df)
        df["liquidity_index"] = self._calculate_liquidity_index(df)

        # === 6. STATISTICAL FEATURES ===
        for window in [10, 20, 50]:
            returns = df["close"].pct_change(1)
            df[f"skewness_{window}"] = returns.rolling(window).apply(
                lambda x: skew(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f"kurtosis_{window}"] = returns.rolling(window).apply(
                lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f"entropy_{window}"] = self._calculate_entropy(df["close"], window)

        # === 7. REGIME DETECTION FEATURES ===
        df["market_regime"] = self._detect_market_regime(df)
        df["regime_stability"] = self._calculate_regime_stability(df)
        df["regime_transition_probability"] = self._calculate_regime_transition_prob(df)

        return df

    def _calculate_volume_profile_score(
        self, df: pd.DataFrame, window: int = 50
    ) -> pd.Series:
        """Calculate volume profile score"""

        def volume_profile(data):
            if len(data) < 10:
                return 0.5

            prices = data["close"].values
            volumes = data["volume"].values

            # Create price bins
            price_min, price_max = prices.min(), prices.max()
            if price_max == price_min:
                return 0.5

            bins = np.linspace(price_min, price_max, 10)

            # Calculate volume at each price level
            volume_at_price = []
            for i in range(len(bins) - 1):
                mask = (prices >= bins[i]) & (prices < bins[i + 1])
                vol_sum = volumes[mask].sum()
                volume_at_price.append(vol_sum)

            # Point of Control (POC) - price level with highest volume
            if sum(volume_at_price) == 0:
                return 0.5

            poc_index = np.argmax(volume_at_price)
            current_price = prices[-1]
            poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2

            # Score based on distance from POC
            distance_ratio = abs(current_price - poc_price) / (
                price_max - price_min + 1e-10
            )
            score = 1 - distance_ratio  # Closer to POC = higher score

            return max(0, min(1, score))

        # Apply rolling calculation
        result = []
        for i in range(len(df)):
            if i < window:
                result.append(0.5)
            else:
                window_data = df.iloc[i - window + 1 : i + 1][["close", "volume"]]
                score = volume_profile(window_data)
                result.append(score)

        return pd.Series(result, index=df.index)

    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume imbalance between buying and selling"""
        up_volume = df["volume"].where(df["close"] > df["open"], 0)
        down_volume = df["volume"].where(df["close"] < df["open"], 0)

        total_volume = up_volume + down_volume
        imbalance = (up_volume - down_volume) / (total_volume + 1e-10)

        return imbalance.rolling(10).mean()

    def _calculate_smart_money_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Smart Money Index (SMI)"""
        price_change = abs(df["close"].pct_change(1))
        volume_norm = df["volume"] / df["volume"].rolling(20).mean()

        smi = volume_norm / (price_change + 1e-10)
        return smi.rolling(10).mean()

    def _calculate_volume_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume breakout strength"""
        volume_ma = df["volume"].rolling(20).mean()
        volume_ratio = df["volume"] / volume_ma

        price_breakout = (
            (df["close"] > df["close"].rolling(20).max().shift(1))
            | (df["close"] < df["close"].rolling(20).min().shift(1))
        ).astype(int)

        breakout_strength = volume_ratio * price_breakout
        return breakout_strength.rolling(5).mean()

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect volatility regime"""
        returns = df["close"].pct_change(1)
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(50).mean()

        regime = pd.Series(1, index=df.index)  # Default normal
        regime[volatility < vol_ma * 0.7] = 0  # Low volatility
        regime[volatility > vol_ma * 1.5] = 2  # High volatility

        return regime

    def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility persistence"""
        returns = df["close"].pct_change(1)
        volatility = returns.rolling(5).std()

        persistence = volatility.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
        )

        return persistence

    def _calculate_volatility_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility mean reversion tendency"""
        returns = df["close"].pct_change(1)
        volatility = returns.rolling(10).std()
        vol_ma = volatility.rolling(50).mean()

        vol_zscore = (volatility - vol_ma) / (volatility.rolling(50).std() + 1e-10)
        mean_reversion = -vol_zscore

        return mean_reversion

    def _classify_momentum_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify momentum regime"""
        mom_3 = df["close"].pct_change(3)
        mom_8 = df["close"].pct_change(8)
        mom_21 = df["close"].pct_change(21)

        regime = pd.Series(0, index=df.index)  # Neutral

        strong_up = (mom_3 > 0.02) & (mom_8 > 0.05) & (mom_21 > 0.1)
        regime[strong_up] = 2

        mod_up = (mom_3 > 0) & (mom_8 > 0) & (mom_21 > 0) & ~strong_up
        regime[mod_up] = 1

        mod_down = (mom_3 < 0) & (mom_8 < 0) & (mom_21 < 0) & (mom_21 > -0.1)
        regime[mod_down] = -1

        strong_down = (mom_3 < -0.02) & (mom_8 < -0.05) & (mom_21 < -0.1)
        regime[strong_down] = -2

        return regime

    def _calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum divergence strength"""
        price_momentum = df["close"].pct_change(10)

        if "rsi" in df.columns:
            rsi_momentum = df["rsi"].diff(10)
        else:
            rsi_momentum = pd.Series(0, index=df.index)

        volume_momentum = df["volume"].pct_change(10)

        # Normalize momentums using rolling z-score
        price_norm = (price_momentum - price_momentum.rolling(50).mean()) / (
            price_momentum.rolling(50).std() + 1e-10
        )
        rsi_norm = (rsi_momentum - rsi_momentum.rolling(50).mean()) / (
            rsi_momentum.rolling(50).std() + 1e-10
        )
        volume_norm = (volume_momentum - volume_momentum.rolling(50).mean()) / (
            volume_momentum.rolling(50).std() + 1e-10
        )

        price_rsi_div = abs(price_norm - rsi_norm)
        price_volume_div = abs(price_norm - volume_norm)

        divergence_strength = (price_rsi_div + price_volume_div) / 2
        return divergence_strength.rolling(5).mean()

    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        price_impact = (df["close"] - df["open"]) / df["open"]
        volume_impact = df["volume"] / df["volume"].rolling(20).mean()

        flow_imbalance = price_impact * volume_impact
        return flow_imbalance.rolling(5).mean()

    def _calculate_liquidity_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market liquidity index"""
        spread = (df["high"] - df["low"]) / df["close"]
        volume_norm = df["volume"] / df["volume"].rolling(50).mean()

        liquidity = volume_norm / (spread + 1e-10)
        return liquidity.rolling(10).mean()

    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate information entropy"""

        def entropy(data):
            if len(data) < 5:
                return 0

            returns = np.diff(data) / (data[:-1] + 1e-10)

            bins = np.histogram_bin_edges(returns, bins=10)
            hist, _ = np.histogram(returns, bins=bins)

            probs = hist / (hist.sum() + 1e-10)
            probs = probs[probs > 0]

            ent = -np.sum(probs * np.log2(probs + 1e-10))
            return ent

        return series.rolling(window).apply(entropy, raw=False)

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect overall market regime"""
        if "trend_strength" in df.columns:
            trend_regime = np.sign(df["trend_strength"])
        else:
            trend_regime = pd.Series(0, index=df.index)

        vol_regime = self._calculate_volatility_regime(df) - 1
        momentum_regime = self._classify_momentum_regime(df)

        market_regime = trend_regime * 0.4 + vol_regime * 0.3 + momentum_regime * 0.3

        return market_regime.rolling(5).mean()

    def _calculate_regime_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime stability"""
        regime = self._detect_market_regime(df)
        regime_changes = abs(regime.diff(1))
        stability = 1 / (regime_changes.rolling(20).mean() + 1e-10)
        return stability

    def _calculate_regime_transition_prob(self, df: pd.DataFrame) -> pd.Series:
        """Calculate probability of regime transition"""
        regime = self._detect_market_regime(df)

        transitions = []
        for i in range(1, len(regime)):
            if not (pd.isna(regime.iloc[i]) or pd.isna(regime.iloc[i - 1])):
                transition = abs(regime.iloc[i] - regime.iloc[i - 1]) > 0.5
                transitions.append(transition)
            else:
                transitions.append(False)

        transition_prob = pd.Series([False] + transitions, index=regime.index)
        prob_smooth = transition_prob.astype(int).rolling(20).mean()

        return prob_smooth

    # === V3 HELPER METHODS FOR ENHANCED FEATURES ===

    def _calculate_beta(
        self, asset_returns: pd.Series, market_returns: pd.Series, window: int
    ) -> pd.Series:
        """Calculate rolling beta coefficient"""
        covariance = asset_returns.rolling(window).cov(market_returns)
        market_variance = market_returns.rolling(window).var()
        return covariance / (market_variance + 1e-10)

    def _estimate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Simplified GARCH(1,1) volatility estimation"""
        # Simplified implementation without external dependencies
        vol = returns.rolling(20).std()
        # Add persistence and mean reversion
        alpha = 0.1  # Shock persistence
        beta = 0.85  # Volatility persistence
        omega = 0.05  # Long-term variance weight

        garch_vol = vol.copy()
        for i in range(21, len(returns)):
            garch_vol.iloc[i] = np.sqrt(
                omega
                + alpha * returns.iloc[i - 1] ** 2
                + beta * garch_vol.iloc[i - 1] ** 2
            )
        return garch_vol

    def _calculate_kyle_lambda(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Kyle's lambda (price impact coefficient)"""
        returns = df["close"].pct_change()
        signed_volume = df["volume"] * np.sign(returns)
        # Rolling regression of returns on signed volume
        lambda_series = pd.Series(index=df.index, dtype=float)
        for i in range(20, len(df)):
            X = signed_volume.iloc[i - 20 : i].values.reshape(-1, 1)
            y = returns.iloc[i - 20 : i].values
            if not np.isnan(X).any() and not np.isnan(y).any():
                coef = np.linalg.lstsq(X, y, rcond=None)[0][0]
                lambda_series.iloc[i] = abs(coef)
        return lambda_series.fillna(0)

    def _calculate_roll_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Roll's implied spread"""
        returns = df["close"].pct_change()
        # Roll spread = 2 * sqrt(-cov(r_t, r_{t-1}))
        spread = pd.Series(index=df.index, dtype=float)
        for i in range(20, len(df)):
            ret_window = returns.iloc[i - 20 : i]
            cov = ret_window.cov(ret_window.shift(1))
            if cov < 0:
                spread.iloc[i] = 2 * np.sqrt(-cov)
            else:
                spread.iloc[i] = 0
        return spread.fillna(0)

    def _calculate_mutual_information(self, df: pd.DataFrame) -> pd.Series:
        """Simplified mutual information between price and volume"""
        # Discretize data for MI calculation
        price_bins = pd.qcut(df["close"], q=10, labels=False, duplicates="drop")
        volume_bins = pd.qcut(df["volume"], q=10, labels=False, duplicates="drop")

        mi_series = pd.Series(index=df.index, dtype=float)
        for i in range(50, len(df)):
            # Calculate joint and marginal probabilities
            p_window = price_bins.iloc[i - 50 : i]
            v_window = volume_bins.iloc[i - 50 : i]

            if p_window.notna().sum() > 10 and v_window.notna().sum() > 10:
                # Simple MI approximation
                joint_counts = pd.crosstab(p_window, v_window, normalize=True)
                p_marginal = p_window.value_counts(normalize=True)
                v_marginal = v_window.value_counts(normalize=True)

                mi = 0
                for p_val in p_marginal.index:
                    for v_val in v_marginal.index:
                        if (
                            p_val in joint_counts.index
                            and v_val in joint_counts.columns
                        ):
                            joint_prob = joint_counts.loc[p_val, v_val]
                            if joint_prob > 0:
                                mi += joint_prob * np.log2(
                                    joint_prob
                                    / (p_marginal[p_val] * v_marginal[v_val] + 1e-10)
                                )
                mi_series.iloc[i] = mi
        return mi_series.fillna(0)

    def _calculate_transfer_entropy(self, df: pd.DataFrame) -> pd.Series:
        """Simplified transfer entropy from volume to price"""
        # Simplified implementation
        returns = df["close"].pct_change()
        volume_change = df["volume"].pct_change()

        # Use correlation as proxy for transfer entropy
        te_proxy = volume_change.shift(1).rolling(20).corr(returns)
        return te_proxy.fillna(0)

    def _calculate_hurst_exponent(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Hurst exponent"""

        def hurst(data):
            if len(data) < 20:
                return 0.5

            # R/S analysis
            lags = range(2, min(20, len(data) // 2))
            tau = [
                np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags
            ]

            if len(tau) > 2:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            return 0.5

        return series.rolling(window).apply(hurst, raw=False).fillna(0.5)

    def _calculate_lyapunov_exponent(self, series: pd.Series) -> pd.Series:
        """Simplified Lyapunov exponent calculation"""
        # Use return divergence as proxy
        returns = series.pct_change()
        abs_returns = abs(returns)

        # Rolling log of divergence rate
        lyap_proxy = np.log(abs_returns.rolling(20).mean() + 1e-10)
        return lyap_proxy.fillna(0)

    def _calculate_feature_importance(self, df: pd.DataFrame, pair: str) -> pd.Series:
        """Calculate feature importance score based on correlation with target"""
        # If we have a model, use its feature importances
        if pair in self.models and self.models[pair]:
            # Return a constant high importance for now
            return pd.Series(0.7, index=df.index)
        return pd.Series(0.5, index=df.index)

    def _calculate_prediction_confidence(
        self, df: pd.DataFrame, pair: str
    ) -> pd.Series:
        """Calculate model prediction confidence"""
        # Base confidence on recent model performance
        if pair in self.model_performance:
            recent_perf = self.model_performance[pair].get("f1_score", 0.5)
            return pd.Series(recent_perf, index=df.index)
        return pd.Series(0.5, index=df.index)

    def _calculate_kaufman_efficiency_ratio(
        self, series: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Kaufman Efficiency Ratio - measures trend efficiency"""
        direction = abs(series.diff(period))
        volatility = series.diff().abs().rolling(period).sum()
        ker = direction / (volatility + 1e-10)
        return ker.fillna(0)

    def create_target_variable(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        profit_threshold: float | None = None,
        dynamic: bool = True,
        quantile: float = 0.85,
        k_atr: float = 1.2,
        k_vol: float = 1.5,
        min_abs: float = 0.003,
        max_abs: float = 0.05,
    ) -> pd.Series:
        """Create target variable with optional dynamic profit threshold.

        Dynamic threshold logic (if dynamic=True and profit_threshold not provided):
          1. Compute ATR% (14) and rolling return volatility (20).
          2. base_series = k_atr * ATR% + k_vol * vola20
          3. base_scalar = median(base_series)
          4. q_thr = 85th percentile of forward_returns (future move distribution)
          5. blended = 0.5 * base_scalar + 0.5 * q_thr
          6. final_thr = clip(blended, min_abs, max_abs)
        This produces a stable scalar threshold per training batch (reproducible) rather than per-row noise.
        """
        # Forward returns used by several strategies
        forward_returns = (
            df["close"].pct_change(forward_periods).shift(-forward_periods)
        )

        # === DYNAMIC THRESHOLD CALCULATION ===
        if dynamic and (profit_threshold is None):
            try:
                # ATR% calculation
                high = df["high"]
                low = df["low"]
                close = df["close"]
                prev_close = close.shift(1)
                tr = np.maximum(
                    high - low,
                    np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
                )
                atr = tr.rolling(14).mean()
                atr_pct = (atr / close).clip(lower=0)

                # Return volatility
                returns_1 = close.pct_change()
                vola20 = returns_1.rolling(20).std()

                base_series = k_atr * atr_pct + k_vol * vola20
                base_scalar = (
                    float(np.nanmedian(base_series.tail(300)))
                    if len(base_series.dropna()) > 30
                    else float(np.nanmedian(base_series))
                )

                # Distribution-based quantile of forward returns (future info okay for label construction stage)
                q_thr = (
                    float(forward_returns.quantile(quantile))
                    if forward_returns.notna().any()
                    else min_abs
                )
                if not np.isfinite(q_thr):
                    q_thr = min_abs
                blended = 0.5 * base_scalar + 0.5 * q_thr
                profit_threshold = float(np.clip(blended, min_abs, max_abs))
            except Exception as e:
                logger.warning(
                    f"Dynamic threshold failed ({e}), falling back to default 0.015"
                )
                profit_threshold = 0.015
        elif profit_threshold is None:
            profit_threshold = 0.015

        # === STRATEGY 1: SIMPLE FORWARD RETURNS ===
        simple_target = (forward_returns > profit_threshold).astype(int)

        # === STRATEGY 2: MAXIMUM PROFIT POTENTIAL ===
        forward_highs = (
            df["high"].rolling(forward_periods).max().shift(-forward_periods)
        )
        max_profit_potential = (forward_highs - df["close"]) / df["close"]
        profit_target = (max_profit_potential > profit_threshold).astype(int)

        # === STRATEGY 3: RISK-ADJUSTED RETURNS ===
        forward_lows = df["low"].rolling(forward_periods).min().shift(-forward_periods)
        max_loss_potential = (forward_lows - df["close"]) / df["close"]
        risk_adjusted_return = forward_returns / (abs(max_loss_potential) + 1e-10)
        risk_target = (
            (forward_returns > profit_threshold * 0.7) & (risk_adjusted_return > 0.5)
        ).astype(int)

        # === STRATEGY 4: VOLATILITY-ADJUSTED TARGET ===
        returns_std = df["close"].pct_change().rolling(20).std()
        volatility_adjusted_threshold = profit_threshold * (1 + returns_std)
        vol_target = (forward_returns > volatility_adjusted_threshold).astype(int)

        # === ENSEMBLE VOTE ===
        combined_target = simple_target + profit_target + risk_target + vol_target
        final_target = (combined_target >= 2).astype(int)

        positive_ratio = final_target.mean()
        logger.info(
            f"Target created (forward={forward_periods}) dynamic_thr={profit_threshold:.4f} "
            f"positives={final_target.sum()}/{len(final_target)} ratio={positive_ratio:.3f}"
        )

        # Only log imbalance now; do not auto-alter labels (professional reproducibility)
        if positive_ratio < 0.05:
            logger.warning(
                f"Very low positive ratio ({positive_ratio:.3f}) at threshold {profit_threshold:.4f}"
            )
        elif positive_ratio > 0.45:
            logger.warning(
                f"High positive ratio ({positive_ratio:.3f}) at threshold {profit_threshold:.4f}"
            )

        return final_target

    def train_predictive_models_async(self, df: pd.DataFrame, pair: str) -> None:
        """V4: Asynchronous training wrapper to prevent blocking"""
        # Check if already training for this pair
        if self.training_in_progress.get(pair, False):
            logger.info(f"[ML-V4] Training already in progress for {pair}, skipping")
            return

        # Mark as training
        self.training_in_progress[pair] = True
        self.training_start_times[pair] = time.time()

        # Submit to executor
        future = self.training_executor.submit(
            self._train_models_worker, df.copy(), pair
        )

        # Add callback to handle completion
        def on_complete(fut):
            try:
                result = fut.result()
                elapsed = time.time() - self.training_start_times.get(pair, 0)
                logger.info(
                    f"[ML-V4] Async training completed for {pair} in {elapsed:.1f}s"
                )
                if result.get("status") == "success":
                    # Atomic model swap happens inside _train_models_worker
                    self.model_versions[pair] = self.model_versions.get(pair, 0) + 1
                    logger.info(
                        f"[ML-V4] {pair} promoted to model v{self.model_versions[pair]}"
                    )
            except Exception as e:
                logger.error(f"[ML-V4] Async training failed for {pair}: {e}")
            finally:
                self.training_in_progress[pair] = False

        future.add_done_callback(on_complete)
        logger.info(f"[ML-V4] Started async training for {pair}")

    def _train_models_worker(self, df: pd.DataFrame, pair: str) -> dict:
        """V4: Worker function for async training with atomic swap"""
        # Create temp directory for this training session
        safe_pair = pair.replace("/", "_").replace(":", "_")
        temp_dir = self.models_dir / f"{safe_pair}_v{int(time.time())}_tmp"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Train models (calls the original train_predictive_models)
            result = self.train_predictive_models(df, pair, output_dir=temp_dir)

            if result.get("status") == "success":
                # Atomic swap: rename temp files to production
                for temp_file in temp_dir.glob("*.pkl"):
                    prod_file = self.models_dir / temp_file.name
                    # Use rename for atomic operation
                    temp_file.replace(prod_file)
                logger.info(f"[ML-V4] Atomic swap completed for {pair}")

            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            return result

        except Exception as e:
            logger.error(f"[ML-V4] Training worker error for {pair}: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"status": "error", "message": str(e)}

    def train_predictive_models(
        self, df: pd.DataFrame, pair: str, output_dir: Optional[Path] = None
    ) -> dict:
        """Train advanced ensemble of predictive models with V4 improvements:
        - Purged Cross-Validation
        - Walk-Forward Validation
        - Better telemetry
        """
        if not SKLEARN_AVAILABLE:
            return {"status": "sklearn_not_available"}

        try:
            # Decide training slice: initial window or sliding window on retrain
            if pair not in self.is_trained or not self.is_trained[pair]:
                # First training: cut to last initial_train_candles
                if len(df) > self.initial_train_candles:
                    base_df = df.iloc[-self.initial_train_candles :].copy()
                else:
                    base_df = df.copy()
            else:
                # Incremental retrain: extend previous cached window with new rows since last_train_index
                prev_df = self.training_cache.get(pair)
                if prev_df is None:
                    prev_df = (
                        df.iloc[-self.initial_train_candles :].copy()
                        if len(df) > self.initial_train_candles
                        else df.copy()
                    )
                # New rows (simple index-based diff). If dataframe has 'date', we could filter last 24h.
                new_rows = df.iloc[self.last_train_index.get(pair, 0) :].copy()
                if len(new_rows) == 0:
                    base_df = prev_df
                else:
                    combined = pd.concat([prev_df, new_rows], ignore_index=True)
                    # Keep only most recent window (rolling window behaviour)
                    if len(combined) > self.initial_train_candles:
                        base_df = combined.iloc[-self.initial_train_candles :].copy()
                    else:
                        base_df = combined

            # === ENHANCED FEATURE ENGINEERING V3 ===
            feature_df = self.extract_advanced_features_v3(base_df, pair)

            # === ADAPTIVE TARGET VARIABLE V3 ===
            # Dynamically adjust target based on market conditions
            market_volatility = (
                base_df["close"].pct_change().rolling(20).std().iloc[-1]
                if len(base_df) > 20
                else 0.02
            )
            adaptive_horizon = self.prediction_horizon
            if market_volatility > 0.03:  # High volatility
                adaptive_horizon = max(3, self.prediction_horizon - 2)
            elif market_volatility < 0.01:  # Low volatility
                adaptive_horizon = min(10, self.prediction_horizon + 2)

            dynamic = self.adaptive_targets.get(pair, True)
            target = self.create_target_variable(
                base_df, forward_periods=adaptive_horizon, dynamic=dynamic
            )

            feature_columns = []
            exclude_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "date",
                "enter_long",
                "enter_short",
                "exit_long",
                "exit_short",
            ]

            for col in feature_df.columns:
                if (
                    col not in exclude_cols
                    and feature_df[col].dtype in ["float64", "int64"]
                    and not col.startswith("enter_")
                    and not col.startswith("exit_")
                ):
                    feature_columns.append(col)

            X = feature_df[feature_columns].fillna(0)
            y = target.fillna(0)

            valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 100:
                return {"status": "insufficient_data"}

            # === FEATURE QUALITY VALIDATION ===
            # Remove constant features (zero variance)
            feature_variance = X.var()
            non_constant_features = feature_variance[
                feature_variance > 1e-10
            ].index.tolist()

            logger.info(
                f"Removed {len(feature_columns) - len(non_constant_features)} "
                f"constant features out of {len(feature_columns)}"
            )

            if len(non_constant_features) < 5:
                logger.warning(
                    f"Too few variable features ({len(non_constant_features)})"
                )
                return {"status": "insufficient_features"}

            X = X[non_constant_features]
            feature_columns = non_constant_features

            # === PURGED CROSS-VALIDATION IMPLEMENTATION V3 ===
            # Prevent look-ahead bias with purged time series split
            from sklearn.model_selection import TimeSeriesSplit

            n_splits = 5
            embargo_pct = 0.02  # 2% embargo between train and test

            # Calculate purge gap based on prediction horizon
            purge_gap = max(adaptive_horizon, 5)  # At least 5 periods gap

            # V4: Use purged TimeSeriesSplit with gap to prevent leakage
            tscv = GapTimeSeriesSplit(n_splits=n_splits, gap=purge_gap)

            # === CLASS BALANCE VALIDATION ===
            positive_count = y.sum()
            negative_count = len(y) - positive_count
            positive_ratio = positive_count / len(y)

            logger.info(
                f"Class distribution: {positive_count} positive, "
                f"{negative_count} negative ({positive_ratio:.3f} ratio)"
            )

            # Check for severe class imbalance
            if positive_count < 10:
                logger.warning(
                    f"Too few positive examples ({positive_count}), "
                    f"adjusting target variable"
                )
                # Create more lenient target
                relaxed_target = self.create_target_variable(
                    df, forward_periods=3, profit_threshold=0.01
                )
                y = relaxed_target[valid_mask].fillna(0)
                positive_count = y.sum()
                positive_ratio = positive_count / len(y)
                logger.info(
                    f"Adjusted class distribution: {positive_count} positive "
                    f"({positive_ratio:.3f} ratio)"
                )

            if positive_count < 5:
                return {"status": "insufficient_positive_examples"}

            # Advanced feature selection
            if len(feature_columns) > 30:
                selector = SelectKBest(
                    score_func=f_classif, k=min(30, len(feature_columns))
                )
                X_selected = selector.fit_transform(X, y)
                selected_features = [
                    feature_columns[i] for i in selector.get_support(indices=True)
                ]
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                feature_columns = selected_features
                logger.info(
                    f"Selected {len(selected_features)} best features for {pair}"
                )

            # === TRUE WALK-FORWARD VALIDATION V4 ===
            # Implement real WFV with multiple temporal windows
            wfv_results = {"f1_scores": [], "auc_scores": [], "accuracy_scores": []}
            n_wfv_splits = 4  # Use 4 windows for Walk-Forward Validation

            # Create temporal splits with purged gaps
            wfv_splitter = GapTimeSeriesSplit(n_splits=n_wfv_splits, gap=purge_gap)

            logger.info(
                f"[ML-V4] Starting Walk-Forward Validation with {n_wfv_splits} windows, gap={purge_gap}"
            )

            # Collect WFV metrics across all temporal windows
            for fold_idx, (train_idx, test_idx) in enumerate(wfv_splitter.split(X)):
                if len(train_idx) < 50 or len(test_idx) < 20:
                    continue  # Skip if fold is too small

                X_fold_train = X.iloc[train_idx]
                X_fold_test = X.iloc[test_idx]
                y_fold_train = y.iloc[train_idx]
                y_fold_test = y.iloc[test_idx]

                # Scale features for this fold
                fold_scaler = RobustScaler()
                X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
                X_fold_test_scaled = fold_scaler.transform(X_fold_test)

                # Train a simple RF for WFV evaluation (fast)
                rf_wfv = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42 + fold_idx,
                    n_jobs=-1,
                    class_weight="balanced",
                )
                rf_wfv.fit(X_fold_train_scaled, y_fold_train)

                # Evaluate on this temporal window
                y_pred = rf_wfv.predict(X_fold_test_scaled)
                fold_f1 = f1_score(y_fold_test, y_pred, zero_division=0)
                fold_acc = accuracy_score(y_fold_test, y_pred)

                if hasattr(rf_wfv, "predict_proba") and len(np.unique(y_fold_test)) > 1:
                    y_proba = rf_wfv.predict_proba(X_fold_test_scaled)[:, 1]
                    fold_auc = roc_auc_score(y_fold_test, y_proba)
                else:
                    fold_auc = 0.5

                wfv_results["f1_scores"].append(fold_f1)
                wfv_results["accuracy_scores"].append(fold_acc)
                wfv_results["auc_scores"].append(fold_auc)

                logger.info(
                    f"[ML-V4] WFV Fold {fold_idx+1}/{n_wfv_splits}: "
                    f"F1={fold_f1:.3f}, Acc={fold_acc:.3f}, AUC={fold_auc:.3f}"
                )

            # Calculate WFV statistics
            if wfv_results["f1_scores"]:
                wfv_f1_mean = np.mean(wfv_results["f1_scores"])
                wfv_f1_std = np.std(wfv_results["f1_scores"])
                wfv_acc_mean = np.mean(wfv_results["accuracy_scores"])
                wfv_auc_mean = np.mean(wfv_results["auc_scores"])

                logger.info(
                    f"[ML-V4] {pair} WFV Summary: F1={wfv_f1_mean:.3f}±{wfv_f1_std:.3f}, "
                    f"Acc={wfv_acc_mean:.3f}, AUC={wfv_auc_mean:.3f}"
                )
            else:
                logger.warning(f"[ML-V4] No valid WFV folds for {pair}")
                wfv_f1_mean = wfv_f1_std = wfv_acc_mean = wfv_auc_mean = 0

            # === FINAL MODEL TRAINING SPLIT ===
            # After WFV evaluation, train final models on 70/30 split with purge
            split_idx = int(len(X) * 0.7)
            train_end_idx = split_idx - purge_gap
            test_start_idx = split_idx

            if train_end_idx <= 0 or test_start_idx >= len(X):
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            else:
                X_train = X[:train_end_idx]
                X_test = X[test_start_idx:]
                y_train = y[:train_end_idx]
                y_test = y[test_start_idx:]

            logger.info(
                f"Final split - Train: {len(X_train)}, Test: {len(X_test)}, Gap: {purge_gap}"
            )

            # Use RobustScaler for better handling of outliers
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # V4: Determine calibration method based on dataset size
            calibration_method = "sigmoid" if len(X_train) >= 3000 else "isotonic"
            logger.info(
                f"[ML-V4] Using {calibration_method} calibration for {pair} (n={len(X_train)})"
            )

            # V4: Create purged CV for GridSearch to prevent leakage
            grid_search_cv = GapTimeSeriesSplit(n_splits=3, gap=purge_gap)

            models = {}
            results = {}
            cv_scores = {}  # Store cross-validation scores

            # === MODEL 1: OPTIMIZED RANDOM FOREST ===
            # Full parameter grid for comprehensive optimization
            rf_params_full = {
                "n_estimators": [150, 200, 250],
                "max_depth": [15, 20, 25],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 5, 8],
                "max_features": ["sqrt", "log2", 0.8],
            }

            # Quick parameter grid for faster training (used by default)
            rf_params_quick = {
                "n_estimators": [150, 200],
                "max_depth": [15, 20],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [2, 5],
                "max_features": ["sqrt", 0.8],
            }

            rf_base = RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight="balanced"
            )

            # Use comprehensive grid search for datasets with sufficient data
            use_full_params = len(X_train) > 600  # More data = more thorough search
            selected_params = rf_params_full if use_full_params else rf_params_quick

            logger.info(
                f"Using {'full' if use_full_params else 'quick'} RF parameters "
                f"for {len(X_train)} training samples"
            )

            # Adaptive grid search based on dataset size
            rf_grid = GridSearchCV(
                rf_base,
                param_grid=selected_params,
                cv=grid_search_cv,  # V4: Use purged CV
                scoring="f1",
                n_jobs=-1,
            )
            rf_grid.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            from sklearn.calibration import CalibratedClassifierCV

            rf_calibrated = CalibratedClassifierCV(
                rf_grid.best_estimator_, method=calibration_method, cv=3
            )
            rf_calibrated.fit(X_train_scaled, y_train)
            models["random_forest"] = rf_calibrated

            # === MODEL 2: OPTIMIZED GRADIENT BOOSTING ===
            gb_base = GradientBoostingClassifier(
                random_state=42, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
            )

            gb_grid = GridSearchCV(
                gb_base,
                param_grid={
                    "n_estimators": [150, 200],
                    "max_depth": [6, 8, 10],
                    "learning_rate": [0.05, 0.1, 0.15],
                    "min_samples_split": [10, 20],
                    "min_samples_leaf": [5, 10],
                },
                cv=grid_search_cv,  # V4: Use purged CV
                scoring="f1",
                n_jobs=-1,
            )
            gb_grid.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            gb_calibrated = CalibratedClassifierCV(
                gb_grid.best_estimator_, method=calibration_method, cv=3
            )
            gb_calibrated.fit(X_train_scaled, y_train)
            models["gradient_boosting"] = gb_calibrated

            # === MODEL 3: EXTRA TREES (EXTREMELY RANDOMIZED TREES) ===
            et = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            et.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            et_calibrated = CalibratedClassifierCV(et, method=calibration_method, cv=3)
            et_calibrated.fit(X_train_scaled, y_train)
            models["extra_trees"] = et_calibrated

            # === MODEL 4: ADAPTIVE BOOSTING ===
            ada = AdaBoostClassifier(
                n_estimators=80,
                learning_rate=0.8,
                algorithm="SAMME",  # Use SAMME to avoid deprecation warning
                random_state=42,
            )
            ada.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            ada_calibrated = CalibratedClassifierCV(
                ada, method=calibration_method, cv=3
            )
            ada_calibrated.fit(X_train_scaled, y_train)
            models["ada_boost"] = ada_calibrated

            # === MODEL 5: SUPPORT VECTOR MACHINE (for small datasets) ===
            if (
                len(X_train) < 2000
            ):  # Only for smaller datasets due to computational cost
                svm = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,  # Enable for predict_proba (accepts double calibration trade-off)
                    random_state=42,
                    class_weight="balanced",
                )
                svm.fit(X_train_scaled, y_train)
                # V4: Calibrate probability with adaptive method (SVM especially benefits)
                svm_calibrated = CalibratedClassifierCV(
                    svm, method=calibration_method, cv=3
                )
                svm_calibrated.fit(X_train_scaled, y_train)
                models["svm"] = svm_calibrated

            # === MODEL 6: HISTOGRAM GRADIENT BOOSTING (modern, fast) ===
            hist_gb = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                learning_rate=0.1,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=10,
                random_state=42,
            )
            # V4: Fix warning - ensure consistent data type for HistGB
            X_train_hgb = (
                X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            )
            hist_gb.fit(X_train_hgb, y_train)  # HGB handles its own scaling
            # V4: Calibrate probability with adaptive method
            hist_gb_calibrated = CalibratedClassifierCV(
                hist_gb, method=calibration_method, cv=3
            )
            hist_gb_calibrated.fit(
                X_train_hgb, y_train
            )  # Use same format for consistency
            models["hist_gradient_boosting"] = hist_gb_calibrated

            y_pred_hist = hist_gb.predict(X_test)
            hist_f1 = f1_score(y_test, y_pred_hist, zero_division=0)
            hist_accuracy = accuracy_score(y_test, y_pred_hist)

            results["hist_gradient_boosting"] = {
                "model": hist_gb,
                "accuracy": hist_accuracy,
                "f1_score": hist_f1,
                "feature_importance": {},
            }

            logger.info(
                f"{pair} HistGradientBoosting: Acc={hist_accuracy:.3f}, F1={hist_f1:.3f}"
            )

            # === MODEL 7: LOGISTIC REGRESSION (baseline) ===
            lr = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                random_state=42,
                class_weight="balanced",
                max_iter=1000,
            )
            lr.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method (even for LR, for consistency)
            lr_calibrated = CalibratedClassifierCV(lr, method=calibration_method, cv=3)
            lr_calibrated.fit(X_train_scaled, y_train)
            models["logistic_regression"] = lr_calibrated

            # === EVALUATE ALL MODELS ===
            # Store WFV results for reporting
            wfv_summary = {
                "wfv_f1_mean": wfv_f1_mean,
                "wfv_f1_std": wfv_f1_std,
                "wfv_acc_mean": wfv_acc_mean,
                "wfv_auc_mean": wfv_auc_mean,
                "n_wfv_folds": len(wfv_results["f1_scores"]),
            }

            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = (
                    model.predict_proba(X_test_scaled)[:, 1]
                    if hasattr(model, "predict_proba")
                    else y_pred
                )

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Calculate AUC metrics using probabilities when available
                try:
                    # from sklearn.metrics import roc_auc_score, average_precision_score

                    if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 1:
                        auc_roc = roc_auc_score(y_test, y_pred_proba)
                        auc_pr = average_precision_score(y_test, y_pred_proba)
                    else:
                        # Fallback for models without predict_proba
                        auc_roc = (
                            roc_auc_score(y_test, y_pred)
                            if len(np.unique(y_test)) > 1
                            else 0.5
                        )
                        auc_pr = (
                            average_precision_score(y_test, y_pred)
                            if len(np.unique(y_test)) > 1
                            else 0.5
                        )
                except ImportError:
                    auc_roc = 0.5
                    auc_pr = 0.5

                # V4: Cross-validation with purged gap
                purged_cv = GapTimeSeriesSplit(n_splits=3, gap=purge_gap)
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=purged_cv, scoring="f1"
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                results[name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "probabilities": y_pred_proba,  # Store probabilities for ensemble
                    "feature_importance": self._get_feature_importance(
                        model, feature_columns
                    ),
                }

                logger.info(
                    f"{pair} {name}: Acc={accuracy:.3f}, F1={f1:.3f}, "
                    f"AUC={auc_roc:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}"
                )

            # === CREATE VOTING ENSEMBLE ===
            # Select top 3 models based on F1 score
            sorted_models = sorted(
                results.items(), key=lambda x: x[1]["f1_score"], reverse=True
            )
            top_models = [
                (name, results[name]["model"]) for name, _ in sorted_models[:3]
            ]

            if len(top_models) >= 2:
                voting_classifier = VotingClassifier(
                    estimators=top_models, voting="soft"  # Use probability averaging
                )
                voting_classifier.fit(X_train_scaled, y_train)
                models["voting_ensemble"] = voting_classifier

                # Evaluate ensemble
                y_pred_ensemble = voting_classifier.predict(X_test_scaled)
                ensemble_f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
                ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

                results["voting_ensemble"] = {
                    "model": voting_classifier,
                    "accuracy": ensemble_accuracy,
                    "f1_score": ensemble_f1,
                    "feature_importance": {},  # Ensemble doesn't have direct feature importance
                }

                logger.info(
                    f"{pair} Voting Ensemble: Acc={ensemble_accuracy:.3f}, F1={ensemble_f1:.3f}"
                )

            # === STACKING CLASSIFIER (Meta-learner) ===
            # Create stacking with top models
            if len(results) >= 3:
                # Get top 3 models by F1 score for stacking
                top_models_for_stacking = sorted(
                    [(name, res["model"]) for name, res in results.items()],
                    key=lambda x: results[x[0]]["f1_score"],
                    reverse=True,
                )[:3]

                # Create stacking classifier with logistic regression meta-learner
                stacking_classifier = StackingClassifier(
                    estimators=top_models_for_stacking,
                    final_estimator=LogisticRegression(C=1.0, random_state=42),
                    cv=3,  # Use cross-validation to train meta-learner
                    stack_method="predict_proba",
                    n_jobs=-1,
                )

                stacking_classifier.fit(X_train_scaled, y_train)

                # Evaluate stacking
                y_pred_stacking = stacking_classifier.predict(X_test_scaled)
                stacking_f1 = f1_score(y_test, y_pred_stacking, zero_division=0)
                stacking_accuracy = accuracy_score(y_test, y_pred_stacking)

                results["stacking_ensemble"] = {
                    "model": stacking_classifier,
                    "accuracy": stacking_accuracy,
                    "f1_score": stacking_f1,
                    "feature_importance": {},
                }

                logger.info(
                    f"{pair} Stacking Ensemble: Acc={stacking_accuracy:.3f}, F1={stacking_f1:.3f}"
                )

                # Add stacking to models if it performs better than voting
                if stacking_f1 > ensemble_f1:
                    models["stacking_ensemble"] = stacking_classifier
                    logger.info(
                        f"{pair} Stacking outperformed Voting (F1: {stacking_f1:.3f} > {ensemble_f1:.3f})"
                    )

            # V4: Models already calibrated individually with adaptive method
            # No need for double calibration
            self.models[pair] = models
            self.scalers[pair] = scaler
            self.feature_importance[pair] = results
            self.is_trained[pair] = True

            # V4: Store feature columns and calibration method in metadata
            if not hasattr(self, "model_metadata"):
                self.model_metadata = {}
            self.model_metadata[pair] = {
                "feature_columns": feature_columns,  # Canonical list of features
                "calibration_method": calibration_method,
                "n_features": len(feature_columns),
                "n_samples_train": len(X_train),
                "wfv_summary": wfv_summary,
            }
            # Update retrain metadata and cache
            self.last_train_time[pair] = datetime.utcnow()
            self.last_train_index[pair] = len(df)
            self.training_cache[pair] = base_df.copy()

            # Save models to disk for persistence
            # V4: Save to custom directory if provided (for async training)
            self._save_models_to_disk(pair, output_dir)

            # Find best model
            best_model_name = max(results.keys(), key=lambda k: results[k]["f1_score"])
            best_f1 = results[best_model_name]["f1_score"]

            return {
                "status": "success",
                "results": results,
                "feature_columns": feature_columns,
                "n_samples": len(X),
                "best_model": best_model_name,
                "best_f1_score": best_f1,
                "n_models": len(models),
            }

        except Exception as e:
            logger.warning(f"Model training failed for {pair}: {e}")
            return {"status": "error", "message": str(e)}

    def _get_feature_importance(self, model, feature_columns):
        """Extract feature importance from different model types"""
        try:
            if hasattr(model, "feature_importances_"):
                return dict(
                    zip(feature_columns, model.feature_importances_, strict=True)
                )
            elif hasattr(model, "coef_"):
                # For linear models like LogisticRegression
                importance = abs(model.coef_[0])
                return dict(zip(feature_columns, importance, strict=True))
            else:
                return {}
        except Exception:
            return {}

    def predict_entry_probability(self, df: pd.DataFrame, pair: str) -> pd.Series:
        """Predict probability of profitable entry using advanced ensemble models"""
        if (
            not SKLEARN_AVAILABLE
            or pair not in self.is_trained
            or not self.is_trained[pair]
        ):
            return pd.Series(0.5, index=df.index)

        try:
            # Use same feature extraction as training (V3)
            feature_df = self.extract_advanced_features_v3(df, pair)

            # V4: Get feature columns from metadata (canonical list)
            if hasattr(self, "model_metadata") and pair in self.model_metadata:
                feature_columns = self.model_metadata[pair].get("feature_columns", [])
                logger.debug(
                    f"[ML-V4] Using {len(feature_columns)} canonical features for {pair}"
                )
            elif (
                pair in self.feature_importance
                and "random_forest" in self.feature_importance[pair]
            ):
                # Fallback to feature importance keys
                feature_columns = list(
                    self.feature_importance[pair]["random_forest"][
                        "feature_importance"
                    ].keys()
                )
            else:
                # Last fallback: use all available numeric columns
                exclude_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "date",
                    "enter_long",
                    "enter_short",
                    "exit_long",
                    "exit_short",
                ]
                feature_columns = [
                    col
                    for col in feature_df.columns
                    if col not in exclude_cols
                    and feature_df[col].dtype in ["float64", "int64"]
                ]

            X = feature_df[feature_columns].fillna(0)
            X_scaled = self.scalers[pair].transform(X)

            # Get predictions from all available models
            model_predictions = {}
            model_weights = {}

            for model_name, model in self.models[pair].items():
                try:
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        # For models without probability prediction
                        pred = model.predict(X_scaled)
                        prob = (
                            pred + 1
                        ) / 2  # Convert -1,1 to 0,1 or similar normalization

                    model_predictions[model_name] = prob

                    # Weight based on model performance (F1 score or accuracy)
                    if (
                        pair in self.feature_importance
                        and model_name in self.feature_importance[pair]
                    ):
                        performance_metrics = self.feature_importance[pair][model_name]
                        # Use F1 score if available, otherwise accuracy
                        weight = performance_metrics.get(
                            "f1_score", performance_metrics.get("accuracy", 0.5)
                        )
                    else:
                        weight = 0.5

                    model_weights[model_name] = max(
                        weight, 0.1
                    )  # Minimum weight of 0.1

                except Exception as e:
                    logger.warning(
                        f"Failed to get predictions from {model_name} for {pair}: {e}"
                    )
                    continue

            if not model_predictions:
                return pd.Series(0.5, index=df.index)

            # === V3 ENSEMBLE PREDICTION (PERFORMANCE-BASED ONLY) ===

            # Use only performance-based weights (F1 scores)
            # No regime-based adjustments - let the models prove themselves

            # Method 1: Weighted average by performance
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                weighted_avg = np.zeros(len(X))
                for model_name, predictions in model_predictions.items():
                    weight = model_weights[model_name] / total_weight
                    weighted_avg += predictions * weight
            else:
                weighted_avg = np.mean(list(model_predictions.values()), axis=0)

            # Method 2: Voting ensemble (if available)
            if "voting_ensemble" in model_predictions:
                ensemble_pred = model_predictions["voting_ensemble"]
                # Combine weighted average with voting ensemble
                final_prediction = 0.6 * ensemble_pred + 0.4 * weighted_avg
            else:
                final_prediction = weighted_avg

            # Method 3: Dynamic model selection based on market conditions
            # V4 FIX: Corrected hasattr to check for the actual function name
            if hasattr(self, "_detect_market_condition"):
                market_regime = self._detect_market_condition(df)

                # Adjust predictions based on market conditions
                if market_regime == "trending":
                    # In trending markets, prefer gradient boosting
                    if "gradient_boosting" in model_predictions:
                        final_prediction = (
                            0.5 * final_prediction
                            + 0.5 * model_predictions["gradient_boosting"]
                        )
                elif market_regime == "volatile":
                    # In volatile markets, prefer random forest
                    if "random_forest" in model_predictions:
                        final_prediction = (
                            0.5 * final_prediction
                            + 0.5 * model_predictions["random_forest"]
                        )
                elif market_regime == "ranging":
                    # In ranging markets, prefer SVM or logistic regression
                    if "svm" in model_predictions:
                        final_prediction = (
                            0.6 * final_prediction + 0.4 * model_predictions["svm"]
                        )
                    elif "logistic_regression" in model_predictions:
                        final_prediction = (
                            0.6 * final_prediction
                            + 0.4 * model_predictions["logistic_regression"]
                        )

            # === C. IMPROVED MODEL AGREEMENT WITH STRONG MODELS ===

            # Focus on agreement among strong models (HistGB, RF, ExtraTrees)
            strong_models = ["histgradient_boosting", "random_forest", "extra_trees"]
            strong_predictions = []

            for model_name in strong_models:
                if model_name in model_predictions:
                    strong_predictions.append(model_predictions[model_name])

            # Calculate agreement metrics
            if len(strong_predictions) >= 2:
                # Count how many strong models agree with high confidence
                strong_array = np.array(strong_predictions)

                # For each sample, check agreement
                agreement_scores = []
                for i in range(len(X)):
                    sample_preds = strong_array[:, i]
                    # Count models above threshold (will be checked against thr_dyn later)
                    high_conf_count = np.sum(sample_preds > 0.35)  # Base threshold
                    agreement_score = high_conf_count / len(strong_predictions)
                    agreement_scores.append(agreement_score)

                # Store agreement scores for later use
                df["ml_strong_agreement"] = agreement_scores

                # Calculate overall model agreement (standard deviation based)
                predictions_array = np.array(list(model_predictions.values()))
                prediction_std = np.std(predictions_array, axis=0)

                # Higher standard deviation = lower confidence
                confidence_factor = 1 - np.clip(prediction_std * 2, 0, 0.3)

                # Adjust predictions toward neutral when confidence is low
                final_prediction = final_prediction * confidence_factor + 0.5 * (
                    1 - confidence_factor
                )
            else:
                # Not enough strong models, use standard agreement
                df["ml_strong_agreement"] = 0.5

                if len(model_predictions) > 1:
                    predictions_array = np.array(list(model_predictions.values()))
                    prediction_std = np.std(predictions_array, axis=0)
                    confidence_factor = 1 - np.clip(prediction_std * 2, 0, 0.3)
                    final_prediction = final_prediction * confidence_factor + 0.5 * (
                        1 - confidence_factor
                    )

            # === OUTLIER DETECTION AND SMOOTHING ===

            # Apply rolling smoothing to reduce noise
            result_series = pd.Series(final_prediction, index=df.index)
            smoothed_result = result_series.rolling(window=3, center=True).mean()
            smoothed_result = smoothed_result.fillna(result_series)

            # Ensure values are in valid range [0, 1]
            smoothed_result = smoothed_result.clip(0, 1)

            return smoothed_result

        except Exception as e:
            logger.warning(f"Advanced prediction failed for {pair}: {e}")
            return pd.Series(0.5, index=df.index)

    def _detect_market_condition(self, df: pd.DataFrame) -> str:
        """Detect current market condition for dynamic model selection"""
        try:
            # Simple market regime detection
            recent_data = df.tail(20)

            if len(recent_data) < 10:
                return "unknown"

            # Calculate volatility
            returns = recent_data["close"].pct_change().dropna()
            volatility = returns.std()

            # Calculate trend strength
            price_change = (
                recent_data["close"].iloc[-1] - recent_data["close"].iloc[0]
            ) / recent_data["close"].iloc[0]

            if volatility > 0.03:  # High volatility threshold
                return "volatile"
            elif abs(price_change) > 0.05:  # Strong trend threshold
                return "trending"
            else:
                return "ranging"

        except Exception:
            return "unknown"

    # Removed _get_regime_based_weights method - using performance-based weights only


# Initialize the predictive engine globally
predictive_engine = AdvancedPredictiveEngine()


def calculate_advanced_predictive_signals(
    dataframe: pd.DataFrame, pair: str, base_threshold: float = 0.02
) -> pd.DataFrame:
    """Main function to calculate advanced predictive signals with enhanced models.

    V4 Improvements:
    - Async training (non-blocking)
    - Better telemetry
    - Dynamic threshold tracking

    Training logic:
    1. If assets missing -> train immediately (async)
    2. If assets exist but 48h+ passed since strategy start -> retrain (async)
    3. Otherwise skip training
    """
    # V4: Track timing
    t_start = time.time()

    try:
        need_training = False
        assets_exist = False
        try:
            assets_exist = predictive_engine._assets_exist(pair)
        except Exception:
            assets_exist = False

        # Check time since strategy startup
        now_utc = datetime.utcnow()
        hours_since_startup = (
            now_utc - predictive_engine.strategy_start_time
        ).total_seconds() / 3600.0

        if not assets_exist:
            # Missing assets -> train immediately
            if len(dataframe) >= 200:
                need_training = True
        elif (
            predictive_engine.enable_startup_retrain
            and hours_since_startup >= predictive_engine.retrain_after_startup_hours
        ):
            # Assets exist but 48h+ passed since startup -> retrain
            if len(dataframe) >= 200:
                need_training = True
                logger.info(
                    f"[ML] Triggering 48h startup retrain for {pair} "
                    f"(startup+{hours_since_startup:.1f}h)"
                )

        # V4: Drift detection - retrain if prediction distribution shifts significantly
        if (
            not need_training
            and "ml_entry_probability" in dataframe.columns
            and len(dataframe) > 500
        ):
            # Calculate KS statistic between recent and historical predictions
            recent_preds = dataframe["ml_entry_probability"].tail(100).dropna()
            historical_preds = (
                dataframe["ml_entry_probability"].iloc[-500:-100].dropna()
            )

            if len(recent_preds) > 50 and len(historical_preds) > 50:
                from scipy.stats import ks_2samp

                ks_stat, p_value = ks_2samp(recent_preds, historical_preds)

                # If distributions are significantly different (drift detected)
                if ks_stat > 0.3 and p_value < 0.05:
                    need_training = True
                    logger.info(
                        f"[ML-V4] Drift detected for {pair} (KS={ks_stat:.3f}, p={p_value:.4f})"
                    )

        if need_training:
            logger.info(
                f"[ML-V4] Triggering async training for {pair} (len={len(dataframe)}) | assets_exist={assets_exist}"
            )
            # V4: Use async training to prevent blocking
            predictive_engine.train_predictive_models_async(dataframe, pair)
            # Note: Training happens in background, we continue with existing models

        # Enhanced ML probability prediction
        dataframe["ml_entry_probability"] = predictive_engine.predict_entry_probability(
            dataframe, pair
        )

        # === B2 DYNAMIC THRESHOLD (Percentile-based auto-calibration) ===
        # Calculate dynamic threshold based on recent prediction distribution
        window = 300  # ~300 candles lookback
        # base_threshold is now passed as parameter

        # === EXPECTED VALUE (EV) BASED THRESHOLD ===
        # 1. ALINEAR HORIZONTE EV con horizonte de predicción del modelo
        h = getattr(
            predictive_engine, "prediction_horizon", 5
        )  # Usar mismo horizonte que el modelo
        returns = dataframe["close"].pct_change(h).shift(-h)  # h-period forward returns

        # Estimate win/loss magnitudes based on recent ML predictions
        high_conf_mask = dataframe["ml_entry_probability"].shift(h) > 0.6
        low_conf_mask = dataframe["ml_entry_probability"].shift(h) <= 0.6

        # V4 FIX: Use default fee since this is a standalone function (not in class)
        # Cannot access self.config here - using default fee
        fee_estimate = 0.001  # Default 0.1% fee

        # Calculate average win/loss for high confidence predictions
        avg_win = 0.01  # Default 1% win
        avg_loss = 0.01  # Default 1% loss
        if high_conf_mask.sum() > 10:
            win_returns = returns[high_conf_mask & (returns > 0)]
            loss_returns = returns[high_conf_mask & (returns < 0)]
            if len(win_returns) > 0:
                avg_win = win_returns.mean()
            if len(loss_returns) > 0:
                avg_loss = abs(loss_returns.mean())

        # Calculate 80th percentile of recent predictions
        dataframe["pred_q80"] = (
            dataframe["ml_entry_probability"]
            .rolling(window, min_periods=50)
            .quantile(0.80)
        )

        # Apply smoothing and clipping to create dynamic threshold
        # V4 FIX: Auditor identified critical bug - threshold was clipped too low (0.01-0.06)
        # Now using adaptive range based on break-even probability

        # --- HYBRID ADAPTIVE THRESHOLD IMPLEMENTATION ---
        # Combines break-even economics with model temperature and market volatility

        # 1) Base económica: break-even
        if pd.notna(avg_win) and pd.notna(avg_loss) and (avg_win + avg_loss) > 0:
            p_be = (avg_loss + fee_estimate) / (avg_win + avg_loss)
        else:
            p_be = 0.55  # Default conservador

        base_floor = np.clip(p_be + 0.03, 0.50, 0.65)  # Suelo "económico" de V4

        # 2) Temperatura del modelo y volatilidad (suavizadas)
        # pred_q80 ya está calculado arriba con rolling(240)
        pred_q80_smooth = dataframe["pred_q80"].ewm(alpha=0.2, adjust=False).mean()
        model_temp = float(pred_q80_smooth.fillna(0.5).iloc[-1])

        # Volatilidad relativa suavizada
        if "atr" in dataframe.columns:
            rel_atr = (
                (dataframe["atr"] / dataframe["close"])
                .rolling(20, min_periods=5)
                .mean()
            )
            vol_factor = float(
                (1.0 + rel_atr.fillna(0.01) * 10.0).clip(1.0, 1.5).iloc[-1]
            )
        else:
            vol_factor = 1.0

        # 3) A. UMBRALES DINÁMICOS BASADOS EN RSI (endurecer fuera de oversold)
        # RSI determina la exigencia del umbral
        current_rsi = float(dataframe["rsi"].iloc[-1] if "rsi" in dataframe else 50)

        if current_rsi < 35:  # Oversold real
            # RSI < 35 → thr_dyn_low = 0.26 (similar a hoy)
            adj_low, adj_high = 0.26 * vol_factor, 0.35 * vol_factor
        elif current_rsi < 50:  # Zona media
            # 35 ≤ RSI < 50 → thr_dyn_mid = 0.34
            adj_low, adj_high = 0.34 * vol_factor, 0.42 * vol_factor
        else:  # RSI alto (no oversold)
            # RSI ≥ 50 → thr_dyn_high = 0.42
            adj_low, adj_high = 0.42 * vol_factor, 0.50 * vol_factor

        # 4) B. FALLBACK: Si EV≈0, elevar thr_dyn según RSI
        # Como EV_mean=0 no reacciona, usamos el RSI como referencia
        if abs(float(dataframe.get("expected_value", pd.Series(0)).iloc[-1])) < 0.001:
            # EV cercano a 0, usar fallback basado en RSI
            if current_rsi < 35:
                economic_floor = 0.26
            elif current_rsi < 50:
                economic_floor = 0.33
            else:
                economic_floor = 0.38
        else:
            economic_floor = max(0.25, min(0.35, p_be - 0.10))

        thr_low_raw = max(economic_floor, adj_low)
        thr_high_raw = max(thr_low_raw + 0.08, adj_high)

        # 5) Suavizado EMA para evitar "dientes de sierra"
        # Usar variables globales para mantener estado entre llamadas
        global thr_low_ema_cache, thr_high_ema_cache

        if "thr_low_ema_cache" not in globals():
            thr_low_ema_cache = {}
            thr_high_ema_cache = {}

        if pair not in thr_low_ema_cache:
            thr_low_ema_cache[pair] = thr_low_raw
            thr_high_ema_cache[pair] = thr_high_raw

        thr_low = thr_low_ema_cache[pair] * 0.8 + thr_low_raw * 0.2
        thr_high = thr_high_ema_cache[pair] * 0.8 + thr_high_raw * 0.2
        thr_low_ema_cache[pair], thr_high_ema_cache[pair] = thr_low, thr_high

        # Log de diagnóstico cada cierto tiempo
        if pair and random.random() < 0.25:  # 25% del tiempo para mejor visibilidad
            logger.info(
                f"[ADAPTIVE] {pair} model_temp={model_temp:.3f} vol_factor={vol_factor:.2f} "
                f"p_be={p_be:.3f} thr_low={thr_low:.3f} thr_high={thr_high:.3f}"
            )

        # 6) Construcción de thr_dyn con el clip nuevo adaptativo
        dataframe["thr_dyn"] = (
            dataframe["pred_q80"]
            .fillna(base_floor)
            .ewm(alpha=0.2, adjust=False)
            .mean()
            .clip(lower=thr_low, upper=thr_high)  # Ahora con rangos adaptativos
        )

        # Calculate Expected Value (EV) for filtering
        # EV = p * avg_win - (1-p) * avg_loss - fees
        if pd.notna(avg_win) and pd.notna(avg_loss) and high_conf_mask.sum() > 10:
            dataframe["expected_value"] = (
                dataframe["ml_entry_probability"] * avg_win
                - (1 - dataframe["ml_entry_probability"]) * avg_loss
                - fee_estimate
            )

            # Only enter when EV is positive AND probability exceeds base threshold
            dataframe["ev_filter"] = (dataframe["expected_value"] > 0).astype(int)
        else:
            # Don't filter if insufficient data
            dataframe["expected_value"] = 0
            dataframe["ev_filter"] = 1

        # Get momentum and volatility regime safely
        momentum_regime = dataframe.get("momentum_regime")
        volatility_regime = dataframe.get("volatility_regime")
        quantum_coherence = dataframe.get("quantum_momentum_coherence")
        neural_pattern = dataframe.get("neural_pattern_score")

        # Advanced confidence scoring with safe comparisons
        ml_high_conf_conditions = dataframe["ml_entry_probability"] > 0.8

        if momentum_regime is not None:
            ml_high_conf_conditions &= momentum_regime > 0

        if volatility_regime is not None:
            ml_high_conf_conditions &= volatility_regime < 2

        dataframe["ml_high_confidence"] = ml_high_conf_conditions.astype(int)

        # Ultra-high confidence entries with safe checks
        ml_ultra_conf_conditions = dataframe["ml_entry_probability"] > 0.9

        if quantum_coherence is not None:
            ml_ultra_conf_conditions &= quantum_coherence > 0.7
        else:
            # Use fallback threshold if quantum analysis not available
            ml_ultra_conf_conditions &= dataframe["ml_entry_probability"] > 0.92

        if neural_pattern is not None:
            ml_ultra_conf_conditions &= neural_pattern > 0.8
        else:
            # Use fallback threshold if neural analysis not available
            ml_ultra_conf_conditions &= dataframe["ml_entry_probability"] > 0.93

        dataframe["ml_ultra_confidence"] = ml_ultra_conf_conditions.astype(int)

        # Enhanced score combination
        if "ultimate_score" in dataframe.columns:
            # Dynamic weighting based on model performance - fix Series comparison
            ml_volatility = (
                dataframe["ml_entry_probability"].rolling(20).std().fillna(0.3)
            )
            ml_weight = ml_volatility.clip(
                upper=0.5
            )  # Safe way to apply min with Series
            traditional_weight = 1 - ml_weight

            dataframe["ml_enhanced_score"] = (
                dataframe["ultimate_score"] * traditional_weight
                + dataframe["ml_entry_probability"] * ml_weight
            )
        else:
            dataframe["ml_enhanced_score"] = dataframe["ml_entry_probability"]

        # C. IMPROVED MODEL AGREEMENT - Focus on strong models
        if pair in predictive_engine.models and len(predictive_engine.models[pair]) > 1:
            # Use strong model agreement if available
            if "ml_strong_agreement" in dataframe:
                # At least 2 of {HistGB, RF, ExtraTrees} should agree
                dataframe["ml_model_agreement"] = dataframe["ml_strong_agreement"]
            else:
                # Fallback to standard agreement measure
                prob_std = dataframe["ml_entry_probability"].rolling(5).std().fillna(0)
                dataframe["ml_model_agreement"] = (1 - prob_std).clip(lower=0, upper=1)
        else:
            dataframe["ml_model_agreement"] = 0.5  # Default agreement prudente

        return dataframe

    except Exception as e:
        logger.warning(f"Advanced predictive analysis failed for {pair}: {e}")
        dataframe["ml_entry_probability"] = 0.5
        dataframe["ml_enhanced_score"] = dataframe.get("ultimate_score", 0.5)
        dataframe["ml_high_confidence"] = 0
        dataframe["ml_ultra_confidence"] = 0
        dataframe["ml_model_agreement"] = 0.5
        return dataframe


def calculate_quantum_momentum_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Quantum-inspired momentum analysis for ultra-precise predictions"""
    try:
        momentum_periods = [3, 5, 8, 13, 21, 34]
        momentum_matrix = pd.DataFrame()

        for period in momentum_periods:
            momentum_matrix[f"mom_{period}"] = dataframe["close"].pct_change(period)

        dataframe["quantum_momentum_coherence"] = momentum_matrix.std(axis=1) / (
            momentum_matrix.mean(axis=1).abs() + 1e-10
        )

        # Calculate momentum entanglement using correlation matrix
        def calculate_entanglement(window_data):
            if len(window_data) < 10:
                return 0
            try:
                corr_matrix = window_data.corr()
                if corr_matrix.empty or corr_matrix.isna().all().all():
                    return 0
                # Get upper triangular correlation values (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                correlations = corr_matrix.values[upper_tri_indices]
                # Remove NaN values and calculate mean
                valid_correlations = correlations[~np.isnan(correlations)]
                return valid_correlations.mean() if len(valid_correlations) > 0 else 0
            except Exception:
                return 0

        entanglement_values = []
        for i in range(len(momentum_matrix)):
            if i < 20:
                entanglement_values.append(0.5)
            else:
                window_data = momentum_matrix.iloc[i - 19 : i + 1]
                entanglement = calculate_entanglement(window_data)
                entanglement_values.append(entanglement)

        dataframe["momentum_entanglement"] = pd.Series(
            entanglement_values, index=dataframe.index
        )

        price_uncertainty = dataframe["close"].rolling(20).std()
        momentum_uncertainty = momentum_matrix["mom_8"].rolling(20).std()
        dataframe["heisenberg_uncertainty"] = price_uncertainty * momentum_uncertainty

        if "maxima_sort_threshold" in dataframe.columns:
            resistance_distance = (
                dataframe["maxima_sort_threshold"] - dataframe["close"]
            ) / dataframe["close"]
            dataframe["quantum_tunnel_up_prob"] = np.exp(
                -resistance_distance.abs() * 10
            )
        else:
            dataframe["quantum_tunnel_up_prob"] = 0.5

        if "minima_sort_threshold" in dataframe.columns:
            support_distance = (
                dataframe["close"] - dataframe["minima_sort_threshold"]
            ) / dataframe["close"]
            dataframe["quantum_tunnel_down_prob"] = np.exp(-support_distance.abs() * 10)
        else:
            dataframe["quantum_tunnel_down_prob"] = 0.5

        return dataframe

    except Exception as e:
        logger.warning(f"Quantum momentum analysis failed: {e}")
        dataframe["quantum_momentum_coherence"] = 0.5
        dataframe["momentum_entanglement"] = 0.5
        dataframe["heisenberg_uncertainty"] = 1.0
        dataframe["quantum_tunnel_up_prob"] = 0.5
        dataframe["quantum_tunnel_down_prob"] = 0.5
        return dataframe


def calculate_neural_pattern_recognition(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Neural pattern recognition for complex market patterns"""
    try:
        dataframe["body_size"] = (
            abs(dataframe["close"] - dataframe["open"]) / dataframe["close"]
        )
        dataframe["upper_shadow"] = (
            dataframe["high"] - np.maximum(dataframe["open"], dataframe["close"])
        ) / dataframe["close"]
        dataframe["lower_shadow"] = (
            np.minimum(dataframe["open"], dataframe["close"]) - dataframe["low"]
        ) / dataframe["close"]
        dataframe["candle_range"] = (dataframe["high"] - dataframe["low"]) / dataframe[
            "close"
        ]

        pattern_memory = []
        for i in range(len(dataframe)):
            if i < 5:
                pattern_memory.append(0)
                continue

            recent_patterns = dataframe[
                ["body_size", "upper_shadow", "lower_shadow"]
            ].iloc[i - 4 : i + 1]
            pattern_signature = recent_patterns.values.flatten()
            pattern_norm = np.linalg.norm(pattern_signature)

            if pattern_norm > 0:
                pattern_score = min(1.0, pattern_norm / 0.1)
            else:
                pattern_score = 0

            pattern_memory.append(pattern_score)

        dataframe["neural_pattern_score"] = pd.Series(
            pattern_memory, index=dataframe.index
        )
        dataframe["pattern_prediction_confidence"] = (
            dataframe["neural_pattern_score"].rolling(10).std()
        )

        return dataframe

    except Exception as e:
        logger.warning(f"Neural pattern recognition failed: {e}")
        dataframe["neural_pattern_score"] = 0.5
        dataframe["pattern_prediction_confidence"] = 0.5
        dataframe["body_size"] = 0.01
        dataframe["candle_range"] = 0.02
        return dataframe


def calculate_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced exit signals based on market deterioration"""
    # === MOMENTUM DETERIORATION ===
    dataframe["momentum_deteriorating"] = (
        (dataframe["momentum_quality"] < dataframe["momentum_quality"].shift(1))
        & (dataframe["momentum_acceleration"] < 0)
        & (dataframe["price_momentum"] < dataframe["price_momentum"].shift(1))
    ).astype(int)

    # === VOLUME DETERIORATION ===
    dataframe["volume_deteriorating"] = (
        (dataframe["volume_strength"] < 0.8)
        & (dataframe["selling_pressure"] > dataframe["buying_pressure"])
        & (dataframe["volume_pressure"] < 0)
    ).astype(int)

    # === STRUCTURE DETERIORATION ===
    dataframe["structure_deteriorating"] = (
        (dataframe["structure_score"] < -1)
        & (dataframe["bearish_structure"] > dataframe["bullish_structure"])
        & (dataframe["structure_break_down"] == 1)
    ).astype(int)

    # === CONFLUENCE BREAKDOWN ===
    dataframe["confluence_breakdown"] = (
        (dataframe["confluence_score"] < 2)
        & (dataframe["near_resistance"] == 1)
        & (dataframe["volume_spike"] == 0)
    ).astype(int)

    # === TREND WEAKNESS ===
    dataframe["trend_weakening"] = (
        (dataframe["trend_strength"] < 0)
        & (dataframe["close"] < dataframe["ema50"])
        & (dataframe["strong_downtrend"] == 1)
    ).astype(int)

    # === ULTIMATE EXIT SCORE ===
    dataframe["exit_pressure"] = (
        dataframe["momentum_deteriorating"] * 2
        + dataframe["volume_deteriorating"] * 2
        + dataframe["structure_deteriorating"] * 2
        + dataframe["confluence_breakdown"] * 1
        + dataframe["trend_weakening"] * 1
    )

    # === RSI OVERBOUGHT WITH DIVERGENCE ===
    dataframe["rsi_exit_signal"] = (
        (dataframe["rsi"] > 75)
        & (
            (dataframe["rsi_divergence_bear"] == 1)
            | (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["close"] < dataframe["close"].shift(1))
        )
    ).astype(int)

    # === PROFIT TAKING LEVELS ===
    mml_resistance_levels = ["[6/8]P", "[8/8]P"]
    dataframe["near_resistance_level"] = 0

    for level in mml_resistance_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe["close"] >= dataframe[level] * 0.99)
                & (dataframe["close"] <= dataframe[level] * 1.02)
            ).astype(int)
            dataframe["near_resistance_level"] += near_level

    # === VOLATILITY SPIKE EXIT ===
    dataframe["volatility_spike"] = (
        dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 1.5
    ).astype(int)

    # === EXHAUSTION SIGNALS ===
    dataframe["bullish_exhaustion"] = (
        (dataframe["consecutive_green"] >= 4)
        & (dataframe["rsi"] > 70)
        & (dataframe["volume"] < dataframe["avg_volume"] * 0.8)
        & (dataframe["momentum_acceleration"] < 0)
    ).astype(int)

    return dataframe


def calculate_dynamic_profit_targets(
    dataframe: pd.DataFrame, entry_type_col: str = "entry_type"
) -> pd.DataFrame:
    """Calculate dynamic profit targets based on entry quality and market conditions"""

    # Base profit targets based on ATR
    dataframe["base_profit_target"] = dataframe["atr"] * 2

    # Adjust based on entry type
    dataframe["profit_multiplier"] = 1.0
    if entry_type_col in dataframe.columns:
        dataframe.loc[dataframe[entry_type_col] == 3, "profit_multiplier"] = (
            2.0  # High quality
        )
        dataframe.loc[dataframe[entry_type_col] == 2, "profit_multiplier"] = (
            1.5  # Medium quality
        )
        dataframe.loc[dataframe[entry_type_col] == 1, "profit_multiplier"] = (
            1.2  # Backup
        )
        dataframe.loc[dataframe[entry_type_col] == 4, "profit_multiplier"] = (
            2.5  # Breakout
        )
        dataframe.loc[dataframe[entry_type_col] == 5, "profit_multiplier"] = (
            1.8  # Reversal
        )

    # Final profit target
    dataframe["dynamic_profit_target"] = (
        dataframe["base_profit_target"] * dataframe["profit_multiplier"]
    )

    return dataframe


def calculate_advanced_stop_loss(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["base_stop_loss"] = dataframe["atr"] * 1.5
    if "minima_sort_threshold" in dataframe.columns:
        dataframe["support_stop_loss"] = (
            dataframe["close"] - dataframe["minima_sort_threshold"]
        )
        dataframe["support_stop_loss"] = dataframe["support_stop_loss"].clip(
            dataframe["base_stop_loss"] * 0.5,
            dataframe["base_stop_loss"] * 1.5,  # Reduced from 2.0
        )
        dataframe["final_stop_loss"] = np.minimum(
            dataframe["base_stop_loss"], dataframe["support_stop_loss"]
        ).clip(
            -0.15, -0.01
        )  # Hard cap at -15%
    else:
        dataframe["final_stop_loss"] = dataframe["base_stop_loss"].clip(-0.15, -0.01)
    return dataframe


def calculate_confluence_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor confluence analysis - much better than BTC correlation"""

    # Support/Resistance Confluence
    dataframe["near_support"] = (
        (dataframe["close"] <= dataframe["minima_sort_threshold"] * 1.02)
        & (dataframe["close"] >= dataframe["minima_sort_threshold"] * 0.98)
    ).astype(int)

    dataframe["near_resistance"] = (
        (dataframe["close"] <= dataframe["maxima_sort_threshold"] * 1.02)
        & (dataframe["close"] >= dataframe["maxima_sort_threshold"] * 0.98)
    ).astype(int)

    # MML Level Confluence
    mml_levels = ["[0/8]P", "[2/8]P", "[4/8]P", "[6/8]P", "[8/8]P"]
    dataframe["near_mml"] = 0

    for level in mml_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe["close"] <= dataframe[level] * 1.015)
                & (dataframe["close"] >= dataframe[level] * 0.985)
            ).astype(int)
            dataframe["near_mml"] += near_level

    # Volume Confluence
    dataframe["volume_spike"] = (
        dataframe["volume"] > dataframe["avg_volume"] * 1.5
    ).astype(int)

    # RSI Confluence Zones
    dataframe["rsi_oversold"] = (dataframe["rsi"] < 30).astype(int)
    dataframe["rsi_overbought"] = (dataframe["rsi"] > 70).astype(int)
    dataframe["rsi_neutral"] = (
        (dataframe["rsi"] >= 40) & (dataframe["rsi"] <= 60)
    ).astype(int)

    # EMA Confluence
    dataframe["above_ema"] = (dataframe["close"] > dataframe["ema50"]).astype(int)

    # CONFLUENCE SCORE (0-6)
    dataframe["confluence_score"] = (
        dataframe["near_support"]
        + dataframe["near_mml"].clip(0, 2)  # Max 2 points for MML
        + dataframe["volume_spike"]
        + dataframe["rsi_oversold"]
        + dataframe["above_ema"]
        + (dataframe["trend_strength"] > 0.01).astype(int)  # Positive trend
    )

    return dataframe


def calculate_smart_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced volume analysis - beats any external correlation"""

    # Volume-Price Trend (VPT)
    price_change_pct = (dataframe["close"] - dataframe["close"].shift(1)) / dataframe[
        "close"
    ].shift(1)
    dataframe["vpt"] = (dataframe["volume"] * price_change_pct).fillna(0).cumsum()

    # Volume moving averages
    dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
    dataframe["volume_sma50"] = dataframe["volume"].rolling(50).mean()

    # Volume strength
    dataframe["volume_strength"] = dataframe["volume"] / dataframe["volume_sma20"]

    # Smart money indicators
    dataframe["accumulation"] = (
        (dataframe["close"] > dataframe["open"])  # Green candle
        & (dataframe["volume"] > dataframe["volume_sma20"] * 1.2)  # High volume
        & (
            dataframe["close"] > (dataframe["high"] + dataframe["low"]) / 2
        )  # Close in upper half
    ).astype(int)

    dataframe["distribution"] = (
        (dataframe["close"] < dataframe["open"])  # Red candle
        & (dataframe["volume"] > dataframe["volume_sma20"] * 1.2)  # High volume
        & (
            dataframe["close"] < (dataframe["high"] + dataframe["low"]) / 2
        )  # Close in lower half
    ).astype(int)

    # Buying/Selling pressure
    dataframe["buying_pressure"] = dataframe["accumulation"].rolling(5).sum()
    dataframe["selling_pressure"] = dataframe["distribution"].rolling(5).sum()

    # Net volume pressure
    dataframe["volume_pressure"] = (
        dataframe["buying_pressure"] - dataframe["selling_pressure"]
    )

    # Volume trend
    dataframe["volume_trend"] = (
        dataframe["volume_sma20"] > dataframe["volume_sma50"]
    ).astype(int)

    # Money flow
    typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
    money_flow = typical_price * dataframe["volume"]
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.rolling(14).sum()

    dataframe["money_flow_ratio"] = positive_flow_sum / (negative_flow_sum + 1e-10)
    dataframe["money_flow_index"] = 100 - (100 / (1 + dataframe["money_flow_ratio"]))

    return dataframe


def calculate_advanced_momentum(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe momentum system - superior to BTC correlation"""

    # Multi-timeframe momentum
    dataframe["momentum_3"] = dataframe["close"].pct_change(6)
    dataframe["momentum_7"] = dataframe["close"].pct_change(14)
    dataframe["momentum_14"] = dataframe["close"].pct_change(28)
    dataframe["momentum_21"] = dataframe["close"].pct_change(21)

    # Momentum acceleration
    dataframe["momentum_acceleration"] = dataframe["momentum_3"] - dataframe[
        "momentum_3"
    ].shift(3)

    # Momentum consistency
    dataframe["momentum_consistency"] = (
        (dataframe["momentum_3"] > 0).astype(int)
        + (dataframe["momentum_7"] > 0).astype(int)
        + (dataframe["momentum_14"] > 0).astype(int)
    )

    # Momentum divergence with volume
    dataframe["price_momentum_rank"] = (
        dataframe["momentum_7"].rolling(20).rank(pct=True)
    )
    dataframe["volume_momentum_rank"] = (
        dataframe["volume_strength"].rolling(20).rank(pct=True)
    )

    dataframe["momentum_divergence"] = (
        dataframe["price_momentum_rank"] - dataframe["volume_momentum_rank"]
    ).abs()

    # Momentum strength
    dataframe["momentum_strength"] = (
        dataframe["momentum_3"].abs()
        + dataframe["momentum_7"].abs()
        + dataframe["momentum_14"].abs()
    ) / 3

    # Momentum quality score (0-5)
    dataframe["momentum_quality"] = (
        (dataframe["momentum_3"] > 0).astype(int)
        + (dataframe["momentum_7"] > 0).astype(int)
        + (dataframe["momentum_acceleration"] > 0).astype(int)
        + (dataframe["volume_strength"] > 1.1).astype(int)
        + (dataframe["momentum_divergence"] < 0.3).astype(int)
    )

    # Rate of Change
    dataframe["roc_5"] = dataframe["close"].pct_change(5) * 100
    dataframe["roc_10"] = dataframe["close"].pct_change(10) * 100
    dataframe["roc_20"] = dataframe["close"].pct_change(20) * 100

    # Momentum oscillator
    dataframe["momentum_oscillator"] = (
        dataframe["roc_5"] + dataframe["roc_10"] + dataframe["roc_20"]
    ) / 3

    return dataframe


def calculate_market_structure(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Market structure analysis - intrinsic trend recognition"""

    # Higher highs, higher lows detection
    dataframe["higher_high"] = (
        (dataframe["high"] > dataframe["high"].shift(1))
        & (dataframe["high"].shift(1) > dataframe["high"].shift(2))
    ).astype(int)

    dataframe["higher_low"] = (
        (dataframe["low"] > dataframe["low"].shift(1))
        & (dataframe["low"].shift(1) > dataframe["low"].shift(2))
    ).astype(int)

    dataframe["lower_high"] = (
        (dataframe["high"] < dataframe["high"].shift(1))
        & (dataframe["high"].shift(1) < dataframe["high"].shift(2))
    ).astype(int)

    dataframe["lower_low"] = (
        (dataframe["low"] < dataframe["low"].shift(1))
        & (dataframe["low"].shift(1) < dataframe["low"].shift(2))
    ).astype(int)

    # Market structure scores
    dataframe["bullish_structure"] = (
        dataframe["higher_high"].rolling(5).sum()
        + dataframe["higher_low"].rolling(5).sum()
    )

    dataframe["bearish_structure"] = (
        dataframe["lower_high"].rolling(5).sum()
        + dataframe["lower_low"].rolling(5).sum()
    )

    dataframe["structure_score"] = (
        dataframe["bullish_structure"] - dataframe["bearish_structure"]
    )

    # Swing highs and lows
    # Live-safe pivot detection without using future candles (no shift(-1))
    # Confirm swing at the PREVIOUS candle using only information up to current bar.
    # A swing high at t-1 is when high[t-1] > high[t-2] and high[t-1] > high[t].
    prev_high = dataframe["high"].shift(1)
    prev_low = dataframe["low"].shift(1)
    dataframe["swing_high"] = (
        (prev_high > dataframe["high"].shift(2)) & (prev_high > dataframe["high"])
    ).astype(int)

    # A swing low at t-1 is when low[t-1] < low[t-2] and low[t-1] < low[t].
    dataframe["swing_low"] = (
        (prev_low < dataframe["low"].shift(2)) & (prev_low < dataframe["low"])
    ).astype(int)

    # Market structure breaks
    # Use previous candle values where the swing was confirmed to avoid lookahead bias
    swing_highs = prev_high.where(dataframe["swing_high"] == 1)
    swing_lows = prev_low.where(dataframe["swing_low"] == 1)

    # Structure break detection
    dataframe["structure_break_up"] = (dataframe["close"] > swing_highs.ffill()).astype(
        int
    )

    dataframe["structure_break_down"] = (
        dataframe["close"] < swing_lows.ffill()
    ).astype(int)

    # Trend strength based on structure
    dataframe["structure_trend_strength"] = (
        dataframe["structure_score"] / 10  # Normalize
    ).clip(-1, 1)

    # Support and resistance strength
    dataframe["support_strength"] = dataframe["swing_low"].rolling(20).sum()
    dataframe["resistance_strength"] = dataframe["swing_high"].rolling(20).sum()

    return dataframe


def calculate_advanced_entry_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced entry signal generation"""

    # Multi-factor signal strength
    dataframe["signal_strength"] = 0

    # Confluence signals
    dataframe["confluence_signal"] = (dataframe["confluence_score"] >= 3).astype(int)
    dataframe["signal_strength"] += dataframe["confluence_signal"] * 2

    # Volume signals
    dataframe["volume_signal"] = (
        (dataframe["volume_pressure"] >= 2) & (dataframe["volume_strength"] > 1.2)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["volume_signal"] * 2

    # Momentum signals
    dataframe["momentum_signal"] = (
        (dataframe["momentum_quality"] >= 3) & (dataframe["momentum_acceleration"] > 0)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["momentum_signal"] * 2

    # Structure signals
    dataframe["structure_signal"] = (
        (dataframe["structure_score"] > 0) & (dataframe["structure_break_up"] == 1)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["structure_signal"] * 1

    # RSI position signal
    dataframe["rsi_signal"] = (
        (dataframe["rsi"] > 30) & (dataframe["rsi"] < 70)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["rsi_signal"] * 1

    # Trend alignment signal
    dataframe["trend_signal"] = (
        (dataframe["close"] > dataframe["ema50"]) & (dataframe["trend_strength"] > 0)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["trend_signal"] * 1

    # Money flow signal
    dataframe["money_flow_signal"] = (dataframe["money_flow_index"] > 50).astype(int)
    dataframe["signal_strength"] += dataframe["money_flow_signal"] * 1

    return dataframe


class AlexNexusForgeV8AIV4_SPOT(IStrategy):
    # General strategy parameters
    timeframe = "1h"
    startup_candle_count: int = 1000
    stoploss = -0.11
    trailing_stop = True
    trailing_stop_positive = 0.005  # Trail at 0.5% below peak profit
    trailing_stop_positive_offset = 0.03  # Start trailing only at 3% profit
    trailing_only_offset_is_reached = (
        True  # Ensure trailing only starts after offset is reached
    )
    use_custom_stoploss = True
    can_short = False  # Modificado para SPOT trading
    use_exit_signal = True
    ignore_roi_if_entry_signal = (
        False  # CHANGED: Allow ROI to work even with entry signals
    )
    process_only_new_candles = (
        True  # More efficient in live trading, avoids intra-candle re-evaluations
    )
    use_custom_exits_advanced = False
    use_emergency_exits = True

    regime_change_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    regime_change_sensitivity = DecimalParameter(
        0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=True, load=True
    )

    # Flash Move Detection
    flash_move_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    flash_move_threshold = DecimalParameter(
        0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=True, load=True
    )
    flash_move_candles = IntParameter(
        3, 10, default=5, space="sell", optimize=True, load=True
    )

    # Volume Spike Detection
    volume_spike_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    volume_spike_multiplier = DecimalParameter(
        2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True, load=True
    )

    # Emergency Exit Protection
    emergency_exit_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    emergency_exit_profit_threshold = DecimalParameter(
        0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True
    )

    # Trailing Stop Exit Control (NEW: Fix for "Blocking trailing stop exit")
    trailing_exit_min_profit = DecimalParameter(
        -0.03, 0.02, default=0.0, decimals=3, space="sell", optimize=True, load=True
    )
    # 3. REMOVIDO timed_exit_hours - usando ROI table para salidas temporales (24h/48h)
    strong_threshold = DecimalParameter(
        0.005, 0.08, default=0.020, decimals=3, space="buy", optimize=True, load=True
    )
    allow_trailing_exit_when_negative = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )

    # Market Sentiment Protection
    sentiment_protection_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    sentiment_shift_threshold = DecimalParameter(
        0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=True, load=True
    )

    # 🔧ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(
        0.8, 2.0, default=1.2, decimals=1, space="sell", optimize=True, load=True
    )
    atr_stoploss_minimum = DecimalParameter(
        -0.25, -0.10, default=-0.12, decimals=2, space="sell", optimize=True, load=True
    )
    atr_stoploss_maximum = DecimalParameter(
        -0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=True, load=True
    )
    atr_stoploss_ceiling = DecimalParameter(
        -0.10, -0.06, default=-0.08, decimals=2, space="sell", optimize=True, load=True
    )
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02,
        high=-0.01,
        default=-0.018,
        decimals=3,
        space="buy",
        optimize=True,
        load=True,
    )
    max_safety_orders = IntParameter(
        1, 3, default=1, space="buy", optimize=True, load=True
    )
    safety_order_step_scale = DecimalParameter(
        low=1.05,
        high=1.5,
        default=1.25,
        decimals=2,
        space="buy",
        optimize=True,
        load=True,
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1,
        high=2.0,
        default=1.4,
        decimals=1,
        space="buy",
        optimize=True,
        load=True,
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005,
        high=1.002,
        default=1.001,
        decimals=4,
        space="buy",
        optimize=True,
        load=True,
    )
    last_entry_price: Optional[float] = None

    # G. Protection parameters - más estrictos
    cooldown_lookback = IntParameter(
        2, 48, default=3, space="protection", optimize=True
    )  # Subido a 3 velas
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(
        default=True, space="protection", optimize=True
    )

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(
        1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True
    )
    mml_const2 = DecimalParameter(
        0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True
    )
    indicator_mml_window = IntParameter(
        32, 128, default=64, space="buy", optimize=True, load=True
    )

    # Dynamic Stoploss parameters
    # Add these parameters
    stoploss_atr_multiplier = DecimalParameter(
        1.0, 3.0, default=1.5, space="sell", optimize=True
    )
    stoploss_max_reasonable = DecimalParameter(
        -0.30, -0.15, default=-0.20, space="sell", optimize=True
    )

    # === Hyperopt Parameters ===
    dominance_threshold = IntParameter(1, 10, default=3, space="buy", optimize=True)
    tightness_factor = DecimalParameter(
        0.5, 2.0, default=1.0, space="buy", optimize=True
    )
    long_rsi_threshold = IntParameter(50, 65, default=50, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=35, space="sell", optimize=True)

    # Leverage parameters commented out - not applicable for SPOT trading
    # leverage_window_size = IntParameter(20, 100, default=70, space="buy", optimize=True, load=True)
    # leverage_base = DecimalParameter(5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=True, load=True)
    # leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    # leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    # leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True, load=True)
    # leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True, load=True)
    # leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy", optimize=True, load=True)
    # leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True, load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(
        3, 15, default=8, space="buy", optimize=True, load=True
    )  # War 5
    indicator_mml_window = IntParameter(
        50, 200, default=50, space="buy", optimize=True, load=True
    )  # War 50
    indicator_rolling_window_threshold = IntParameter(
        20, 100, default=50, space="buy", optimize=True, load=True
    )  # War 20
    indicator_rolling_check_window = IntParameter(
        5, 20, default=10, space="buy", optimize=True, load=True
    )  # War 5

    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(
        0.3, 0.6, default=0.45, space="buy", optimize=True
    )

    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(
        default=True, space="buy", optimize=True
    )
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)

    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(
        24, 168, default=48, space="buy", optimize=True
    )  # hours

    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(
        default=False, space="buy", optimize=True
    )  # Optional
    fear_greed_extreme_threshold = IntParameter(
        20, 30, default=25, space="buy", optimize=True
    )
    fear_greed_greed_threshold = IntParameter(
        70, 80, default=75, space="buy", optimize=True
    )
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(
        0.01, 0.05, default=0.02, space="buy", optimize=True
    )
    momentum_confirmation_candles = IntParameter(
        1, 5, default=2, space="buy", optimize=True
    )

    # Dynamic exit based on entry quality
    dynamic_exit_enabled = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )
    exit_on_confluence_loss = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )
    exit_on_structure_break = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )

    # Profit target multipliers based on entry type
    high_quality_profit_multiplier = DecimalParameter(
        1.2, 3.0, default=2.0, space="sell", optimize=True, load=True
    )
    medium_quality_profit_multiplier = DecimalParameter(
        1.0, 2.5, default=1.5, space="sell", optimize=True, load=True
    )
    backup_profit_multiplier = DecimalParameter(
        0.8, 2.0, default=1.2, space="sell", optimize=True, load=True
    )

    # Advanced exit thresholds
    volume_decline_exit_threshold = DecimalParameter(
        0.3, 0.8, default=0.5, space="sell", optimize=True, load=True
    )
    momentum_decline_exit_threshold = IntParameter(
        1, 4, default=2, space="sell", optimize=True, load=True
    )
    structure_deterioration_threshold = DecimalParameter(
        -3.0, 0.0, default=-1.5, space="sell", optimize=True, load=True
    )

    # RSI exit levels
    rsi_overbought_exit = IntParameter(
        70, 85, default=75, space="sell", optimize=True, load=True
    )
    rsi_divergence_exit_enabled = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )

    # Trailing stop improvements
    use_advanced_trailing = BooleanParameter(
        default=False, space="sell", optimize=False, load=True
    )
    trailing_stop_positive_offset_high_quality = DecimalParameter(
        0.02, 0.08, default=0.04, space="sell", optimize=True, load=True
    )
    trailing_stop_positive_offset_medium_quality = DecimalParameter(
        0.015, 0.06, default=0.03, space="sell", optimize=True, load=True
    )

    # === NEUE ADVANCED PARAMETERS ===
    # Confluence Analysis
    confluence_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    confluence_threshold = DecimalParameter(
        2.0, 4.0, default=2.5, space="buy", optimize=True, load=True
    )  # War 3.0

    # Volume Analysis
    volume_analysis_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    volume_strength_threshold = DecimalParameter(
        1.1, 2.0, default=1.3, space="buy", optimize=True, load=True
    )
    volume_pressure_threshold = IntParameter(
        1, 3, default=1, space="buy", optimize=True, load=True
    )  # War 2

    # Momentum Analysis
    momentum_analysis_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    momentum_quality_threshold = IntParameter(
        2, 4, default=2, space="buy", optimize=True, load=True
    )  # War 3

    # Market Structure Analysis
    structure_analysis_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    structure_score_threshold = DecimalParameter(
        -2.0, 5.0, default=0.5, space="buy", optimize=True, load=True
    )

    # Ultimate Score
    ultimate_score_threshold = DecimalParameter(
        0.5, 3.0, default=1.5, space="buy", optimize=True, load=True
    )

    # Advanced Entry Filters
    require_volume_confirmation = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    require_momentum_confirmation = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    require_structure_confirmation = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )

    # G. ROI con timed exits para swing corto (24-48h)
    # Evitar "muertes por mil cortes" con salidas temporales definidas
    minimal_roi = {
        "0": 0.06,
        "5": 0.055,
        "10": 0.04,
        "20": 0.03,
        "40": 0.025,
        "80": 0.02,
        "160": 0.015,
        "320": 0.01,
        "1440": 0.005,  # 24h - salida mínima
        "2880": 0,  # 48h - salida forzada
    }

    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {
            # Trend indicators
            "ema50": {"color": "gray", "type": "line"},
            # Support/Resistance
            "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
            "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
        },
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "maxima": {"color": "#a29db9", "type": "scatter"},
                "minima": {"color": "#aac7fc", "type": "scatter"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},  # 50% MML
                "[6/8]P": {"color": "green", "type": "line"},  # 75% MML
                "[2/8]P": {"color": "orange", "type": "line"},  # 25% MML
                "[8/8]P": {"color": "red", "type": "line"},  # 100% MML
                "[0/8]P": {"color": "red", "type": "line"},  # 0% MML
                "mmlextreme_oscillator": {"color": "purple", "type": "line"},
            },
            "rsi_analysis": {
                "rsi": {"color": "purple", "type": "line"},
                "rsi_divergence_bull": {"color": "green", "type": "scatter"},
                "rsi_divergence_bear": {"color": "red", "type": "scatter"},
            },
            "confluence_analysis": {
                "confluence_score": {"color": "gold", "type": "line"},
                "near_support": {"color": "green", "type": "scatter"},
                "near_resistance": {"color": "red", "type": "scatter"},
                "near_mml": {"color": "blue", "type": "line"},
                "volume_spike": {"color": "orange", "type": "scatter"},
            },
            "volume_analysis": {
                "volume_strength": {"color": "cyan", "type": "line"},
                "volume_pressure": {"color": "magenta", "type": "line"},
                "buying_pressure": {"color": "green", "type": "line"},
                "selling_pressure": {"color": "red", "type": "line"},
                "money_flow_index": {"color": "yellow", "type": "line"},
            },
            "momentum_analysis": {
                "momentum_quality": {"color": "brown", "type": "line"},
                "momentum_acceleration": {"color": "pink", "type": "line"},
                "momentum_consistency": {"color": "lime", "type": "line"},
                "momentum_oscillator": {"color": "navy", "type": "line"},
            },
            "structure_analysis": {
                "structure_score": {"color": "teal", "type": "line"},
                "bullish_structure": {"color": "green", "type": "line"},
                "bearish_structure": {"color": "red", "type": "line"},
                "structure_break_up": {"color": "lime", "type": "scatter"},
                "structure_break_down": {"color": "crimson", "type": "scatter"},
            },
            "trend_strength": {
                "trend_strength": {"color": "indigo", "type": "line"},
                "trend_strength_5": {"color": "lightblue", "type": "line"},
                "trend_strength_10": {"color": "mediumblue", "type": "line"},
                "trend_strength_20": {"color": "darkblue", "type": "line"},
            },
            "ultimate_signals": {
                "ultimate_score": {"color": "gold", "type": "line"},
                "signal_strength": {"color": "silver", "type": "line"},
                "high_quality_setup": {"color": "lime", "type": "scatter"},
                "entry_type": {"color": "white", "type": "line"},
            },
            "market_conditions": {
                "strong_uptrend": {"color": "green", "type": "scatter"},
                "strong_downtrend": {"color": "red", "type": "scatter"},
                "ranging": {"color": "yellow", "type": "scatter"},
                "strong_up_momentum": {"color": "lime", "type": "scatter"},
                "strong_down_momentum": {"color": "crimson", "type": "scatter"},
            },
            "di_analysis": {
                "DI_values": {"color": "orange", "type": "line"},
                "DI_catch": {"color": "red", "type": "scatter"},
                "plus_di": {"color": "green", "type": "line"},
                "minus_di": {"color": "red", "type": "line"},
            },
        },
    }

    # Helper method to check if we have an active position in the opposite direction
    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @staticmethod
    def _calculate_mml_core(
        mn: float, finalH: float, mx: float, finalL: float, mml_c1: float, mml_c2: float
    ) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if (
            dmml_calc == 0
            or np.isinf(dmml_calc)
            or np.isnan(dmml_calc)
            or finalH == finalL
        ):
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels_optimized(
        self, df: pd.DataFrame, window_size: int
    ) -> Dict[str, pd.Series]:
        """
        OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
        """
        murrey_levels_data: Dict[str, list] = {
            key: [np.nan] * len(df) for key in MML_LEVEL_NAMES
        }
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value

        calculation_step = 5

        for i in range(0, len(df), calculation_step):
            if i < window_size:
                continue

            # Use data up to the previous candle for the rolling window
            window_end = i - 1
            window_start = window_end - window_size + 1
            if window_start < 0:
                window_start = 0

            window_data = df.iloc[window_start:window_end]
            mn_period = window_data["low"].min()
            mx_period = window_data["high"].max()
            current_close = (
                df["close"].iloc[window_end] if window_end > 0 else df["close"].iloc[0]
            )

            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][window_end] = current_close
                continue

            levels = self._calculate_mml_core(
                mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2
            )

            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][window_end] = levels.get(key, current_close)

        # Interpolate using only past data up to each point
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index)
            # Interpolate forward only up to the current point, avoiding future data
            series = (
                series.expanding().mean().ffill()
            )  # Use expanding mean as a safe alternative
            murrey_levels_data[key] = series.tolist()

        return {
            key: pd.Series(data, index=df.index)
            for key, data in murrey_levels_data.items()
        }

    def calculate_synthetic_market_breadth(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate synthetic market breadth using technical indicators
        Simulates market sentiment based on multiple factors
        """
        try:
            # RSI component (30% weight)
            rsi_component = (dataframe["rsi"] - 50) / 50  # Normalize to -1 to 1

            # Volume component (25% weight)
            volume_ma = dataframe["volume"].rolling(20).mean()
            volume_component = (dataframe["volume"] / volume_ma - 1).clip(-1, 1)

            # Momentum component (25% weight)
            momentum_3 = dataframe["close"].pct_change(3)
            momentum_component = np.tanh(momentum_3 * 100)  # Smooth normalization

            # Volatility component (20% weight) - inverted (lower vol = higher breadth)
            atr_normalized = dataframe["atr"] / dataframe["close"]
            atr_ma = atr_normalized.rolling(20).mean()
            volatility_component = -(atr_normalized / atr_ma - 1).clip(-1, 1)

            # Combine components with weights
            synthetic_breadth = (
                rsi_component * 0.30
                + volume_component * 0.25
                + momentum_component * 0.25
                + volatility_component * 0.20
            )

            # Normalize to 0-1 range (market breadth percentage)
            synthetic_breadth = (synthetic_breadth + 1) / 2

            # Smooth with rolling average
            synthetic_breadth = synthetic_breadth.rolling(3).mean()

            return synthetic_breadth.fillna(0.5)

        except Exception as e:
            logger.warning(f"Synthetic market breadth calculation failed: {e}")
            return pd.Series(0.5, index=dataframe.index)

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends
        """

        # Linear regression slope
        def calc_slope(series, period=10):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.iloc[-period:].values
            if np.isnan(y).any() or np.isinf(y).any():
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope

        # Calculate trend strength using multiple timeframes
        df["slope_5"] = (
            df["close"].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        )
        df["slope_10"] = (
            df["close"].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        )
        df["slope_20"] = (
            df["close"].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)
        )

        df["trend_strength_5"] = df["slope_5"] / df["close"] * 100
        df["trend_strength_10"] = df["slope_10"] / df["close"] * 100
        df["trend_strength_20"] = df["slope_20"] / df["close"] * 100

        # Combined trend strength
        df["trend_strength"] = (
            df["trend_strength_5"] + df["trend_strength_10"] + df["trend_strength_20"]
        ) / 3

        # Trend classification
        strong_threshold = float(self.strong_threshold.value)  # Use parametrized value
        df["strong_uptrend"] = df["trend_strength"] > strong_threshold
        df["strong_downtrend"] = df["trend_strength"] < -strong_threshold
        df["ranging"] = df["trend_strength"].abs() < (strong_threshold * 0.5)

        return df

    @property
    def protections(self):
        """
        Protections moved from config (deprecated in Freqtrade 2024.10+)
        Use --enable-protections flag for backtesting
        """
        prot = []

        # G. CooldownPeriod - evitar re-entrada ansiosa
        prot.append(
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value,  # 3 velas por defecto (subido de 1)
            }
        )

        # MaxDrawdown - stop trading on heavy drawdowns
        prot.append(
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,  # ~2 days in 1h
                "trade_limit": 20,
                "stop_duration_candles": 12,  # ~12h
                "max_allowed_drawdown": 0.10,  # 10%
                "only_per_pair": False,
            }
        )

        # StoplossGuard - stop after multiple stoplosses
        if self.use_stop_protection.value:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 24,  # 1 day in 1h
                    "trade_limit": 4,
                    "stop_duration_candles": self.stop_duration.value,  # 6 by default
                    "only_per_pair": False,
                }
            )

        # G. LowProfitPairs - congelar pares tibios antes
        prot.append(
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 360,  # 15 days in 1h
                "trade_limit": 8,  # need 8 trades to evaluate
                "stop_duration_candles": 12,  # block for 12h
                "required_profit": 0.02,  # 2% minimum profit (subido de 1%)
                "only_per_pair": False,
            }
        )

        return prot

    def informative_pairs(self):
        """
        V4: Define additional pairs and timeframes for multi-timeframe analysis
        Fixed pairs for BTC/ETH to avoid WebSocket errors
        """
        pairs = []

        # V4: Always include BTC and ETH as fixed references (avoid WebSocket unsub errors)
        # These are market leaders and provide stable correlation signals
        reference_pairs = [
            ("BTC/USDT", "1h"),
            ("BTC/USDT", "4h"),
            ("BTC/USDT", "1d"),
            ("ETH/USDT", "1h"),
            ("ETH/USDT", "4h"),
            ("ETH/USDT", "1d"),
        ]
        pairs.extend(reference_pairs)

        # Define timeframes for multi-timeframe analysis
        informative_timeframes = ["4h", "8h", "1d"]

        # Add current pair with different timeframes
        for tf in informative_timeframes:
            pairs.extend([(pair, tf) for pair in self.dp.current_whitelist()])

        # Note: Removed dynamic BTC/ETH fetching to prevent WebSocket unsubscribe errors
        if self.timeframe:
            pairs.append(("BTC/USDT", self.timeframe))
            pairs.append(("ETH/USDT", self.timeframe))
            pairs.append(("BNB/USDT", self.timeframe))

        # Add major market indicators with higher timeframes for trend analysis
        for tf in informative_timeframes:
            pairs.append(("BTC/USDT", tf))
            pairs.append(("ETH/USDT", tf))

        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

        return unique_pairs

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )

        if dataframe.empty or "atr" not in dataframe.columns:
            return self.stoploss  # Use strategy stoploss (-0.15) as fallback

        atr = dataframe["atr"].iat[-1]
        if pd.isna(atr) or atr <= 0:
            return self.stoploss  # Fallback to -0.15

        atr_percent = atr / current_rate

        # Profit-based multiplier adjustment
        if current_profit > 0.15:
            multiplier = 1.0
        elif current_profit > 0.08:
            multiplier = 1.2
        elif current_profit > 0.03:
            multiplier = 1.4
        else:
            multiplier = 1.6

        calculated_stoploss = -(
            atr_percent * multiplier * self.atr_stoploss_multiplier.value
        )

        # Initialize trailing_offset
        trailing_offset = 0.0

        # Enhanced trailing logic with multiple profit levels
        if current_profit > 0.01:  # Start trailing at 1% profit instead of 3%
            if current_profit > self.trailing_stop_positive_offset:  # 0.03 (3% profit)
                # Full trailing at 3%+ profit
                trailing_offset = max(
                    0, current_profit - self.trailing_stop_positive
                )  # Trail 0.5% below peak
            elif current_profit > 0.02:  # 2% profit
                # Moderate trailing at 2-3% profit
                trailing_offset = max(0, current_profit - 0.01)  # Trail 1% below peak
            else:  # 1-2% profit
                # Minimal trailing at 1-2% profit
                trailing_offset = max(
                    0, current_profit - 0.015
                )  # Trail 1.5% below peak

            # Apply trailing adjustment to calculated stoploss
            if trailing_offset > 0:
                calculated_stoploss = max(
                    calculated_stoploss, -trailing_offset
                )  # Trail up in profit correctly

        final_stoploss = max(
            min(calculated_stoploss, self.atr_stoploss_ceiling.value),
            self.atr_stoploss_maximum.value,
        )

        logger.info(
            f"{pair} Custom SL: {final_stoploss:.3f} | ATR: {atr:.6f} | "
            f"Profit: {current_profit:.3f} | Trailing: {trailing_offset:.3f}"
        )
        return final_stoploss

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        """
        SPOT trading doesn't use leverage. Return 1.0 always.
        This method is kept for compatibility but not used in SPOT mode.
        """
        return 1.0

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        ULTIMATE indicator calculations with advanced market analysis
        """
        # === V4 FIX: Initialize feature cache if not exists ===
        if not hasattr(self, "feature_cache"):
            self.feature_cache = {}
            self.last_cache_update = {}
            self.cache_expiry_candles = 5  # Cache valid for 5 candles

        # === ML ASSET STARTUP CHECK ===
        pair = metadata["pair"]
        if hasattr(self, "predictive_engine") and self.predictive_engine is not None:
            # Mark pair as trained if assets exist (startup optimization)
            self.predictive_engine.mark_trained_if_assets(pair)

        # === EXTERNAL DATA INTEGRATION ===
        try:
            # Add BTC data for correlation analysis using informative pairs
            if metadata["pair"] != "BTC/USDT":
                btc_info = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
                if not btc_info.empty and len(btc_info) >= len(dataframe):
                    # Take only the last N rows to match our dataframe length
                    btc_close_data = (
                        btc_info["close"].tail(len(dataframe)).reset_index(drop=True)
                    )
                    dataframe["btc_close"] = btc_close_data.values
                    logger.info(
                        f"{metadata['pair']} BTC correlation data added successfully"
                    )
                else:
                    # Fallback: use current pair data
                    dataframe["btc_close"] = dataframe["close"]
                    logger.warning(
                        f"{metadata['pair']} BTC data unavailable, using pair data as fallback"
                    )
            else:
                # For BTC pairs, use own data
                dataframe["btc_close"] = dataframe["close"]

            # Add ETH data for relative strength calculations
            if metadata["pair"] != "ETH/USDT":
                eth_info = self.dp.get_pair_dataframe("ETH/USDT", self.timeframe)
                if not eth_info.empty and len(eth_info) >= len(dataframe):
                    eth_close_data = (
                        eth_info["close"].tail(len(dataframe)).reset_index(drop=True)
                    )
                    dataframe["eth_close"] = eth_close_data.values
                    logger.info(
                        f"{metadata['pair']} ETH correlation data added successfully"
                    )
                else:
                    dataframe["eth_close"] = dataframe["close"]  # fallback
                    logger.warning(
                        f"{metadata['pair']} ETH data unavailable, using pair data as fallback"
                    )
            else:
                # For ETH pairs, use own data
                dataframe["eth_close"] = dataframe["close"]

        except Exception as e:
            logger.warning(f"{metadata['pair']} External data integration failed: {e}")
            dataframe["btc_close"] = dataframe["close"]  # Safe fallback
            dataframe["eth_close"] = dataframe["close"]  # Safe fallback

        # === MULTI-TIMEFRAME INTEGRATION ===
        try:
            from freqtrade.strategy import merge_informative_pair

            # 4h timeframe indicators for trend confirmation
            inf_4h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="4h")
            if not inf_4h.empty:
                inf_4h["rsi_4h"] = ta.RSI(inf_4h["close"], timeperiod=14)
                inf_4h["ema50_4h"] = ta.EMA(inf_4h["close"], timeperiod=50)
                inf_4h["atr_4h"] = ta.ATR(
                    inf_4h["high"], inf_4h["low"], inf_4h["close"], timeperiod=14
                )
                dataframe = merge_informative_pair(
                    dataframe, inf_4h, self.timeframe, "4h", ffill=True
                )

            # 8h timeframe for medium-term trend
            inf_8h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="8h")
            if not inf_8h.empty:
                inf_8h["rsi_8h"] = ta.RSI(inf_8h["close"], timeperiod=14)
                inf_8h["ema50_8h"] = ta.EMA(inf_8h["close"], timeperiod=50)
                inf_8h["atr_8h"] = ta.ATR(
                    inf_8h["high"], inf_8h["low"], inf_8h["close"], timeperiod=14
                )
                dataframe = merge_informative_pair(
                    dataframe, inf_8h, self.timeframe, "8h", ffill=True
                )

            # 1d timeframe for major trend
            inf_1d = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1d")
            if not inf_1d.empty:
                inf_1d["rsi_1d"] = ta.RSI(inf_1d["close"], timeperiod=14)
                inf_1d["ema20_1d"] = ta.EMA(inf_1d["close"], timeperiod=20)
                dataframe = merge_informative_pair(
                    dataframe, inf_1d, self.timeframe, "1d", ffill=True
                )

            # BTC 4h for market regime (avoid column collisions)
            if metadata["pair"] != "BTC/USDT":
                btc_4h = self.dp.get_pair_dataframe("BTC/USDT", "4h")
                if not btc_4h.empty:
                    # Keep only date and close, rename close to avoid collision
                    btc_4h_clean = btc_4h[["date", "close", "high", "low"]].copy()
                    btc_4h_clean = btc_4h_clean.rename(
                        columns={
                            "close": "btc_close",
                            "high": "btc_high",
                            "low": "btc_low",
                        }
                    )
                    # Calculate indicators using renamed columns
                    btc_4h_clean["btc_rsi_4h"] = ta.RSI(
                        btc_4h_clean["btc_close"], timeperiod=14
                    )
                    btc_4h_clean["btc_ema50_4h"] = ta.EMA(
                        btc_4h_clean["btc_close"], timeperiod=50
                    )
                    # Merge with proper suffix handling
                    dataframe = merge_informative_pair(
                        dataframe, btc_4h_clean, self.timeframe, "4h", ffill=True
                    )

        except Exception as e:
            logger.warning(
                f"{metadata['pair']} Multi-timeframe integration failed: {e}"
            )
            # Set default values for missing columns
            for col in [
                "rsi_4h",
                "ema50_4h",
                "atr_4h",
                "rsi_8h",
                "ema50_8h",
                "atr_8h",
                "rsi_1d",
                "ema20_1d",
                "btc_rsi_4h",
                "btc_ema50_4h",
            ]:
                if col not in dataframe.columns:
                    dataframe[col] = 50 if "rsi" in col else dataframe["close"]

        # === STANDARD INDICATORS ===
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema100"] = ta.EMA(
            dataframe["close"], timeperiod=100
        )  # Neu hinzufÃ¼gen
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=10
        )

        # === SYNTHETIC MARKET BREADTH CALCULATION ===
        try:
            # Calculate synthetic market breadth using multiple indicators
            # (after RSI and ATR are available)
            dataframe["market_breadth"] = self.calculate_synthetic_market_breadth(
                dataframe
            )
            logger.info(f"{metadata['pair']} Synthetic market breadth calculated")
        except Exception as e:
            logger.warning(f"{metadata['pair']} Market breadth calculation failed: {e}")
            dataframe["market_breadth"] = 0.5  # Neutral fallback
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # === EXTREMA DETECTION ===
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"]
            == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"]
            == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # === HEIKIN-ASHI ===
        dataframe["ha_close"] = (
            dataframe["open"]
            + dataframe["high"]
            + dataframe["low"]
            + dataframe["close"]
        ) / 4

        # === ROLLING EXTREMA ===
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(
            dataframe, self.h2.value
        )
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(
            dataframe, self.h1.value
        )
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(
            dataframe, self.h0.value
        )
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(
            dataframe, self.cp.value
        )

        # === MURREY MATH LEVELS ===
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(
            dataframe, window_size=mml_window
        )

        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # === MML OSCILLATOR ===
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")

        if (
            mml_4_8 is not None
            and mml_plus_3_8 is not None
            and mml_minus_3_8 is not None
        ):
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * (
                (dataframe["close"] - mml_4_8) / osc_denominator
            )
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # === DI CATCH ===
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1
        )

        # === ROLLING THRESHOLDS ===
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = (
            dataframe["close"]
            .rolling(window=rolling_window_threshold, min_periods=1)
            .min()
        )
        dataframe["maxima_sort_threshold"] = (
            dataframe["close"]
            .rolling(window=rolling_window_threshold, min_periods=1)
            .max()
        )

        # === EXTREMA CHECKS ===
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"]
            .rolling(window=rolling_check_window, min_periods=1)
            .sum()
            == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"]
            .rolling(window=rolling_check_window, min_periods=1)
            .sum()
            == 0
        ).astype(int)

        # === VOLATILITY INDICATORS ===
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = (
            dataframe["volatility_range"].rolling(window=50).mean()
        )
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=50).mean()

        # === TREND STRENGTH INDICATORS ===
        # Use enhanced Wavelet+FFT method with fallback
        try:
            # Advanced wavelet & FFT method
            # V4 FIX: Pass pair and cache parameters for proper functionality
            dataframe = calculate_advanced_trend_strength_with_wavelets(
                dataframe,
                float(self.strong_threshold.value),
                pair=metadata.get("pair", "unknown"),
                feature_cache=self.feature_cache,
                last_cache_update=self.last_cache_update,
            )

            # Use advanced trend strength as primary
            dataframe["trend_strength"] = dataframe["trend_strength_cycle_adjusted"]
            dataframe["strong_uptrend"] = dataframe["strong_uptrend_advanced"]
            dataframe["strong_downtrend"] = dataframe["strong_downtrend_advanced"]
            dataframe["ranging"] = dataframe["ranging_advanced"]

            logger.info(f"{metadata['pair']} Using advanced Wavelet+FFT trend analysis")

        except Exception as e:
            # Fallback to original enhanced method if advanced fails
            logger.warning(
                f"{metadata['pair']} Wavelet/FFT analysis failed: {e}. "
                "Using enhanced method."
            )

            def calc_slope(series, period):
                """Enhanced slope calculation as fallback"""
                if len(series) < period:
                    return 0
                y = series.values[-period:]
                if np.isnan(y).any() or np.isinf(y).any():
                    return 0
                if np.all(y == y[0]):
                    return 0
                x = np.linspace(0, period - 1, period)
                try:
                    coefficients = np.polyfit(x, y, 1)
                    slope = coefficients[0]
                    if np.isnan(slope) or np.isinf(slope):
                        return 0
                    max_reasonable_slope = np.std(y) / period
                    if abs(slope) > max_reasonable_slope * 10:
                        return np.sign(slope) * max_reasonable_slope * 10
                    return slope
                except Exception:
                    try:
                        simple_slope = (y[-1] - y[0]) / (period - 1)
                        return (
                            simple_slope
                            if not (np.isnan(simple_slope) or np.isinf(simple_slope))
                            else 0
                        )
                    except Exception:
                        return 0

            # Original slope calculations
            dataframe["slope_5"] = (
                dataframe["close"]
                .rolling(5)
                .apply(lambda x: calc_slope(x, 5), raw=False)
            )
            dataframe["slope_10"] = (
                dataframe["close"]
                .rolling(10)
                .apply(lambda x: calc_slope(x, 10), raw=False)
            )
            dataframe["slope_20"] = (
                dataframe["close"]
                .rolling(20)
                .apply(lambda x: calc_slope(x, 20), raw=False)
            )

            dataframe["trend_strength_5"] = (
                dataframe["slope_5"] / dataframe["close"] * 100
            )
            dataframe["trend_strength_10"] = (
                dataframe["slope_10"] / dataframe["close"] * 100
            )
            dataframe["trend_strength_20"] = (
                dataframe["slope_20"] / dataframe["close"] * 100
            )

            dataframe["trend_strength"] = (
                dataframe["trend_strength_5"]
                + dataframe["trend_strength_10"]
                + dataframe["trend_strength_20"]
            ) / 3

            strong_threshold = float(
                self.strong_threshold.value
            )  # Use parametrized value
            dataframe["strong_uptrend"] = dataframe["trend_strength"] > strong_threshold
            dataframe["strong_downtrend"] = (
                dataframe["trend_strength"] < -strong_threshold
            )
            dataframe["ranging"] = dataframe["trend_strength"].abs() < (
                strong_threshold * 0.5
            )

        # === MOMENTUM INDICATORS ===
        dataframe["price_momentum"] = dataframe["close"].pct_change(3)
        dataframe["momentum_increasing"] = dataframe["price_momentum"] > dataframe[
            "price_momentum"
        ].shift(1)
        dataframe["momentum_decreasing"] = dataframe["price_momentum"] < dataframe[
            "price_momentum"
        ].shift(1)

        dataframe["volume_momentum"] = (
            dataframe["volume"].rolling(3).mean()
            / dataframe["volume"].rolling(20).mean()
        )

        dataframe["rsi_divergence_bull"] = (
            dataframe["close"] < dataframe["close"].shift(5)
        ) & (dataframe["rsi"] > dataframe["rsi"].shift(5))
        dataframe["rsi_divergence_bear"] = (
            dataframe["close"] > dataframe["close"].shift(5)
        ) & (dataframe["rsi"] < dataframe["rsi"].shift(5))

        # === CANDLE PATTERNS ===
        dataframe["green_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["red_candle"] = dataframe["close"] < dataframe["open"]
        dataframe["consecutive_green"] = dataframe["green_candle"].rolling(3).sum()
        dataframe["consecutive_red"] = dataframe["red_candle"].rolling(3).sum()

        # Define strong_threshold for momentum calculations
        strong_threshold = float(self.strong_threshold.value)  # Use parametrized value

        dataframe["strong_up_momentum"] = (
            (dataframe["consecutive_green"] >= 3)
            & (dataframe["volume"] > dataframe["avg_volume"])
            & (dataframe["trend_strength"] > strong_threshold)
        )
        dataframe["strong_down_momentum"] = (
            (dataframe["consecutive_red"] >= 3)
            & (dataframe["volume"] > dataframe["avg_volume"])
            & (dataframe["trend_strength"] < -strong_threshold)
        )

        # === ADVANCED ANALYSIS MODULES ===

        # 1. CONFLUENCE ANALYSIS
        if self.confluence_enabled.value:
            dataframe = calculate_confluence_score(dataframe)
        else:
            dataframe["confluence_score"] = 0

        # 2. SMART VOLUME ANALYSIS
        if self.volume_analysis_enabled.value:
            dataframe = calculate_smart_volume(dataframe)
        else:
            dataframe["volume_pressure"] = 0
            dataframe["volume_strength"] = 1.0
            dataframe["money_flow_index"] = 50

        # 3. ADVANCED MOMENTUM
        if self.momentum_analysis_enabled.value:
            dataframe = calculate_advanced_momentum(dataframe)
        else:
            dataframe["momentum_quality"] = 0
            dataframe["momentum_acceleration"] = 0

        # 4. MARKET STRUCTURE
        if self.structure_analysis_enabled.value:
            dataframe = calculate_market_structure(dataframe)
        else:
            dataframe["structure_score"] = 0
            dataframe["structure_break_up"] = 0

        # 5. ADVANCED ENTRY SIGNALS
        dataframe = calculate_advanced_entry_signals(dataframe)

        # === ULTIMATE MARKET SCORE ===
        dataframe["ultimate_score"] = (
            dataframe["confluence_score"] * 0.25  # 25% confluence
            + dataframe["volume_pressure"] * 0.2  # 20% volume pressure
            + dataframe["momentum_quality"] * 0.2  # 20% momentum quality
            + (dataframe["structure_score"] / 5) * 0.15  # 15% structure (normalized)
            + (dataframe["signal_strength"] / 10) * 0.2  # 20% signal strength
        )

        # Normalize ultimate score to 0-1 range
        dataframe["ultimate_score"] = dataframe["ultimate_score"].clip(0, 5) / 5

        # === FINAL QUALITY CHECKS ===
        dataframe["high_quality_setup"] = (
            (dataframe["ultimate_score"] > self.ultimate_score_threshold.value)
            & (dataframe["signal_strength"] >= 5)
            & (dataframe["volume_strength"] > 1.1)
            & (dataframe["rsi"] > 30)
            & (dataframe["rsi"] < 70)
        ).astype(int)

        # === DEBUG INFO ===
        if metadata["pair"] in ["BTC/USDT", "ETH/USDT"]:  # Only log for major pairs
            latest_score = dataframe["ultimate_score"].iloc[-1]
            latest_signal = dataframe["signal_strength"].iloc[-1]
            logger.info(
                f"{metadata['pair']} Ultimate Score: {latest_score:.3f}, Signal Strength: {latest_signal}"
            )

        # ===========================================
        # REGIME CHANGE DETECTION
        # ===========================================

        if self.regime_change_enabled.value:

            # ===========================================
            # FLASH MOVE DETECTION
            # ===========================================

            flash_candles = self.flash_move_candles.value
            flash_threshold = self.flash_move_threshold.value

            # Schnelle Preisbewegungen
            dataframe["price_change_fast"] = dataframe["close"].pct_change(
                flash_candles
            )
            dataframe["flash_pump"] = dataframe["price_change_fast"] > flash_threshold
            dataframe["flash_dump"] = dataframe["price_change_fast"] < -flash_threshold
            dataframe["flash_move"] = dataframe["flash_pump"] | dataframe["flash_dump"]

            # ===========================================
            # VOLUME SPIKE DETECTION
            # ===========================================

            volume_ma20 = dataframe["volume"].rolling(20).mean()
            volume_multiplier = self.volume_spike_multiplier.value
            dataframe["volume_spike"] = dataframe["volume"] > (
                volume_ma20 * volume_multiplier
            )

            # Volume + Bewegung kombiniert
            dataframe["volume_pump"] = (
                dataframe["volume_spike"] & dataframe["flash_pump"]
            )
            dataframe["volume_dump"] = (
                dataframe["volume_spike"] & dataframe["flash_dump"]
            )

            # ===========================================
            # MARKET SENTIMENT DETECTION
            # ===========================================

            # Market Breadth Change
            if "market_breadth" in dataframe.columns:
                dataframe["market_breadth_change"] = dataframe["market_breadth"].diff(3)
                sentiment_threshold = self.sentiment_shift_threshold.value
                dataframe["sentiment_shift_bull"] = (
                    dataframe["market_breadth_change"] > sentiment_threshold
                )
                dataframe["sentiment_shift_bear"] = (
                    dataframe["market_breadth_change"] < -sentiment_threshold
                )
            else:
                dataframe["sentiment_shift_bull"] = False
                dataframe["sentiment_shift_bear"] = False

            # ===========================================
            # BTC CORRELATION MONITORING
            # ===========================================

            # BTC Flash Moves
            if "btc_close" in dataframe.columns:
                dataframe["btc_change_fast"] = dataframe["btc_close"].pct_change(
                    flash_candles
                )
                dataframe["btc_flash_pump"] = (
                    dataframe["btc_change_fast"] > flash_threshold
                )
                dataframe["btc_flash_dump"] = (
                    dataframe["btc_change_fast"] < -flash_threshold
                )

                # Correlation Break
                pair_movement = dataframe["price_change_fast"].abs()
                btc_movement = dataframe["btc_change_fast"].abs()
                dataframe["correlation_break"] = (btc_movement > flash_threshold) & (
                    pair_movement < flash_threshold * 0.4
                )
            else:
                dataframe["btc_flash_pump"] = False
                dataframe["btc_flash_dump"] = False
                dataframe["correlation_break"] = False

            # ===========================================
            # REGIME CHANGE SCORE
            # ===========================================

            regime_signals = [
                "flash_move",
                "volume_spike",
                "sentiment_shift_bull",
                "sentiment_shift_bear",
                "btc_flash_pump",
                "btc_flash_dump",
                "correlation_break",
            ]

            dataframe["regime_change_score"] = 0
            for signal in regime_signals:
                if signal in dataframe.columns:
                    dataframe["regime_change_score"] += dataframe[signal].astype(int)

            # Normalisiere auf 0-1
            max_signals = len(regime_signals)
            dataframe["regime_change_intensity"] = (
                dataframe["regime_change_score"] / max_signals
            )

            # Alert Level
            sensitivity = self.regime_change_sensitivity.value
            dataframe["regime_alert"] = (
                dataframe["regime_change_intensity"] >= sensitivity
            )

        else:
            # Falls Regime Detection deaktiviert
            dataframe["flash_pump"] = False
            dataframe["flash_dump"] = False
            dataframe["volume_pump"] = False
            dataframe["volume_dump"] = False
            dataframe["regime_alert"] = False
            dataframe["regime_change_intensity"] = 0.0

        # === ADVANCED PREDICTIVE ANALYSIS ===
        try:
            # V4 FIX: Define t_start for timing telemetry
            t_start = time.time()
            pair = metadata.get("pair", "UNKNOWN")
            dataframe = calculate_advanced_predictive_signals(
                dataframe, pair, float(self.strong_threshold.value)
            )
            dataframe = calculate_quantum_momentum_analysis(dataframe)
            dataframe = calculate_neural_pattern_recognition(dataframe)

            # V4: Enhanced telemetry
            t_elapsed = time.time() - t_start
            if "thr_dyn" in dataframe.columns and len(dataframe) > 100:
                thr_stats = dataframe["thr_dyn"].tail(100)
                ev_stats = dataframe.get("expected_value", pd.Series([0]))
                logger.info(
                    f"[ML-V4] {pair} Analysis completed in {t_elapsed:.1f}s | "
                    f"thr_dyn=[{thr_stats.min():.3f},{thr_stats.mean():.3f},{thr_stats.max():.3f}] | "
                    f"EV_mean={ev_stats.tail(100).mean():.4f} | "
                    f"Models={len(predictive_engine.models.get(pair, {}))} | "
                    f"Training={'YES' if predictive_engine.training_in_progress.get(pair, False) else 'NO'}"
                )
            else:
                logger.info(
                    f"[ML-V4] {pair} Advanced predictive analysis completed in {t_elapsed:.1f}s"
                )
        except Exception as e:
            logger.warning(f"Advanced predictive analysis failed: {e}")
            dataframe["ml_entry_probability"] = 0.5
            dataframe["ml_enhanced_score"] = dataframe.get("ultimate_score", 0.5)
            dataframe["ml_high_confidence"] = 0
            dataframe["ml_ultra_confidence"] = 0
            dataframe["quantum_momentum_coherence"] = 0.5
            dataframe["momentum_entanglement"] = 0.5
            dataframe["quantum_tunnel_up_prob"] = 0.5
            dataframe["neural_pattern_score"] = 0.5

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        AI-CENTRIC ENTRY LOGIC - Simplified and functional
        """

        # Debug log to verify populate_entry_trend is being called
        pair = metadata.get("pair", "UNKNOWN")
        logger.info(f"ENTRY_TREND: Processing {pair} with {len(dataframe)} candles")

        # ===========================================
        # INITIALIZE ENTRY COLUMNS
        # ===========================================
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # V4 FIX: Skip SHORT calculations in SPOT mode to save CPU
        if not self.can_short:
            # For SPOT trading, we only calculate LONG signals
            # This avoids unnecessary computation and potential conflicts
            pass  # Will only process LONG signals below
        dataframe["entry_type"] = 0

        # ===========================================
        # CORE AI SIGNALS (Primary Decision Makers)
        # ===========================================

        # Ensure AI probability exists and is valid
        ml_prob = dataframe.get(
            "ml_entry_probability", pd.Series(0.5, index=dataframe.index)
        )
        ml_enhanced = dataframe.get(
            "ml_enhanced_score", pd.Series(0.5, index=dataframe.index)
        )

        # ===========================================
        # BASIC SAFETY FILTERS (Minimal Requirements)
        # ===========================================

        # 4. ATR ADAPTATIVO por mediana histórica del par
        atr_rel = dataframe["atr"] / dataframe["close"]
        atr_ok = atr_rel < (
            atr_rel.rolling(200).median().fillna(atr_rel.median()) * 1.8
        )

        basic_safety = (
            (dataframe["rsi"] > 15)
            & (dataframe["rsi"] < 85)  # Not extreme RSI
            & (dataframe["volume"] > dataframe["avg_volume"] * 0.3)  # Some volume
            & atr_ok  # ATR adaptativo por mediana (auto-escala por par)
        )

        # ===========================================
        # AI-DRIVEN LONG ENTRIES (Tiered System)
        # ===========================================

        # Dynamic threshold gate with EV filtering
        thr_dyn = dataframe.get("thr_dyn", pd.Series(0.5, index=dataframe.index))
        ev_filter = dataframe.get("ev_filter", pd.Series(1, index=dataframe.index))
        # 5. AJUSTAR DEFAULT ml_agreement a 0.5 para mayor prudencia inicial
        ml_agreement = dataframe.get(
            "ml_model_agreement", pd.Series(0.5, index=dataframe.index)
        )

        # 2. EXTENDER FILTRO DE RÉGIMEN al gate principal
        # Evitar whipsaws cuando RSI≥50 y no hay empuje de tendencia
        trend_strength = dataframe.get(
            "trend_strength", pd.Series(0, index=dataframe.index)
        )
        trend_ok = (dataframe["rsi"] < 50) | (trend_strength > 0)

        # Main gate con filtro de tendencia adicional
        gate = (
            (ml_prob >= thr_dyn)
            & (ev_filter == 1)
            & (ml_agreement > 0.6)
            & trend_ok  # Filtro de régimen extendido
        )

        # A. Gate con requisitos basados en RSI
        current_rsi = dataframe["rsi"]

        # Requisitos de ml_enhanced según RSI
        ml_enhanced_required = pd.Series(0.10, index=dataframe.index)
        ml_enhanced_required[current_rsi >= 50] = 0.18  # RSI ≥ 50 → ml_enhanced ≥ 0.18
        ml_enhanced_required[(current_rsi >= 35) & (current_rsi < 50)] = (
            0.12  # 35 ≤ RSI < 50 → ml_enhanced ≥ 0.12
        )
        # RSI < 35 mantiene 0.10 por defecto

        # E. FILTRO ADICIONAL: trend_strength para RSI >= 50 (ya definido arriba)
        trend_filter = (current_rsi < 50) | (
            trend_strength > 0
        )  # Solo exigir trend si RSI >= 50

        # Gate especial para opportunistic con requisitos ajustados por RSI
        oppo_gate = (
            (ev_filter == 1)  # EV no negativo
            & (ml_agreement > 0.5)  # Consenso mínimo
            & (ml_enhanced >= ml_enhanced_required)  # Requisito dinámico según RSI
            & trend_filter  # Trend positivo si RSI >= 50
        )

        # Debug: Log opportunistic conditions
        if len(dataframe) > 0:
            last_rsi = dataframe["rsi"].iloc[-1] if "rsi" in dataframe else -1
            last_ml_enhanced = (
                ml_enhanced.iloc[-1] if isinstance(ml_enhanced, pd.Series) else -1
            )
            last_ml_prob = ml_prob.iloc[-1] if isinstance(ml_prob, pd.Series) else -1
            last_ml_agreement = (
                ml_agreement.iloc[-1] if isinstance(ml_agreement, pd.Series) else -1
            )
            last_ev_filter = (
                ev_filter.iloc[-1] if isinstance(ev_filter, pd.Series) else -1
            )

            # ALWAYS log to see what's happening
            logger.info(
                f"OPPO_VALUES: {pair} | RSI={last_rsi:.2f} | ml_enhanced={last_ml_enhanced:.3f} | ml_prob={last_ml_prob:.3f} | ml_agreement={last_ml_agreement:.3f} | ev_filter={last_ev_filter}"
            )

            # Check if opportunistic conditions are being met
            if last_rsi > 0 and last_rsi < 35:
                logger.info(
                    f"OPPO_TRIGGER: {pair} RSI={last_rsi:.2f} < 35! Checking other conditions..."
                )

        # TIER 1: Ultra AI Confidence (with dynamic threshold)
        ai_ultra_long = (
            gate & (ml_enhanced > 0.65) & basic_safety  # Enhanced score still high
        )

        # TIER 2: High AI Confidence
        ai_high_long = (
            gate
            & (ml_enhanced > 0.55)  # Medium enhanced score
            & basic_safety
            & ~ai_ultra_long  # Only if not ultra
        )

        # TIER 3: Standard AI Confidence
        ai_standard_long = (
            gate  # Must pass dynamic threshold and EV
            & (ml_enhanced > 0.50)  # Lower enhanced score
            & basic_safety
            & ~(ai_ultra_long | ai_high_long)  # Only if not higher tier
        )

        # TIER 4: Opportunistic AI (Market conditions favorable)
        ai_opportunistic_long = (
            oppo_gate  # Uses special gate without strict ml_prob requirement
            & (dataframe["rsi"] < 35)  # RSI oversold real para entradas de calidad
            & (
                dataframe["volume"] > dataframe["avg_volume"] * 1.2
            )  # Volumen reducido a 1.2x
            & basic_safety
            & ~(ai_ultra_long | ai_high_long | ai_standard_long)
        )

        # Debug: Check if any opportunistic signals triggered
        if ai_opportunistic_long.any():
            # Find indices where signals are true
            signal_indices = dataframe.index[ai_opportunistic_long].tolist()
            if signal_indices:
                last_signal_idx = signal_indices[-1]
                logger.info(
                    f"OPPO_SIGNAL: Found {ai_opportunistic_long.sum()} opportunistic entry signals!"
                )
                logger.info(
                    f"  Last signal at index {last_signal_idx}: RSI={dataframe.loc[last_signal_idx, 'rsi']:.2f}"
                )

        # ===========================================
        # AI-DRIVEN SHORT ENTRIES (Mirror Logic)
        # ===========================================

        # V4 FIX: Only calculate SHORT signals if shorting is enabled (saves CPU in SPOT)
        if self.can_short:
            # TIER 1: Ultra AI Short Confidence
            ai_ultra_short = (
                (ml_prob < 0.3)  # Low probability = good for shorts
                & (ml_enhanced < 0.35)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])  # Below EMA for shorts
            )

            # TIER 2: High AI Short Confidence
            ai_high_short = (
                (ml_prob < 0.4)
                & (ml_enhanced < 0.45)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
                & ~ai_ultra_short
            )

            # TIER 3: Standard AI Short
            ai_standard_short = (
                (ml_prob < 0.48)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
                & ~(ai_ultra_short | ai_high_short)
            )

            # TIER 4: Opportunistic Short
            ai_opportunistic_short = (
                (ml_prob < 0.52)
                & (dataframe["rsi"] > 65)  # Overbought opportunity
                & (dataframe["volume"] > dataframe["avg_volume"] * 1.5)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
                & ~(ai_ultra_short | ai_high_short | ai_standard_short)
            )
        else:
            # In SPOT mode, no SHORT signals are calculated
            ai_ultra_short = False
            ai_high_short = False
            ai_standard_short = False
            ai_opportunistic_short = False

        # ===========================================
        # ENHANCED AI ENTRIES (Optional Boost)
        # ===========================================

        # Check for enhanced AI indicators safely
        quantum_coherence = dataframe.get(
            "quantum_momentum_coherence", pd.Series(0.5, index=dataframe.index)
        )
        neural_pattern = dataframe.get(
            "neural_pattern_score", pd.Series(0.5, index=dataframe.index)
        )
        ml_agreement = dataframe.get(
            "ml_model_agreement", pd.Series(0.5, index=dataframe.index)
        )

        # Enhanced long entries (when advanced AI agrees)
        ai_enhanced_long = (
            (ml_prob > 0.6)
            & (quantum_coherence > 0.6)  # Reduced threshold
            & (neural_pattern > 0.6)  # Reduced threshold
            & (ml_agreement > 0.6)  # Reduced threshold
            & basic_safety
        )

        # Enhanced short entries
        # V4 FIX: Only calculate if shorting is enabled
        if self.can_short:
            ai_enhanced_short = (
                (ml_prob < 0.4)
                & (quantum_coherence < 0.4)  # Inverted for shorts
                & (neural_pattern < 0.4)  # Inverted for shorts
                & (ml_agreement > 0.6)  # Models agree on direction
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
            )
        else:
            ai_enhanced_short = False

        # ===========================================
        # FALLBACK TECHNICAL ENTRIES (If AI fails)
        # ===========================================

        # Simple technical long (backup when AI is neutral)
        technical_long = (
            (ml_prob.between(0.45, 0.55))  # AI neutral
            & (dataframe["rsi"] < 30)  # Oversold
            & (dataframe["close"] > dataframe["ema50"])  # Above EMA
            & (dataframe["volume"] > dataframe["avg_volume"] * 2)  # High volume
            & basic_safety
        )

        # Simple technical short (backup when AI is neutral)
        # V4 FIX: Only calculate if shorting is enabled
        if self.can_short:
            technical_short = (
                (ml_prob.between(0.45, 0.55))  # AI neutral
                & (dataframe["rsi"] > 70)  # Overbought
                & (dataframe["close"] < dataframe["ema50"])  # Below EMA
                & (dataframe["volume"] > dataframe["avg_volume"] * 2)  # High volume
                & basic_safety
            )
        else:
            technical_short = False

        # ===========================================
        # APPLY ENTRY SIGNALS (Hierarchical Priority)
        # ===========================================

        # LONG ENTRIES (Highest priority first)

        # Enhanced AI Long (Priority 1)
        dataframe.loc[ai_enhanced_long, "enter_long"] = 1
        dataframe.loc[ai_enhanced_long, "entry_type"] = 15
        dataframe.loc[ai_enhanced_long, "enter_tag"] = "ai_enhanced_long"

        # Ultra AI Long (Priority 2)
        mask = ai_ultra_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 14
        dataframe.loc[mask, "enter_tag"] = "ai_ultra_long"

        # High AI Long (Priority 3)
        mask = ai_high_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 13
        dataframe.loc[mask, "enter_tag"] = "ai_high_long"

        # Standard AI Long (Priority 4)
        mask = ai_standard_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 12
        dataframe.loc[mask, "enter_tag"] = "ai_standard_long"

        # Opportunistic AI Long (Priority 5)
        mask = ai_opportunistic_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 11
        dataframe.loc[mask, "enter_tag"] = "ai_opportunistic_long"

        # Technical Long (Priority 6 - Fallback)
        mask = technical_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 10
        dataframe.loc[mask, "enter_tag"] = "technical_long"

        # SHORT ENTRIES (If shorting enabled)

        if self.can_short:
            # Enhanced AI Short (Priority 1)
            dataframe.loc[ai_enhanced_short, "enter_short"] = 1
            dataframe.loc[ai_enhanced_short, "entry_type"] = 25
            dataframe.loc[ai_enhanced_short, "enter_tag"] = "ai_enhanced_short"

            # Ultra AI Short (Priority 2)
            mask = ai_ultra_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 24
            dataframe.loc[mask, "enter_tag"] = "ai_ultra_short"

            # High AI Short (Priority 3)
            mask = ai_high_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 23
            dataframe.loc[mask, "enter_tag"] = "ai_high_short"

            # Standard AI Short (Priority 4)
            mask = ai_standard_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 22
            dataframe.loc[mask, "enter_tag"] = "ai_standard_short"

            # Opportunistic AI Short (Priority 5)
            mask = ai_opportunistic_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 21
            dataframe.loc[mask, "enter_tag"] = "ai_opportunistic_short"

            # Technical Short (Priority 6 - Fallback)
            mask = technical_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 20
            dataframe.loc[mask, "enter_tag"] = "technical_short"

        # ===========================================
        # ENTRY DEBUGGING & MONITORING
        # ===========================================

        if metadata["pair"] in ["BTC/USDT", "ETH/USDT"]:
            # Count total entries in last 10 candles
            recent_long_entries = dataframe["enter_long"].tail(10).sum()
            recent_short_entries = dataframe["enter_short"].tail(10).sum()

            if recent_long_entries > 0 or recent_short_entries > 0:
                latest_ml_prob = ml_prob.iloc[-1]
                latest_enhanced = ml_enhanced.iloc[-1]
                latest_entry_type = dataframe["entry_type"].iloc[-1]
                latest_tag = dataframe["enter_tag"].iloc[-1]

                entry_types = {
                    10: "Technical Long",
                    11: "AI Opportunistic Long",
                    12: "AI Standard Long",
                    13: "AI High Long",
                    14: "AI Ultra Long",
                    15: "AI Enhanced Long",
                    20: "Technical Short",
                    21: "AI Opportunistic Short",
                    22: "AI Standard Short",
                    23: "AI High Short",
                    24: "AI Ultra Short",
                    25: "AI Enhanced Short",
                }

                logger.info(f"🎯 {metadata['pair']} ENTRY SIGNAL DETECTED!")
                logger.info(
                    f"   Type: {entry_types.get(latest_entry_type, 'Unknown')} ({latest_tag})"
                )
                logger.info(f"   🤖 ML Probability: {latest_ml_prob:.3f}")
                logger.info(f"   📈 ML Enhanced Score: {latest_enhanced:.3f}")
                logger.info(f"   📊 RSI: {dataframe['rsi'].iloc[-1]:.1f}")
                logger.info(
                    f"   💧 Volume Strength: {dataframe.get('volume_strength', pd.Series([1.0])).iloc[-1]:.2f}"
                )
                logger.info(
                    f"   Recent Entries: {recent_long_entries} Long, {recent_short_entries} Short"
                )

                # Alert if no AI indicators available
                if latest_ml_prob == 0.5 and latest_enhanced == 0.5:
                    logger.warning(
                        f"⚠️  {metadata['pair']} Using fallback - AI indicators may not be working!"
                    )

        # ===========================================
        # FINAL SAFETY CHECK
        # ===========================================

        # Ensure we don't have conflicting signals
        conflict_mask = (dataframe["enter_long"] == 1) & (dataframe["enter_short"] == 1)
        if conflict_mask.any():
            logger.warning(
                f"{metadata['pair']} Resolving {conflict_mask.sum()} conflicting signals"
            )
            # Resolve conflicts: prefer higher entry_type (more confident signal)
            long_priority = dataframe["entry_type"].where(
                dataframe["enter_long"] == 1, 0
            )
            short_priority = dataframe["entry_type"].where(
                dataframe["enter_short"] == 1, 0
            )

            # Keep the higher priority signal
            keep_long = long_priority >= short_priority
            dataframe.loc[conflict_mask & ~keep_long, "enter_long"] = 0
            dataframe.loc[conflict_mask & keep_long, "enter_short"] = 0

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        UNIFIED EXIT SYSTEM - Choose between Custom MML Exits or Simple Opposite Signal Exits
        """
        # ===========================================
        # INITIALIZE EXIT COLUMNS
        # ===========================================
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        # ===========================================
        # CHOOSE EXIT SYSTEM
        # ===========================================
        if self.use_custom_exits_advanced:
            # Use Alex's Advanced MML-based Exit System
            return self._populate_custom_exits_advanced(dataframe, metadata)
        else:
            # Use Simple Opposite Signal Exit System
            return self._populate_simple_exits(dataframe, metadata)

    def _populate_custom_exits_advanced(
        self, df: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        ALEX'S ADVANCED MML-BASED EXIT SYSTEM
        Profit-protecting exit strategy with better signal coordination
        """

        # ===========================================
        # MML MARKET STRUCTURE FOR EXITS
        # ===========================================

        # Bullish/Bearish structure (same as entry)
        bullish_mml = (df["close"] > df["[6/8]P"]) | (
            (df["close"] > df["[4/8]P"])
            & (df["close"].shift(5) < df["[4/8]P"].shift(5))
        )

        bearish_mml = (df["close"] < df["[2/8]P"]) | (
            (df["close"] < df["[4/8]P"])
            & (df["close"].shift(5) > df["[4/8]P"].shift(5))
        )

        # MML resistance/support levels for exits
        at_resistance = (
            (df["high"] >= df["[6/8]P"])  # At 75%
            | (df["high"] >= df["[7/8]P"])  # At 87.5%
            | (df["high"] >= df["[8/8]P"])  # At 100%
        )

        at_support = (
            (df["low"] <= df["[2/8]P"])  # At 25%
            | (df["low"] <= df["[1/8]P"])  # At 12.5%
            | (df["low"] <= df["[0/8]P"])  # At 0%
        )

        # ===========================================
        # AI-BASED INTELLIGENT EXIT SIGNALS
        # ===========================================

        # Calculate AI prediction stability and direction indicators
        current_profit_signal = pd.Series([False] * len(df), index=df.index)
        ai_stability_signal = pd.Series([False] * len(df), index=df.index)
        ai_degradation_signal = pd.Series([False] * len(df), index=df.index)

        try:
            # 1. AI STABILITY ANALYSIS
            # Check if ML predictions are stable and maintain direction
            ml_prob = df.get("ml_entry_probability", pd.Series([0.5] * len(df)))
            ml_enhanced = df.get("ml_enhanced_score", pd.Series([0.5] * len(df)))

            # Calculate AI stability metrics
            ml_prob_sma_5 = ml_prob.rolling(5).mean().fillna(ml_prob)
            ml_prob_sma_10 = ml_prob.rolling(10).mean().fillna(ml_prob)
            ml_prob_std_5 = ml_prob.rolling(5).std().fillna(0.1)

            # For LONG positions: AI should maintain high probability (>0.6) for continuation
            ai_long_stable = (
                (ml_prob > 0.6)  # Current prediction supports long
                & (ml_prob_sma_5 > 0.6)  # Recent average supports long
                & (ml_prob_std_5 < 0.15)  # Low volatility in predictions (stable)
                & (ml_enhanced > 0.65)  # Enhanced score supports direction
                & (ml_prob > ml_prob.shift(1))  # Predictions improving or stable
            )

            # For SHORT positions: AI should maintain low probability (<0.4) for continuation
            ai_short_stable = (
                (ml_prob < 0.4)
                & (ml_prob_sma_5 < 0.4)  # Recent average supports short
                & (ml_prob_std_5 < 0.15)  # Low volatility in predictions (stable)
                & (ml_enhanced < 0.35)  # Enhanced score supports direction
                & (ml_prob < ml_prob.shift(1))  # Predictions improving toward short
            )

            # 2. F. AI DEGRADATION WITH HISTERESIS (anti-ruido)
            # Evitar salidas rápidas agregando histeresis

            # Calcular trend_strength para filtro adicional
            trend_strength = df.get("trend_strength", pd.Series(0, index=df.index))

            # Pre-degradación: marcar cuando ml_prob < 0.48 y trend cae
            pre_degradation_long = (
                ml_prob < 0.48
            ) & (  # Umbral más alto para pre-degradación
                trend_strength < trend_strength.shift(3)
            )  # Trend cayendo

            # Degradación confirmada: necesita 2 velas consecutivas o volume spike
            volume_spike = df["volume"] > df["volume"].rolling(20).mean() * 2.0

            # For LONG positions: Exit con histeresis
            ai_long_degradation = (
                # Debe cumplir condiciones de histeresis
                (
                    # Opción 1: Dos velas consecutivas en pre-degradación
                    (pre_degradation_long & pre_degradation_long.shift(1))
                    |
                    # Opción 2: Pre-degradación con volume spike confirmando
                    (pre_degradation_long & volume_spike)
                )
                &
                # Y cumplir al menos una condición de degradación
                (
                    (ml_prob < 0.45)  # Prediction dropped below neutral
                    | (ml_prob < ml_prob.shift(2) - 0.15)  # 15% drop (reducido de 20%)
                    | (ml_prob_sma_5 < ml_prob_sma_10 - 0.1)  # Trend deteriorating
                    | (ml_prob_std_5 > 0.30)  # Alta volatilidad (subido de 0.25)
                    | ((ml_prob > 0.6) & (ml_enhanced < 0.4))  # Conflicting signals
                )
            )

            # Pre-degradación SHORT
            pre_degradation_short = (
                ml_prob > 0.52
            ) & (  # Umbral para pre-degradación short
                trend_strength > trend_strength.shift(3)
            )  # Trend subiendo

            # For SHORT positions: Exit con histeresis
            ai_short_degradation = (
                # Debe cumplir condiciones de histeresis
                (
                    # Opción 1: Dos velas consecutivas en pre-degradación
                    (pre_degradation_short & pre_degradation_short.shift(1))
                    |
                    # Opción 2: Pre-degradación con volume spike
                    (pre_degradation_short & volume_spike)
                )
                &
                # Y cumplir al menos una condición de degradación
                (
                    (ml_prob > 0.55)  # Prediction moved above neutral
                    | (ml_prob > ml_prob.shift(2) + 0.15)  # 15% rise (reducido de 20%)
                    | (ml_prob_sma_5 > ml_prob_sma_10 + 0.1)  # Trend improving
                    | (ml_prob_std_5 > 0.30)  # Alta volatilidad
                    | ((ml_prob < 0.4) & (ml_enhanced > 0.6))  # Conflicting signals
                )
            )

            # 3. COMBINED AI EXIT LOGIC WITH PROFIT PROTECTION
            # Protección de trailing profit > 2%
            # Nota: El profit real viene del objeto trade, aquí es aproximación

            # Exit LONG con protección (no salir si hay buen profit)
            ai_long_exit = ai_long_degradation & (~ai_long_stable)
            # Nota: La protección de profit real se maneja en custom_exit

            # Exit SHORT con protección similar
            ai_short_exit = ai_short_degradation & (~ai_short_stable)

            # Store signals for use in exit combinations
            ai_stability_signal = ai_long_stable | ai_short_stable
            ai_degradation_signal = ai_long_exit | ai_short_exit

            # Simple profit conditions (backup to AI logic)
            rolling_high = df["high"].rolling(20).max()
            current_drawdown = (rolling_high - df["close"]) / rolling_high

            profit_exit_signal = (
                df["close"] > df["close"].shift(20) * 1.06
            ) & (  # 6%+ gain from 20 candles ago
                current_drawdown > 0.02
            )  # But now dropped 2%+ from recent high

            resistance_profit_exit = (
                at_resistance
                & (
                    df["close"] > df["close"].shift(10) * 1.04
                )  # 4%+ gain from 10 candles ago
                & (df["rsi"] > 65)  # Overbought
                & (df["close"] < df["high"])  # Didn't close at high
            )

            # Combine AI exits with traditional profit-taking
            current_profit_signal = (
                ai_degradation_signal  # AI degradation (primary)
                | (
                    profit_exit_signal & (~ai_stability_signal)
                )  # Profit exit only if AI unstable
                | (
                    resistance_profit_exit & (~ai_stability_signal)
                )  # Resistance exit only if AI unstable
            )

        except Exception as e:
            # If any error, continue with normal exit logic
            logger.warning(
                f"AI exit logic error for {metadata.get('pair', 'unknown')}: {e}"
            )
            # Fallback to simple profit signals
            try:
                rolling_high = df["high"].rolling(20).max()
                current_drawdown = (rolling_high - df["close"]) / rolling_high
                current_profit_signal = (df["close"] > df["close"].shift(20) * 1.06) & (
                    current_drawdown > 0.02
                )
            except Exception:
                current_profit_signal = pd.Series([False] * len(df), index=df.index)

        # ===========================================
        # LONG EXIT SIGNALS (ADVANCED MML SYSTEM)
        # ===========================================

        # 1. Profit-Taking Exits
        long_exit_resistance_profit = (
            at_resistance
            & (df["close"] < df["high"])  # Failed to close at high
            & (df["rsi"] > 65)  # Overbought
            & (df["maxima"] == 1)  # Local top
            & (df["volume"] > df["volume"].rolling(10).mean())
        )

        long_exit_extreme_overbought = (
            (df["close"] > df["[7/8]P"])
            & (df["rsi"] > 75)
            & (df["close"] < df["close"].shift(1))  # Price turning down
            & (df["maxima"] == 1)
        )

        long_exit_volume_exhaustion = (
            at_resistance
            & (
                df["volume"] < df["volume"].rolling(20).mean() * 0.6
            )  # Tightened from 0.8
            & (df["rsi"] > 70)
            & (df["close"] < df["close"].shift(1))
            & (df["close"] < df["close"].rolling(3).mean())  # Added price confirmation
        )

        # 2. Structure Breakdown (Improved with strong filters)
        long_exit_structure_breakdown = (
            (df["close"] < df["[4/8]P"])
            & (df["close"].shift(1) >= df["[4/8]P"].shift(1))
            & bullish_mml.shift(1)
            & (df["close"] < df["[4/8]P"] * 0.995)
            & (df["close"] < df["close"].shift(1))
            & (df["close"] < df["close"].shift(2))
            & (df["rsi"] < 45)  # Tightened from 50
            & (
                df["volume"] > df["volume"].rolling(15).mean() * 2.0
            )  # Increased from 1.5
            & (df["close"] < df["open"])
            & (df["low"] < df["low"].shift(1))
            & (df["close"] < df["close"].rolling(3).mean())
            & (df["momentum_quality"] < 0)  # Added momentum check
        )

        # 3. Momentum Divergence
        long_exit_momentum_divergence = (
            at_resistance
            & (df["rsi"] < df["rsi"].shift(1))  # RSI falling
            & (df["rsi"].shift(1) < df["rsi"].shift(2))  # RSI was falling
            & (df["rsi"] < df["rsi"].shift(3))  # 3-candle RSI decline
            & (df["close"] >= df["close"].shift(1))  # Price still up/flat
            & (df["maxima"] == 1)
            & (df["rsi"] > 60)  # Only in overbought territory
        )

        # 4. Range Exit
        long_exit_range = (
            (df["close"] >= df["[2/8]P"])
            & (df["close"] <= df["[6/8]P"])  # In range
            & (df["high"] >= df["[6/8]P"])  # HIGH touched 75%, not close
            & (df["close"] < df["[6/8]P"] * 0.995)  # But closed below
            & (df["rsi"] > 65)  # More conservative RSI
            & (df["maxima"] == 1)
            & (
                df["volume"] > df["volume"].rolling(10).mean() * 1.2
            )  # Volume confirmation
        )

        # 5. Emergency Exit
        long_exit_emergency = (
            (
                (df["close"] < df["[0/8]P"])
                & (df["rsi"] < 20)  # Changed from 15
                & (
                    df["volume"] > df["volume"].rolling(20).mean() * 2.5
                )  # Reduced from 3
                & (df["close"] < df["close"].shift(1))
                & (df["close"] < df["close"].shift(2))
                & (df["close"] < df["open"])
            )
            if self.use_emergency_exits
            else pd.Series([False] * len(df), index=df.index)
        )

        # ===========================================
        # AI-ENHANCED EXIT COMBINATION
        # ===========================================

        try:
            # Get ML prediction history for stability analysis
            ml_prob = df.get("ml_entry_probability", pd.Series([0.5] * len(df)))

            # Calculate AI trend and stability over longer period
            ml_prob_sma_20 = ml_prob.rolling(20).mean().fillna(ml_prob)
            ml_trend_strength = (ml_prob - ml_prob_sma_20).abs()

            # AI Override Logic: Don't exit if AI shows strong consistent signals
            ai_override_long = (
                ai_stability_signal  # AI is stable
                & (ml_prob > 0.7)  # High confidence for long
                & (ml_trend_strength < 0.1)  # Low deviation from 20-period average
            )

            # Traditional MML exit signals
            traditional_mml_exits = (
                long_exit_resistance_profit
                | long_exit_extreme_overbought
                | long_exit_volume_exhaustion
                | long_exit_structure_breakdown
                | long_exit_momentum_divergence
                | long_exit_range
                | long_exit_emergency
            )

            # Final AI-enhanced exit decision
            any_long_exit = (
                # AI degradation signal (highest priority - always exit)
                ai_degradation_signal
                |
                # Traditional profit-taking when AI not stable
                (current_profit_signal & (~ai_stability_signal))
                |
                # Traditional MML exits unless AI strongly overrides
                (traditional_mml_exits & (~ai_override_long))
            )

            # Log AI decision for debugging (only when signal changes)
            if len(df) > 1:
                current_ai_exit = (
                    ai_degradation_signal.iloc[-1]
                    if len(ai_degradation_signal) > 0
                    else False
                )
                current_ai_stable = (
                    ai_stability_signal.iloc[-1]
                    if len(ai_stability_signal) > 0
                    else False
                )
                current_ml_prob = ml_prob.iloc[-1] if len(ml_prob) > 0 else 0.5
                current_ai_override = (
                    ai_override_long.iloc[-1] if len(ai_override_long) > 0 else False
                )

                # Initialize logging state if needed
                if not hasattr(self, "_last_ai_state"):
                    self._last_ai_state = {}

                pair_key = metadata.get("pair", "UNKNOWN")
                last_state = self._last_ai_state.get(pair_key, {})

                # Log significant AI state changes
                if (
                    abs(current_ml_prob - last_state.get("ml_prob", 0.5)) > 0.1
                    or current_ai_exit != last_state.get("ai_exit", False)
                    or current_ai_stable != last_state.get("ai_stable", False)
                    or current_ai_override != last_state.get("ai_override", False)
                ):

                    logger.info(
                        f"AI Exit Analysis {pair_key}: "
                        f"ML_Prob={current_ml_prob:.3f}, "
                        f"AI_Exit={current_ai_exit}, "
                        f"AI_Stable={current_ai_stable}, "
                        f"AI_Override={current_ai_override}"
                    )

                # Store current state
                self._last_ai_state[pair_key] = {
                    "ml_prob": current_ml_prob,
                    "ai_exit": current_ai_exit,
                    "ai_stable": current_ai_stable,
                    "ai_override": current_ai_override,
                }

        except Exception as e:
            logger.warning(
                f"AI exit combination error for {metadata.get('pair', 'unknown')}: {e}"
            )
            # Fallback to traditional logic
            any_long_exit = (
                current_profit_signal
                | long_exit_resistance_profit
                | long_exit_extreme_overbought
                | long_exit_volume_exhaustion
                | long_exit_structure_breakdown
                | long_exit_momentum_divergence
                | long_exit_range
                | long_exit_emergency
            )

        # ===========================================
        # SHORT EXIT SIGNALS (if enabled)
        # ===========================================

        if self.can_short:
            # 1. Profit-Taking Exits
            short_exit_support_profit = (
                at_support
                & (df["close"] > df["low"])  # Failed to close at low
                & (df["rsi"] < 35)  # Oversold
                & (df["minima"] == 1)  # Local bottom
                & (df["volume"] > df["volume"].rolling(10).mean())
            )

            short_exit_extreme_oversold = (
                (df["close"] < df["[1/8]P"])
                & (df["rsi"] < 25)
                & (df["close"] > df["close"].shift(1))  # Price turning up
                & (df["minima"] == 1)
            )

            short_exit_volume_exhaustion = (
                at_support
                & (
                    df["volume"] < df["volume"].rolling(20).mean() * 0.6
                )  # Tightened from 0.8
                & (df["rsi"] < 30)
                & (df["close"] > df["close"].shift(1))
                & (
                    df["close"] > df["close"].rolling(3).mean()
                )  # Added price confirmation
            )

            # 2. Structure Breakout
            short_exit_structure_breakout = (
                (df["close"] > df["[4/8]P"])
                & (df["close"].shift(1) <= df["[4/8]P"].shift(1))
                & bearish_mml.shift(1)
                & (df["close"] > df["[4/8]P"] * 1.005)
                & (df["close"] > df["close"].shift(1))
                & (df["close"] > df["close"].shift(2))
                & (df["rsi"] > 55)  # Tightened from 50
                & (
                    df["volume"] > df["volume"].rolling(15).mean() * 2.0
                )  # Increased from 1.5
                & (df["close"] > df["open"])
                & (df["high"] > df["high"].shift(1))
                & (df["momentum_quality"] > 0)  # Added momentum check
            )

            # 3. Momentum Divergence
            short_exit_momentum_divergence = (
                at_support
                & (df["rsi"] > df["rsi"].shift(1))  # RSI rising
                & (df["rsi"].shift(1) > df["rsi"].shift(2))  # RSI was rising
                & (df["rsi"] > df["rsi"].shift(3))  # 3-candle RSI rise
                & (df["close"] <= df["close"].shift(1))  # Price still down/flat
                & (df["minima"] == 1)
                & (df["rsi"] < 40)  # Only in oversold territory
            )

            # 4. Range Exit
            short_exit_range = (
                (df["close"] >= df["[2/8]P"])
                & (df["close"] <= df["[6/8]P"])  # In range
                & (df["low"] <= df["[2/8]P"])  # LOW touched 25%
                & (df["close"] > df["[2/8]P"] * 1.005)  # But closed above
                & (df["rsi"] < 35)  # More conservative RSI
                & (df["minima"] == 1)
                & (
                    df["volume"] > df["volume"].rolling(10).mean() * 1.2
                )  # Volume confirmation
            )

            # 5. Emergency Exit
            short_exit_emergency = (
                (
                    (df["close"] > df["[8/8]P"])
                    & (df["rsi"] > 80)  # Changed from 85
                    & (
                        df["volume"] > df["volume"].rolling(20).mean() * 2.5
                    )  # Reduced from 3
                    & (df["close"] > df["close"].shift(1))
                    & (df["close"] > df["close"].shift(2))
                    & (df["close"] > df["open"])
                )
                if self.use_emergency_exits
                else pd.Series([False] * len(df), index=df.index)
            )

            # ===========================================
            # AI-ENHANCED SHORT EXIT COMBINATION
            # ===========================================

            try:
                # AI Override Logic for SHORT positions
                ai_override_short = (
                    ai_stability_signal  # AI is stable
                    & (ml_prob < 0.3)  # High confidence for short
                    & (ml_trend_strength < 0.1)  # Low deviation from 20-period average
                )

                # Traditional MML short exit signals
                traditional_short_exits = (
                    short_exit_support_profit
                    | short_exit_extreme_oversold
                    | short_exit_volume_exhaustion
                    | short_exit_structure_breakout
                    | short_exit_momentum_divergence
                    | short_exit_range
                    | short_exit_emergency
                )

                # Final AI-enhanced SHORT exit decision
                any_short_exit = (
                    # AI degradation signal (always exit shorts too)
                    ai_degradation_signal
                    |
                    # Traditional short exits unless AI strongly overrides
                    (traditional_short_exits & (~ai_override_short))
                )

            except Exception as e:
                logger.warning(
                    f"AI short exit error for {metadata.get('pair', 'unknown')}: {e}"
                )
                # Fallback to traditional logic
                any_short_exit = (
                    short_exit_support_profit
                    | short_exit_extreme_oversold
                    | short_exit_volume_exhaustion
                    | short_exit_structure_breakout
                    | short_exit_momentum_divergence
                    | short_exit_range
                    | short_exit_emergency
                )
        else:
            any_short_exit = pd.Series([False] * len(df), index=df.index)

        # ===========================================
        # COORDINATION WITH ENTRY SIGNALS
        # ===========================================

        # If we have new Entry signals, they override Exit signals
        has_long_entry = "enter_long" in df.columns and (df["enter_long"] == 1).any()
        has_short_entry = "enter_short" in df.columns and (df["enter_short"] == 1).any()

        if has_long_entry:
            long_entry_mask = df["enter_long"] == 1
            any_long_exit = any_long_exit & (~long_entry_mask)

        if has_short_entry and self.can_short:
            short_entry_mask = df["enter_short"] == 1
            any_short_exit = any_short_exit & (~short_entry_mask)

        # ===========================================
        # SET FINAL EXIT SIGNALS AND TAGS
        # ===========================================

        # Long Exits
        df.loc[any_long_exit, "exit_long"] = 1

        # Tags for Long Exits (Priority: AI > Emergency > Structure > Profit)
        df.loc[any_long_exit & ai_degradation_signal, "exit_tag"] = (
            "AI_Degradation_Exit"
        )
        df.loc[any_long_exit & long_exit_emergency, "exit_tag"] = (
            "MML_Emergency_Long_Exit"
        )
        df.loc[
            any_long_exit & current_profit_signal & (df["exit_tag"] == ""), "exit_tag"
        ] = "AI_Profit_Taking"
        df.loc[
            any_long_exit & long_exit_structure_breakdown & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Structure_Breakdown_Confirmed"
        df.loc[
            any_long_exit & long_exit_resistance_profit & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Resistance_Profit"
        df.loc[
            any_long_exit & long_exit_extreme_overbought & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Extreme_Overbought"
        df.loc[
            any_long_exit & long_exit_volume_exhaustion & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Volume_Exhaustion_Long"
        df.loc[
            any_long_exit & long_exit_momentum_divergence & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Momentum_Divergence_Long"
        df.loc[any_long_exit & long_exit_range & (df["exit_tag"] == ""), "exit_tag"] = (
            "MML_Range_Exit_Long"
        )

        # Short Exits
        if self.can_short:
            df.loc[any_short_exit, "exit_short"] = 1

            # Tags for Short Exits (Priority: AI > Emergency > Structure > Profit)
            df.loc[any_short_exit & ai_degradation_signal, "exit_tag"] = (
                "AI_Degradation_Exit"
            )
            df.loc[any_short_exit & short_exit_emergency, "exit_tag"] = (
                "MML_Emergency_Short_Exit"
            )
            df.loc[
                any_short_exit & short_exit_structure_breakout & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Structure_Breakout_Confirmed"
            df.loc[
                any_short_exit & short_exit_support_profit & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Support_Profit"
            df.loc[
                any_short_exit & short_exit_extreme_oversold & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Extreme_Oversold"
            df.loc[
                any_short_exit & short_exit_volume_exhaustion & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Volume_Exhaustion_Short"
            df.loc[
                any_short_exit
                & short_exit_momentum_divergence
                & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Momentum_Divergence_Short"
            df.loc[
                any_short_exit & short_exit_range & (df["exit_tag"] == ""), "exit_tag"
            ] = "MML_Range_Exit_Short"

        return df

    def _populate_simple_exits(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        SIMPLE OPPOSITE SIGNAL EXIT SYSTEM - SYNTAX FIXED
        """

        # Exit LONG when any SHORT signal appears
        long_exit_on_short = dataframe["enter_short"] == 1

        # Exit SHORT when any LONG signal appears
        short_exit_on_long = dataframe["enter_long"] == 1

        # Emergency exits (if enabled)
        if self.use_emergency_exits:
            emergency_long_exit = (
                (dataframe["rsi"] > 85)
                & (dataframe["volume"] > dataframe["avg_volume"] * 3)
                & (dataframe["close"] < dataframe["open"])
                & (dataframe["close"] < dataframe["low"].shift(1))
            ) | (
                (dataframe.get("structure_break_down", 0) == 1)
                & (dataframe["volume"] > dataframe["avg_volume"] * 2.5)
                & (dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 2)
            )

            emergency_short_exit = (
                (dataframe["rsi"] < 15)
                & (dataframe["volume"] > dataframe["avg_volume"] * 3)
                & (dataframe["close"] > dataframe["open"])
                & (dataframe["close"] > dataframe["high"].shift(1))
            ) | (
                (dataframe.get("structure_break_up", 0) == 1)
                & (dataframe["volume"] > dataframe["avg_volume"] * 2.5)
                & (dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 2)
            )
        else:
            emergency_long_exit = pd.Series(
                [False] * len(dataframe), index=dataframe.index
            )
            emergency_short_exit = pd.Series(
                [False] * len(dataframe), index=dataframe.index
            )

        # Apply exits
        dataframe.loc[long_exit_on_short, "exit_long"] = 1
        dataframe.loc[long_exit_on_short, "exit_tag"] = "trend_reversal"

        dataframe.loc[short_exit_on_long, "exit_short"] = 1
        dataframe.loc[short_exit_on_long, "exit_tag"] = "trend_reversal"

        # Emergency exits
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_long"] = 1
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_tag"] = (
            "emergency_exit"
        )

        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_short"] = 1
        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_tag"] = (
            "emergency_exit"
        )

        # DEBUGGING (FIXED THE ERROR HERE)
        if metadata["pair"] in ["BTC/USDT", "ETH/USDT"]:
            recent_exits = (
                dataframe["exit_long"].tail(5).sum()
                + dataframe["exit_short"].tail(5).sum()
            )
            if recent_exits > 0:
                exit_tag = dataframe["exit_tag"].iloc[-1]
                logger.info(f"{metadata['pair']} EXIT SIGNAL - Tag: {exit_tag}")
                # âœ… FIXED: Use the correct attribute name
                logger.info(
                    f"  Exit System: {'Custom MML' if self.use_custom_exits_advanced else 'Simple Opposite'}"
                )
                logger.info(f"  RSI: {dataframe['rsi'].iloc[-1]:.1f}")

        return dataframe

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_entry_profit: float,
        current_exit_rate: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """
        V4: DCA (Dollar Cost Averaging) implementation
        Custom trade adjustment logic for position sizing
        """
        try:
            # Only do DCA if enabled and we're in a loss
            if current_profit > self.initial_safety_order_trigger.value:
                return None

            # Check if we've reached max safety orders
            filled_entries = trade.nr_of_successful_entries
            if (
                filled_entries >= self.max_safety_orders.value + 1
            ):  # +1 for initial order
                return None

            # Calculate the trigger for this safety order
            # Each subsequent order triggers at a larger loss
            trigger = self.initial_safety_order_trigger.value
            for i in range(1, filled_entries):
                trigger = trigger * self.safety_order_step_scale.value

            # Check if we've hit the trigger for next safety order
            if current_profit <= trigger:
                # Calculate DCA order size
                # Each order is larger than the previous
                dca_amount = trade.stake_amount
                for i in range(filled_entries):
                    dca_amount = dca_amount * self.safety_order_volume_scale.value

                # Ensure we don't exceed max_stake
                if dca_amount > max_stake:
                    dca_amount = max_stake

                # Ensure we meet min_stake
                if min_stake and dca_amount < min_stake:
                    return None

                logger.info(
                    f"[DCA-V4] {trade.pair} triggering safety order {filled_entries} "
                    f"at {current_profit:.2%} (trigger: {trigger:.2%}), "
                    f"amount: {dca_amount:.4f}"
                )

                return dca_amount

        except Exception as e:
            logger.error(f"[DCA-V4] Error in adjust_trade_position: {e}")

        return None

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
        current_profit_ratio = trade.calc_profit_ratio(rate)
        trade_duration = (
            current_time - trade.open_date_utc
        ).total_seconds() / 3600  # Hours

        always_allow = [
            "stoploss",
            "stop_loss",
            "custom_stoploss",
            "roi",
            "trend_reversal",
            "emergency_exit",
        ]

        # Allow regime protection exits (icons)
        if any(char in exit_reason for char in ["⚡", "🔊", "🌊", "🎯", "₿"]):
            return True

        # Allow known good exits
        if exit_reason in always_allow:
            return True

        # FIXED: Previously blocked trailing stops if profit <= 0.
        # Now configurable & allows controlled negative trailing exits (prevents deeper drawdowns).
        if exit_reason in ["trailing_stop_loss", "trailing_stop"]:
            # If enabled, allow exit when profit >= configured minimal threshold
            if self.allow_trailing_exit_when_negative.value:
                if current_profit_ratio >= self.trailing_exit_min_profit.value:
                    logger.info(
                        f"{pair} Allow trailing exit (thr={self.trailing_exit_min_profit.value:.3f}) "
                        f"Profit: {current_profit_ratio:.2%} Reason: {exit_reason}"
                    )
                    return True
                else:
                    logger.info(
                        f"{pair} Blocking trailing exit below min threshold "
                        f"(profit {current_profit_ratio:.2%} < {self.trailing_exit_min_profit.value:.2%})"
                    )
                    return False
            else:
                # Legacy behaviour (only positive)
                if current_profit_ratio > 0:
                    logger.info(
                        f"{pair} Allow trailing exit (legacy >0). Profit: {current_profit_ratio:.2%}"
                    )
                    return True
                logger.info(
                    f"{pair} Blocking trailing exit (legacy rule). Profit: {current_profit_ratio:.2%}"
                )
                return False

        # 3. Timed exits manejados por ROI table (24h: 0.5%, 48h: salida forzada)
        # Ya no necesitamos lógica redundante aquí

        return True