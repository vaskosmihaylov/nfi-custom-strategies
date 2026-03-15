# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
from datetime import datetime
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
	IStrategy,
	informative,
	stoploss_from_absolute,
	IntParameter,
)


class MtfScalper(IStrategy):
	INTERFACE_VERSION = 3

	# Base timeframe
	timeframe = "5m"
	process_only_new_candles = True
	startup_candle_count = 240
	can_short = True

	# Baseline risk config (will be overridden by hyperopt and config files)
	minimal_roi = {
		"0": 0.04,
		"30": 0.02,
		"60": 0.01,
	}
	stoploss = -0.10

	# Parameters aligned with user's initial confirmation
	atr_length: int = 14
	atr_multiplier: float = 1.5
	ema_fast_len: int = 9
	ema_slow_len: int = 21
	ema_trend_len: int = 200
	adx_len: int = 14
	adx_threshold: int = 25

	# Hyperoptable parameters
	buy_rsi: IntParameter = IntParameter(low=40, high=70, default=55, space="buy", optimize=True, load=True)
	sell_rsi: IntParameter = IntParameter(low=40, high=70, default=55, space="sell", optimize=True, load=True)
	adx_thr_buy: IntParameter = IntParameter(low=20, high=35, default=25, space="buy", optimize=True, load=True)
	adx_thr_sell: IntParameter = IntParameter(low=20, high=35, default=25, space="sell", optimize=True, load=True)
	atr_threshold: IntParameter = IntParameter(low=1, high=10, default=5, space="buy", optimize=True, load=True)

	# Target risk per trade (fraction of equity), and default leverage
	risk_per_trade: float = 0.02

	# --- Informative higher TF indicators ---
	@informative("15m")
	def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_len)
		dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_len)
		dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_len)
		return dataframe

	@informative("1h")
	def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_len)
		dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_len)
		dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_len)
		return dataframe

	def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		# Base TF indicators
		dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_len)
		dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_len)
		dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.ema_trend_len)
		dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
		dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_len)

		# ATR on base TF
		# ta.ATR expects columns high/low/close
		dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_length)

		# Additional indicators for LSTM (from Netanelshoshan/freqAI-LSTM)
		dataframe["ma"] = ta.SMA(dataframe, timeperiod=10)
		macd_data = ta.MACD(dataframe)
		dataframe["macd"] = macd_data["macd"]  # Only the main line
		dataframe["roc"] = ta.ROC(dataframe, timeperiod=2)
		dataframe["momentum"] = ta.MOM(dataframe, timeperiod=4)
		dataframe["bb_upper"], _, dataframe["bb_lower"] = ta.BBANDS(dataframe, timeperiod=20)
		dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)
		dataframe["stoch"] = ta.STOCH(dataframe)["slowk"]
		dataframe["obv"] = ta.OBV(dataframe)

		# Normalization (z-score) - only numeric columns
		numeric_cols = ["rsi", "atr", "ma", "macd", "roc", "momentum", "bb_upper", "cci", "stoch", "obv"]
		for col in numeric_cols:
			if col in dataframe.columns and dataframe[col].dtype in ['float64', 'int64']:
				dataframe[f"normalized_{col}"] = (dataframe[col] - dataframe[col].rolling(14).mean()) / dataframe[col].rolling(14).std()

		# Dynamic Weighting
		trend_strength = abs(dataframe["ema_fast"] - dataframe["close"])
		strong_trend = trend_strength > trend_strength.rolling(14).mean() + 1.5 * trend_strength.rolling(14).std()
		dataframe["w_momentum"] = strong_trend.astype(int) * 1.5 + 1.0

		# Regime Filter R (Bollinger)
		dataframe["R"] = 0
		if dataframe["bb_upper"].dtype in ['float64', 'int64'] and dataframe["ma"].dtype in ['float64', 'int64']:
			dataframe.loc[(dataframe["close"] > dataframe["bb_upper"]) & (dataframe["close"] > dataframe["ma"]), "R"] = 1
			dataframe.loc[(dataframe["close"] < dataframe["bb_lower"]) & (dataframe["close"] < dataframe["ma"]), "R"] = -1

		# Volatility V
		dataframe["V"] = 1 / dataframe["atr"]

		# Aggregate Score S
		dataframe["S"] = dataframe["normalized_rsi"] * dataframe["w_momentum"]

		# Target T
		dataframe["&-target"] = dataframe["S"] * dataframe["R"] * dataframe["V"]

		# Momentum-Strength filter: ABS(log_return 20bar) / ATR(20) > 0.6
		dataframe["momentum_strength"] = abs(dataframe["close"].pct_change(20)) / dataframe["atr"]

		# --- New Feature Engineering (from user's suggestion) ---
		# Returns lagged
		for lag in [1, 3, 5, 10]:
			dataframe[f'return_lag_{lag}'] = dataframe['close'].pct_change(lag)

		# Rolling volatility
		for w in [30, 60, 120]:
			dataframe[f'vol_{w}'] = dataframe['close'].pct_change().rolling(w).std()

		# Volume features
		dataframe['vol_change_1'] = dataframe['volume'].pct_change(1)
		for w in [30, 60]:
			dataframe[f'vol_mean_{w}'] = dataframe['volume'].rolling(w).mean()
			dataframe[f'vol_over_mean_{w}'] = dataframe['volume'] / (dataframe[f'vol_mean_{w}'] + 1e-9)

		# EMA diff
		dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)
		dataframe['ema_30'] = ta.EMA(dataframe, timeperiod=30)
		dataframe['ema_diff_10_30'] = dataframe['ema_10'] - dataframe['ema_30']

		# Spread proxy (high-low)
		dataframe['hl_spread'] = (dataframe['high'] - dataframe['low']) / ((dataframe['high'] + dataframe['low']) / 2 + 1e-9)

		# --- Advanced Features ---
		# Lagged RSI
		for lag in [3, 5]:
			dataframe[f'rsi_lag_{lag}'] = dataframe['rsi'].shift(lag)

		# MACD histogram
		macd, macdsignal, macdhist = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
		dataframe['macd_hist'] = macdhist

		# Improved spread proxy: rolling average of hl_spread
		dataframe['hl_spread_mean_30'] = dataframe['hl_spread'].rolling(30).mean()
		dataframe['hl_spread_over_mean'] = dataframe['hl_spread'] / (dataframe['hl_spread_mean_30'] + 1e-9)

		# Fillna
		dataframe.fillna(0, inplace=True)

		return dataframe

	def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		dataframe["enter_long"] = 0
		dataframe["enter_short"] = 0

		# --- Session Filter: Removed for new pairs (DOGE/SOL) as per user request ---

		# Trend conditions base TF
		main_trend_up = (dataframe["ema_fast"] > dataframe["ema_slow"]) & (dataframe["close"] > dataframe["ema_trend"])  # noqa: E501
		main_trend_down = (dataframe["ema_fast"] < dataframe["ema_slow"]) & (dataframe["close"] < dataframe["ema_trend"])  # noqa: E501
		main_strong_trend_buy = dataframe["adx"] > float(self.adx_thr_buy.value)
		main_strong_trend_sell = dataframe["adx"] > float(self.adx_thr_sell.value)

		# Confirmation 15m (columns from informative decorator are suffixed with _15m)
		confirm_trend_up = dataframe["ema_fast_15m"] > dataframe["ema_slow_15m"]
		confirm_trend_down = dataframe["ema_fast_15m"] < dataframe["ema_slow_15m"]
		confirm_strong_trend_buy = dataframe["adx_15m"] > float(self.adx_thr_buy.value)
		confirm_strong_trend_sell = dataframe["adx_15m"] > float(self.adx_thr_sell.value)

		# Filter 1h (columns suffixed with _1h)
		filter_trend_up = dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]
		filter_trend_down = dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]
		filter_strong_trend_buy = dataframe["adx_1h"] > float(self.adx_thr_buy.value)
		filter_strong_trend_sell = dataframe["adx_1h"] > float(self.adx_thr_sell.value)

		aligned_bullish = main_trend_up & confirm_trend_up & filter_trend_up
		aligned_bearish = main_trend_down & confirm_trend_down & filter_trend_down

		# Volatility filter: ATR as percentage of price (more fair for different price levels)
		atr_pct = (dataframe["atr"] / dataframe["close"]) * 100  # Convert to percentage
		volatility_filter = atr_pct < float(self.atr_threshold.value)  # Default 5% threshold

		# Simplified buy/sell conditions inspired by Pine (no lookahead, candle-close only)
		buy_cond = (
			aligned_bullish
			& main_strong_trend_buy
			& confirm_strong_trend_buy
			& filter_strong_trend_buy
			& (dataframe["rsi"] > float(self.buy_rsi.value))
			& (dataframe["close"] > dataframe["open"])
			& volatility_filter  # Dynamic volatility filter
		)

		sell_cond = (
			aligned_bearish
			& main_strong_trend_sell
			& confirm_strong_trend_sell
			& filter_strong_trend_sell
			& (dataframe["rsi"] < float(self.sell_rsi.value))
			& (dataframe["close"] < dataframe["open"])
			& volatility_filter  # Dynamic volatility filter
		)

		dataframe.loc[buy_cond, "enter_long"] = 1
		dataframe.loc[sell_cond, "enter_short"] = 1

		return dataframe

	def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		# Use custom_stoploss for ATR-based exits; keep exit columns empty
		dataframe["exit_long"] = 0
		dataframe["exit_short"] = 0
		return dataframe

	# --- Protections ---
	@property
	def protections(self):
		return [
			{
				"method": "CooldownPeriod",
				"stop_duration_candles": 2,
			},
			{
				"method": "MaxDrawdown",
				"lookback_period_candles": 48,
				"trade_limit": 20,
				"stop_duration_candles": 12,
						"max_allowed_drawdown": 0.1,
			},
			{
				"method": "StoplossGuard",
				"lookback_period_candles": 24,
				"trade_limit": 4,
				"stop_duration_candles": 4,
				"only_per_pair": False,
				"only_per_side": False,
			},
			{
				"method": "LowProfitPairs",
				"lookback_period_candles": 6,
				"trade_limit": 2,
				"stop_duration_candles": 2,
				"required_profit": 0.02,
			},
		]

	# --- ATR-based stoploss respecting futures semantics ---
	def custom_stoploss(
		self,
		pair: str,
		trade,  # Trade
		current_time: datetime,
		current_rate: float,
		current_profit: float,
		**kwargs,
	) -> float:
		"""
		In futures, custom_stoploss must return RISK (fraction of stake) considering leverage.
		We'll compute stop from ATR distance at current candle and convert to percentage.
		"""
		# Need candle ATR around current bar; fallback to static stoploss if not available
		if trade is None or trade.open_date_utc is None:
			return self.stoploss

		try:
			# Get latest analyzed dataframe to read ATR
			dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
			if dataframe is None or dataframe.empty:
				return self.stoploss
			last = dataframe.iloc[-1]
			atr_val = float(last.get("atr", 0.0))
			if atr_val <= 0:
				return self.stoploss

			# Absolute stop distance in price units
			stop_distance = self.atr_multiplier * atr_val

			# Time-Stop: بعد از 12 بار (1 ساعت) اگر زیر breakeven، ببند
			import pandas as pd
			open_time = trade.open_date_utc
			elapsed_bars = int((current_time - open_time).total_seconds() / (5 * 60))  # 5m bars
			if elapsed_bars >= 12:
				return 0.0  # ببند

			# Dynamic stoploss: اگر current_profit > 0.02، stop را به breakeven + 0.005 منتقل کنید
			if current_profit > 0.02:
				# Move stop to breakeven + 0.005 (0.5% profit protection)
				breakeven_plus = 0.005
				return float(breakeven_plus)

			# Percentage move to stop (from current rate)
			move_pct = stop_distance / current_rate

			# Return as fraction of stake (positive for loss)
			return float(-move_pct)  # Negative for stoploss
		except Exception:
			# Fall back to static if anything goes wrong
			return self.stoploss

	# --- Risk-based position sizing targeting ~4% of equity per trade ---
	def custom_stake_amount(
		self,
		pair: str,
		current_time: datetime,
		current_rate: float,
		proposed_stake: float,
		min_stake: float | None,
		max_stake: float,
		leverage: float,
		entry_tag: str | None,
		side: str,
		**kwargs,
	) -> float:
		try:
			# Get latest analyzed dataframe to read ATR
			dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
			if dataframe is None or dataframe.empty:
				return proposed_stake
			last = dataframe.iloc[-1]
			atr_val = float(last.get("atr", 0.0))
			if atr_val <= 0 or current_rate <= 0:
				return proposed_stake

			# Absolute stop distance in price units
			stop_distance = self.atr_multiplier * atr_val
			if stop_distance <= 0:
				return proposed_stake

			# Percentage move to stop
			move_pct = stop_distance / current_rate
			if move_pct <= 0:
				return proposed_stake

			# Target absolute risk in stake currency (approximate equity * risk)
			# Use available stake as equity proxy respecting tradable_balance_ratio
			available_equity = self.wallets.get_total_stake_amount()
			risk_amount = max(0.0, self.risk_per_trade) * available_equity
			if risk_amount <= 0:
				return proposed_stake

			# Loss at stop ~ position_notional * move_pct = (stake * leverage) * move_pct
			desired_stake = risk_amount / max(1e-12, leverage * move_pct)

			# Safety cap: limit stake to a fraction of equity to avoid oversizing
			max_equity_cap = 0.05 * available_equity  # 5% of equity cap
			stake_cap = min(float(max_stake), max_equity_cap)

			# Respect exchange/account constraints and safety cap
			if min_stake is not None:
				desired_stake = max(desired_stake, float(min_stake))
			desired_stake = min(desired_stake, stake_cap)

			# Also never exceed a small multiple of proposed stake (gradual ramp-up)
			return float(min(desired_stake, proposed_stake * 2.0))
		except Exception:
			# Fall back to proposed if anything goes wrong
			return proposed_stake

	# --- Futures: leverage callback ---
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
		# Target leverage 3x, capped by exchange max
		return float(min(3.0, max_leverage))