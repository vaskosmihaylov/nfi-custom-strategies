# AwesomeEWOLambo Dual Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add AwesomeEWOLambo (longs) and AwesomeEWOLambo_Shorts to multi-strategy deployment

**Architecture:** Create shorts variant using conversion patterns (RSI/EMA inversions, tighter risk management), integrate both strategies into existing multi-bot infrastructure with ports 8108-8109, separate databases, NGINX routing, and environment configurations.

**Tech Stack:** FreqTrade, Docker Compose, NGINX, Python 3.12

---

## Analysis: AwesomeEWOLambo Strategy Structure

### Current Strategy Features (Longs Only)
- **Entry Signals** (4 modes):
  - `buy1ewo`: High EWO (>1.001) + Low RSI (<35) + Below MA
  - `buy2ewo`: Very high EWO (>-3.585) + Very low RSI_14 (<25) + Below MA
  - `lambo2`: Price below EMA_14 + Low RSI_4 (<44) + Low RSI_14 (<39)
  - `buyewolow`: Low EWO (<-2.289) + Low RSI_fast (<35) + Below MA

- **Risk Management**:
  - ROI: 2% → 1.2% over 180 minutes
  - Stoploss: -70% (very wide)
  - Trailing stop: Activates at +1.2% profit
  - Unclog: Exit at -4% after 10 days

- **Advanced Features**:
  - Position adjustment (DCA): Max 8 safety orders, 1.4x volume scale
  - Custom stoploss: 5% trailing from max_rate
  - Pump/dump protection: Volume spike detection
  - Protections: Cooldown, MaxDrawdown, StoplossGuard, LowProfitPairs

### Conversion Requirements for Shorts
- Invert RSI thresholds: `100 - value`
- Invert EMA offsets: `2 - value`
- Invert EWO logic: `> becomes <`, `< becomes >`
- More conservative ROI: 7% target (vs 2% for longs)
- Tighter stoploss: -18.9% (vs -70% for longs)
- Tighter unclog: 3 days (vs 10 days for longs)
- Max 4-8 short positions enforced

---

## Task 1: Create AwesomeEWOLambo_Shorts Strategy

**Files:**
- Read: `user_data/strategies/AwesomeEWOLambo/AwesomeEWOLambo.py`
- Create: `user_data/strategies/AwesomeEWOLambo_Shorts/AwesomeEWOLambo_Shorts.py`
- Create: `user_data/strategies/AwesomeEWOLambo_Shorts/__init__.py`

### Step 1: Read the long strategy completely

```bash
# Verify strategy location
ls -la user_data/strategies/AwesomeEWOLambo/
```

**Expected**: AwesomeEWOLambo.py exists

### Step 2: Create shorts strategy directory

```bash
mkdir -p user_data/strategies/AwesomeEWOLambo_Shorts
```

### Step 3: Create __init__.py for shorts directory

Create empty `user_data/strategies/AwesomeEWOLambo_Shorts/__init__.py`

### Step 4: Create AwesomeEWOLambo_Shorts.py with header and imports

```python
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, DecimalParameter, stoploss_from_open, IntParameter
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
from freqtrade.persistence import Trade
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta, timezone
from freqtrade.vendor.qtpylib.indicators import heikinashi, tdi, awesome_oscillator, sma
import math
import logging
from technical.indicators import ichimoku
logger = logging.getLogger(__name__)

def EWO(dataframe, ema_length=5, ema2_length=3):
  df = dataframe.copy()
  ema1 = ta.EMA(df, timeperiod=ema_length)
  ema2 = ta.EMA(df, timeperiod=ema2_length)
  emadif = (ema1 - ema2) / df['close'] * 100
  return emadif
```

### Step 5: Create class definition with inverted parameters

**Key Inversions**:
- `can_short = True` (enable shorts)
- `minimal_roi = {"0": 0.07}` (more conservative: 7% vs 2%)
- `stoploss = -0.189` (tighter: -18.9% vs -70%)
- `max_short_trades = 4` (new parameter)

**Parameter Inversions** (RSI: 100 - value):
```python
class AwesomeEWOLambo_Shorts(IStrategy):
  INTERFACE_VERSION: int = 3
  can_short = True

  minimal_roi = {"0": 0.07, "20": 0.05, "40": 0.03, "60": 0.015}
  stoploss = -0.189
  timeframe = '5m'

  # Short-specific
  max_short_trades = 4

  fast_ewo = 50
  slow_ewo = 200

  # Sell hyperspace params (INVERTED from buy_params)
  sell_params = {
    "base_nb_candles_sell": 12,
    "ewo_high": -1.001,  # Inverted sign
    "ewo_high_2": 3.585,  # Inverted sign
    "low_offset": 1.013,  # 2 - 0.987 = 1.013
    "low_offset_2": 1.058,  # 2 - 0.942 = 1.058
    "ewo_low": 2.289,  # Inverted sign
    "rsi_sell": 42,  # 100 - 58 = 42
    "lambo2_ema_14_factor": 1.019,  # 2 - 0.981 = 1.019
    "lambo2_rsi_14_limit": 61,  # 100 - 39 = 61
    "lambo2_rsi_4_limit": 56,  # 100 - 44 = 56
  }

  # Exit hyperspace params (INVERTED from sell_params)
  cover_params = {
    "base_nb_candles_cover": 22,
    "high_offset": 0.986,  # 2 - 1.014 = 0.986
    "high_offset_2": 0.99,  # 2 - 1.01 = 0.99
  }
```

### Step 6: Create parameter declarations (inverted)

```python
  base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
  base_nb_candles_cover = IntParameter(8, 20, default=cover_params['base_nb_candles_cover'], space='sell', optimize=False)
  low_offset = DecimalParameter(1.005, 1.015, default=sell_params['low_offset'], space='sell', optimize=True)
  low_offset_2 = DecimalParameter(1.01, 1.1, default=sell_params['low_offset_2'], space='sell', optimize=False)
  high_offset = DecimalParameter(0.985, 0.995, default=cover_params['high_offset'], space='sell', optimize=True)
  high_offset_2 = DecimalParameter(0.980, 0.990, default=cover_params['high_offset_2'], space='sell', optimize=True)

  ewo_low = DecimalParameter(8.0, 20.0, default=sell_params['ewo_low'], space='sell', optimize=True)
  ewo_high = DecimalParameter(-3.4, -3.0, default=sell_params['ewo_high'], space='sell', optimize=True)
  rsi_sell = IntParameter(30, 70, default=sell_params['rsi_sell'], space='sell', optimize=False)
  ewo_high_2 = DecimalParameter(-12.0, 6.0, default=sell_params['ewo_high_2'], space='sell', optimize=False)

  lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=sell_params['lambo2_ema_14_factor'], space='sell', optimize=True)
  lambo2_rsi_4_limit = IntParameter(40, 95, default=sell_params['lambo2_rsi_4_limit'], space='sell', optimize=True)
  lambo2_rsi_14_limit = IntParameter(40, 95, default=sell_params['lambo2_rsi_14_limit'], space='sell', optimize=True)

  # Keep same trailing stop config (managed by custom_stoploss)
  trailing_stop = True
  trailing_only_offset_is_reached = True
  trailing_stop_positive = 0.001
  trailing_stop_positive_offset = 0.012

  process_only_new_candles = True
  startup_candle_count = 96

  use_exit_signal = True
  exit_profit_only = True
  ignore_roi_if_entry_signal = False
  use_custom_stoploss = True

  # DCA settings (same as longs)
  initial_safety_order_trigger = -0.018
  max_safety_orders = 8
  safety_order_step_scale = 1.2
  safety_order_volume_scale = 1.4
  position_adjustment_enable = True
  threshold = 0.3

  slippage_protection = {
    'retries': 3,
    'max_slippage': -0.02
  }
```

### Step 7: Copy protections method (unchanged)

Protections work the same for shorts, copy directly from longs.

### Step 8: Copy pump_dump_protection method (unchanged)

Volume protection works same way for shorts.

### Step 9: Create custom_stoploss (INVERTED RSI logic)

```python
  def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                      current_rate: float, current_profit: float, **kwargs) -> float:
    max_reached_price = trade.max_rate
    trailing_percentage = 0.05
    new_stoploss = max_reached_price * (1 - trailing_percentage)
    return max(new_stoploss, self.stoploss)
```

**Note**: For shorts, this works correctly as-is because FreqTrade handles the rate inversions.

### Step 10: Create custom_exit (tighter unclog)

```python
  def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
    # Tighter unclog: 3 days vs 10 days for longs (shorts pay funding fees)
    if current_profit < -0.04 and (current_time - trade.open_date_utc).days >= 3:
      return 'unclog_short'
```

### Step 11: Create confirm_trade_entry (enforce short limits)

```python
  def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                         time_in_force: str, current_time: datetime, entry_tag: str,
                         side: str, **kwargs) -> bool:
    # Only allow shorts
    if side == "long":
      return False

    # Count open shorts
    short_count = sum(1 for t in Trade.get_trades_proxy(is_open=True) if t.is_short)

    # Enforce max shorts limit
    if short_count >= self.max_short_trades:
      return False

    return True
```

### Step 12: Copy confirm_trade_exit (INVERTED HMA/EMA logic)

```python
  def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                         rate: float, time_in_force: str, sell_reason: str,
                         current_time: datetime, **kwargs) -> bool:

    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last_candle = dataframe.iloc[-1]

    if (last_candle is not None):
      if (sell_reason in ['exit_signal']):
        # INVERTED: Block exit if price is ABOVE ema100 (shorts still profitable)
        if (last_candle['hma_50']*0.851 < last_candle['ema100']) and (last_candle['close'] > last_candle['ema100']*1.049):
          return False

    # Slippage protection (same logic)
    try:
      state = self.slippage_protection['__pair_retries']
    except KeyError:
      state = self.slippage_protection['__pair_retries'] = {}

    candle = dataframe.iloc[-1].squeeze()
    slippage = (rate / candle['close']) - 1
    if slippage < self.slippage_protection['max_slippage']:
      pair_retries = state.get(pair, 0)
      if pair_retries < self.slippage_protection['retries']:
        state[pair] = pair_retries + 1
        return False

    state[pair] = 0
    return True
```

### Step 13: Copy informative_pairs (unchanged)

```python
  def informative_pairs(self):
    return []
```

### Step 14: Copy populate_indicators (unchanged)

All indicators calculate the same way for shorts. Copy entire method.

### Step 15: Create populate_entry_trend (INVERTED for shorts)

**Signal Inversions**:
- `rsi_fast < 35` → `rsi_fast > 65` (100 - 35)
- `close < ma * offset` → `close > ma * offset`
- `EWO > high` → `EWO < -high`
- `EWO < low` → `EWO > -low`
- `enter_long` → `enter_short`
- `enter_tag` suffix: `buy_` → `short_`

```python
  def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    conditions = []

    # Short signal 1: EWO high (negative for shorts)
    sell1ewo = (
      (dataframe['rsi_fast'] > 65) &  # INVERTED: 100 - 35
      (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset.value)) &  # INVERTED: >
      (dataframe['EWO'] < self.ewo_high.value) &  # INVERTED: < (ewo_high is negative)
      (dataframe['rsi_14'] > self.rsi_sell.value) &  # INVERTED: >
      (dataframe['volume'] > 0) &
      (dataframe['close'] > (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.high_offset.value))  # INVERTED: >
    )
    dataframe.loc[sell1ewo, 'enter_tag'] += 'short_ewo_high_rsi_'
    conditions.append(sell1ewo)

    # Short signal 2: EWO very high with very high RSI
    sell2ewo = (
      (dataframe['rsi_fast'] > 65) &  # INVERTED
      (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset_2.value)) &  # INVERTED
      (dataframe['EWO'] < self.ewo_high_2.value) &  # INVERTED
      (dataframe['rsi_14'] > self.rsi_sell.value) &  # INVERTED
      (dataframe['volume'] > 0) &
      (dataframe['close'] > (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.high_offset.value)) &  # INVERTED
      (dataframe['rsi_14'] > 75)  # INVERTED: 100 - 25 = 75
    )
    dataframe.loc[sell2ewo, 'enter_tag'] += 'short_ewo2_high_rsi_'
    conditions.append(sell2ewo)

    # Lambo2 short: Price above EMA with high RSI
    lambo2_short = (
      (dataframe['close'] > (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &  # INVERTED: >
      (dataframe['rsi_4'] > int(self.lambo2_rsi_4_limit.value)) &  # INVERTED: >
      (dataframe['rsi_14'] > int(self.lambo2_rsi_14_limit.value))  # INVERTED: >
    )
    dataframe.loc[lambo2_short, 'enter_tag'] += 'short_lambo2_'
    conditions.append(lambo2_short)

    # Short signal: EWO low (positive for shorts = market overheated)
    sellewolow = (
      (dataframe['rsi_fast'] > 65) &  # INVERTED
      (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset.value)) &  # INVERTED
      (dataframe['EWO'] > self.ewo_low.value) &  # INVERTED: > (ewo_low is positive)
      (dataframe['volume'] > 0) &
      (dataframe['close'] > (dataframe[f'ma_cover_{self.base_nb_candles_cover.value}'] * self.high_offset.value))  # INVERTED
    )
    dataframe.loc[sellewolow, 'enter_tag'] += 'short_ewo_low_rsi_'
    conditions.append(sellewolow)

    if conditions:
      dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_short'] = 1  # INVERTED: enter_short

    # Pump/dump protection still applies
    dont_sell_conditions = []
    dont_sell_conditions.append((dataframe['pnd_volume_warn'] == -1))
    if dont_sell_conditions:
      dataframe.loc[reduce(lambda x, y: x | y, dont_sell_conditions), 'enter_short'] = 0

    return dataframe
```

### Step 16: Create populate_exit_trend (INVERTED for shorts)

```python
  def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    conditions = []

    # Exit short when downtrend signal appears
    coversignal = (
      (dataframe['ha_close'] < dataframe['ha_open']) &  # INVERTED: Red candle = exit short
      (dataframe['difference_signal'] <= -1.9)  # INVERTED: <= (threshold inverted)
    )
    dataframe.loc[coversignal, 'exit_tag'] += 'cover_signal'
    conditions.append(coversignal)

    if conditions:
      dataframe.loc[reduce(lambda x, y: x | y, conditions), 'exit_short'] = 1  # INVERTED: exit_short

    return dataframe
```

### Step 17: Copy adjust_trade_position (works same for shorts)

Position adjustment logic works identically for shorts - copy entire method.

### Step 18: Add docstring to class

```python
"""
AwesomeEWOLambo_Shorts Strategy

A shorts-only variant of AwesomeEWOLambo, designed to profit during overbought
conditions and bear markets using Elliott Wave Oscillator momentum detection.

Original AwesomeEWOLambo (Longs):
- Author: xNighbloodx Natblida
- Entry: 4 modes (EWO high/low, lambo2, RSI divergence)
- Risk: -70% stoploss, 2% ROI target
- DCA: 8 safety orders with 1.4x scaling

Strategy Concept (Shorts):
Enter shorts when market is overbought (high RSI, price above EMAs, negative EWO momentum).
Uses 4 entry modes mirroring the long strategy but inverted for bearish conditions.

Entry Conditions:
1. short_ewo_high_rsi: High RSI (>65) + Price above MA + Negative EWO
2. short_ewo2_high_rsi: Very high RSI (>75) + Price above MA + Very negative EWO
3. short_lambo2: Price above EMA_14 + High RSI_4/RSI_14
4. short_ewo_low_rsi: High RSI + Price above MA + Positive EWO (overheated)

Exit Conditions:
- Signal: Red candle + negative difference signal
- ROI: 7% → 1.5% over 60 minutes
- Stop Loss: -18.9%
- Trailing Stop: Activates at +1.2% profit
- Unclog: -4% after 3 days

Key Differences from Long Strategy:
- Tighter stop loss: -18.9% vs -70%
- More aggressive ROI: 7% vs 2%
- Shorter unclog: 3 days vs 10 days
- Max 4 short positions (vs 8+ longs)
- All RSI thresholds inverted (100 - value)
- All EMA offsets inverted (2 - value)

Author: Derived from AwesomeEWOLambo by xNighbloodx
Version: 1.0.0
"""
```

### Step 19: Verify shorts strategy file

```bash
python -m py_compile user_data/strategies/AwesomeEWOLambo_Shorts/AwesomeEWOLambo_Shorts.py
```

**Expected**: No syntax errors

### Step 20: Commit shorts strategy

```bash
git add user_data/strategies/AwesomeEWOLambo_Shorts/
git commit -m "feat: add AwesomeEWOLambo_Shorts strategy

- Created shorts variant of AwesomeEWOLambo
- Inverted RSI thresholds (100 - value)
- Inverted EMA offsets (2 - value)
- Inverted EWO logic for short entries
- More conservative risk: -18.9% stoploss, 7% ROI
- Tighter unclog: 3 days vs 10 days
- Max 4 short positions enforced
- 4 entry modes: short_ewo_high_rsi, short_ewo2_high_rsi, short_lambo2, short_ewo_low_rsi

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Environment Files

**Files:**
- Create: `env-files/awesomeewolambo.env`
- Create: `env-files/awesomeewolambo_shorts.env`

### Step 1: Create awesomeewolambo.env (longs)

```bash
# AwesomeEWOLambo Strategy (Longs Only)
FREQTRADE__BOT_NAME=Vasko_AwesomeEWOLambo
FREQTRADE__STRATEGY=AwesomeEWOLambo
FREQTRADE__TRADING_MODE=futures
FREQTRADE__MAX_OPEN_TRADES=8
FREQTRADE__STAKE_AMOUNT=unlimited
FREQTRADE__DRY_RUN=true

# Exchange Configuration
FREQTRADE__EXCHANGE__NAME=bybit
FREQTRADE__EXCHANGE__KEY=Your_Real_API_Key
FREQTRADE__EXCHANGE__SECRET=Your_Real_API_Secret

# API Server Configuration
FREQTRADE__API_SERVER__ENABLED=true
FREQTRADE__API_SERVER__LISTEN_IP_ADDRESS=0.0.0.0
FREQTRADE__API_SERVER__LISTEN_PORT=8080
FREQTRADE__API_SERVER__USERNAME=awesomeewolambo_user
FREQTRADE__API_SERVER__PASSWORD=awesomeewolambo_secure_password
FREQTRADE__API_SERVER__JWT_SECRET_KEY=AwesomeEWOLambo_JWTSecretKey2026
FREQTRADE__API_SERVER__WS_TOKEN=AwesomeEWOLambo_WSToken2026

# CORS Configuration (comma-separated, NO QUOTES, NO BRACKETS)
FREQTRADE__API_SERVER__CORS_ORIGINS=https://freq.gaiaderma.com,http://freq.gaiaderma.com
FREQTRADE__API_SERVER__FORWARDED_ALLOW_IPS=*
```

**Port**: 8108 (next available)

### Step 2: Create awesomeewolambo_shorts.env

```bash
# AwesomeEWOLambo_Shorts Strategy (Shorts Only)
FREQTRADE__BOT_NAME=Vasko_AwesomeEWOLambo_Shorts
FREQTRADE__STRATEGY=AwesomeEWOLambo_Shorts
FREQTRADE__TRADING_MODE=futures
FREQTRADE__MAX_OPEN_TRADES=4
FREQTRADE__STAKE_AMOUNT=unlimited
FREQTRADE__DRY_RUN=true

# Exchange Configuration
FREQTRADE__EXCHANGE__NAME=bybit
FREQTRADE__EXCHANGE__KEY=Your_Real_API_Key
FREQTRADE__EXCHANGE__SECRET=Your_Real_API_Secret

# API Server Configuration
FREQTRADE__API_SERVER__ENABLED=true
FREQTRADE__API_SERVER__LISTEN_IP_ADDRESS=0.0.0.0
FREQTRADE__API_SERVER__LISTEN_PORT=8080
FREQTRADE__API_SERVER__USERNAME=awesomeewolambo_shorts_user
FREQTRADE__API_SERVER__PASSWORD=awesomeewolambo_shorts_secure_password
FREQTRADE__API_SERVER__JWT_SECRET_KEY=AwesomeEWOLamboShorts_JWTSecretKey2026
FREQTRADE__API_SERVER__WS_TOKEN=AwesomeEWOLamboShorts_WSToken2026

# CORS Configuration
FREQTRADE__API_SERVER__CORS_ORIGINS=https://freq.gaiaderma.com,http://freq.gaiaderma.com
FREQTRADE__API_SERVER__FORWARDED_ALLOW_IPS=*
```

**Port**: 8109

### Step 3: Verify env files created

```bash
ls -la env-files/ | grep awesomeewolambo
```

**Expected**: Two files listed

### Step 4: Commit env files

```bash
git add env-files/awesomeewolambo*.env
git commit -m "feat: add environment files for AwesomeEWOLambo strategies

- awesomeewolambo.env: Longs configuration (port 8108, max 8 trades)
- awesomeewolambo_shorts.env: Shorts configuration (port 8109, max 4 trades)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Docker Compose Configuration

**Files:**
- Modify: `docker-compose-multi-strategies.yml`

### Step 1: Read current docker-compose file

```bash
tail -100 docker-compose-multi-strategies.yml
```

### Step 2: Add AwesomeEWOLambo service definition

Add after the last strategy (ETCG_Shorts on port 8103):

```yaml
  # AwesomeEWOLambo Strategy (Longs)
  freqtrade-awesomeewolambo:
    <<: *common-settings
    container_name: freqtrade-awesomeewolambo
    ports:
      - "127.0.0.1:8108:8080"
    env_file:
      - path: ./env-files/awesomeewolambo.env
        required: true
    command: >
      trade
      --config user_data/strategies/config.json
      --db-url sqlite:////freqtrade/user_data/awesomeewolambo-tradesv3.sqlite
      --log-file user_data/logs/awesomeewolambo.log
      --strategy-path user_data/strategies/AwesomeEWOLambo
      --strategy AwesomeEWOLambo
```

### Step 3: Add AwesomeEWOLambo_Shorts service definition

```yaml
  # AwesomeEWOLambo_Shorts Strategy (Shorts)
  freqtrade-awesomeewolambo_shorts:
    <<: *common-settings
    container_name: freqtrade-awesomeewolambo_shorts
    ports:
      - "127.0.0.1:8109:8080"
    env_file:
      - path: ./env-files/awesomeewolambo_shorts.env
        required: true
    command: >
      trade
      --config user_data/strategies/config.json
      --db-url sqlite:////freqtrade/user_data/awesomeewolambo_shorts-tradesv3.sqlite
      --log-file user_data/logs/awesomeewolambo_shorts.log
      --strategy-path user_data/strategies/AwesomeEWOLambo_Shorts
      --strategy AwesomeEWOLambo_Shorts
```

### Step 4: Validate docker-compose syntax

```bash
docker compose -f docker-compose-multi-strategies.yml config > /dev/null
```

**Expected**: No errors

### Step 5: Commit docker-compose changes

```bash
git add docker-compose-multi-strategies.yml
git commit -m "feat: add AwesomeEWOLambo strategies to docker-compose

- freqtrade-awesomeewolambo: Port 8108, longs only
- freqtrade-awesomeewolambo_shorts: Port 8109, shorts only
- Separate databases and log files per strategy

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update NGINX Configuration

**Files:**
- Modify: `nginx-freqtrade-multi.conf`

### Step 1: Read current NGINX configuration

```bash
grep -A 5 "location /etcg_shorts" nginx-freqtrade-multi.conf
```

### Step 2: Add AwesomeEWOLambo location block

Add after ETCG_Shorts section:

```nginx
  # AwesomeEWOLambo Strategy (Longs)
  location /awesomeewolambo/ {
    proxy_pass http://127.0.0.1:8108/;
    include /etc/nginx/freqtrade-proxy-common.conf;
  }

  # Health check for AwesomeEWOLambo
  location = /health/awesomeewolambo {
    proxy_pass http://127.0.0.1:8108/api/v1/ping;
    include /etc/nginx/freqtrade-proxy-common.conf;
  }
```

### Step 3: Add AwesomeEWOLambo_Shorts location block

```nginx
  # AwesomeEWOLambo_Shorts Strategy (Shorts)
  location /awesomeewolambo_shorts/ {
    proxy_pass http://127.0.0.1:8109/;
    include /etc/nginx/freqtrade-proxy-common.conf;
  }

  # Health check for AwesomeEWOLambo_Shorts
  location = /health/awesomeewolambo_shorts {
    proxy_pass http://127.0.0.1:8109/api/v1/ping;
    include /etc/nginx/freqtrade-proxy-common.conf;
  }
```

### Step 4: Test NGINX configuration

```bash
sudo nginx -t
```

**Expected**: Configuration test successful

### Step 5: Commit NGINX changes

```bash
git add nginx-freqtrade-multi.conf
git commit -m "feat: add NGINX routing for AwesomeEWOLambo strategies

- Added /awesomeewolambo/ route to port 8108
- Added /awesomeewolambo_shorts/ route to port 8109
- Added health check endpoints for both strategies

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update Deployment Script

**Files:**
- Modify: `deploy-multi-strategies.sh`

### Step 1: Add strategies to STRATEGIES array

Find the STRATEGIES array declaration and add:

```bash
STRATEGIES=(
  "nfi-x7"
  "bandtastic"
  "elliotv5_sma"
  "binclucmadv1"
  "nasosv4"
  "rsiquiv5"
  "elliotv5_sma_shorts"
  "e0v1e"
  "e0v1e_shorts"
  "ei4_t4c0s_v2_2"
  "ei4_t4c0s_v2_2_shorts"
  "etcg"
  "etcg_shorts"
  "cluchanix_hhll"
  "cluchanix_hhll_shorts"
  "awesomeewolambo"           # NEW
  "awesomeewolambo_shorts"    # NEW
)
```

### Step 2: Add port mappings

Find port mapping section and add:

```bash
declare -A STRATEGY_PORTS=(
  ["nfi-x7"]="8080"
  # ... existing mappings ...
  ["etcg_shorts"]="8103"
  ["cluchanix_hhll"]="8106"
  ["cluchanix_hhll_shorts"]="8107"
  ["awesomeewolambo"]="8108"           # NEW
  ["awesomeewolambo_shorts"]="8109"    # NEW
)
```

### Step 3: Test deployment script syntax

```bash
bash -n deploy-multi-strategies.sh
```

**Expected**: No syntax errors

### Step 4: Commit deployment script changes

```bash
git add deploy-multi-strategies.sh
git commit -m "feat: add AwesomeEWOLambo strategies to deployment script

- Added awesomeewolambo (port 8108)
- Added awesomeewolambo_shorts (port 8109)
- Updated STRATEGIES array and port mappings

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `MULTI_STRATEGY_SETUP.md`

### Step 1: Add strategies to architecture diagram

Update line 19-33 to include:

```
                           ├── AwesomeEWOLambo (Port 8108)
                           └── AwesomeEWOLambo_Shorts (Port 8109)
```

### Step 2: Add to bot configuration table

Add to table starting at line 131:

```markdown
| **AwesomeEWOLambo** | `Vasko_AwesomeEWOLambo` | `http://freq.gaiaderma.com/awesomeewolambo` | `awesomeewolambo_user` | `awesomeewolambo_secure_password` |
| **AwesomeEWOLambo_Shorts** | `Vasko_AwesomeEWOLambo_Shorts` | `http://freq.gaiaderma.com/awesomeewolambo_shorts` | `awesomeewolambo_shorts_user` | `awesomeewolambo_shorts_secure_password` |
```

### Step 3: Update port allocation table

Add to port allocation table:

```markdown
- AwesomeEWOLambo: 8108
- AwesomeEWOLambo_Shorts: 8109
```

### Step 4: Update database list

Add to database section:

```markdown
- `awesomeewolambo-tradesv3.sqlite`
- `awesomeewolambo_shorts-tradesv3.sqlite`
```

### Step 5: Add health check endpoints

Add to health monitoring section:

```bash
curl http://127.0.0.1:8108/api/v1/ping  # AwesomeEWOLambo
curl http://127.0.0.1:8109/api/v1/ping  # AwesomeEWOLambo_Shorts

# Test through NGINX
curl http://freq.gaiaderma.com/awesomeewolambo/api/v1/ping
curl http://freq.gaiaderma.com/awesomeewolambo_shorts/api/v1/ping
```

### Step 6: Update validation tests section

Add to validation tests:

```bash
curl http://freq.gaiaderma.com/awesomeewolambo/api/v1/ping
curl http://freq.gaiaderma.com/awesomeewolambo_shorts/api/v1/ping
curl http://freq.gaiaderma.com/health/awesomeewolambo
curl http://freq.gaiaderma.com/health/awesomeewolambo_shorts
```

### Step 7: Commit documentation changes

```bash
git add MULTI_STRATEGY_SETUP.md
git commit -m "docs: add AwesomeEWOLambo strategies to setup guide

- Added architecture diagram entries
- Added bot configuration table entries
- Added port allocation (8108, 8109)
- Added health check endpoints
- Added validation test commands

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Strategy Status Memory

**Files:**
- Update memory: `strategy_implementation_status`

### Step 1: Read current memory

```bash
mcp__serena__read_memory strategy_implementation_status
```

### Step 2: Add new strategies to table

Update the status table to include:

```markdown
| 13 | **awesomeewolambo** | **8108** | **✅ NEW - Longs EWO/DCA** | **-** |
| 14 | **awesomeewolambo_shorts** | **8109** | **✅ NEW - Shorts EWO/DCA** | **-** |
```

### Step 3: Add to env files list

```markdown
  - awesomeewolambo.env (NEW - EWO momentum + DCA)
  - awesomeewolambo_shorts.env (NEW - EWO shorts + DCA)
```

### Step 4: Add strategy details section

Add new section:

```markdown
### AwesomeEWOLambo Strategy (Longs - Port 8108) - January 2026
- **Created**: January 2026
- **Version**: 1.0.0
- **Strategy**: EWO-based multi-entry with DCA/safety orders
- **Original Author**: xNighbloodx Natblida
- **Features**:
  - 4 entry modes: buy1ewo, buy2ewo, lambo2, buyewolow
  - Elliott Wave Oscillator momentum detection
  - Position adjustment: 8 safety orders with 1.4x volume scaling
  - Pump/dump protection via volume analysis
  - Comprehensive protections (Cooldown, MaxDrawdown, StoplossGuard, LowProfitPairs)
  - ROI: 2% → 1.2% over 180 minutes
  - Stoploss: -70% (wide, managed by trailing)
  - Trailing stop: +1.2% offset, 5% trailing from max
  - Unclog: -4% after 10 days
  - Max 8 open trades
- **Port**: 8108
- **Database**: `awesomeewolambo-tradesv3.sqlite`

### AwesomeEWOLambo_Shorts Strategy (Shorts - Port 8109) - January 2026
- **Created**: January 2026
- **Version**: 1.0.0
- **Strategy**: Shorts variant of AwesomeEWOLambo
- **Features**:
  - 4 inverted entry modes: short_ewo_high_rsi, short_ewo2_high_rsi, short_lambo2, short_ewo_low_rsi
  - All RSI thresholds inverted (100 - value)
  - All EMA offsets inverted (2 - value)
  - More aggressive ROI: 7% → 1.5% over 60 minutes
  - Tighter stoploss: -18.9% (vs -70% longs)
  - Tighter unclog: 3 days (vs 10 days longs)
  - Max 4 short positions enforced via confirm_trade_entry()
  - Same DCA settings: 8 safety orders, 1.4x scaling
  - Same protections as long strategy
- **Port**: 8109
- **Database**: `awesomeewolambo_shorts-tradesv3.sqlite`
```

### Step 5: Update port allocation table

Add to port table:

```markdown
| 8108 | awesomeewolambo | **Longs only** | - | ✅ Active |
| 8109 | awesomeewolambo_shorts | **Shorts only** | - | ✅ Active |
```

### Step 6: Update quick commands

```bash
# Health check all active strategies (16 total)
for s in nfi-x7 bandtastic elliotv5_sma binclucmadv1 nasosv4 rsiquiv5 elliotv5_sma_shorts e0v1e e0v1e_shorts ei4_t4c0s_v2_2 ei4_t4c0s_v2_2_shorts etcg etcg_shorts cluchanix_hhll cluchanix_hhll_shorts awesomeewolambo awesomeewolambo_shorts; do
  curl -s http://freq.gaiaderma.com/$s/api/v1/ping | grep -q pong && echo "$s ✅" || echo "$s ❌"
done
```

### Step 7: Update memory file

Use `mcp__serena__edit_memory` or manually update the memory file.

---

## Task 8: Test and Verify

### Step 1: Start containers in dry-run mode

```bash
docker compose -f docker-compose-multi-strategies.yml up -d freqtrade-awesomeewolambo freqtrade-awesomeewolambo_shorts
```

**Expected**: Both containers start successfully

### Step 2: Check container status

```bash
docker compose -f docker-compose-multi-strategies.yml ps | grep awesomeewolambo
```

**Expected**: Both containers "Up" and healthy

### Step 3: Check container logs

```bash
docker compose -f docker-compose-multi-strategies.yml logs freqtrade-awesomeewolambo | tail -50
docker compose -f docker-compose-multi-strategies.yml logs freqtrade-awesomeewolambo_shorts | tail -50
```

**Expected**: No errors, strategies loaded successfully

### Step 4: Test direct API access

```bash
curl http://127.0.0.1:8108/api/v1/ping
curl http://127.0.0.1:8109/api/v1/ping
```

**Expected**: Both return `{"status":"pong"}`

### Step 5: Reload NGINX configuration

```bash
sudo nginx -t && sudo nginx -s reload
```

**Expected**: Configuration test passes, reload successful

### Step 6: Test NGINX proxied access

```bash
curl http://freq.gaiaderma.com/awesomeewolambo/api/v1/ping
curl http://freq.gaiaderma.com/awesomeewolambo_shorts/api/v1/ping
```

**Expected**: Both return `{"status":"pong"}`

### Step 7: Test health endpoints

```bash
curl http://freq.gaiaderma.com/health/awesomeewolambo
curl http://freq.gaiaderma.com/health/awesomeewolambo_shorts
```

**Expected**: Both return `{"status":"pong"}`

### Step 8: Verify database files created

```bash
ls -lh user_data/*.sqlite | grep awesomeewolambo
```

**Expected**: Two database files listed

### Step 9: Verify log files created

```bash
ls -lh user_data/logs/ | grep awesomeewolambo
```

**Expected**: Two log files listed

### Step 10: Check FreqUI connectivity

1. Open FreqUI: `http://freq.gaiaderma.com`
2. Add bot: `Vasko_AwesomeEWOLambo`
   - URL: `http://freq.gaiaderma.com/awesomeewolambo`
   - Username: `awesomeewolambo_user`
   - Password: `awesomeewolambo_secure_password`
3. Add bot: `Vasko_AwesomeEWOLambo_Shorts`
   - URL: `http://freq.gaiaderma.com/awesomeewolambo_shorts`
   - Username: `awesomeewolambo_shorts_user`
   - Password: `awesomeewolambo_shorts_secure_password`

**Expected**: Both bots connect successfully, show status

---

## Task 9: Production Preparation

### Step 1: Update API credentials in env files

**IMPORTANT**: Before live trading, update both env files:

```bash
# In awesomeewolambo.env and awesomeewolambo_shorts.env
FREQTRADE__EXCHANGE__KEY=Your_REAL_Bybit_API_Key
FREQTRADE__EXCHANGE__SECRET=Your_REAL_Bybit_API_Secret
FREQTRADE__DRY_RUN=false  # ONLY after testing!
```

### Step 2: Test with small stake in dry-run

Monitor for 24-48 hours in dry-run mode to verify:
- Entry signals generating correctly
- Exit signals working
- No errors in logs
- DCA orders triggering appropriately
- Max trades limits enforced

### Step 3: Enable live trading gradually

1. Start with smaller `STAKE_AMOUNT` (e.g., 50 USDT per trade)
2. Monitor for 1 week
3. Gradually increase if performing well

### Step 4: Monitor key metrics

**For AwesomeEWOLambo (Longs)**:
- Win rate target: >60%
- Avg profit target: >0.5%
- Monitor DCA effectiveness
- Watch for excessive safety orders

**For AwesomeEWOLambo_Shorts**:
- Win rate target: >55% (shorts typically harder)
- Avg profit target: >0.3%
- Monitor funding fee impact
- Ensure 4-trade limit enforced

---

## Summary

**Total Files Modified**: 5
- `docker-compose-multi-strategies.yml`
- `nginx-freqtrade-multi.conf`
- `deploy-multi-strategies.sh`
- `MULTI_STRATEGY_SETUP.md`
- Memory: `strategy_implementation_status`

**Total Files Created**: 5
- `user_data/strategies/AwesomeEWOLambo_Shorts/AwesomeEWOLambo_Shorts.py`
- `user_data/strategies/AwesomeEWOLambo_Shorts/__init__.py`
- `env-files/awesomeewolambo.env`
- `env-files/awesomeewolambo_shorts.env`
- This plan document

**Port Allocation**:
- 8108: AwesomeEWOLambo (Longs)
- 8109: AwesomeEWOLambo_Shorts (Shorts)

**Key Features**:
- 4 entry modes per strategy (EWO-based momentum detection)
- DCA/safety orders: 8 orders max, 1.4x volume scaling
- Comprehensive protections (Cooldown, MaxDrawdown, StoplossGuard, LowProfitPairs)
- Pump/dump volume protection
- Slippage protection
- Separate databases and logs
- NGINX reverse proxy routing
- Health check endpoints

**Testing Checklist**:
- [ ] Shorts strategy syntax valid
- [ ] Docker containers start successfully
- [ ] API endpoints respond
- [ ] NGINX routing works
- [ ] Health checks pass
- [ ] FreqUI connects to both bots
- [ ] Logs show no errors
- [ ] Database files created
- [ ] Entry signals generating (dry-run)
- [ ] Exit signals working (dry-run)
- [ ] DCA orders triggering (dry-run)
- [ ] Max trades limits enforced

**Next Steps After Implementation**:
1. Test in dry-run for 24-48 hours
2. Update API credentials for live trading
3. Start with small stake amounts
4. Monitor win rates and profitability
5. Adjust parameters if needed (hyperopt)

---

**Plan Complete** ✅

Would you like to proceed with:
1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach would you prefer?
