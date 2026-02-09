# Hyperparameter Optimization (Hyperopt) Guide

Step-by-step guide for running hyperopt on individual strategies using the multi-strategy Docker setup.

## Prerequisites

- Docker and docker-compose installed
- The strategy file exists in `user_data/strategies/`
- Exchange API keys configured in env files (needed for data download)
- Historical data downloaded (Step 1)

## Config File

The hyperopt config (`user_data/strategies/config-hyperopt.json`) is a self-contained minimal config
designed specifically for hyperopt/backtesting. It includes:

- `dry_run`, `stake_currency`, `stake_amount`, `entry_pricing`, `exit_pricing`
- Chains: `trading_mode-futures.json`, `blacklist-bybit.json`, `pairlist-backtest-static-bybit-futures-usdt.json`

It intentionally does **NOT** include `exampleconfig.json` or `exampleconfig_secret.json` to avoid
overriding strategy-specific settings (like `timeframe`). Each strategy's own timeframe is used
automatically. Exchange keys and name come from the Docker container's environment variables.

The pairs for download and backtesting are defined in `configs/pairlist-backtest-static-bybit-futures-usdt.json`
(28 top liquid Bybit futures pairs) and are read automatically by FreqTrade.

## Quick Reference (Copy-Paste Commands)

```bash
# === STEP 1: Download data (run ONCE, reuse for all strategies) ===
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  download-data \
  --config user_data/strategies/config-hyperopt.json \
  --exchange bybit \
  --timeframes 5m 15m 1h \
  --days 90 \
  --trading-mode futures

# === STEP 2: Run hyperopt ===
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  hyperopt \
  --config user_data/strategies/config-hyperopt.json \
  --strategy HarmonicDivergence_fix \
  --strategy-path user_data/strategies/Harmonic-divergence \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces roi stoploss trailing \
  --timerange 20251101-20260201 \
  --epochs 500 \
  -j -1

# === STEP 3: View results ===
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  hyperopt-show \
  --config user_data/strategies/config-hyperopt.json \
  --best
```

## Detailed Step-by-Step

### Step 1: Download Historical Data

Hyperopt needs historical OHLCV data. You only need to download this once - all strategies
share the same data directory. Pairs are read automatically from the config's `StaticPairList`.

```bash
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  download-data \
  --config user_data/strategies/config-hyperopt.json \
  --exchange bybit \
  --timeframes 5m 15m 1h \
  --days 90 \
  --trading-mode futures
```

**Parameters explained:**
- No `--pairs` needed - pairs are read from the config's `StaticPairList` automatically
- `--timeframes 5m 15m 1h` - Download all timeframes used by strategies (5m for most, 15m for HarmonicDivergence, 1h for BB_RPB)
- `--days 90` - 3 months of data (good balance of speed vs coverage)
- `--trading-mode futures` - Download futures data, not spot
- Any container can be used for download (data is shared via volume mount)

**Note:** Data is saved to `user_data/data/bybit/futures/` and shared across all containers
via the volume mount.

### Step 2: Run Hyperopt

#### For HarmonicDivergence (Longs):

```bash
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  hyperopt \
  --config user_data/strategies/config-hyperopt.json \
  --strategy HarmonicDivergence_fix \
  --strategy-path user_data/strategies/Harmonic-divergence \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces roi stoploss trailing \
  --timerange 20251101-20260201 \
  --epochs 500 \
  -j -1
```

#### For HarmonicDivergence_Shorts:

```bash
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence_shorts \
  hyperopt \
  --config user_data/strategies/config-hyperopt.json \
  --strategy HarmonicDivergence_Shorts \
  --strategy-path user_data/strategies/Harmonic-divergence \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces roi stoploss trailing \
  --timerange 20251101-20260201 \
  --epochs 500 \
  -j -1
```

**Parameters explained:**
- `run --rm` - Creates a temporary container (doesn't affect the live trading container)
- `--config` - Self-contained hyperopt config (does NOT override strategy timeframe or settings)
- `--strategy` - The strategy class name
- `--strategy-path` - Directory containing the strategy file
- `--hyperopt-loss` - The objective function to minimize (see options below)
- `--spaces roi stoploss trailing` - What to optimize:
  - `roi` - Minimal ROI table (tiered take-profit targets)
  - `stoploss` - Stop loss percentage
  - `trailing` - Trailing stop parameters (positive, offset, only_offset)
- `--timerange` - Date range for backtesting (YYYYMMDD-YYYYMMDD)
- `--epochs 500` - Number of optimization iterations (more = better but slower)
- `-j -1` - Use all CPU cores

#### Available Loss Functions:

| Loss Function | Best For |
|---|---|
| `SharpeHyperOptLossDaily` | **Recommended** - Balances returns vs risk |
| `SortinoHyperOptLossDaily` | Penalizes downside volatility only |
| `MaxDrawDownHyperOptLoss` | Minimizes maximum drawdown |
| `CalmarHyperOptLoss` | Return/max-drawdown ratio |
| `ProfitDrawDownHyperOptLoss` | Profit weighted by drawdown |
| `OnlyProfitHyperOptLoss` | Pure profit maximization (can overfit) |

### Step 3: View Results

```bash
# Show the best result
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  hyperopt-show \
  --config user_data/strategies/config-hyperopt.json \
  --best

# Show top 10 results
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  hyperopt-show \
  --config user_data/strategies/config-hyperopt.json \
  --best -n 10

# Show a specific result by index
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  hyperopt-show \
  --config user_data/strategies/config-hyperopt.json \
  -n 42
```

### Step 4: Interpret the Output

Hyperopt will print results like:

```
Best result:

    42/500:    120 trades. 78/32/10 Wins/Draws/Losses.
    Avg profit: 1.25%. Median profit: 0.80%.
    Total profit: 0.15000000 USDT (15.00%).
    Avg duration: 4:30:00.
    Objective: -0.12345

    # ROI table:
    minimal_roi = {
        "0": 0.035,
        "30": 0.025,
        "60": 0.015,
        "120": 0.005
    }

    # Stoploss:
    stoploss = -0.025

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
```

**Key metrics to evaluate:**
- **Win Rate** > 60% is good
- **Avg profit** > 0.5% per trade
- **Profit Factor** (total wins / total losses) > 1.5
- **Max Drawdown** < 20% (especially important with 3x leverage)
- **Number of trades** > 50 (too few = unreliable)

### Step 5: Apply Results to Strategy

After finding good parameters, update the strategy file:

```python
# In HarmonicDivergence.py - replace the existing values:
minimal_roi = {
    "0": 0.035,
    "30": 0.025,
    "60": 0.015,
    "120": 0.005
}
stoploss = -0.025
trailing_stop = True
trailing_stop_positive = 0.01
trailing_stop_positive_offset = 0.02
trailing_only_offset_is_reached = True
```

### Step 6: Validate with Backtest

Before going live, always validate with a full backtest:

```bash
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-harmonicdivergence \
  backtesting \
  --config user_data/strategies/config-hyperopt.json \
  --strategy HarmonicDivergence_fix \
  --strategy-path user_data/strategies/Harmonic-divergence \
  --timerange 20251101-20260201
```

### Step 7: Restart the Live Container

After updating the strategy file with new parameters:

```bash
# Restart the specific strategy container
./deploy-multi-strategies.sh restart harmonicdivergence

# Or for shorts:
./deploy-multi-strategies.sh restart harmonicdivergence_shorts
```

## Running Hyperopt for ANY Strategy

The pattern is the same for all strategies. Just change:
1. The container name (`freqtrade-XXXX`)
2. The `--strategy` name (class name)
3. The `--strategy-path` (directory containing the strategy file)

The strategy's own timeframe is used automatically (no override from config).
Data only needs to be downloaded once (Step 1) - shared across all containers.

### Examples for Other Strategies:

```bash
# ETCG (5m timeframe)
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-etcg \
  hyperopt \
  --config user_data/strategies/config-hyperopt.json \
  --strategy Etcg \
  --strategy-path user_data/strategies/ETCG \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces roi stoploss trailing \
  --timerange 20251101-20260201 \
  --epochs 500 \
  -j -1

# AwesomeEWOLambo (5m timeframe)
docker compose -f docker-compose-multi-strategies.yml run --rm \
  freqtrade-awesomeewolambo \
  hyperopt \
  --config user_data/strategies/config-hyperopt.json \
  --strategy AwesomeEWOLambo \
  --strategy-path user_data/strategies/AwesomeEWOLambo \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces roi stoploss trailing \
  --timerange 20251101-20260201 \
  --epochs 500 \
  -j -1
```

## Tips

1. **Start with 100-200 epochs** to get quick results, then increase to 500-1000 for fine-tuning
2. **Use `SharpeHyperOptLossDaily`** as default - it balances profit with consistency
3. **With 3x leverage**, effective stoploss is 3x the value. A -0.02 stoploss = -6% actual loss
4. **Never trust hyperopt blindly** - always validate with a separate backtest timerange
5. **Overfitting warning**: If hyperopt gives amazing results but backtest on different dates
   is poor, reduce epochs or use a wider timerange
6. **Download data once**, reuse for all strategies (shared volume mount)
7. **Data includes 5m, 15m, 1h** - covers all strategy timeframes in one download
8. **To add/remove pairs**: Edit `configs/pairlist-backtest-static-bybit-futures-usdt.json`

## Timeframe Reference

| Strategy | Timeframe | Notes |
|----------|-----------|-------|
| HarmonicDivergence | 15m | Uses its own 15m timeframe |
| ETCG / ETCG_Shorts | 5m | |
| E0V1E / E0V1E_Shorts | 5m | |
| AwesomeEWOLambo | 5m | |
| BB_RPB_TSL_RNG_TBS_GOLD | 5m | Also uses 1h informative |
| KamaFama | 5m | |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `StaticPairList requires pair_whitelist` | A chained config overrides `pair_whitelist` to empty | Ensure `config-hyperopt.json` does NOT chain `exampleconfig_secret.json` |
| `No history for X found` | Data not downloaded for the strategy's timeframe | Re-run Step 1 with all needed timeframes (`5m 15m 1h`) |
| `Override strategy 'timeframe'` | A config file overrides the strategy's timeframe | Ensure `config-hyperopt.json` does NOT chain `exampleconfig.json` |
| `KeyError: 'exit_pricing'` | Missing pricing config | Already included in `config-hyperopt.json` |
| `Pair X is not compatible with exchange` | Delisted pair in pairlist | Remove from `configs/pairlist-backtest-static-bybit-futures-usdt.json` |
| `zsh: no such file or directory` on pairs | Line breaks in `--pairs` argument | Don't use `--pairs` - pairs are read from config automatically |

**Last Updated**: February 9, 2026
