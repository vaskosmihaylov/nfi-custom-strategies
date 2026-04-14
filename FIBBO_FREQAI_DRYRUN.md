# Fibbo FreqAI Dry-Run Setup

This stack isolates the `Fibbo` strategy on a dedicated `stable_freqai` Docker Compose file.

## Files

- `docker-compose-fibbo-freqai.yml`: dedicated dry-run compose stack
- `docker/Dockerfile.freqai`: custom image based on `freqtradeorg/freqtrade:stable_freqai`
- `configs/fibbo-backtest.json`: Fibbo backtest entrypoint using the static Bybit futures pairlist
- `configs/pairlist-backtest-top5-bybit-futures-usdt.json`: reduced validation pairlist with `BTC`, `ETH`, `SOL`, `BNB`, `XRP`
- `configs/blacklist-bybit-fibbo-backtest.json`: backtest-specific Bybit blacklist with pairs that currently have no leverage tiers in Freqtrade futures backtests
- `configs/fibbo-freqai.json`: Fibbo-specific FreqAI configuration
- `configs/fibbo-freqai-backtest-override.json`: backtest-only FreqAI override to reduce the run to a single training window
- `env-files/fibbo_freqai.env`: runtime environment overrides
- `user_data/strategies/FibboFreqAI/Fibbo.py`: isolated strategy wrapper

## Validation

Render the compose file:

```bash
docker compose -f docker-compose-fibbo-freqai.yml config
```

List strategies from the isolated path:

```bash
docker compose -f docker-compose-fibbo-freqai.yml run --rm fibbo-freqai \
  list-strategies \
  --config user_data/strategies/config.json \
  --config configs/fibbo-freqai.json \
  --strategy-path user_data/strategies/FibboFreqAI
```

## Historical Data

Download Bybit futures data for the Fibbo base and informative timeframes:

```bash
docker compose -f docker-compose-fibbo-freqai.yml run --rm fibbo-freqai \
  download-data \
  --prepend \
  --config configs/fibbo-backtest.json \
  --config configs/fibbo-freqai.json \
  --exchange bybit \
  --trading-mode futures \
  --data-format-ohlcv feather \
  --timeframes 15m 1h 4h \
  --timerange 20260220-20260410
```

The Fibbo FreqAI config uses Bybit futures correlation pairs `BTC/USDT:USDT` and `ETH/USDT:USDT`.
The backtest config keeps `configs/pairlist-backtest-static-bybit-futures-usdt.json` as the source pairlist and applies `configs/blacklist-bybit-fibbo-backtest.json` to exclude pairs that currently fail futures backtests because Bybit leverage tiers are unavailable in Freqtrade.

Check available timeranges after download:

```bash
docker compose -f docker-compose-fibbo-freqai.yml run --rm fibbo-freqai \
  list-data \
  --config configs/fibbo-backtest.json \
  --config configs/fibbo-freqai.json \
  --show-timerange \
  --data-format-ohlcv feather
```

## Backtest

Run a Fibbo backtest against the static Bybit futures whitelist:

```bash
docker compose -f docker-compose-fibbo-freqai.yml run --rm fibbo-freqai \
  backtesting \
  --config configs/fibbo-backtest.json \
  --config configs/fibbo-freqai.json \
  --config configs/fibbo-freqai-backtest-override.json \
  --strategy-path user_data/strategies/FibboFreqAI \
  --strategy Fibbo \
  --cache none \
  --timerange 20260401-20260410
```

The backtest override changes `freqai.backtest_period_days` to `10`, which keeps the full static Bybit futures pairlist but reduces the FreqAI run to a single training window for practical validation.

For a faster smoke test, override the pairlist with the reduced top-5 Bybit futures config:

```bash
docker compose -f docker-compose-fibbo-freqai.yml run --rm fibbo-freqai \
  backtesting \
  --cache none \
  --config configs/fibbo-backtest.json \
  --config configs/fibbo-freqai.json \
  --config configs/fibbo-freqai-backtest-override.json \
  --config configs/pairlist-backtest-top5-bybit-futures-usdt.json \
  --strategy-path user_data/strategies/FibboFreqAI \
  --strategy Fibbo \
  --timerange 20260401-20260410
```

## Start the Bot

```bash
docker compose -f docker-compose-fibbo-freqai.yml up -d --build
```

## PRD NGINX and FreqUI

The canonical external base URL for this standalone service is:

```text
http://freq.gaiaderma.com/fibbo_freqai
```

Use that exact base URL in FreqUI. Do not append `/api/v1/`.

Suggested FreqUI bot entry:

- Bot Name: `Vasko_Fibbo_FreqAI`
- API URL: `http://freq.gaiaderma.com/fibbo_freqai`
- Username: `fibbo_freqai_user`
- Password: `fibbo_freqai_secure_password`

Health check through NGINX:

```bash
curl http://freq.gaiaderma.com/health/fibbo_freqai
```

Direct local health check on the server:

```bash
curl http://127.0.0.1:8140/api/v1/ping
```
