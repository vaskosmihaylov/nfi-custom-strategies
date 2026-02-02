# Market Regime Switching System - Design Document

**Date:** 2026-02-02
**Status:** Approved
**Implementation:** Hybrid automatic detection with manual override

---

## 1. Executive Summary

Implement a market regime detection and switching system that automatically enables/disables long and short trading strategies based on market conditions (BULL/BEAR/SIDEWAYS), with manual override capability.

### Key Goals

1. **Predictive Detection**: Use leading indicators to detect bear markets BEFORE they fully develop
2. **Risk Management**: Only run strategies proven to work in current market regime
3. **Flexibility**: Support manual override when needed
4. **Production Ready**: Deploy 4 strategy pairs (AwesomeEWOLambo, ETCG, e0v1e, ei4_t4c0s_v2_2)

### Current Status (Feb 2026)

- Market: Bearish consolidation since October 2025
- Performance: SHORT strategies profitable, LONG strategies losing
- Strategy pairs analyzed: 4 pairs ready for regime switching

---

## 2. Market Regime Detection Strategy

### Leading Indicators Framework

#### Tier 1 - Early Warning (Days to Weeks Ahead)

**1. Funding Rates** (Most Responsive)
- Source: Bybit/Binance API
- Update: Every 8 hours
- Signals:
  - `>0.01%` = Dangerous leverage buildup → Bear market coming
  - `0.005% - 0.01%` = Neutral
  - `<0.005%` = Capitulation/bottom → Bull market coming
- Weight: 35%

**2. MVRV Z-Score** (Predicts tops within 2 weeks)
- Source: Glassnode/CoinGlass API
- Update: Daily
- Signals:
  - `>3.7` = Extreme overvaluation → Sell signal
  - `2.5 - 3.7` = Elevated → Caution
  - `1.0 - 2.5` = Fair value → Normal
  - `<1.0` = Undervaluation → Accumulation
- Current: 1.2 (near bottom)
- Weight: 30%

**3. ETF Flows** (Institutional Money)
- Source: Bloomberg/CoinGlass API
- Update: Daily
- Signals:
  - Net inflows → Bullish
  - Net outflows → Bearish
- Current: Net sellers since Nov 2025
- Weight: 20%

#### Tier 2 - Confirmation Signals

**4. Open Interest + Price Divergence**
- Source: Exchange APIs
- Signals:
  - Rising OI + Falling Price = Bear warning
  - Falling OI + Rising Price = Reversal warning
- Weight: 10%

**5. NUPL (Net Unrealized Profit/Loss)**
- Source: Glassnode API
- Signals:
  - `>0.75` = Euphoria → Top near
  - `0.50 - 0.75` = Greed
  - `0 - 0.50` = Neutral
  - `<0` = Capitulation → Bottom near
- Current: 0.522 (overextended)
- Weight: 5%

### Regime Classification

```python
# Weighted score: 0-100 (0=extreme bear, 100=extreme bull)

if score >= 70:
    regime = "BULL"
    # Enable: Long strategies
    # Disable: Short strategies (except proven pairs)

elif score <= 30:
    regime = "BEAR"
    # Enable: Short strategies
    # Disable: Long strategies (except proven pairs)

else:  # 30 < score < 70
    regime = "SIDEWAYS"
    # Enable: Both long and short for approved pairs
    # Disable: Unproven strategies
```

### Sources
- [Bitcoin Bear Market Indicators 2026](https://beincrypto.com/bitcoin-bear-market-indicators-2026/)
- [MVRV Z-Score Analysis](https://www.bitcoinmagazinepro.com/charts/mvrv-zscore/)
- [Derivatives Market Signals](https://web3.gate.com/crypto-wiki/article/what-are-crypto-derivatives-market-signals-and-how-do-they-predict-price-movements-using-futures-open-interest-funding-rates-and-liquidation-data-20260128)
- [On-Chain Metrics Analysis](https://www.ainvest.com/news/bitcoin-entering-mild-bear-market-deep-dive-chain-derivatives-signals-2512/)

---

## 3. System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                  Market Regime Engine                    │
│  ┌──────────────┐      ┌───────────────┐               │
│  │ Data Fetcher │─────→│ Regime        │               │
│  │ (APIs)       │      │ Classifier    │               │
│  └──────────────┘      └───────┬───────┘               │
│         │                      │                         │
│         │              ┌───────▼────────┐               │
│         │              │ Manual Override │               │
│         │              │ (Config/Env)    │               │
│         │              └───────┬─────────┘               │
│         │                      │                         │
│         │              ┌───────▼────────┐               │
│         └─────────────→│ Strategy       │               │
│                        │ Controller     │               │
│                        └───────┬────────┘               │
└────────────────────────────────┼─────────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
        ┌───────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
        │ Long Strategies│  │Short Strats│  │Both (if OK)│
        │ (Enabled/     │  │(Enabled/   │  │            │
        │  Disabled)    │  │ Disabled)  │  │            │
        └───────────────┘  └───────────┘  └─────────────┘
```

### Core Components

#### 1. Market Regime Engine (`regime_engine/market_regime_engine.py`)

**Responsibilities:**
- Fetch data from APIs (CoinGlass, Glassnode, Bybit)
- Calculate weighted regime score (0-100)
- Output regime classification (BULL/BEAR/SIDEWAYS)
- Cache data to avoid API rate limits
- Run on schedule (every 1 hour, configurable)

**Key Methods:**
```python
def fetch_funding_rate() -> float
def fetch_mvrv_zscore() -> float
def fetch_etf_flows() -> dict
def fetch_open_interest() -> dict
def fetch_nupl() -> float
def calculate_regime_score() -> int
def classify_regime(score: int) -> str
```

#### 2. Strategy Controller (`regime_engine/strategy_controller.py`)

**Responsibilities:**
- Read current regime from engine
- Check manual override (takes precedence)
- Enable/disable strategy containers via docker-compose
- Implement safety delays (30s between stops/starts)
- Log all regime changes to database
- Send notifications (Telegram/Discord optional)

**Key Methods:**
```python
def get_current_regime() -> str
def check_manual_override() -> Optional[str]
def enable_strategies(strategy_list: List[str])
def disable_strategies(strategy_list: List[str])
def apply_regime_change(regime: str)
```

#### 3. Configuration Files

**`regime_engine/config/strategy_mapping.yaml`**
```yaml
strategies:
  awesomeewolambo:
    long_container: freqtrade-awesomeewolambo
    short_container: freqtrade-awesomeewolambo_shorts
    approved_modes: [BULL, BEAR, SIDEWAYS]
    status: both_ready

  etcg:
    long_container: freqtrade-etcg
    short_container: freqtrade-etcg_shorts
    approved_modes: [BULL, BEAR, SIDEWAYS]
    status: both_ready

  e0v1e:
    long_container: freqtrade-e0v1e
    short_container: freqtrade-e0v1e_shorts
    approved_modes: [BEAR]  # Long needs validation
    status: short_only

  ei4_t4c0s_v2_2:
    long_container: freqtrade-ei4_t4c0s_v2_2
    short_container: freqtrade-ei4_t4c0s_v2_2_shorts
    approved_modes: [BEAR]  # Long needs validation
    status: short_only

# Regime behavior
regime_rules:
  BULL:
    enable_longs: true
    enable_shorts: false  # Unless status=both_ready

  BEAR:
    enable_longs: false  # Unless status=both_ready
    enable_shorts: true

  SIDEWAYS:
    enable_longs: true   # Only if status=both_ready
    enable_shorts: true  # Only if status=both_ready
```

**`regime_engine/config/indicators.yaml`**
```yaml
indicators:
  funding_rate:
    weight: 0.35
    source: bybit
    update_interval: 8h
    thresholds:
      bear: 0.005
      bull: 0.01

  mvrv_zscore:
    weight: 0.30
    source: glassnode
    update_interval: 24h
    thresholds:
      undervalued: 1.0
      fair: 2.5
      overvalued: 3.7

  etf_flows:
    weight: 0.20
    source: coinglass
    update_interval: 24h

  open_interest:
    weight: 0.10
    source: coinglass
    update_interval: 1h

  nupl:
    weight: 0.05
    source: glassnode
    update_interval: 24h
    thresholds:
      capitulation: 0.0
      neutral: 0.5
      euphoria: 0.75

cache:
  enabled: true
  ttl: 3600  # 1 hour
```

**`regime_engine/config/manual_override.json`**
```json
{
  "enabled": false,
  "regime": null,
  "reason": "",
  "set_at": null,
  "set_by": "manual"
}
```

#### 4. Database Schema

**`regime_history` table:**
```sql
CREATE TABLE regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    regime VARCHAR(20) NOT NULL,
    score INTEGER NOT NULL,
    funding_rate FLOAT,
    mvrv_zscore FLOAT,
    etf_flow_btc FLOAT,
    open_interest FLOAT,
    nupl FLOAT,
    is_manual_override BOOLEAN DEFAULT 0,
    override_reason TEXT,
    strategies_enabled TEXT,
    strategies_disabled TEXT
);
```

---

## 4. Data Flow

### Hourly Regime Check

```
1. Cron/Scheduler triggers regime_engine
   ↓
2. Fetch indicator data from APIs
   - Funding rate (Bybit)
   - MVRV Z-Score (Glassnode/CoinGlass)
   - ETF flows (CoinGlass)
   - Open Interest (CoinGlass)
   - NUPL (Glassnode)
   ↓
3. Calculate weighted score (0-100)
   ↓
4. Classify regime (BULL/BEAR/SIDEWAYS)
   ↓
5. Check manual override
   - If enabled: use override regime
   - If disabled: use calculated regime
   ↓
6. Compare with previous regime
   - If SAME: Log and exit
   - If DIFFERENT: Proceed to step 7
   ↓
7. Log regime change to database
   ↓
8. Send notifications (optional)
   - Telegram message
   - Discord webhook
   ↓
9. Strategy Controller applies changes
   - Load strategy mapping
   - Determine which strategies to enable/disable
   - Stop containers (with 30s delay)
   - Start containers (with 30s delay)
   ↓
10. Log final status
```

### Manual Override Flow

```
1. User updates manual_override.json
   {
     "enabled": true,
     "regime": "BEAR",
     "reason": "Fed meeting tomorrow",
     "set_at": "2026-02-02T10:00:00Z",
     "set_by": "admin"
   }
   ↓
2. Next regime check detects override
   ↓
3. Uses "BEAR" instead of calculated regime
   ↓
4. Applies strategy changes
   ↓
5. Logs with is_manual_override=true
```

---

## 5. Strategy Deployment Matrix

### Current Production Status (Feb 2026)

| Strategy | Long Status | Short Status | Deployment Mode |
|----------|-------------|--------------|-----------------|
| AwesomeEWOLambo | ✅ Profitable ($321) | ✅ Profitable ($1,510) | BULL/BEAR/SIDEWAYS |
| ETCG | ✅ Profitable ($373) | ✅ Profitable ($2,359) | BULL/BEAR/SIDEWAYS |
| e0v1e | ❌ Loss (-$3,704) | ✅ Profitable ($762) | BEAR only |
| ei4_t4c0s_v2_2 | ❌ Loss (-$4,608) | ✅ Profitable ($2,008) | BEAR only |

### Regime-Based Behavior

**BEAR Market (Current):**
```
ENABLED:
- awesomeewolambo_shorts
- etcg_shorts
- e0v1e_shorts
- ei4_t4c0s_v2_2_shorts

DISABLED:
- awesomeewolambo (long)
- etcg (long)
- e0v1e (long)
- ei4_t4c0s_v2_2 (long)
```

**BULL Market (Future):**
```
ENABLED:
- awesomeewolambo (long)
- etcg (long)

DISABLED:
- awesomeewolambo_shorts
- etcg_shorts
- e0v1e (all - needs validation)
- ei4_t4c0s_v2_2 (all - needs validation)
```

**SIDEWAYS Market:**
```
ENABLED (Both long AND short):
- awesomeewolambo (both)
- etcg (both)

DISABLED:
- e0v1e (needs validation)
- ei4_t4c0s_v2_2 (needs validation)
```

---

## 6. Error Handling & Safety

### Safety Mechanisms

1. **Gradual Transitions**: 30-second delay between container stops/starts
2. **Health Checks**: Verify container is healthy before marking as enabled
3. **Rollback**: If container fails to start, revert to previous state
4. **Rate Limiting**: Cache API responses to avoid rate limits
5. **Fallback**: If APIs fail, maintain current regime (don't change)

### Error Scenarios

**API Failure:**
- Retry 3 times with exponential backoff
- If all fail: use cached data (if <6 hours old)
- If cache stale: maintain current regime, log warning

**Container Failure:**
- Log error with full stack trace
- Send alert notification
- Do not disable other working containers
- Attempt restart after 5 minutes

**Invalid Configuration:**
- Validate YAML/JSON on load
- Fail fast with clear error message
- Do not apply partial changes

---

## 7. Monitoring & Alerting

### Metrics to Track

1. **Regime Changes**: Timestamp, old regime, new regime, score
2. **Indicator Values**: Store all indicator values for analysis
3. **Strategy Status**: Which strategies are running in each regime
4. **Performance**: Profit/loss per regime period
5. **API Health**: Response times, error rates

### Alerts

**Critical:**
- Regime change detected
- Container failed to start/stop
- All APIs failed

**Warning:**
- API response slow (>5s)
- Manual override active for >24 hours
- Score near regime boundary (±5 points)

**Info:**
- Hourly regime check completed
- Indicator data cached
- Strategy enabled/disabled

### Dashboard (Future Enhancement)

- Current regime indicator (big display)
- Historical regime chart
- Indicator values over time
- Strategy performance by regime
- Manual override controls

---

## 8. Testing Strategy

### Unit Tests

- Indicator calculation functions
- Regime scoring logic
- Configuration parsing
- API response mocking

### Integration Tests

- Full regime check cycle with mocked APIs
- Strategy enable/disable with dry-run mode
- Database operations
- Manual override behavior

### Production Testing

**Phase 1: Dry Run (Week 1-2)**
- Run regime engine WITHOUT controlling containers
- Log what WOULD happen
- Validate indicator data quality
- Tune thresholds if needed

**Phase 2: Single Strategy (Week 3)**
- Enable regime control for AwesomeEWOLambo only
- Monitor closely for 1 week
- Validate transitions work correctly

**Phase 3: All Strategies (Week 4+)**
- Enable all 4 strategy pairs
- Full production deployment
- Continue monitoring

---

## 9. Future Enhancements

### Short Term (1-2 months)

1. **Web Dashboard**: Real-time regime display, manual controls
2. **Telegram Bot**: Check regime, toggle override via chat
3. **Backtesting**: Simulate regime switching on historical data
4. **Alert Tuning**: ML-based threshold optimization

### Long Term (3-6 months)

1. **ML-Based Regime Detection**: Train model on historical data
2. **Multi-Timeframe Analysis**: Different regimes for different timeframes
3. **Per-Asset Regimes**: BTC regime might differ from ETH regime
4. **Risk Scoring**: Dynamic position sizing based on regime confidence
5. **Strategy Validation Pipeline**: Auto-test long strategies in dry-run during bear markets

---

## 10. Implementation Checklist

- [ ] Set up project structure
- [ ] Implement Market Regime Engine
  - [ ] Data fetchers for each API
  - [ ] Weighted scoring algorithm
  - [ ] Regime classifier
  - [ ] Cache layer
- [ ] Implement Strategy Controller
  - [ ] Docker-compose integration
  - [ ] Strategy mapping parser
  - [ ] Enable/disable logic
  - [ ] Health checks
- [ ] Configuration files
  - [ ] strategy_mapping.yaml
  - [ ] indicators.yaml
  - [ ] manual_override.json
- [ ] Database setup
  - [ ] regime_history table
  - [ ] Migration scripts
- [ ] Testing
  - [ ] Unit tests (80%+ coverage)
  - [ ] Integration tests
  - [ ] Dry-run testing (1-2 weeks)
- [ ] Deployment
  - [ ] Cron job setup
  - [ ] Logging configuration
  - [ ] Alert setup (Telegram/Discord)
- [ ] Documentation
  - [ ] API documentation
  - [ ] Operations runbook
  - [ ] Troubleshooting guide

---

## 11. API Requirements

### Required APIs

1. **Bybit API** (Free)
   - Endpoint: `/v5/market/funding/history`
   - Rate Limit: 50 requests/second
   - Data: Funding rate history

2. **CoinGlass API** (Free tier sufficient)
   - Endpoints:
     - `/api/futures/openInterest/chart`
     - `/api/index/mvrv-z-score`
     - `/api/etf/flows`
   - Rate Limit: 100 requests/day (free tier)
   - Cache aggressively

3. **Glassnode API** (Paid, or use CoinGlass alternative)
   - Endpoints:
     - `/v1/metrics/market/mvrv_z_score`
     - `/v1/metrics/indicators/net_unrealized_profit_loss`
   - Rate Limit: Depends on plan
   - Alternative: Use CoinGlass for free equivalents

### API Key Storage

```bash
# .env file (DO NOT commit)
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
COINGLASS_API_KEY=your_key
GLASSNODE_API_KEY=your_key  # Optional
TELEGRAM_BOT_TOKEN=your_token  # Optional
TELEGRAM_CHAT_ID=your_chat_id  # Optional
```

---

## 12. Deployment Architecture

### File Structure

```
/opt/trading/nfi-custom-strategies/
├── regime_engine/
│   ├── __init__.py
│   ├── market_regime_engine.py
│   ├── strategy_controller.py
│   ├── data_fetchers.py
│   ├── scoring.py
│   ├── config/
│   │   ├── strategy_mapping.yaml
│   │   ├── indicators.yaml
│   │   └── manual_override.json
│   ├── database/
│   │   ├── models.py
│   │   └── migrations/
│   └── utils/
│       ├── logger.py
│       ├── cache.py
│       └── alerts.py
├── regime_engine.db
├── logs/
│   └── regime_engine.log
└── scripts/
    └── run_regime_check.sh
```

### Cron Job

```bash
# Check regime every hour
0 * * * * /opt/trading/nfi-custom-strategies/scripts/run_regime_check.sh >> /var/log/regime_engine_cron.log 2>&1
```

---

## 13. Success Criteria

### Technical Success

- [ ] System runs reliably for 30 days without manual intervention
- [ ] API failures handled gracefully (no crashes)
- [ ] Regime changes applied within 60 seconds
- [ ] All strategy transitions successful (100%)
- [ ] Manual override works correctly

### Business Success

- [ ] Profitable shorts continue generating returns in BEAR regime
- [ ] System correctly detects next regime change
- [ ] No losses from running wrong strategies in wrong regime
- [ ] Long strategies validated and approved for BULL regime

### Validation Metrics

After 30 days, compare:
- P&L with regime switching vs without (backtest simulation)
- Expected improvement: 30-50% better risk-adjusted returns
- Drawdown reduction: 20-30%

---

## Approval & Sign-off

**Design Approved By:** User
**Date:** 2026-02-02
**Next Steps:**
1. Set up git worktree for isolated development
2. Create detailed implementation plan
3. Begin Phase 1 implementation

---

*End of Design Document*
