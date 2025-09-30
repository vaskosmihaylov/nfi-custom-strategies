# FVGAdvancedStrategy_V2

## Overview
Advanced Fair Value Gap (FVG) trading strategy with dual-sided trading capabilities (long and short positions). The strategy identifies price inefficiencies through gap detection and combines them with trend, momentum, and volatility indicators for high-probability trade setups.

## Strategy Type
- **Timeframe:** 5m (primary), 1h (informative)
- **Trading Mode:** Futures with leverage support
- **Direction:** Bidirectional (long and short)
- **Interface Version:** 3

## Key Features

### 1. Fair Value Gap (FVG) Detection
- Identifies price gaps where trading activity skipped price levels
- **Bull FVG:** Gap between candle highs and lows indicating upward momentum
- **Bear FVG:** Gap between candle lows and highs indicating downward momentum
- Confirmation period of 1-3 bars (default: 2)
- Minimum gap size based on ATR multiples (default: 0.6x ATR)

### 2. Multi-Timeframe Analysis
- **5m timeframe:** Primary signals and entry/exit execution
- **1h timeframe:** Trend confirmation and regime filtering
- Merges both timeframes for comprehensive market view

### 3. Market Regime Scoring
Composite 0-100 score combining:
- **ADX Component:** Trend strength measurement
- **RSI Component:** Momentum deviation from neutral
- **Volatility Component:** ATR-based volatility assessment
- **Bollinger Band Component:** Price expansion/contraction
- **Momentum Component:** Fast/slow EMA relationship

### 4. Dynamic Leverage Management
- **Default Leverage:** 2.0x
- **Range:** 1.0x - 3.0x
- **Adjustments based on:**
  - Market score (higher score = higher leverage)
  - ADX trend strength
  - RSI extremes
  - 1h timeframe trend confirmation
  
### 5. Position Management
- **DCA (Dollar Cost Averaging):** Up to 1 additional entry
- **Scale Out:** Partial profit taking at 3%+ profit (25% position size)
- **Maximum DCA Multiplier:** 1.5x
- **DCA Profit Range:** -4% to -12%

## Entry Conditions

### Long Entry
All conditions must be met:
1. ✅ Bull FVG detected and confirmed
2. ✅ Fast EMA > Slow EMA (uptrend)
3. ✅ 1h uptrend confirmed
4. ✅ Market score > 65
5. ✅ Volume ratio > 1.05 (above average)
6. ✅ ATR percentile rank < 0.85 (controlled volatility)
7. ✅ Price > EMA 200 (long-term uptrend)

### Short Entry
All conditions must be met:
1. ✅ Bear FVG detected and confirmed
2. ✅ Fast EMA < Slow EMA (downtrend)
3. ✅ 1h downtrend confirmed
4. ✅ Market score < 35
5. ✅ Volume ratio > 1.05 (above average)
6. ✅ ATR percentile rank < 0.85 (controlled volatility)
7. ✅ Price < EMA 200 (long-term downtrend)

## Exit Conditions

### Long Exit
Any of the following triggers exit:
- ❌ Bear FVG detected (reversal signal)
- ❌ Uptrend broken (Fast EMA < Slow EMA)
- ❌ Market score drops below 50 (score - 15)
- ❌ Price falls below Fast EMA
- ❌ Volume ratio < 0.8 (weak volume)

### Short Exit
Any of the following triggers exit:
- ❌ Bull FVG detected (reversal signal)
- ❌ Downtrend broken (Fast EMA > Slow EMA)
- ❌ Market score rises above 50 (score + 15)
- ❌ Price rises above Fast EMA
- ❌ Volume ratio < 0.8 (weak volume)

## Risk Management

### Fixed Parameters
- **Minimal ROI:**
  - 6% immediate
  - 4% after 60 minutes
  - 2% after 180 minutes
  - 0% after 360 minutes
- **Stop Loss:** -9%

### Tunable Parameters
- **filter_width:** 0.1-2.0 (default 0.6) - Gap size threshold in ATR multiples
- **tp_mult:** 0.8-3.0 (default 1.6) - Take profit ATR multiplier
- **sl_mult:** 0.5-2.0 (default 0.9) - Stop loss ATR multiplier
- **fvg_confirmation_bars:** 1-3 (default 2) - Bars needed to confirm FVG

## Technical Indicators

### Trend Indicators
- **EMA Fast:** 21 periods
- **EMA Slow:** 55 periods
- **EMA Long:** 200 periods

### Volatility Indicators
- **ATR:** 14 periods
- **Bollinger Bands:** 20 periods, 2 standard deviations

### Momentum Indicators
- **RSI:** 14 periods
- **ADX:** 14 periods

### Volume Analysis
- **Volume Mean:** 40-period rolling average
- **Volume Ratio:** Current vs. average volume

## Recent Fixes

### 2025-09-30: Boolean Type Error Fix ✅
**Issue:** Strategy crashed with `TypeError: unsupported operand type(s) for &: 'float' and 'bool'`

**Root Cause:** The `bull_fvg` and `bear_fvg` columns were created using `.rolling().max()` which returns float values instead of boolean values.

**Fix Applied:**
```python
# Before (caused error)
dataframe["bull_fvg"] = bull_gap.rolling(confirm_window).max().fillna(False)

# After (fixed)
dataframe["bull_fvg"] = bull_gap.rolling(confirm_window).max().fillna(0).astype(bool)
```

**Status:** ✅ Production ready - No configuration changes required

## Performance Optimization

### Computational Efficiency
- Vectorized operations throughout
- Minimal DataFrame copies
- Efficient rolling window calculations
- Cached market score computations

### Memory Management
- NaN handling with proper fillna strategies
- Clipping values to prevent overflow
- Efficient data type usage (bool, float32 where appropriate)

## Configuration Requirements

### Freqtrade Config
```json
{
  "use_exit_signal": true,
  "exit_profit_only": false,
  "ignore_roi_if_entry_signal": false,
  "timeframe": "5m",
  "can_short": true,
  "trading_mode": "futures",
  "margin_mode": "isolated"
}
```

### Recommended Pairlist
- 40-80 pairs
- Prefer stablecoin pairs (USDT, USDC)
- Avoid leveraged tokens (*BULL, *BEAR, *UP, *DOWN)
- Volume-based pairlist recommended

### Stake Configuration
- **Open Trades:** 4-6 recommended
- **Stake Amount:** Unlimited stake recommended
- Adjust based on account size and risk tolerance

## Monitoring & Maintenance

### Key Metrics to Watch
- FVG detection frequency
- Market score distribution
- Leverage utilization
- Win rate by market regime
- Average trade duration
- DCA trigger frequency

### Common Issues
1. **Too few trades:** Lower `market_score_long/short` thresholds
2. **Too many false signals:** Increase `fvg_confirmation_bars`
3. **Large drawdowns:** Reduce `max_leverage` or increase filters
4. **Missed opportunities:** Lower `filter_width` or volume requirements

## Backtesting Recommendations

### Data Requirements
- Minimum 3 months historical data
- Include both trending and ranging markets
- Test across different volatility regimes

### Optimization Targets
1. Filter width for FVG detection
2. Market score thresholds
3. Volume ratio requirements
4. Leverage parameters

### Validation
- Out-of-sample testing mandatory
- Walk-forward analysis recommended
- Multiple market conditions validation

## Dependencies
- freqtrade >= 2023.1
- pandas >= 1.5.0
- numpy >= 1.23.0
- ta-lib >= 0.4.24

## Version History
- **V2:** Current version with enhanced market regime scoring and leverage management
- **V1:** Initial FVG detection with basic trend filtering

## Support & Contributing
For issues, questions, or contributions, please refer to the main NostalgiaForInfinity repository.

## License
Same as parent NostalgiaForInfinity project

---
**Last Updated:** 2025-09-30
**Status:** Production Ready ✅
**Maintenance:** Active