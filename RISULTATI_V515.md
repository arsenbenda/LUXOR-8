# 📊 LUXOR V7 v5.1.5 - RISULTATI UFFICIALI

## ✅ Versione Stabile: v5.1.5 HYBRID

**Data Backtest**: 2026-01-12  
**Dataset**: BTC/USDT Daily (750 bars, 2024-2026)  
**Capitale Iniziale**: $10,000  
**Capitale Finale**: $13,087.34

---

## 🎯 METRICHE PRINCIPALI

| Metrica | Valore | Target |
|---------|--------|--------|
| **Total Trades** | 90 | - |
| **Win Rate** | 45.56% | ✅ 45-50% |
| **Total Return** | +30.87% | ✅ >20% |
| **Profit Factor** | 1.51x | ⚠️ Target: 2.0x |
| **Max Drawdown** | -6.56% | ✅ <-8% |
| **Sharpe Ratio** | 0.17 | ⚠️ Target: >0.5 |

---

## 💰 ANALISI TRADE QUALITY

| Metrica | Winners | Losers |
|---------|---------|--------|
| **Numero** | 41 | 49 |
| **Avg Profit/Loss** | $221.80 | $-122.58 |
| **Avg R Multiple** | +1.86R | -0.99R |
| **Total PnL** | $9,093.80 | $-6,006.42 |

**Avg R per Trade**: +0.31R  
**Avg Bars Held**: 6.9  

---

## 🔄 TRAILING STOP STATS

- **Trailing Exits**: 38 trades (42.2%)
- **Total Trailing Events**: 56
- **Break-Even Activation**: +1.5R

### Regime-Aware Multipliers:

**TRENDING** (0.8x - 1.2x ATR):
- Stage 0 (profit < 1.0R): 1.5x
- Stage 1 (1.0R - 2.0R): 1.2x
- Stage 2 (profit ≥ 2.0R): 1.0x

**RANGING** (0.5x - 1.5x ATR):
- Stage 0 (profit < 0.5R): 1.8x
- Stage 1 (profit ≥ 0.5R): 0.6x

**VOLATILE** (1.0x - 1.6x ATR):
- Stage 0 (profit < 1.0R): 2.0x
- Stage 1 (profit ≥ 1.0R): 1.3x

---

## 🎪 FEATURES v5.1.5

✅ **Hybrid Trailing Stops** (regime-aware)  
✅ **Multi-Timeframe Regime Detection** (1D, 3D, 1W, 1M)  
✅ **ATR-based Stop Loss** (2.5x ATR minimum)  
✅ **Break-Even Protection** (+1.5R)  
✅ **Profit-Based Stage Management**  
✅ **Confidence-Based Entry Filter**  

---

## 📁 STRUTTURA FILE

```
luxor-v7-python/
├── luxor_v7_prana.py           # Core system v5.1.3
├── backtest_v515.py            # Backtest engine v5.1.5
├── data/
│   └── btcusdt_daily_1000.csv  # Dataset BTC/USDT
├── results/
│   └── backtest_v515_hybrid_baseline.json  # Risultati
├── CHANGELOG.md                # Changelog dettagliato
└── DEPLOYMENT_GUIDE.md         # Guida deployment
```

---

## 🚀 PROSSIMI SVILUPPI

### FASE 1 - Entry Filters (Priorità Alta)
- Confidence HIGH filter
- Candlestick pattern confirmation
- MTF alignment check
- Volume > 1.2x average

**Target miglioramento**:
- Win Rate: 45.6% → 50-55%
- Profit Factor: 1.51x → 2.0-2.5x
- Trades: -30% (solo setup migliori)

### FASE 2 - Risk Management
- Position sizing dinamico
- Correlation filter
- Portfolio heat management

### FASE 3 - Exit Optimization
- Multi-target exits
- Time-based exits
- Momentum-based trailing

---

## 📝 NOTE TECNICHE

**Core System**: luxor_v7_prana.py v5.1.3 (stabile)  
**Backtest Engine**: v5.1.5 Hybrid Trailing  
**Python Version**: ≥3.8  
**Dependencies**: pandas, numpy, ta-lib  

---

## ⚠️ DISCLAIMER

Risultati di backtest su dati storici. Performance passate non garantiscono risultati futuri.
Sistema progettato per BTC/USDT timeframe daily. Testare su paper trading prima del live.

---

**Ultima Verifica**: 2026-01-12  
**Status**: ✅ Production Ready  
**File Risultati**: `results/backtest_v515_hybrid_baseline.json`
