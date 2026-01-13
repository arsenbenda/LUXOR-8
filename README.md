# LUXOR v7.1 AGGRESSIVE Trading System

**High-Performance BTC/USDT Trading Strategy**

## 📊 Performance Metrics (2024-2026)
- **Total P&L:** +251.64% (2 years)
- **Win Rate:** 89.7% (61W/7L)
- **Total Trades:** 68
- **Avg Win:** +5.38% | **Avg Loss:** -10.90%
- **LONG:** +110.79% (34 trades) | **SHORT:** +140.86% (34 trades)
- **Annualized Return:** ~125%

## 🎯 Strategy Core
1. **Swing Detection:** 5-day pivot analysis
2. **Dynamic Scoring:** Multi-layer technical indicators (RSI, EMA, Volume, ATR)
3. **Aggressive Entry:** Min score 60, Max score 100
4. **Smart Exits:** Partial exits (50% @ +0.5R, 30% @ +1R) + ATR trailing stops
5. **Direction Balance:** LONG + SHORT simultaneously active

## 🔧 Key Parameters
```python
CONFIG = {
    'min_score': 60,
    'max_score': 100,
    'ideal_volatility_min': 0.5,
    'ideal_volatility_max': 8.0,
    'trailing_atr_multiplier': 1.5,
    'breakeven_at': 0.3,
    'partial_exit_1': 0.50,  # 50% @ +0.5R
    'partial_exit_2': 0.30,  # 30% @ +1R
    'sl_atr_multiplier': 1.2
}
```

## 📂 Repository Structure
```
luxor-v7.1-aggressive/
├── README.md
├── LICENSE
├── requirements.txt
├── config.json
├── src/
│   ├── __init__.py
│   ├── strategy.py          # Core trading logic
│   ├── indicators.py        # Technical indicators
│   ├── backtester.py        # Backtest engine
│   └── utils.py             # Helper functions
├── data/
│   └── btcusdt_daily_1000.csv
├── results/
│   ├── backtest_v710_results.csv
│   └── performance_report.txt
├── notebooks/
│   └── analysis.ipynb
└── tests/
    └── test_strategy.py
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/luxor-v7.1-aggressive.git
cd luxor-v7.1-aggressive
pip install -r requirements.txt
```

### Run Backtest
```bash
python -m src.backtester --start-date 2024-01-01 --end-date 2026-01-12
```

### Configuration
Edit `config.json` to adjust strategy parameters.

## 📈 Top 5 Winning Trades
1. **2024-02-23 LONG:** +16.03%
2. **2024-11-04 LONG:** +14.58%
3. **2024-05-01 LONG:** +12.47%
4. **2025-01-30 SHORT:** +11.04%
5. **2024-05-10 LONG:** +10.32%

## ⚠️ Risk Disclosure
- **High volatility strategy:** Avg loss -10.90%
- **Leveraged positions:** Use proper risk management
- **Backtest results:** Past performance ≠ future results
- **Recommended:** Start with paper trading

## 📝 Version History
- **v7.1 AGGRESSIVE:** Current (251% P&L, 89.7% WR)
- **v7.0:** ML-based optimizer (no trades)
- **v5.3.0:** Ultra optimized (12 trades, 83% WR)
- **v5.2.0:** Baseline (42 trades, 52% WR, +41% P&L)

## 📞 Support
- Issues: [GitHub Issues](https://github.com/yourusername/luxor-v7.1-aggressive/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/luxor-v7.1-aggressive/discussions)

## 📄 License
MIT License - See LICENSE file for details

---
**⚡ Built for maximum profit with intelligent risk management**
