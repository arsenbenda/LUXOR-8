# LUXOR v7.1 AGGRESSIVE

Bitcoin Trading Strategy with Gann High/Low Integration

## 📊 Features

- **Multi-Timeframe EMA Alignment** - 21, 50, 200 period EMAs
- **RSI Momentum** - Overbought/oversold detection
- **ATR-Based Stops** - Dynamic stop loss calculation
- **Gann High/Low** - Adaptive lookback based on volatility
- **Confidence Scoring** - 0-100 signal quality score
- **Risk Management** - Position sizing, R/R ratios
- **API Integration** - REST endpoint for n8n/Telegram

## 🚀 Quick Start

### 1. File Structure

```
LUXOR-8/
├── strategy.py          # Main strategy (THIS FILE)
├── indicators.py        # Technical indicators
├── gann_high_low.py     # Gann module
├── signal_api.py        # Flask API
├── config.py            # Configuration
├── requirements.txt     # Dependencies
└── README.md            # This file
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/arsenbenda/LUXOR-8.git
cd LUXOR-8

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage

```python
from strategy import LuxorV7AggressiveStrategy
import pandas as pd

# Load your data
df = pd.read_csv('btc_data.csv')

# Initialize strategy
strategy = LuxorV7AggressiveStrategy(
    initial_capital=10000,
    risk_per_trade=0.02,
    timeframe="1D"
)

# Generate signal
signal = strategy.generate_signal(df)

print(f"Signal: {signal.signal}")
print(f"Score: {signal.score}/100")
print(f"Entry: ${signal.entry_price:.2f}")
print(f"Stop: ${signal.stop_loss:.2f}")
print(f"Target: ${signal.take_profit:.2f}")
```

## 📈 Signal Generation

### Scoring System (0-100)

| Component | Weight | Description |
|-----------|--------|-------------|
| **Trend** | 30 pts | Price vs EMA 200 |
| **Momentum** | 25 pts | EMA Fast vs Slow |
| **RSI** | 20 pts | Overbought/oversold |
| **Gann Position** | 15 pts | Price relative to Gann levels |
| **Gann Confidence** | 10 pts | Quality of Gann levels |

### Signal Thresholds

- **LONG**: Score ≥ 60 + Gann confidence ≥ 50%
- **SHORT**: Score ≤ 15 + Gann confidence ≥ 50%
- **NEUTRAL**: Otherwise

## 🎯 Gann High/Low

### Lookback Periods

| Timeframe | Min | Max | Optimal |
|-----------|-----|-----|---------|
| 1D | 52 | 260 | 90 |
| 3D | 40 | 80 | 60 |
| 1W | 26 | 52 | 39 |
| 1M | 12 | 24 | 18 |

### Adaptive Adjustments

- **EXTREME volatility**: -30% lookback
- **HIGH volatility**: -15% lookback
- **NORMAL**: Standard lookback
- **LOW volatility**: +20% lookback

## 🔧 Configuration

Create `.env` file:

```bash
# Strategy
INITIAL_CAPITAL=10000
RISK_PER_TRADE=0.02
DEFAULT_TIMEFRAME=1D

# Indicators
EMA_FAST=21
EMA_SLOW=50
EMA_TREND=200
RSI_PERIOD=14

# Gann
USE_ADAPTIVE_GANN=true
GANN_MIN_CONFIDENCE=50.0

# Scoring
MIN_LONG_SCORE=60
MIN_SHORT_SCORE=85

# Market
MARKET=BTC/USDT
EXCHANGE=Binance
```

## 🌐 API Usage

Start the Flask API:

```bash
python signal_api.py
```

Endpoints:

- `GET /health` - Health check
- `GET /signal/daily` - Get daily signal
- `GET /signal/test` - Test with mock data

Example response:

```json
{
  "signal": "LONG",
  "score": 85.3,
  "price": 45230.00,
  "gann": {
    "high": 46800.00,
    "low": 44000.00,
    "confidence": 75.5,
    "volatility_regime": "normal"
  },
  "levels": {
    "entry": 45200.00,
    "stop_loss": 44400.00,
    "take_profit": 46800.00
  },
  "risk": {
    "risk_usd": 400.00,
    "risk_pct": 2.0,
    "position_size": 0.015,
    "risk_reward": 2.0
  }
}
```

## 🔌 n8n Integration

1. Create webhook trigger in n8n
2. Add HTTP Request node pointing to LUXOR API
3. Use Code Node to format for Telegram
4. Send via Telegram node

See `luxor-n8n-workflow.json` for complete workflow.

## 📊 Backtesting

```python
from backtester import Backtester

backtester = Backtester(
    strategy=strategy,
    start_date="2023-01-01",
    end_date="2024-12-31"
)

results = backtester.run(df)
backtester.plot_results()
```

## 📝 Signal Validation

Signals are validated for:

- ✅ Gann confidence ≥ minimum threshold
- ✅ Risk/Reward ≥ 1.5:1
- ✅ Volatility < 5% (extreme)
- ✅ Score meets minimum for signal type

## 🛠️ Deployment

### Coolify

1. Push to GitHub
2. Create new application in Coolify
3. Select Dockerfile deployment
4. Set environment variables
5. Deploy!

### Docker

```bash
docker build -t luxor-v7 .
docker run -p 5000:5000 \
  -e INITIAL_CAPITAL=10000 \
  -e RISK_PER_TRADE=0.02 \
  luxor-v7
```

## 📚 Documentation

- **Strategy Logic**: See `strategy.py` docstrings
- **Gann Module**: See `gann_high_low.py`
- **Indicators**: See `indicators.py`
- **API**: See `signal_api.py`

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test strategy
python strategy.py

# Test Gann module
python gann_high_low.py

# Test API
curl http://localhost:5000/health
```

## 📈 Performance Metrics

- **Win Rate**: Track with backtester
- **Average R/R**: Typically 1.5-3.0:1
- **Max Drawdown**: Monitor position sizing
- **Sharpe Ratio**: Calculate from backtest results

## ⚠️ Risk Disclaimer

This is a trading strategy for educational purposes. 
- Always test with paper trading first
- Never risk more than you can afford to lose
- Past performance does not guarantee future results
- Cryptocurrency trading carries significant risk

## 🔄 Updates

### v7.1.0 (Current)
- ✅ Gann High/Low integration
- ✅ Adaptive lookback system
- ✅ Confidence scoring
- ✅ Multi-timeframe support
- ✅ Enhanced risk management

### v7.0.0
- Initial aggressive strategy
- EMA + RSI system
- ATR stops

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📧 Support

- GitHub Issues: [arsenbenda/LUXOR-8](https://github.com/arsenbenda/LUXOR-8/issues)
- Documentation: See inline code comments

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- LUXOR Team
- Community contributors
- Open source trading libraries

---

**Version**: 7.1.0  
**Last Updated**: January 2026  
**Author**: LUXOR Trading Team
