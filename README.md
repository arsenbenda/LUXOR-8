# LUXOR-8 Trading Bot

**v7.2 STABLE** - Pure Python Implementation (Zero Native Dependencies)

## 🎯 Key Features

- **Multi-Indicator Strategy**: RSI, MACD, Bollinger Bands, Gann Levels
- **Pure Python**: No TA-Lib or native C dependencies
- **Production Ready**: FastAPI + Docker + Health Checks
- **Lightweight**: Alpine Linux base (~50MB image)
- **Zero Build Errors**: 100% Python packages only

## 📊 Technical Indicators

| Indicator | Default Period | Signal Logic |
|-----------|----------------|--------------|
| **RSI** | 14 | Oversold <30 (BUY), Overbought >70 (SELL) |
| **MACD** | 12/26/9 | Bullish/Bearish crossover |
| **Bollinger Bands** | 20, 2σ | Price outside bands |
| **Gann Levels** | High/Low | Support/Resistance breaks |

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/arsenbenda/LUXOR-8.git
cd LUXOR-8

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000
```

### Docker Deployment
```bash
# Build image
docker build -t luxor-8:7.2 .

# Run container
docker run -d -p 8000:8000 --name luxor luxor-8:7.2

# Check health
curl http://localhost:8000/health
```

### Docker Compose
```bash
docker-compose up -d
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
# Response: {"status":"healthy","timestamp":"2026-01-13T..."}
```

### Test Signal (Mock Data)
```bash
GET /signal/test
# Response: {"signal":"BUY","price":45234.56,"indicators":{...}}
```

### Live Signal (Production)
```bash
GET /signal/daily
# Fetches real BTC/USDT data from Binance
```

## 🔧 Configuration

Environment variables (`.env`):
```bash
PORT=8000
SYMBOL=BTC/USDT
EXCHANGE=binance
TIMEFRAME=1d
```

Strategy parameters (modify in `strategy.py`):
```python
TradingStrategy(
    rsi_period=14,
    rsi_oversold=30.0,
    rsi_overbought=70.0,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_period=20,
    bb_std=2.0
)
```

## 📦 Dependencies

- **FastAPI**: Web framework
- **ccxt**: Exchange connectivity
- **pandas/numpy**: Data analysis
- **aiohttp==3.10.11**: HTTP client (pinned for ccxt compatibility)

## 🛠️ Troubleshooting

### Build Fails
- **Solution**: This version uses ONLY pure Python packages
- **Check**: `requirements.txt` has no `ta-lib` or `pandas-ta`

### Import Errors
- **Solution**: Verify Python 3.11+ is installed
- **Check**: `python --version` >= 3.11

### Connection Errors
- **Solution**: Check firewall allows port 8000
- **Check**: `docker logs luxor` for details

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with mock data
curl http://localhost:8000/signal/test

# Test live signal
curl http://localhost:8000/signal/daily
```

## 📝 Version History

- **v7.2**: Removed TA-Lib (pure Python)
- **v7.1**: Fixed aiohttp/ccxt conflict
- **v7.0**: Initial production release

## 👤 Author

**Arsen Benda**  
Repository: [arsenbenda/LUXOR-8](https://github.com/arsenbenda/LUXOR-8)

## 📄 License

MIT License - See LICENSE file for details

---

**Status**: ✅ Production Ready | **Build**: 🟢 Passing | **Dependencies**: 🟢 Stable
