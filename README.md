# LUXOR Trading Bot v7.1.0

**BTC Trading Strategy API** with TA-Lib indicators, Gann levels, and FastAPI.

---

## 🚀 **Quick Start**

### **1. Local Development**
```bash
# Clone repository
git clone https://github.com/arsenbenda/LUXOR-8.git
cd LUXOR-8

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **2. Docker Build**
```bash
# Build image
docker build -t luxor-bot:7.1.0 .

# Run container
docker run -d -p 8000:8000 --name luxor luxor-bot:7.1.0

# Check logs
docker logs -f luxor

# Test healthcheck
curl http://localhost:8000/health
```

### **3. Deploy to Coolify**
1. Push code to GitHub: `git push origin main`
2. Coolify auto-detects Dockerfile
3. Monitor build logs
4. Test deployment: `curl https://luxor.arsenbenda.it/health`

---

## 📡 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Healthcheck (Docker) |
| `/signal/test` | GET | Mock signal (testing) |
| `/signal/daily` | GET | Real BTC signal (production) |

### **Example Response** (`/signal/daily`)
```json
{
  "signal": "BUY",
  "price": 45234.56,
  "indicators": {
    "rsi": 28.45,
    "macd": 123.45,
    "macd_signal": 98.32,
    "atr": 456.78,
    "bb_upper": 46000.00,
    "bb_lower": 44000.00
  },
  "gann_levels": {
    "gann_high": [45678, 45890, 46120, 46340, 46580],
    "gann_low": [44320, 44100, 43890, 43680, 43450],
    "pivot": 45000.0
  },
  "timestamp": "2026-01-13T11:30:00"
}
```

---

## 🔧 **Troubleshooting**

### **Dependency Conflict (aiohttp)**
- **Problem**: `ccxt==4.4.37` requires `aiohttp<=3.10.11`
- **Solution**: Already fixed in `requirements.txt` → `aiohttp==3.10.11`

### **UndefinedVar $PYTHONPATH**
- **Problem**: Dockerfile used `$PYTHONPATH` before defining it
- **Solution**: `ENV PYTHONPATH=/app` moved before first usage (line 20)

### **TA-Lib Build Failures**
- **Problem**: Missing build dependencies (gcc, make, etc.)
- **Solution**: Using binary wheels from PyPI (`TA-Lib==0.5.1`), no build deps needed

---

## 📦 **File Structure**
```
LUXOR-8/
├── main.py              # FastAPI app
├── strategy.py          # Trading logic
├── gann_high_low.py     # Gann calculations
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build config
├── .dockerignore        # Docker ignore patterns
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

---

## 🛡️ **Production Checklist**

- [ ] Environment variables configured (if needed)
- [ ] Healthcheck passing (`/health` returns 200)
- [ ] All endpoints tested (`/`, `/signal/test`, `/signal/daily`)
- [ ] Logs clean (no import errors or warnings)
- [ ] Docker build successful (no dependency conflicts)
- [ ] Deployment stable (container running for >5 minutes)

---

## 📄 **Version History**

- **v7.1.0** (2026-01-13): Fixed aiohttp/ccxt conflict, Dockerfile PYTHONPATH, production ready
- **v7.0.x** (2026-01-12): Initial development, pandas-ta issues
- **v6.x**: Legacy versions (deprecated)

---

## 👤 **Author**

**Arsen Benda** ([@arsenbenda](https://github.com/arsenbenda))

---

## 📞 **Support**

Issues? Open a GitHub issue or contact via Coolify dashboard.

---

**Status**: ✅ **PRODUCTION READY** (2026-01-13)
