# 🚀 Quick Start: LUXOR v7.1 → n8n in 5 minuti

## 📥 Download & Setup
```bash
# 1. Scarica ZIP da link fornito
unzip luxor-v7.1-aggressive-n8n.zip
cd luxor-repo-update/

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Verifica installazione
python -c "from src.strategy import LuxorStrategy; print('✅ OK')"
```

## 🎯 Opzione 1: Invio Diretto (Più semplice)
```bash
# Esegui subito
python send_to_n8n.py

# Output atteso:
# ✅ Segnale inviato a n8n: LONG (score: 72.5)
```

**Automazione cron** (esegui ogni giorno alle 00:05):
```bash
crontab -e
# Aggiungi questa riga:
5 0 * * * cd /path/to/luxor-repo-update && python send_to_n8n.py >> logs/n8n.log 2>&1
```

## 🌐 Opzione 2: API Server (Più flessibile)
```bash
# Avvia server
python signal_api.py
# Server attivo su http://localhost:5000

# Test da altro terminale
curl http://localhost:5000/signal/daily
```

**n8n configuration**:
- Webhook Trigger → URL: `http://your-server-ip:5000/signal/daily`
- Metodo: GET o POST
- Response: JSON con `signal`, `score`, `price`, `levels`

## 🔄 Workflow n8n Esempio

```
┌─────────────┐
│  HTTP GET   │  →  ogni ora/giorno
│ /signal/day │
└──────┬──────┘
       │
       v
┌─────────────┐
│  IF Node    │  →  signal != "NEUTRAL" ?
└──────┬──────┘
       │
   YES │  NO → Stop
       v
┌─────────────┐
│  Telegram   │  →  "🚨 LUXOR LONG @ $45230"
│    Bot      │
└─────────────┘
```

## ✅ Test di Validazione
```bash
# Confronta risultati backtest vs API
python run_backtest.py --start-date 2025-01-01 --end-date 2025-12-31
python signal_api.py &
curl http://localhost:5000/backtest/validate

# Deve mostrare: "API usa stessi moduli di backtest - risultati garantiti identici"
```

## 🔧 Personalizzazione

**Cambia webhook n8n**:
```python
# Apri send_to_n8n.py
N8N_WEBHOOK_URL = "http://TUO_WEBHOOK_URL"
```

**Cambia capitale/risk**:
```python
# Apri send_to_n8n.py o signal_api.py
capital = 20000  # era 10000
```

**Usa dati real-time**:
```python
# Sostituisci pd.read_csv(...) con:
import ccxt
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=200)
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
```

## 📊 Esempio Output JSON
```json
{
  "signal": "LONG",
  "score": 72.5,
  "price": 45230.50,
  "levels": {
    "entry": 45230.50,
    "stop_loss": 43730.50,
    "take_profit": 48980.50
  },
  "risk": {
    "risk_usd": 230.0,
    "risk_pct": 2.3
  }
}
```

## 🆘 Troubleshooting

**Errore: ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Errore: n8n webhook timeout**
- Verifica URL webhook n8n attivo
- Test: `curl -X POST http://YOUR_N8N_URL -d '{"test":"ok"}'`

**Segnale sempre NEUTRAL**
- Verifica dati in `data/btcusdt_daily_1000.csv`
- Score minimo = 60 (vedi `config.json`)

## 📚 Documentazione Completa
Leggi `n8n_integration_README.md` per:
- Security best practices
- Deploy produzione
- Monitoring & logging
- FAQ avanzate

---

**Tempo setup**: ~5 minuti  
**Prerequisiti**: Python 3.8+, pip, n8n attivo  
**Support**: https://github.com/arsenbenda/luxor-v7-python
