# 🔗 Integrazione LUXOR v7.1 AGGRESSIVE con n8n

**GARANZIA**: Questi script usano **identici moduli** (`src/strategy.py`, `src/indicators.py`) del backtest originale.  
**Risultati**: 100% invariati - stesso scoring, entry, risk management.

---

## 📦 File Creati

### 1. `signal_api.py` - REST API Server
**Funzione**: Espone endpoint HTTP per n8n  
**Endpoint**:
- `GET/POST /signal/daily` → Segnale LUXOR corrente
- `GET /health` → Health check
- `GET /backtest/validate` → Validazione coerenza backtest vs API

**Avvio**:
```bash
# Sviluppo (Flask dev server)
python signal_api.py

# Produzione (Gunicorn - consigliato)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 signal_api:app
```

**Output JSON esempio**:
```json
{
  "timestamp": "2026-01-12T23:45:00",
  "signal": "LONG",
  "score": 72.5,
  "price": 45230.50,
  "indicators": {
    "rsi": 58.3,
    "ema_fast": 45100.2,
    "ema_slow": 44980.5,
    "atr": 1250.0,
    "volatility": 2.3
  },
  "levels": {
    "entry": 45230.50,
    "stop_loss": 43730.50,
    "take_profit": 48980.50
  },
  "risk": {
    "risk_usd": 230.0,
    "risk_pct": 2.3
  },
  "metadata": {
    "strategy": "LUXOR v7.1 AGGRESSIVE",
    "version": "7.1.0",
    "timeframe": "1D",
    "asset": "BTC/USDT"
  }
}
```

---

### 2. `send_to_n8n.py` - Direct Webhook Sender
**Funzione**: Calcola segnale e invia direttamente a n8n  
**Modalità**: Schedulabile (cron, systemd timer)

**Configurazione interna**:
```python
N8N_WEBHOOK_URL = "http://a80kcgwg80ko0ggow0ss40cs.69.197.134.206.sslip.io/signal/daily"
CONFIG_FILE = "config.json"
DATA_FILE = "data/btcusdt_daily_1000.csv"
```

**Esecuzione manuale**:
```bash
python send_to_n8n.py
```

**Output console**:
```
🚀 LUXOR v7.1 AGGRESSIVE → n8n Sender
⏰ 2026-01-12 23:45:00
--------------------------------------------------
📈 Calcolo segnale LUXOR...

📋 Segnale generato:
{
  "signal": "LONG",
  "score": 72.5,
  ...
}

🔗 Invio a n8n: http://a80k...sslip.io/signal/daily
✅ Segnale inviato a n8n: LONG (score: 72.5)
📊 Prezzo: 45230.50
🎯 Stop Loss: 43730.50
🎯 Take Profit: 48980.50

✅ Operazione completata con successo
```

---

## ⚙️ Setup Completo

### 1. Installa dipendenze aggiuntive
```bash
pip install flask gunicorn requests
```

### 2. Testa localmente
```bash
# Opzione A: API Server
python signal_api.py
# Testa: curl http://localhost:5000/signal/daily

# Opzione B: Direct Sender
python send_to_n8n.py
```

### 3. Scheduler automatico (Opzione A: cron)
```bash
# Esegui ogni giorno alle 00:05 UTC
crontab -e
# Aggiungi:
5 0 * * * cd /path/to/luxor-v7.1-aggressive && python send_to_n8n.py >> logs/n8n_sender.log 2>&1
```

### 4. Scheduler automatico (Opzione B: systemd timer)
```bash
# /etc/systemd/system/luxor-n8n.service
[Unit]
Description=LUXOR v7.1 n8n Signal Sender

[Service]
Type=oneshot
WorkingDirectory=/path/to/luxor-v7.1-aggressive
ExecStart=/usr/bin/python3 send_to_n8n.py
User=your_user

# /etc/systemd/system/luxor-n8n.timer
[Unit]
Description=LUXOR n8n Daily Timer

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target

# Attiva
sudo systemctl enable luxor-n8n.timer
sudo systemctl start luxor-n8n.timer
```

---

## 🔄 Workflow n8n (Esempio)

**Webhook Trigger**:
- URL: `http://a80k...sslip.io/signal/daily`
- Metodo: POST
- Body: JSON payload da LUXOR

**Nodi n8n suggeriti**:
1. **Webhook** → riceve segnale LUXOR
2. **IF**: `{{$json.signal}} != "NEUTRAL"`
3. **TRUE branch**:
   - **Telegram Bot**: invia alert `🚨 LUXOR LONG @ $45230`
   - **HTTP Request**: invia ordine a exchange (Binance/Bybit API)
   - **Google Sheets**: log segnale
4. **FALSE branch**: Log "NEUTRAL - no action"

---

## ✅ Validazione Risultati Identici

### Test automatico:
```bash
# Endpoint validazione
curl http://localhost:5000/backtest/validate
```

### Test manuale (confronto):
```python
# 1. Esegui backtest originale
python run_backtest.py --start-date 2025-01-01 --end-date 2025-12-31

# 2. Avvia API
python signal_api.py &

# 3. Confronta ultimo segnale
curl http://localhost:5000/signal/daily | jq .score
# Deve corrispondere all'ultimo score del backtest
```

---

## 🔐 Security Best Practices

**API Pubblico**:
```bash
# Aggiungi API key in signal_api.py
@app.route('/signal/daily', methods=['POST'])
def get_daily_signal():
    api_key = request.headers.get('X-API-Key')
    if api_key != 'your_secret_key':
        return jsonify({"error": "Unauthorized"}), 401
```

**n8n Webhook**:
- Usa `sslip.io` solo per sviluppo
- Produzione: dominio con HTTPS + authentication
- Considera n8n cloud (webhook con token)

---

## 📊 Monitoring

**Log API**:
```bash
# Produzione: gunicorn con logging
gunicorn -w 4 -b 0.0.0.0:5000 signal_api:app \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

**Log Sender**:
```bash
# Cron con redirect
python send_to_n8n.py >> logs/n8n_sender.log 2>&1
```

**Metriche utili**:
- Latenza API: `curl -w "@curl-format.txt" http://localhost:5000/signal/daily`
- Success rate n8n: conta HTTP 200 vs 4xx/5xx nei log

---

## 🚀 Deploy Produzione

### Opzione 1: VPS/Cloud Server
```bash
# 1. Clone repo GitHub
git clone https://github.com/arsenbenda/luxor-v7-python.git
cd luxor-v7-python

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install flask gunicorn requests

# 3. Avvia API (background)
nohup gunicorn -w 4 -b 0.0.0.0:5000 signal_api:app &

# 4. Setup cron per sender
crontab -e  # vedi sezione scheduler
```

### Opzione 2: Docker
```dockerfile
# Dockerfile (da creare)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install flask gunicorn requests
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "signal_api:app"]
```

---

## ❓ FAQ

**Q: Devo modificare strategy.py per usare n8n?**  
A: NO. Gli script usano i moduli esistenti senza modifiche.

**Q: Come aggiorno i dati per calcolare segnali real-time?**  
A: Sostituisci `pd.read_csv('data/btcusdt_daily_1000.csv')` con:
```python
# Fetch da exchange (esempio Binance)
import ccxt
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=200)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
```

**Q: Posso usare entrambi (API + Sender)?**  
A: Sì. API per query on-demand, Sender per automation scheduled.

---

## 📞 Support

- Issues: https://github.com/arsenbenda/luxor-v7-python/issues
- Email: support@luxor-trading.com (se disponibile)

---

**Versione**: 1.0.0  
**Data**: 2026-01-12  
**Autore**: LUXOR Trading Systems
