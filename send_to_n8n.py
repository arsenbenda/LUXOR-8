"""
LUXOR v7.1 AGGRESSIVE - n8n Sender
Invia segnali daily al tuo webhook n8n
Schedulabile con cron/systemd timer
"""

import requests
import pandas as pd
import sys
import os
from datetime import datetime
import json

# Import moduli LUXOR (stessa logica!)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from strategy import LuxorStrategy
from utils import load_config

# CONFIGURAZIONE
N8N_WEBHOOK_URL = "http://a80kcgwg80ko0ggow0ss40cs.69.197.134.206.sslip.io/signal/daily"
CONFIG_FILE = "config.json"
DATA_FILE = "data/btcusdt_daily_1000.csv"

def get_luxor_signal():
    """
    Calcola segnale LUXOR usando stessi moduli di backtest
    GARANZIA: risultati identici a strategy.py
    """
    # Carica config e strategia
    config = load_config(CONFIG_FILE)
    strategy = LuxorStrategy(config)
    
    # Carica dati
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    
    # Calcola indicatori sugli ultimi 200 candles
    df_recent = df.tail(200).copy()
    df_recent = strategy._calculate_indicators(df_recent)
    
    # Ultimo candle
    latest = df_recent.iloc[-1]
    idx = len(df_recent) - 1
    
    # Calcola score (stessa logica di strategy.py)
    long_score = strategy._calculate_long_score(df_recent, idx)
    short_score = strategy._calculate_short_score(df_recent, idx)
    
    # Determina segnale
    signal = "NEUTRAL"
    score = 0
    direction = None
    
    if long_score >= config['entry']['min_score']:
        signal = "LONG"
        score = long_score
        direction = 1
    elif short_score >= config['entry']['min_score']:
        signal = "SHORT"
        score = short_score
        direction = -1
    
    # Calcola livelli operativi
    atr = latest['atr']
    entry_price = latest['close']
    
    if direction == 1:  # LONG
        stop_loss = entry_price - (atr * config['risk']['stop_loss_atr_multiplier'])
        take_profit = entry_price + (atr * 3.0)
    elif direction == -1:  # SHORT
        stop_loss = entry_price + (atr * config['risk']['stop_loss_atr_multiplier'])
        take_profit = entry_price - (atr * 3.0)
    else:
        stop_loss = None
        take_profit = None
    
    # Risk sizing
    risk_usd = None
    risk_pct = None
    if direction:
        vol = latest['volatility']
        risk_pct = vol * config['position_sizing']['volatility_factor']
        risk_pct = max(config['position_sizing']['min_risk_per_trade'],
                      min(risk_pct, config['position_sizing']['max_risk_per_trade']))
        capital = 10000  # Modifica se necessario
        risk_usd = capital * risk_pct
    
    # Costruisci payload
    payload = {
        "timestamp": datetime.now().isoformat(),
        "signal": signal,
        "score": round(score, 2),
        "price": round(entry_price, 2),
        "indicators": {
            "rsi": round(latest['rsi'], 2),
            "ema_fast": round(latest['ema_fast'], 2),
            "ema_slow": round(latest['ema_slow'], 2),
            "atr": round(atr, 2),
            "volatility": round(latest['volatility']*100, 2)
        },
        "levels": {
            "entry": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None
        },
        "risk": {
            "risk_usd": round(risk_usd, 2) if risk_usd else None,
            "risk_pct": round(risk_pct*100, 2) if risk_pct else None
        },
        "metadata": {
            "strategy": "LUXOR v7.1 AGGRESSIVE",
            "version": "7.1.0",
            "timeframe": "1D",
            "asset": "BTC/USDT",
            "data_date": latest['timestamp'].strftime('%Y-%m-%d')
        }
    }
    
    return payload

def send_to_n8n(payload):
    """
    Invia payload al webhook n8n
    """
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Segnale inviato a n8n: {payload['signal']} (score: {payload['score']})")
            print(f"📊 Prezzo: {payload['price']}")
            if payload['signal'] != "NEUTRAL":
                print(f"🎯 Stop Loss: {payload['levels']['stop_loss']}")
                print(f"🎯 Take Profit: {payload['levels']['take_profit']}")
            return True
        else:
            print(f"❌ Errore n8n: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore connessione n8n: {e}")
        return False

def main():
    """
    Workflow principale
    """
    print("🚀 LUXOR v7.1 AGGRESSIVE → n8n Sender")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Calcola segnale
    print("📈 Calcolo segnale LUXOR...")
    payload = get_luxor_signal()
    
    # Log locale
    print(f"\n📋 Segnale generato:")
    print(json.dumps(payload, indent=2))
    
    # Invia a n8n
    print(f"\n🔗 Invio a n8n: {N8N_WEBHOOK_URL}")
    success = send_to_n8n(payload)
    
    if success:
        print("\n✅ Operazione completata con successo")
        return 0
    else:
        print("\n❌ Operazione fallita")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
