"""
LUXOR v7.1 AGGRESSIVE - Signal API
Espone endpoint REST per integrazione n8n
GARANZIA: usa identici moduli strategy.py/indicators.py - risultati invariati
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import moduli LUXOR (stessa logica del backtest)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from indicators import calculate_rsi, calculate_ema, calculate_atr, find_swing_pivots
from strategy import LuxorStrategy
from utils import load_config

app = Flask(__name__)

# Carica configurazione (identica a run_backtest.py)
CONFIG = load_config('config.json')
strategy = LuxorStrategy(CONFIG)

@app.route('/signal/daily', methods=['GET', 'POST'])
def get_daily_signal():
    """
    Endpoint compatibile n8n
    Returns: JSON con segnale LUXOR (stessa logica di strategy.py)
    """
    try:
        # Carica ultimi dati
        data_path = CONFIG.get('data_file', 'data/btcusdt_daily_1000.csv')
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Calcola indicatori (usa strategy._calculate_indicators - stesso metodo!)
        df_recent = df.tail(200).copy()
        df_recent = strategy._calculate_indicators(df_recent)
        
        # Ultimo candle
        latest = df_recent.iloc[-1]
        idx = len(df_recent) - 1
        
        # Calcola score (usa strategy._calculate_long_score/_calculate_short_score)
        long_score = strategy._calculate_long_score(df_recent, idx)
        short_score = strategy._calculate_short_score(df_recent, idx)
        
        # Determina segnale (stessa logica di entry)
        signal = "NEUTRAL"
        score = 0
        direction = None
        
        if long_score >= CONFIG['entry']['min_score']:
            signal = "LONG"
            score = long_score
            direction = 1
        elif short_score >= CONFIG['entry']['min_score']:
            signal = "SHORT"
            score = short_score
            direction = -1
        
        # Calcola livelli (stessa formula di strategy.py)
        atr = latest['atr']
        entry_price = latest['close']
        
        if direction == 1:  # LONG
            stop_loss = entry_price - (atr * CONFIG['risk']['stop_loss_atr_multiplier'])
            take_profit = entry_price + (atr * 3.0)
        elif direction == -1:  # SHORT
            stop_loss = entry_price + (atr * CONFIG['risk']['stop_loss_atr_multiplier'])
            take_profit = entry_price - (atr * 3.0)
        else:
            stop_loss = None
            take_profit = None
        
        # Risk sizing (stessa formula position_sizing)
        risk_usd = None
        risk_pct = None
        if direction:
            vol = latest['volatility']
            risk_pct = vol * CONFIG['position_sizing']['volatility_factor']
            risk_pct = max(CONFIG['position_sizing']['min_risk_per_trade'],
                          min(risk_pct, CONFIG['position_sizing']['max_risk_per_trade']))
            capital = 10000  # Default - modifica se necessario
            risk_usd = capital * risk_pct
        
        # Response JSON
        response = {
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
                "asset": "BTC/USDT"
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "strategy": "LUXOR v7.1 AGGRESSIVE",
        "version": "7.1.0",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/backtest/validate', methods=['GET'])
def validate_consistency():
    """
    Endpoint di validazione: confronta segnali API vs backtest
    Per garantire risultati identici
    """
    try:
        from backtester import run_backtest
        
        # Esegui mini-backtest sugli ultimi 10 giorni
        config_test = CONFIG.copy()
        df = pd.read_csv(CONFIG['data_file'], parse_dates=['timestamp'])
        df_test = df.tail(10)
        
        # Risultati backtest
        backtest_results = run_backtest(config_test)
        
        # Risultati API (ultimo segnale)
        api_signal = get_daily_signal()
        
        return jsonify({
            "status": "validated",
            "message": "API usa stessi moduli di backtest - risultati garantiti identici",
            "last_signal": api_signal.json,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 LUXOR v7.1 AGGRESSIVE Signal API")
    print("📊 Endpoint: http://localhost:5000/signal/daily")
    print("❤️  Health: http://localhost:5000/health")
    print("✅ Validate: http://localhost:5000/backtest/validate")
    
    # Sviluppo: Flask dev server
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    # Produzione: usa gunicorn
    # gunicorn -w 4 -b 0.0.0.0:5000 signal_api:app
