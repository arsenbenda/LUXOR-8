"""
LUXOR Trading Strategy - BTC Daily Signals
Uses: TA-Lib indicators + Gann High/Low + Entry Logic
Version: 7.1.0 (production ready)
"""

import ccxt
import pandas as pd
import numpy as np
import talib

# Import Gann module (ensure gann_high_low.py is in same directory)
from gann_high_low import calculate_gann_levels

def luxor_strategy():
    """
    Execute LUXOR strategy on BTC/USDT (Binance)
    Returns: dict with signal (BUY/SELL/WAIT) + indicators
    """
    
    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Fetch OHLCV data (1 day timeframe, last 100 candles)
    symbol = 'BTC/USDT'
    timeframe = '1d'
    limit = 100
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate indicators using TA-Lib
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # RSI (14 periods)
    rsi = talib.RSI(close, timeperiod=14)
    
    # MACD (12, 26, 9)
    macd, macd_signal, macd_hist = talib.MACD(close, 
                                               fastperiod=12, 
                                               slowperiod=26, 
                                               signalperiod=9)
    
    # Bollinger Bands (20, 2)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 
                                                   timeperiod=20, 
                                                   nbdevup=2, 
                                                   nbdevdn=2)
    
    # ATR (14 periods) for volatility
    atr = talib.ATR(high, low, close, timeperiod=14)
    
    # Calculate Gann Levels (using last 20 candles)
    gann_levels = calculate_gann_levels(df.tail(20))
    
    # Current values (last candle)
    current_price = float(close[-1])
    current_rsi = float(rsi[-1])
    current_macd = float(macd[-1])
    current_macd_signal = float(macd_signal[-1])
    current_atr = float(atr[-1])
    
    # Entry Logic (simplified LUXOR rules)
    signal = "WAIT"
    
    # BUY conditions
    if (current_rsi < 30 and  # Oversold
        current_macd > current_macd_signal and  # MACD bullish crossover
        current_price < bb_lower[-1]):  # Price below lower BB
        signal = "BUY"
    
    # SELL conditions
    elif (current_rsi > 70 and  # Overbought
          current_macd < current_macd_signal and  # MACD bearish crossover
          current_price > bb_upper[-1]):  # Price above upper BB
        signal = "SELL"
    
    # Build response
    return {
        "signal": signal,
        "price": current_price,
        "indicators": {
            "rsi": round(current_rsi, 2),
            "macd": round(current_macd, 2),
            "macd_signal": round(current_macd_signal, 2),
            "atr": round(current_atr, 2),
            "bb_upper": round(float(bb_upper[-1]), 2),
            "bb_lower": round(float(bb_lower[-1]), 2)
        },
        "gann_levels": gann_levels,
        "timestamp": str(df['timestamp'].iloc[-1])
    }

# Test execution (comment out in production)
if __name__ == "__main__":
    result = luxor_strategy()
    print(result)
