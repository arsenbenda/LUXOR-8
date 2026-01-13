"""
Technical Indicators Module
Calculates RSI, EMA, Volume, ATR and swing pivots
"""

import numpy as np
import pandas as pd

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr

def detect_swing_pivots(highs, lows, lookback=5):
    """
    Detect swing high/low pivots
    Returns: (pivot_highs, pivot_lows) boolean arrays
    """
    n = len(highs)
    pivot_highs = np.zeros(n, dtype=bool)
    pivot_lows = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n - lookback):
        # Swing high: current high > all highs in lookback window
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            pivot_highs[i] = True
        
        # Swing low: current low < all lows in lookback window
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            pivot_lows[i] = True
    
    return pivot_highs, pivot_lows

def calculate_volatility(close, period=20):
    """Calculate rolling volatility (standard deviation of returns)"""
    returns = np.diff(close) / close[:-1]
    volatility = np.zeros(len(close))
    volatility[0] = 0
    
    for i in range(1, len(close)):
        start_idx = max(0, i - period)
        volatility[i] = np.std(returns[start_idx:i]) if i > start_idx else 0
    
    return volatility * 100  # Convert to percentage

def calculate_indicators(df, config):
    """
    Calculate all technical indicators
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary
    
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    # RSI
    rsi_period = config.get('indicators', {}).get('rsi_period', 14)
    df['rsi'] = calculate_rsi(df['close'].values, rsi_period)
    
    # EMA
    ema_fast = config.get('indicators', {}).get('ema_fast', 12)
    ema_slow = config.get('indicators', {}).get('ema_slow', 26)
    df['ema_fast'] = calculate_ema(df['close'].values, ema_fast)
    df['ema_slow'] = calculate_ema(df['close'].values, ema_slow)
    
    # Volume MA
    vol_period = config.get('indicators', {}).get('volume_ma_period', 20)
    df['volume_ma'] = df['volume'].rolling(window=vol_period).mean()
    
    # ATR
    atr_period = config.get('indicators', {}).get('atr_period', 14)
    df['atr'] = calculate_atr(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        atr_period
    )
    
    # Volatility
    df['volatility'] = calculate_volatility(df['close'].values, 20)
    
    # Swing pivots
    lookback = config.get('parameters', {}).get('entry', {}).get('swing_lookback', 5)
    pivot_highs, pivot_lows = detect_swing_pivots(
        df['high'].values,
        df['low'].values,
        lookback
    )
    df['pivot_high'] = pivot_highs
    df['pivot_low'] = pivot_lows
    
    return df
