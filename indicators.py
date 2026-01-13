#!/usr/bin/env python3
"""
LUXOR v7.1 - Technical Indicators
Essential indicators for the strategy
"""

import pandas as pd
import numpy as np
from typing import Union

# ============================================
# MOVING AVERAGES
# ============================================

def EMA(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average
    
    Args:
        series: Price series (typically close)
        period: EMA period
    
    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()

def SMA(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average
    
    Args:
        series: Price series
        period: SMA period
    
    Returns:
        SMA series
    """
    return series.rolling(window=period).mean()

# ============================================
# MOMENTUM INDICATORS
# ============================================

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index
    
    Args:
        series: Price series (typically close)
        period: RSI period (default 14)
    
    Returns:
        RSI series (0-100)
    """
    delta = series.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def MACD(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple:
    """
    Moving Average Convergence Divergence
    
    Args:
        series: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = EMA(series, fast_period)
    slow_ema = EMA(series, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = EMA(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

# ============================================
# VOLATILITY INDICATORS
# ============================================

def ATR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Average True Range
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
    
    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate historical volatility (standard deviation of returns)
    
    Args:
        series: Price series
        period: Lookback period
    
    Returns:
        Volatility series
    """
    returns = series.pct_change()
    volatility = returns.rolling(window=period).std()
    
    return volatility

def BollingerBands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple:
    """
    Bollinger Bands
    
    Args:
        series: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = SMA(series, period)
    std = series.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower

# ============================================
# TREND INDICATORS
# ============================================

def ADX(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Average Directional Index
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
    
    Returns:
        ADX series (0-100)
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate TR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth with Wilder's method
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

# ============================================
# VOLUME INDICATORS
# ============================================

def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume
    
    Args:
        close: Close prices
        volume: Volume data
    
    Returns:
        OBV series
    """
    obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
    return obv

def VWAP(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Volume Weighted Average Price
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
    
    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    return vwap

# ============================================
# SUPPORT/RESISTANCE
# ============================================

def find_pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> dict:
    """
    Calculate pivot points
    
    Args:
        high: High prices (use previous period)
        low: Low prices (use previous period)
        close: Close prices (use previous period)
    
    Returns:
        Dictionary with pivot, resistance, and support levels
    """
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    }

# ============================================
# PATTERN RECOGNITION
# ============================================

def detect_higher_highs_lows(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 20
) -> tuple:
    """
    Detect higher highs and higher lows (uptrend)
    
    Args:
        high: High prices
        low: Low prices
        lookback: Period to check
    
    Returns:
        Tuple of (has_higher_highs, has_higher_lows)
    """
    recent_high = high.iloc[-lookback:]
    recent_low = low.iloc[-lookback:]
    
    has_higher_highs = recent_high.iloc[-1] > recent_high.iloc[:-1].max()
    has_higher_lows = recent_low.iloc[-1] > recent_low.iloc[:-1].min()
    
    return has_higher_highs, has_higher_lows

# ============================================
# UTILITIES
# ============================================

def normalize_series(series: pd.Series, min_val: float = 0, max_val: float = 100) -> pd.Series:
    """
    Normalize series to a specific range
    
    Args:
        series: Input series
        min_val: Minimum value of output range
        max_val: Maximum value of output range
    
    Returns:
        Normalized series
    """
    series_min = series.min()
    series_max = series.max()
    
    if series_max == series_min:
        return pd.Series([50] * len(series), index=series.index)
    
    normalized = (series - series_min) / (series_max - series_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test indicators with synthetic data
    import matplotlib.pyplot as plt
    
    print("Testing LUXOR Indicators...")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close = pd.Series(40000 + np.cumsum(np.random.randn(100) * 200), index=dates)
    high = close + abs(np.random.randn(100) * 100)
    low = close - abs(np.random.randn(100) * 100)
    volume = pd.Series(np.random.uniform(1000, 5000, 100), index=dates)
    
    # Calculate indicators
    ema_21 = EMA(close, 21)
    ema_50 = EMA(close, 50)
    rsi = RSI(close, 14)
    atr = ATR(high, low, close, 14)
    
    print(f"\n✅ All indicators calculated successfully!")
    print(f"Latest Close: ${close.iloc[-1]:.2f}")
    print(f"Latest RSI: {rsi.iloc[-1]:.2f}")
    print(f"Latest ATR: ${atr.iloc[-1]:.2f}")
    print(f"Latest EMA 21: ${ema_21.iloc[-1]:.2f}")
    print(f"Latest EMA 50: ${ema_50.iloc[-1]:.2f}")
