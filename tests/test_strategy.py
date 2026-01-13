"""
Unit Tests for LUXOR Strategy
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.strategy import LuxorStrategy
from src.indicators import calculate_rsi, calculate_ema, detect_swing_pivots
from src.utils import load_config

@pytest.fixture
def config():
    """Load test configuration"""
    return load_config('config.json')

@pytest.fixture
def strategy(config):
    """Create strategy instance"""
    return LuxorStrategy(config)

def test_rsi_calculation():
    """Test RSI calculation"""
    prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42])
    rsi = calculate_rsi(prices, period=3)
    assert len(rsi) == len(prices)
    assert 0 <= rsi[-1] <= 100

def test_ema_calculation():
    """Test EMA calculation"""
    prices = np.array([10, 11, 12, 11, 10, 11, 12, 13])
    ema = calculate_ema(prices, period=3)
    assert len(ema) == len(prices)
    assert ema[-1] > 0

def test_swing_pivot_detection():
    """Test swing pivot detection"""
    highs = np.array([10, 12, 15, 13, 11, 14, 16, 14, 12])
    lows = np.array([8, 9, 11, 10, 8, 10, 12, 11, 9])
    
    pivot_highs, pivot_lows = detect_swing_pivots(highs, lows, lookback=2)
    
    assert len(pivot_highs) == len(highs)
    assert len(pivot_lows) == len(lows)
    assert pivot_highs.dtype == bool
    assert pivot_lows.dtype == bool

def test_score_calculation(strategy):
    """Test score calculation"""
    row = pd.Series({
        'rsi': 35,
        'ema_fast': 100,
        'ema_slow': 95,
        'volume': 1000,
        'volume_ma': 900,
        'volatility': 2.5,
        'pivot_high': False,
        'pivot_low': True
    })
    
    score = strategy.calculate_score(row)
    assert 0 <= score <= 100

def test_position_sizing(strategy):
    """Test position sizing logic"""
    # High confidence
    size_high = strategy.calculate_position_size(score=90)
    # Low confidence
    size_low = strategy.calculate_position_size(score=65)
    
    assert size_high > size_low

def test_stop_loss_calculation(strategy):
    """Test stop loss calculation"""
    entry_price = 100
    atr = 2
    
    sl_long = strategy.calculate_stop_loss(entry_price, 'LONG', atr)
    sl_short = strategy.calculate_stop_loss(entry_price, 'SHORT', atr)
    
    assert sl_long < entry_price
    assert sl_short > entry_price

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
