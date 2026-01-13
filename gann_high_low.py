"""
Gann High/Low Calculator
"""

import pandas as pd
from typing import Dict


def calculate_gann_levels(high: pd.Series, low: pd.Series, lookback: int = 20) -> Dict:
    """
    Calculate Gann Square of 9 levels based on high/low range
    
    Args:
        high: Pandas Series of high prices
        low: Pandas Series of low prices
        lookback: Number of periods to look back (default 20)
    
    Returns:
        Dictionary with Gann levels
    """
    recent_high = float(high.tail(lookback).max())
    recent_low = float(low.tail(lookback).min())
    gann_range = recent_high - recent_low
    
    # Gann 50% (pivot)
    pivot = recent_low + (gann_range * 0.5)
    
    # Main levels
    resistance_2 = recent_low + (gann_range * 0.875)  # 7/8
    resistance_1 = recent_low + (gann_range * 0.75)   # 6/8
    support_1 = recent_low + (gann_range * 0.25)      # 2/8
    support_2 = recent_low + (gann_range * 0.125)     # 1/8
    
    return {
        'resistance_2': float(resistance_2),
        'resistance_1': float(resistance_1),
        'pivot': float(pivot),
        'support_1': float(support_1),
        'support_2': float(support_2),
        'range': float(gann_range),
        'lookback_periods': lookback
    }
