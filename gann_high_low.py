"""
Gann High/Low Levels Calculator - Pure Python Implementation
No external dependencies required

Calculates support and resistance levels based on Gann analysis
"""

import pandas as pd
from typing import Dict


def calculate_gann_levels(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 20
) -> Dict[str, float]:
    """
    Calculate Gann high/low support and resistance levels
    
    Args:
        high: Series of high prices
        low: Series of low prices
        lookback: Number of periods to look back (default: 20)
    
    Returns:
        Dictionary with Gann levels:
        - resistance_2: Strong resistance (high of lookback period)
        - resistance_1: Moderate resistance (75% level)
        - pivot: Pivot point (50% level)
        - support_1: Moderate support (25% level)
        - support_2: Strong support (low of lookback period)
    """
    if len(high) < lookback or len(low) < lookback:
        # Return current values if insufficient data
        current_high = high.iloc[-1] if len(high) > 0 else 0.0
        current_low = low.iloc[-1] if len(low) > 0 else 0.0
        pivot = (current_high + current_low) / 2
        
        return {
            "resistance_2": current_high,
            "resistance_1": pivot + (current_high - pivot) / 2,
            "pivot": pivot,
            "support_1": pivot - (pivot - current_low) / 2,
            "support_2": current_low
        }
    
    # Calculate highest high and lowest low over lookback period
    period_high = high.iloc[-lookback:].max()
    period_low = low.iloc[-lookback:].min()
    
    # Calculate range
    range_hl = period_high - period_low
    
    # Gann levels (quarters of the range)
    pivot = (period_high + period_low) / 2
    resistance_1 = pivot + (range_hl * 0.25)
    resistance_2 = period_high
    support_1 = pivot - (range_hl * 0.25)
    support_2 = period_low
    
    return {
        "resistance_2": float(resistance_2),
        "resistance_1": float(resistance_1),
        "pivot": float(pivot),
        "support_1": float(support_1),
        "support_2": float(support_2),
        "range": float(range_hl),
        "lookback_periods": lookback
    }


def calculate_gann_angles(
    price: float,
    time_units: int = 1,
    base_angle: float = 45.0
) -> Dict[str, float]:
    """
    Calculate Gann angles for trend analysis
    
    Args:
        price: Current price level
        time_units: Number of time units (default: 1)
        base_angle: Base angle in degrees (default: 45° - 1x1 line)
    
    Returns:
        Dictionary with Gann angles:
        - angle_1x8: Steep support (1x8 = 82.5°)
        - angle_1x4: Strong support (1x4 = 75°)
        - angle_1x2: Medium support (1x2 = 63.4°)
        - angle_1x1: Main trend line (1x1 = 45°)
        - angle_2x1: Medium resistance (2x1 = 26.6°)
        - angle_4x1: Strong resistance (4x1 = 14°)
        - angle_8x1: Steep resistance (8x1 = 7.1°)
    """
    import math
    
    # Calculate price changes for different Gann angles
    # Angle = arctan(price/time)
    
    angles = {
        "angle_1x8": price + (time_units * price * 8),      # 82.5°
        "angle_1x4": price + (time_units * price * 4),      # 75°
        "angle_1x2": price + (time_units * price * 2),      # 63.4°
        "angle_1x1": price + (time_units * price * 1),      # 45° (main)
        "angle_2x1": price + (time_units * price * 0.5),    # 26.6°
        "angle_4x1": price + (time_units * price * 0.25),   # 14°
        "angle_8x1": price + (time_units * price * 0.125),  # 7.1°
    }
    
    return {k: float(v) for k, v in angles.items()}
