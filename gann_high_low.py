"""
Gann High/Low Calculation Module
Calculates Gann Square of 9 levels for support/resistance
Version: 1.0.0
"""

import pandas as pd
import numpy as np

def calculate_gann_levels(df, num_levels=5):
    """
    Calculate Gann levels based on Square of 9
    
    Args:
        df: DataFrame with OHLC data (must have 'high', 'low', 'close' columns)
        num_levels: Number of Gann levels to calculate (default 5)
    
    Returns:
        dict: {
            "gann_high": [level1, level2, ...],
            "gann_low": [level1, level2, ...],
            "pivot": float
        }
    """
    
    if len(df) < 3:
        return {"error": "Insufficient data for Gann calculation"}
    
    # Get recent high, low, close
    recent_high = df['high'].max()
    recent_low = df['low'].min()
    recent_close = df['close'].iloc[-1]
    
    # Pivot point (simple pivot)
    pivot = (recent_high + recent_low + recent_close) / 3
    
    # Gann Square of 9 formula (simplified)
    # Upper levels: pivot + (pivot * sqrt(n) * 0.01)
    # Lower levels: pivot - (pivot * sqrt(n) * 0.01)
    
    gann_high = []
    gann_low = []
    
    for i in range(1, num_levels + 1):
        factor = np.sqrt(i) * 0.01
        gann_high.append(round(pivot * (1 + factor), 2))
        gann_low.append(round(pivot * (1 - factor), 2))
    
    return {
        "gann_high": gann_high,
        "gann_low": gann_low,
        "pivot": round(pivot, 2)
    }

# Test execution
if __name__ == "__main__":
    # Mock data for testing
    test_data = {
        'high': [45000, 45500, 46000, 45800, 46200],
        'low': [44000, 44500, 45000, 44800, 45200],
        'close': [44500, 45000, 45500, 45300, 45800]
    }
    df_test = pd.DataFrame(test_data)
    
    levels = calculate_gann_levels(df_test)
    print("Gann Levels Test:")
    print(levels)
