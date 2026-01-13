#!/usr/bin/env python3
"""
LUXOR v7.1 - Gann High/Low Detector
Adaptive lookback based on volatility and timeframe
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class GannLevels:
    """Gann high/low levels with metadata"""
    high: float
    low: float
    high_idx: int
    low_idx: int
    lookback_used: int
    confidence: float  # 0-100
    volatility_regime: str  # "low", "normal", "high", "extreme"
    
    @property
    def range_pct(self) -> float:
        """Calculate percentage range between high and low"""
        return ((self.high - self.low) / self.low) * 100

# ============================================
# LOOKBACK CONFIGURATIONS
# ============================================

LOOKBACK_CONFIG = {
    "1D": {
        "min": 52,
        "max": 260,
        "optimal": 90,
        "description": "~3 months (intermediate cycle)"
    },
    "3D": {
        "min": 40,
        "max": 80,
        "optimal": 60,
        "description": "~6 months (medium cycle)"
    },
    "1W": {
        "min": 26,
        "max": 52,
        "optimal": 39,
        "description": "~9 months (long cycle)"
    },
    "1M": {
        "min": 12,
        "max": 24,
        "optimal": 18,
        "description": "~1.5 years (major cycle)"
    }
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def get_volatility_regime(atr: pd.Series, lookback: int = 50) -> str:
    """
    Determine current volatility regime
    
    Args:
        atr: ATR series
        lookback: Period for volatility comparison
    
    Returns:
        Volatility regime: "low", "normal", "high", "extreme"
    """
    if len(atr) < lookback:
        return "normal"
    
    current_atr = atr.iloc[-1]
    atr_mean = atr.iloc[-lookback:].mean()
    atr_std = atr.iloc[-lookback:].std()
    
    # Calculate z-score
    z_score = (current_atr - atr_mean) / atr_std if atr_std > 0 else 0
    
    if z_score > 2.0:
        return "extreme"
    elif z_score > 1.0:
        return "high"
    elif z_score < -1.0:
        return "low"
    else:
        return "normal"

def adjust_lookback_for_volatility(
    base_lookback: int,
    volatility_regime: str,
    config: Dict
) -> int:
    """
    Adjust lookback based on volatility regime
    
    High volatility → shorter lookback (more reactive)
    Low volatility → longer lookback (more stable)
    
    Args:
        base_lookback: Base lookback period
        volatility_regime: Current volatility regime
        config: Lookback configuration dict
    
    Returns:
        Adjusted lookback period
    """
    min_lookback = config["min"]
    max_lookback = config["max"]
    
    if volatility_regime == "extreme":
        # Very reactive in extreme volatility
        adjusted = int(base_lookback * 0.7)
    elif volatility_regime == "high":
        # More reactive in high volatility
        adjusted = int(base_lookback * 0.85)
    elif volatility_regime == "low":
        # More stable in low volatility
        adjusted = int(base_lookback * 1.2)
    else:
        # Normal volatility - use base
        adjusted = base_lookback
    
    # Clamp to min/max
    return max(min_lookback, min(max_lookback, adjusted))

def calculate_level_confidence(
    df: pd.DataFrame,
    high_idx: int,
    low_idx: int,
    lookback: int
) -> float:
    """
    Calculate confidence score for Gann levels (0-100)
    
    Higher confidence when:
    - Levels are recent but not too recent
    - Clear distinction from surrounding price action
    - Multiple touches/tests of the level
    
    Args:
        df: Price DataFrame
        high_idx: Index of high level
        low_idx: Index of low level
        lookback: Lookback period used
    
    Returns:
        Confidence score (0-100)
    """
    confidence = 50.0  # Base confidence
    
    # Age factor (prefer not too old, not too recent)
    high_age = len(df) - high_idx
    low_age = len(df) - low_idx
    
    optimal_age = lookback // 3  # Optimal ~1/3 of lookback
    
    # Penalize if too recent (< 5 candles) or too old (> 80% of lookback)
    for age in [high_age, low_age]:
        if age < 5:
            confidence -= 10
        elif age > lookback * 0.8:
            confidence -= 15
        elif abs(age - optimal_age) < lookback * 0.2:
            confidence += 10
    
    # Range factor (prefer clear separation)
    high = df.iloc[high_idx]['high']
    low = df.iloc[low_idx]['low']
    range_pct = ((high - low) / low) * 100
    
    if range_pct > 15:  # Good range
        confidence += 15
    elif range_pct > 10:
        confidence += 10
    elif range_pct < 3:  # Too narrow
        confidence -= 20
    
    # Test factor (check if levels were tested/respected)
    recent_prices = df.iloc[-lookback//2:]
    
    high_tests = (recent_prices['high'] > high * 0.99).sum()
    low_tests = (recent_prices['low'] < low * 1.01).sum()
    
    if high_tests >= 2:
        confidence += 10
    if low_tests >= 2:
        confidence += 10
    
    # Clamp to 0-100
    return max(0, min(100, confidence))

# ============================================
# MAIN FUNCTION
# ============================================

def find_gann_high_low(
    df: pd.DataFrame,
    timeframe: str = "1D",
    use_adaptive: bool = True,
    atr_period: int = 14
) -> GannLevels:
    """
    Find Gann high and low levels with adaptive lookback
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe string ("1D", "3D", "1W", "1M")
        use_adaptive: Whether to use adaptive lookback based on volatility
        atr_period: Period for ATR calculation
    
    Returns:
        GannLevels object with high/low levels and metadata
    
    Example:
        >>> df = pd.read_csv('btc_daily.csv')
        >>> levels = find_gann_high_low(df, timeframe="1D", use_adaptive=True)
        >>> print(f"High: ${levels.high:.2f}, Low: ${levels.low:.2f}")
        >>> print(f"Confidence: {levels.confidence:.1f}%")
    """
    # Get lookback config
    if timeframe not in LOOKBACK_CONFIG:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(LOOKBACK_CONFIG.keys())}")
    
    config = LOOKBACK_CONFIG[timeframe]
    base_lookback = config["optimal"]
    
    # Calculate ATR and volatility regime
    atr = calculate_atr(df, period=atr_period)
    volatility_regime = get_volatility_regime(atr)
    
    # Adjust lookback if adaptive
    if use_adaptive:
        lookback = adjust_lookback_for_volatility(base_lookback, volatility_regime, config)
    else:
        lookback = base_lookback
    
    # Ensure we have enough data
    if len(df) < lookback:
        lookback = len(df)
    
    # Find high and low in lookback period
    lookback_data = df.iloc[-lookback:]
    
    high_idx = lookback_data['high'].idxmax()
    low_idx = lookback_data['low'].idxmin()
    
    high_value = df.loc[high_idx, 'high']
    low_value = df.loc[low_idx, 'low']
    
    # Calculate confidence
    confidence = calculate_level_confidence(df, high_idx, low_idx, lookback)
    
    return GannLevels(
        high=high_value,
        low=low_value,
        high_idx=high_idx,
        low_idx=low_idx,
        lookback_used=lookback,
        confidence=confidence,
        volatility_regime=volatility_regime
    )

# ============================================
# MULTI-TIMEFRAME ANALYSIS
# ============================================

def get_multi_timeframe_gann(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, GannLevels]:
    """
    Get Gann levels across multiple timeframes
    
    Args:
        df_dict: Dictionary mapping timeframe to DataFrame
                 e.g., {"1D": df_daily, "1W": df_weekly}
    
    Returns:
        Dictionary mapping timeframe to GannLevels
    
    Example:
        >>> dfs = {
        ...     "1D": pd.read_csv('btc_daily.csv'),
        ...     "1W": pd.read_csv('btc_weekly.csv')
        ... }
        >>> levels = get_multi_timeframe_gann(dfs)
        >>> print(levels["1D"].high, levels["1W"].high)
    """
    results = {}
    
    for timeframe, df in df_dict.items():
        try:
            levels = find_gann_high_low(df, timeframe=timeframe)
            results[timeframe] = levels
        except Exception as e:
            print(f"Error processing {timeframe}: {e}")
    
    return results

# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Example with synthetic data
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    np.random.seed(42)
    
    # Simulate BTC price data
    price = 40000 + np.cumsum(np.random.randn(300) * 500)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(300) * 100,
        'high': price + abs(np.random.randn(300) * 200),
        'low': price - abs(np.random.randn(300) * 200),
        'close': price + np.random.randn(300) * 100,
        'volume': np.random.uniform(1000, 5000, 300)
    })
    
    # Find Gann levels
    print("=" * 60)
    print("LUXOR v7.1 - Gann High/Low Analysis")
    print("=" * 60)
    
    for timeframe in ["1D", "3D", "1W", "1M"]:
        config = LOOKBACK_CONFIG[timeframe]
        print(f"\n📊 {timeframe} Timeframe ({config['description']})")
        print("-" * 60)
        
        # Adaptive lookback
        levels = find_gann_high_low(df, timeframe=timeframe, use_adaptive=True)
        
        print(f"🔼 Gann High:      ${levels.high:,.2f}")
        print(f"🔽 Gann Low:       ${levels.low:,.2f}")
        print(f"📏 Range:          {levels.range_pct:.2f}%")
        print(f"🔢 Lookback Used:  {levels.lookback_used} candles")
        print(f"🎯 Confidence:     {levels.confidence:.1f}/100")
        print(f"💨 Volatility:     {levels.volatility_regime.upper()}")
        
        # Fixed lookback comparison
        levels_fixed = find_gann_high_low(df, timeframe=timeframe, use_adaptive=False)
        print(f"\n   (Fixed lookback: {levels_fixed.lookback_used} candles)")
        print(f"   Difference: {abs(levels.high - levels_fixed.high):.2f} / {abs(levels.low - levels_fixed.low):.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)
