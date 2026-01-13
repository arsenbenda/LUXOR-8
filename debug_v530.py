#!/usr/bin/env python3
"""Debug script for v5.3.0 - Test first pivot"""

import pandas as pd
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, '/tmp/luxor-repo-update')

# Import from luxor_v530
from luxor_v530_ultra_optimized import (
    detect_pivots, calculate_gann_levels, calculate_volatility, 
    calculate_atr, calculate_enneagram_state, calculate_score_v530,
    calculate_dynamic_rr_min, calculate_position_size, CONFIG
)

# Load data
df = pd.read_csv('/tmp/luxor-repo-update/data/btcusdt_daily_1000.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2026-01-12')].reset_index(drop=True)

print("="*80)
print("DEBUG v5.3.0 - First Pivot Analysis")
print("="*80)

# Detect pivots
pivots_df = detect_pivots(df, CONFIG['lookback'])
print(f"\nTotal pivots detected: {len(pivots_df)}")

if len(pivots_df) == 0:
    print("ERROR: No pivots detected!")
    sys.exit(1)

# Test first pivot
first_pivot = pivots_df.iloc[0]
pivot_idx = first_pivot['index']
pivot_type = first_pivot['type']
pivot_price = first_pivot['price']

print(f"\nFirst Pivot:")
print(f"  Type: {pivot_type}")
print(f"  Date: {first_pivot['date']}")
print(f"  Price: {pivot_price:.2f}")
print(f"  Index: {pivot_idx}")

# Test first time bar
time_bars = [0, 7, 9, 13, 14, 18, 20, 21, 26, 28]
analysis_idx = pivot_idx + time_bars[0]

print(f"\nTesting time_bar={time_bars[0]} (analysis_idx={analysis_idx}):")
current_price = df['close'].iloc[analysis_idx]
current_date = df['date'].iloc[analysis_idx]

print(f"  Date: {current_date}")
print(f"  Close: {current_price:.2f}")

# Calculate metrics
volatility = calculate_volatility(df, analysis_idx)
atr = calculate_atr(df, analysis_idx)

print(f"  Volatility: {volatility:.2f}%")
print(f"  ATR: {atr:.2f}")

# GANN levels
gann_levels, gann_low, gann_high = calculate_gann_levels(
    pivot_price, current_price, pivot_type
)

print(f"\nGANN Levels:")
print(f"  Low: {gann_low:.2f}")
print(f"  High: {gann_high:.2f}")
print(f"  Range: {gann_high - gann_low:.2f}")

# Enneagram state
state, gain_pct, momentum = calculate_enneagram_state(df, pivot_idx, analysis_idx)

print(f"\nEnneagram State:")
print(f"  State: {state}")
print(f"  Gain%: {gain_pct:.2f}%")
print(f"  Momentum: {momentum:.2f}")

# Score calculation
time_score = 40
price_distance = min([abs(current_price - level) / current_price * 100 
                     for level in gann_levels.values()])
price_score = max(0, 40 - price_distance * 10)
state_score = 20 if momentum * gain_pct > 0 else 10

direction = 'LONG' if pivot_type == 'LOW' else 'SHORT'

final_score = calculate_score_v530(
    time_score, price_score, state_score, 
    volatility, direction, pivot_type
)

print(f"\nScore Breakdown:")
print(f"  Time Score: {time_score}")
print(f"  Price Score: {price_score:.2f} (distance: {price_distance:.2f}%)")
print(f"  State Score: {state_score}")
print(f"  Base Score: {time_score + price_score + state_score:.2f}")
print(f"  Direction: {direction}")
print(f"  FINAL SCORE v5.3.0: {final_score:.2f}")

# Entry filters
min_score_required = CONFIG['high_vol_min_score'] if volatility > CONFIG['volatility_threshold'] else CONFIG['min_score']

print(f"\nEntry Filters:")
print(f"  Min Score Required: {min_score_required}")
print(f"  HIGH VOL? {volatility > CONFIG['volatility_threshold']}")

if direction == 'SHORT':
    print(f"  SHORT Min Score: {CONFIG['short_min_score']}")
    print(f"  SHORT Filter: {'PASS' if final_score >= CONFIG['short_min_score'] else 'FAIL'}")

print(f"  Score Filter: {'PASS' if final_score >= min_score_required else 'FAIL'}")

# Setup trade
entry_price = current_price

if direction == 'LONG':
    sl_distance = entry_price - gann_levels['0/8']
    if volatility > CONFIG['volatility_threshold']:
        sl_distance *= CONFIG['high_vol_sl_multiplier']
    
    sl_price = entry_price - sl_distance
    tp1 = gann_levels['4/8']
    tp2 = gann_levels['6/8']
    tp3 = gann_levels['8/8']
else:  # SHORT
    sl_distance = gann_levels['8/8'] - entry_price
    if volatility > CONFIG['volatility_threshold']:
        sl_distance *= CONFIG['high_vol_sl_multiplier']
    
    sl_price = entry_price + sl_distance
    tp1 = gann_levels['4/8']
    tp2 = gann_levels['2/8']
    tp3 = gann_levels['0/8']

# R:R check
rr_ratio = abs(tp1 - entry_price) / abs(entry_price - sl_price)
dynamic_rr_min = calculate_dynamic_rr_min(volatility)

print(f"\nTrade Setup:")
print(f"  Entry: {entry_price:.2f}")
print(f"  SL: {sl_price:.2f}")
print(f"  TP1: {tp1:.2f}")
print(f"  TP2: {tp2:.2f}")
print(f"  TP3: {tp3:.2f}")
print(f"  R:R Ratio: {rr_ratio:.2f}")
print(f"  Dynamic R:R Min: {dynamic_rr_min:.2f}")
print(f"  R:R Filter: {'PASS' if rr_ratio >= dynamic_rr_min else 'FAIL'}")

position_size = calculate_position_size(volatility, final_score)
print(f"  Position Size: {position_size:.2f}")

# Final verdict
passes_score = final_score >= min_score_required
passes_short = (direction == 'LONG') or (final_score >= CONFIG['short_min_score'])
passes_rr = rr_ratio >= dynamic_rr_min

print(f"\n{'='*80}")
print(f"TRADE VERDICT: {'✅ EXECUTE' if (passes_score and passes_short and passes_rr) else '❌ REJECT'}")
print(f"{'='*80}")
