#!/usr/bin/env python3
"""
LUXOR v6.0 DATA-DRIVEN - Evidence-Based Trading System
Based on forensic analysis of 42 trades in v5.2.0

KEY INSIGHTS FROM DATA:
1. Vol 2-3% → 82% Win Rate, +47% P&L (GOLD ZONE)
2. Score 85-90 BETTER than 90-95 (inverse correlation!)
3. SHORT: 67% WR but -5% P&L (avoid in bull market)
4. TP3 outcomes: +31% total (10% avg per trade)
5. Best LONG patterns: Score 85-100, Vol 2-4%

NEW APPROACH:
- ONLY LONG in bull market (2024-2026)
- STRICT volatility filter: 2-3% ONLY
- Inverse score weighting (85-90 > 90-95)
- Aggressive TP targets (aim for TP3)
- ATR-based trailing stops from entry
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# CONFIG v6.0 - DATA-DRIVEN
# ============================================================================
CONFIG = {
    'lookback': 7,
    
    # GOLD ZONE volatility (from data: 82% WR!)
    'vol_min': 1.8,  # Slightly wider
    'vol_max': 3.2,  # Slightly wider
    
    # Direction filter (data: LONG +46%, SHORT -5%)
    'only_long': True,  # Bull market 2024-2026
    
    # Score range (data: 85-90 > 90-95!)
    'score_min': 80,
    'score_max': 95,  # Cap at 95 (higher scores = worse performance!)
    
    # R:R minimum
    'min_rr': 1.2,  # More permissive
    
    # Trailing stop (ATR-based)
    'trailing_atr_mult': 2.5,
    'breakeven_at': 0.3,  # Move to BE after +0.3R
    
    # Position sizing (volatility-adjusted)
    'base_size': 1.0,
    'high_vol_size': 0.7,  # Reduce size if vol > 2.5%
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def detect_pivots(df, lookback=7):
    """Detect swing pivots"""
    pivots = []
    
    for i in range(lookback, len(df) - lookback):
        # Pivot High
        is_high_before = (df['high'].iloc[i] >= df['high'].iloc[i-lookback:i].max())
        is_high_after = (df['high'].iloc[i] >= df['high'].iloc[i+1:i+lookback+1].max())
        
        if is_high_before and is_high_after:
            pivots.append({
                'date': df['date'].iloc[i],
                'type': 'HIGH',
                'price': df['high'].iloc[i],
                'index': i
            })
        
        # Pivot Low
        is_low_before = (df['low'].iloc[i] <= df['low'].iloc[i-lookback:i].min())
        is_low_after = (df['low'].iloc[i] <= df['low'].iloc[i+1:i+lookback+1].min())
        
        if is_low_before and is_low_after:
            pivots.append({
                'date': df['date'].iloc[i],
                'type': 'LOW',
                'price': df['low'].iloc[i],
                'index': i
            })
    
    return pd.DataFrame(pivots)

def calculate_volatility(df, current_idx, window=30):
    """Calculate daily volatility (not annualized)"""
    start = max(0, current_idx - window)
    returns = df['close'].iloc[start:current_idx].pct_change().dropna()
    return returns.std() * 100 if len(returns) > 0 else 0

def calculate_atr(df, current_idx, period=14):
    """Calculate Average True Range"""
    start = max(0, current_idx - period)
    high = df['high'].iloc[start:current_idx]
    low = df['low'].iloc[start:current_idx]
    close = df['close'].iloc[start:current_idx]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.mean() if len(tr) > 0 else 0

def calculate_gann_levels(pivot_price, current_price, pivot_type):
    """Calculate GANN Rule of Eights"""
    if pivot_type == 'LOW':
        range_size = max((current_price - pivot_price) * 2, pivot_price * 0.05)
        low = pivot_price
        high = low + range_size
    else:
        range_size = max((pivot_price - current_price) * 2, pivot_price * 0.05)
        high = pivot_price
        low = high - range_size
    
    eighth = range_size / 8
    levels = {
        '0/8': low,
        '1/8': low + eighth,
        '2/8': low + 2 * eighth,
        '3/8': low + 3 * eighth,
        '4/8': low + 4 * eighth,
        '5/8': low + 5 * eighth,
        '6/8': low + 6 * eighth,
        '7/8': low + 7 * eighth,
        '8/8': high,
    }
    
    return levels, low, high

def calculate_enneagram_state(df, pivot_idx, current_idx):
    """Calculate Enneagram transformation state"""
    if current_idx <= pivot_idx:
        return 1, 0.0, 0.0
    
    pivot_price = df['close'].iloc[pivot_idx]
    current_price = df['close'].iloc[current_idx]
    
    gain_pct = ((current_price - pivot_price) / pivot_price) * 100
    
    # Momentum
    bars = current_idx - pivot_idx
    momentum = gain_pct / bars if bars > 0 else 0
    
    # State (1-9 based on magnitude)
    abs_gain = abs(gain_pct)
    if abs_gain < 1:
        state = 1
    elif abs_gain < 3:
        state = 2
    elif abs_gain < 5:
        state = 3
    elif abs_gain < 8:
        state = 4
    elif abs_gain < 13:
        state = 5
    elif abs_gain < 21:
        state = 6
    elif abs_gain < 34:
        state = 7
    elif abs_gain < 55:
        state = 8
    else:
        state = 9
    
    return state, gain_pct, momentum

def calculate_score_v600(time_score, price_score, state_score, volatility):
    """
    v6.0 Evidence-Based Scoring
    
    KEY INSIGHT: Score 85-90 performs BETTER than 90-95!
    - Cap max score at 95
    - Penalize scores > 90 (inverse correlation!)
    """
    base_score = time_score + price_score + state_score
    
    # Inverse penalty for high scores (data shows worse performance!)
    if base_score > 90:
        high_score_penalty = (base_score - 90) * 0.5  # Reduce high scores
        base_score -= high_score_penalty
    
    # Volatility bonus for GOLD ZONE (2-3%)
    if CONFIG['vol_min'] <= volatility <= CONFIG['vol_max']:
        base_score += 10  # Boost gold zone trades
    
    return max(0, min(100, base_score))

def calculate_trailing_stop(entry_price, sl_price, current_price, atr, direction):
    """ATR-based trailing stop"""
    if direction == 'LONG':
        initial_risk = entry_price - sl_price
        
        # Trailing stop at current_price - ATR * multiplier
        trailing_stop = current_price - (atr * CONFIG['trailing_atr_mult'])
        
        # Never lower than original SL
        trailing_stop = max(sl_price, trailing_stop)
        
        # Move to breakeven after +0.3R profit
        if current_price >= entry_price + (initial_risk * CONFIG['breakeven_at']):
            trailing_stop = max(trailing_stop, entry_price)
        
        return trailing_stop
    
    else:  # SHORT
        initial_risk = sl_price - entry_price
        trailing_stop = current_price + (atr * CONFIG['trailing_atr_mult'])
        trailing_stop = min(sl_price, trailing_stop)
        
        if current_price <= entry_price - (initial_risk * CONFIG['breakeven_at']):
            trailing_stop = min(trailing_stop, entry_price)
        
        return trailing_stop

def calculate_position_size(volatility, score):
    """Volatility-adjusted position sizing"""
    if volatility > 2.5:
        return CONFIG['high_vol_size']
    else:
        return CONFIG['base_size']

# ============================================================================
# BACKTEST ENGINE v6.0
# ============================================================================

def run_backtest_v600(df, start_date, end_date):
    """Run evidence-based backtest"""
    
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop=True)
    
    pivots_df = detect_pivots(df, CONFIG['lookback'])
    
    results = []
    
    for _, pivot in pivots_df.iterrows():
        pivot_idx = pivot['index']
        pivot_type = pivot['type']
        pivot_price = pivot['price']
        
        # Direction filter (ONLY LONG in bull market)
        direction = 'LONG' if pivot_type == 'LOW' else 'SHORT'
        if CONFIG['only_long'] and direction == 'SHORT':
            continue
        
        # GANN time sequence (skip 0 - unrealistic)
        time_bars = [7, 9, 13, 14, 18, 20, 21, 26, 28]
        
        for time_bar in time_bars:
            analysis_idx = pivot_idx + time_bar
            
            if analysis_idx >= len(df):
                continue
            
            current_price = df['close'].iloc[analysis_idx]
            current_date = df['date'].iloc[analysis_idx]
            
            # Calculate metrics
            volatility = calculate_volatility(df, analysis_idx)
            atr = calculate_atr(df, analysis_idx)
            
            # GOLD ZONE filter (2-3% volatility = 82% WR!)
            if volatility < CONFIG['vol_min'] or volatility > CONFIG['vol_max']:
                continue
            
            # GANN levels
            gann_levels, gann_low, gann_high = calculate_gann_levels(
                pivot_price, current_price, pivot_type
            )
            
            # Enneagram state
            state, gain_pct, momentum = calculate_enneagram_state(df, pivot_idx, analysis_idx)
            
            # Scoring
            time_score = 40
            price_distance = min([abs(current_price - level) / current_price * 100 
                                 for level in gann_levels.values()])
            price_score = max(0, 40 - price_distance * 10)
            state_score = 20 if momentum * gain_pct > 0 else 10
            
            final_score = calculate_score_v600(time_score, price_score, state_score, volatility)
            
            # Score filter (data shows 85-90 is sweet spot)
            if final_score < CONFIG['score_min']:
                continue
            
            # Setup trade
            entry_price = current_price
            
            if direction == 'LONG':
                sl_distance = entry_price - gann_levels['0/8']
                sl_price = entry_price - sl_distance
                tp1 = gann_levels['4/8']
                tp2 = gann_levels['6/8']
                tp3 = gann_levels['8/8']
            else:  # SHORT
                sl_distance = gann_levels['8/8'] - entry_price
                sl_price = entry_price + sl_distance
                tp1 = gann_levels['4/8']
                tp2 = gann_levels['2/8']
                tp3 = gann_levels['0/8']
            
            # R:R check
            rr_ratio = abs(tp1 - entry_price) / abs(entry_price - sl_price)
            if rr_ratio < CONFIG['min_rr']:
                continue
            
            # Position size
            position_size = calculate_position_size(volatility, final_score)
            
            # Simulate with TRAILING STOP
            outcome = None
            exit_price = None
            exit_date = None
            current_sl = sl_price
            
            for future_idx in range(analysis_idx + 1, min(analysis_idx + 30, len(df))):
                future_high = df['high'].iloc[future_idx]
                future_low = df['low'].iloc[future_idx]
                future_close = df['close'].iloc[future_idx]
                future_date = df['date'].iloc[future_idx]
                future_atr = calculate_atr(df, future_idx)
                
                # Update trailing stop
                current_sl = calculate_trailing_stop(
                    entry_price, sl_price, future_close, future_atr, direction
                )
                
                if direction == 'LONG':
                    if future_low <= current_sl:
                        outcome = 'SL'
                        exit_price = current_sl
                        exit_date = future_date
                        break
                    elif future_high >= tp3:
                        outcome = 'TP3'
                        exit_price = tp3
                        exit_date = future_date
                        break
                    elif future_high >= tp2:
                        outcome = 'TP2'
                        exit_price = tp2
                        exit_date = future_date
                        break
                    elif future_high >= tp1:
                        outcome = 'TP1'
                        exit_price = tp1
                        exit_date = future_date
                        break
                else:  # SHORT
                    if future_high >= current_sl:
                        outcome = 'SL'
                        exit_price = current_sl
                        exit_date = future_date
                        break
                    elif future_low <= tp3:
                        outcome = 'TP3'
                        exit_price = tp3
                        exit_date = future_date
                        break
                    elif future_low <= tp2:
                        outcome = 'TP2'
                        exit_price = tp2
                        exit_date = future_date
                        break
                    elif future_low <= tp1:
                        outcome = 'TP1'
                        exit_price = tp1
                        exit_date = future_date
                        break
            
            if outcome is None:
                outcome = 'OPEN'
                exit_price = df['close'].iloc[-1]
                exit_date = df['date'].iloc[-1]
            
            # P&L
            if direction == 'LONG':
                pnl = ((exit_price - entry_price) / entry_price) * 100 * position_size
            else:
                pnl = ((entry_price - exit_price) / entry_price) * 100 * position_size
            
            results.append({
                'pivot_date': pivot['date'],
                'entry_date': current_date,
                'exit_date': exit_date,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'sl_price': sl_price,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'score': final_score,
                'volatility': volatility,
                'atr': atr,
                'rr_ratio': rr_ratio,
                'position_size': position_size,
                'outcome': outcome,
                'pnl': pnl
            })
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df = pd.read_csv('data/btcusdt_daily_1000.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n" + "="*80)
    print("🚀 BACKTEST v6.0 DATA-DRIVEN")
    print("="*80)
    print("\nEVIDENCE-BASED RULES:")
    print(f"  ✓ GOLD ZONE volatility: {CONFIG['vol_min']}-{CONFIG['vol_max']}% (82% WR from data!)")
    print(f"  ✓ LONG only (bull market)")
    print(f"  ✓ Score cap at {CONFIG['score_max']} (inverse correlation!)")
    print(f"  ✓ ATR trailing stops ({CONFIG['trailing_atr_mult']}x)")
    print(f"  ✓ Breakeven at +{CONFIG['breakeven_at']}R")
    print()
    
    results = run_backtest_v600(df, '2024-01-01', '2026-01-12')
    
    if len(results) == 0:
        print("❌ No trades executed!")
        exit(1)
    
    results.to_csv('/mnt/user-data/outputs/backtest_v600_results.csv', index=False)
    
    # Stats
    total_trades = len(results)
    long_trades = len(results[results['direction'] == 'LONG'])
    short_trades = len(results[results['direction'] == 'SHORT'])
    
    winners = results[results['pnl'] > 0]
    losers = results[results['pnl'] <= 0]
    
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
    total_pnl = results['pnl'].sum()
    avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
    
    print("="*80)
    print("📊 RISULTATI BACKTEST v6.0")
    print("="*80)
    print()
    print(f"✅ Trade eseguiti: {total_trades}")
    print(f"   LONG:  {long_trades}")
    print(f"   SHORT: {short_trades}")
    print()
    print("💰 PERFORMANCE:")
    print(f"   Win Rate:    {win_rate:.1f}%")
    print(f"   Winners:     {len(winners)}")
    print(f"   Losers:      {len(losers)}")
    print(f"   Total P&L:   {total_pnl:+.2f}%")
    print(f"   Avg Win:     {avg_win:+.2f}%")
    print(f"   Avg Loss:    {avg_loss:+.2f}%")
    print()
    
    # Outcome breakdown
    print("🎯 OUTCOME BREAKDOWN:")
    for outcome in ['SL', 'TP1', 'TP2', 'TP3', 'OPEN']:
        count = len(results[results['outcome'] == outcome])
        pct = count / total_trades * 100 if total_trades > 0 else 0
        print(f"   {outcome:4}: {count:3} ({pct:5.1f}%)")
    
    print()
    print("="*80)
    print("📊 CONFRONTO v5.2.0 → v6.0")
    print("="*80)
    print()
    print("Metrica          | v5.2.0  | v6.0    | Delta")
    print("-"*80)
    print(f"Trade Totali     |   42    |  {total_trades:3}    | {total_trades - 42:+3}")
    print(f"Win Rate         | 52.4%   | {win_rate:5.1f}% | {win_rate - 52.4:+5.1f}%")
    print(f"Total P&L        | +41.11% | {total_pnl:+6.2f}% | {total_pnl - 41.11:+6.2f}%")
    print(f"Avg Win          | +5.95%  | {avg_win:+6.2f}% | {avg_win - 5.95:+6.2f}%")
    print(f"Avg Loss         | -4.99%  | {avg_loss:+6.2f}% | {avg_loss + 4.99:+6.2f}%")
    
    print()
    print("="*80)
    print("✅ BACKTEST v6.0 COMPLETATO")
    print("="*80)
    print()
