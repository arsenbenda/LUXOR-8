#!/usr/bin/env python3
"""
LUXOR v5.3.0 ULTRA OPTIMIZED - GANN-ENNEAGRAM System
Ottimizzazioni implementate:
1. Dynamic Trailing Stops (ATR-based)
2. Short Bias Filter (crypto bull market adjustment)
3. Score Recalibration (volatility-weighted)
4. Enhanced Position Sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURAZIONE v5.3.0
# ============================================================================
CONFIG = {
    'lookback': 7,
    'min_score': 70,
    'min_rr': 1.5,
    'volatility_threshold': 3.5,
    'high_vol_min_score': 90,
    'high_vol_sl_multiplier': 1.3,
    'high_vol_size': 0.5,
    
    # NEW v5.3.0
    'short_bias_penalty': 15,  # Penalizzazione score per SHORT
    'short_min_score': 85,      # Score minimo per SHORT
    'trailing_atr_multiplier': 2.0,  # Trailing stop ATR multiplier
    'volatility_score_weight': 0.3,  # Peso volatility nel score
    'dynamic_rr_min': True,     # R:R minimo dinamico basato su volatility
}

# ============================================================================
# FUNZIONI CORE
# ============================================================================

def detect_pivots(df, lookback=7):
    """Detect major pivots (swing high/low)"""
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

def calculate_gann_levels(pivot_price, current_price, pivot_type):
    """Calculate GANN Rule of Eights levels"""
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
        '2/8': low + 2*eighth,
        '3/8': low + 3*eighth,
        '4/8': low + 4*eighth,
        '5/8': low + 5*eighth,
        '6/8': low + 6*eighth,
        '7/8': low + 7*eighth,
        '8/8': high
    }
    
    return levels, low, high

def calculate_enneagram_state(df, pivot_idx, current_idx):
    """Determine Enneagram state based on price action"""
    pivot_price = df['close'].iloc[pivot_idx]
    current_price = df['close'].iloc[current_idx]
    
    gain_pct = ((current_price - pivot_price) / pivot_price) * 100
    
    # Momentum
    momentum_window = min(7, current_idx - pivot_idx)
    if momentum_window > 0:
        momentum = ((df['close'].iloc[current_idx] - df['close'].iloc[current_idx - momentum_window]) / 
                   df['close'].iloc[current_idx - momentum_window]) * 100
    else:
        momentum = 0
    
    # State determination
    if abs(gain_pct) < 2:
        state = 1  # INITIATION
    elif abs(gain_pct) < 5:
        state = 2  # TENSION
    elif abs(gain_pct) < 10:
        state = 3  # ACCELERATION
    else:
        state = 4  # CLIMAX
    
    return state, gain_pct, momentum

def calculate_volatility(df, current_idx, window=30):
    """Calculate historical volatility (daily, not annualized)"""
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

def calculate_score_v530(time_score, price_score, state_score, volatility, direction, pivot_type):
    """
    v5.3.0 Score Recalibration
    - Volatility-weighted scoring
    - Short bias penalty
    """
    base_score = time_score + price_score + state_score
    
    # Volatility penalty
    vol_penalty = 0
    if volatility > CONFIG['volatility_threshold']:
        vol_penalty = min(15, (volatility - CONFIG['volatility_threshold']) * 3)
    
    # Short bias penalty (crypto bull market)
    short_penalty = 0
    if direction == 'SHORT':
        short_penalty = CONFIG['short_bias_penalty']
    
    final_score = base_score - vol_penalty - short_penalty
    
    return max(0, min(100, final_score))

def calculate_dynamic_rr_min(volatility):
    """Dynamic R:R minimum based on volatility"""
    if not CONFIG['dynamic_rr_min']:
        return CONFIG['min_rr']
    
    if volatility > 5.0:
        return 2.0
    elif volatility > CONFIG['volatility_threshold']:
        return 1.8
    else:
        return CONFIG['min_rr']

def calculate_position_size(volatility, score):
    """Enhanced position sizing"""
    base_size = 1.0
    
    # Volatility adjustment
    if volatility > CONFIG['volatility_threshold']:
        base_size = CONFIG['high_vol_size']
    
    # Score adjustment
    if score < 85:
        base_size *= 0.7
    
    return base_size

def calculate_trailing_stop(entry_price, sl_price, current_price, atr, direction):
    """
    Dynamic Trailing Stop (v5.3.0)
    - ATR-based trailing
    - Progressive tightening
    """
    if direction == 'LONG':
        # Initial risk
        initial_risk = entry_price - sl_price
        
        # Trailing level
        trailing_stop = current_price - (atr * CONFIG['trailing_atr_multiplier'])
        
        # Never lower than original SL
        trailing_stop = max(sl_price, trailing_stop)
        
        # Breakeven dopo +0.5R
        if current_price >= entry_price + (initial_risk * 0.5):
            trailing_stop = max(trailing_stop, entry_price)
        
        # Tighten dopo +1R
        if current_price >= entry_price + initial_risk:
            trailing_stop = max(trailing_stop, entry_price + (initial_risk * 0.5))
        
        return trailing_stop
    
    else:  # SHORT
        initial_risk = sl_price - entry_price
        trailing_stop = current_price + (atr * CONFIG['trailing_atr_multiplier'])
        trailing_stop = min(sl_price, trailing_stop)
        
        if current_price <= entry_price - (initial_risk * 0.5):
            trailing_stop = min(trailing_stop, entry_price)
        
        if current_price <= entry_price - initial_risk:
            trailing_stop = min(trailing_stop, entry_price - (initial_risk * 0.5))
        
        return trailing_stop

# ============================================================================
# BACKTEST ENGINE v5.3.0
# ============================================================================

def run_backtest_v530(df, start_date, end_date):
    """Run backtest with v5.3.0 optimizations"""
    
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop=True)
    
    # Detect pivots
    pivots_df = detect_pivots(df, CONFIG['lookback'])
    
    results = []
    
    for _, pivot in pivots_df.iterrows():
        pivot_idx = pivot['index']
        pivot_type = pivot['type']
        pivot_price = pivot['price']
        
        # GANN Time Sequence
        time_bars = [7, 9, 13, 14, 18, 20, 21, 26, 28]  # Removed 0 - entry on pivot is unrealistic
        
        for time_bar in time_bars:
            analysis_idx = pivot_idx + time_bar
            
            if analysis_idx >= len(df):
                continue
            
            current_price = df['close'].iloc[analysis_idx]
            current_date = df['date'].iloc[analysis_idx]
            
            # Calculate metrics
            volatility = calculate_volatility(df, analysis_idx)
            atr = calculate_atr(df, analysis_idx)
            
            # GANN levels
            gann_levels, gann_low, gann_high = calculate_gann_levels(
                pivot_price, current_price, pivot_type
            )
            
            # Enneagram state
            state, gain_pct, momentum = calculate_enneagram_state(df, pivot_idx, analysis_idx)
            
            # Scoring
            time_score = 40  # Simplified
            
            # Price score
            price_distance = min([abs(current_price - level) / current_price * 100 
                                 for level in gann_levels.values()])
            price_score = max(0, 40 - price_distance * 10)
            
            # State score
            state_score = 20 if momentum * gain_pct > 0 else 10
            
            # Direction
            direction = 'LONG' if pivot_type == 'LOW' else 'SHORT'
            
            # v5.3.0 Score
            final_score = calculate_score_v530(
                time_score, price_score, state_score, 
                volatility, direction, pivot_type
            )
            
            # Entry filters
            min_score_required = CONFIG['high_vol_min_score'] if volatility > CONFIG['volatility_threshold'] else CONFIG['min_score']
            
            # SHORT additional filter
            if direction == 'SHORT' and final_score < CONFIG['short_min_score']:
                continue
            
            if final_score < min_score_required:
                continue
            
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
            
            if rr_ratio < dynamic_rr_min:
                continue
            
            # Position size
            position_size = calculate_position_size(volatility, final_score)
            
            # Simulate outcome with TRAILING STOP
            outcome = None
            exit_price = None
            exit_date = None
            current_sl = sl_price  # Trailing SL
            
            for future_idx in range(analysis_idx + 1, min(analysis_idx + 30, len(df))):
                future_high = df['high'].iloc[future_idx]
                future_low = df['low'].iloc[future_idx]
                future_close = df['close'].iloc[future_idx]
                future_date = df['date'].iloc[future_idx]
                future_atr = calculate_atr(df, future_idx)
                
                # Update trailing stop
                current_sl = calculate_trailing_stop(
                    entry_price, sl_price, future_close, 
                    future_atr, direction
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
    # Load data
    df = pd.read_csv('data/btcusdt_daily_1000.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n" + "="*80)
    print("🚀 BACKTEST v5.3.0 ULTRA OPTIMIZED")
    print("="*80)
    print("\nOTTIMIZZAZIONI:")
    print("  ✓ Dynamic Trailing Stops (ATR-based)")
    print("  ✓ Short Bias Filter (−15 score penalty)")
    print("  ✓ Score Recalibration (volatility-weighted)")
    print("  ✓ Dynamic R:R minimum")
    print("  ✓ Enhanced Position Sizing")
    print()
    
    # Run backtest
    results = run_backtest_v530(df, '2024-01-01', '2026-01-12')
    
    # Save results
    results.to_csv('/mnt/user-data/outputs/backtest_v530_results.csv', index=False)
    
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
    print("📊 RISULTATI BACKTEST v5.3.0")
    print("="*80)
    print()
    print(f"✅ Trade eseguiti: {total_trades}")
    print(f"   LONG:  {long_trades}")
    print(f"   SHORT: {short_trades}")
    print()
    print(f"💰 PERFORMANCE:")
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
        print(f"   {outcome:4s}: {count:3d} ({pct:5.1f}%)")
    print()
    
    # Comparison v5.2.0 vs v5.3.0
    print("="*80)
    print("📊 CONFRONTO v5.2.0 → v5.3.0")
    print("="*80)
    print()
    print("Metrica          | v5.2.0  | v5.3.0  | Delta")
    print("-" * 80)
    print(f"Trade Totali     |   42    |   {total_trades:2d}    | {total_trades-42:+3d}")
    print(f"Win Rate         | 52.4%   | {win_rate:5.1f}% | {win_rate-52.4:+5.1f}%")
    print(f"Total P&L        | +41.11% | {total_pnl:+6.2f}% | {total_pnl-41.11:+6.2f}%")
    print(f"SL Rate          | 73.8%   | {len(results[results['outcome']=='SL'])/total_trades*100:5.1f}% | {len(results[results['outcome']=='SL'])/total_trades*100-73.8:+5.1f}%")
    print()
    
    print("="*80)
    print("✅ BACKTEST v5.3.0 COMPLETATO")
    print("="*80)
    print()
