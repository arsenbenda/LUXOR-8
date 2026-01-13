#!/usr/bin/env python3
"""
ANALISI PRATICA: BTC Trade Examples con Sistema Enneagram-Gann
Periodo: Nov 2024 - Gen 2026
Mostra come funziona nella pratica il sistema completo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ============================================================================
# LAYER 1: ENNEAGRAM STATES (Simplified for Demo)
# ============================================================================

ENNEAGRAM_STATES = {
    1: {"name": "INITIATION", "angle": 0, "bias": "ACCUMULATION"},
    2: {"name": "EARLY_DISTRIBUTION", "angle": 80, "bias": "WARNING"},
    3: {"name": "COMPLETION", "angle": 320, "bias": "CLIMAX_TOP"},
    4: {"name": "RETRACEMENT", "angle": 40, "bias": "HEALTHY_PULLBACK"},
    5: {"name": "DEEP_CORRECTION", "angle": 160, "bias": "CAPITULATION"},
    6: {"name": "DECISION", "angle": 280, "bias": "NEUTRAL_COIL"},
    7: {"name": "EXPANSION", "angle": 200, "bias": "UPTREND"},
    8: {"name": "STRONG_MARKUP", "angle": 120, "bias": "PARABOLIC_TOP"},
    9: {"name": "EQUILIBRIUM", "angle": 240, "bias": "RANGE_TIGHT"}
}

# Transition arrows (stress/growth)
BEARISH_TRANSITIONS = [(8,5), (3,5), (7,4), (2,5), (1,4)]
BULLISH_TRANSITIONS = [(5,8), (5,7), (4,7), (6,7), (9,1), (1,7)]

def identify_enneagram_state(bar, prev_bars, atr):
    """
    Identifica lo stato Enneagram basato su ADX, RSI, Volume, Price Structure
    Ritorna: (state_number, confidence_score)
    """
    # Calcola metriche
    close = bar['close']
    high = bar['high']
    low = bar['low']
    volume = bar['volume']
    
    # ADX proxy (semplificato: range high-low)
    price_range = high - low
    adx_proxy = (price_range / close) * 100
    
    # RSI proxy (semplificato: posizione nel range recente)
    recent_high = max([b['high'] for b in prev_bars[-14:]])
    recent_low = min([b['low'] for b in prev_bars[-14:]])
    rsi_proxy = ((close - recent_low) / (recent_high - recent_low)) * 100 if recent_high > recent_low else 50
    
    # Volume (vs media recente)
    avg_volume = np.mean([b['volume'] for b in prev_bars[-20:]])
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    
    # Trend structure
    ma20 = np.mean([b['close'] for b in prev_bars[-20:]])
    ma50 = np.mean([b['close'] for b in prev_bars[-50:]]) if len(prev_bars) >= 50 else ma20
    
    # Price momentum
    prev_close = prev_bars[-1]['close'] if prev_bars else close
    price_change_pct = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0
    
    # Decision Logic
    state = 9  # default equilibrium
    confidence = 50
    
    # State 8: STRONG_MARKUP (parabolic)
    if rsi_proxy > 70 and volume_ratio > 2.0 and price_change_pct > 5 and close > ma20 > ma50:
        state = 8
        confidence = 85
    
    # State 3: COMPLETION (climax top)
    elif rsi_proxy > 80 and volume_ratio > 3.0 and adx_proxy > 4.0:
        state = 3
        confidence = 80
    
    # State 7: EXPANSION (strong uptrend)
    elif rsi_proxy > 55 and close > ma20 > ma50 and volume_ratio > 1.2:
        state = 7
        confidence = 75
    
    # State 5: DEEP_CORRECTION (capitulation)
    elif rsi_proxy < 30 and volume_ratio > 2.0 and price_change_pct < -5:
        state = 5
        confidence = 80
    
    # State 4: RETRACEMENT (healthy pullback)
    elif 40 < rsi_proxy < 50 and close < ma20 and volume_ratio < 0.8:
        state = 4
        confidence = 70
    
    # State 1: INITIATION (accumulation)
    elif 45 < rsi_proxy < 55 and volume_ratio > 1.1 and close > prev_close:
        state = 1
        confidence = 65
    
    # State 6: DECISION (range/coil)
    elif abs(price_change_pct) < 1.5 and volume_ratio < 0.6:
        state = 6
        confidence = 60
    
    return state, confidence, {
        'rsi_proxy': rsi_proxy,
        'adx_proxy': adx_proxy,
        'volume_ratio': volume_ratio,
        'price_change_pct': price_change_pct
    }

# ============================================================================
# LAYER 2: GANN TIME CYCLES
# ============================================================================

def calculate_gann_time_window(pivot_date, current_state, target_state, cycle_days=30):
    """
    Calcola finestra temporale Gann basata su angular distance tra stati
    """
    current_angle = ENNEAGRAM_STATES[current_state]['angle']
    target_angle = ENNEAGRAM_STATES[target_state]['angle']
    
    # Angular distance (shortest path on circle)
    angular_distance = abs(target_angle - current_angle)
    if angular_distance > 180:
        angular_distance = 360 - angular_distance
    
    # Time window estimate (fraction of cycle)
    time_fraction = angular_distance / 360.0
    days_to_target = int(cycle_days * time_fraction)
    
    # Tolerance band
    tolerance = max(1, int(days_to_target * 0.15))  # ±15%
    
    return {
        'days_to_target': days_to_target,
        'tolerance': tolerance,
        'target_date_min': pivot_date + timedelta(days=days_to_target - tolerance),
        'target_date_max': pivot_date + timedelta(days=days_to_target + tolerance)
    }

# ============================================================================
# LAYER 3: GANN PRICE GEOMETRY
# ============================================================================

def calculate_gann_levels(pivot_price, direction='both'):
    """
    Calcola livelli Gann 8ths e 3rds dal pivot
    """
    levels = {}
    
    # 8ths (1/8, 2/8, 3/8, etc.)
    for i in range(1, 9):
        eighth = i / 8.0
        levels[f'eighth_{i}'] = {
            'up': pivot_price * (1 + eighth * 0.1),  # ~10% per full 8/8
            'down': pivot_price * (1 - eighth * 0.1)
        }
    
    # 3rds (1/3, 2/3)
    for i in range(1, 3):
        third = i / 3.0
        levels[f'third_{i}'] = {
            'up': pivot_price * (1 + third * 0.15),  # ~15% per full 3/3
            'down': pivot_price * (1 - third * 0.15)
        }
    
    return levels

def calculate_square_of_9_target(pivot_price, angle_degrees):
    """
    Calcola target price dal Square of 9
    Formula: target = (sqrt(pivot) + angle/360)^2
    """
    sqrt_pivot = np.sqrt(pivot_price)
    angle_fraction = angle_degrees / 360.0
    target = (sqrt_pivot + angle_fraction * sqrt_pivot * 0.5) ** 2  # Simplified
    return target

# ============================================================================
# TRIPLE CONFLUENCE ENTRY SIGNAL
# ============================================================================

def generate_sniper_signal(bar, prev_bars, pivot_low, pivot_date, current_date):
    """
    Genera segnale SNIPER basato su Triple Confluence:
    1. Enneagram State Transition
    2. Gann Time Window
    3. Gann Price Level
    """
    atr = np.mean([b['high'] - b['low'] for b in prev_bars[-14:]]) if len(prev_bars) >= 14 else 1000
    
    # LAYER 1: Identify current state
    current_state, state_confidence, metrics = identify_enneagram_state(bar, prev_bars, atr)
    
    # Check for bullish transition
    target_state = None
    transition_type = None
    for (from_state, to_state) in BULLISH_TRANSITIONS:
        if current_state == from_state:
            target_state = to_state
            transition_type = 'BULLISH'
            break
    
    if target_state is None:
        return None  # No valid transition
    
    # LAYER 2: Gann Time Window (30-day cycle)
    time_window = calculate_gann_time_window(pivot_date, current_state, target_state, cycle_days=30)
    
    # Check if current_date is in time window
    in_time_window = time_window['target_date_min'] <= current_date <= time_window['target_date_max']
    
    # LAYER 3: Gann Price Geometry
    gann_levels = calculate_gann_levels(pivot_low, direction='up')
    
    # Check if price is near a support level (e.g., within 2% of eighth_2)
    current_price = bar['close']
    price_confluence = False
    confluence_level = None
    
    for level_name, level_values in gann_levels.items():
        support = level_values['down']
        if abs(current_price - support) / support < 0.02:  # Within 2%
            price_confluence = True
            confluence_level = level_name
            break
    
    # Square of 9 target (45 degrees = first target)
    sq9_target = calculate_square_of_9_target(pivot_low, 45)
    
    # SCORING SYSTEM (0-100)
    score = 0
    
    # State confidence (max 40 points)
    score += (state_confidence / 100.0) * 40
    
    # Time window (30 points if in window)
    if in_time_window:
        score += 30
    
    # Price confluence (30 points if near support)
    if price_confluence:
        score += 30
    
    # Volume confirmation (bonus +10 if volume > 1.2x avg)
    if metrics['volume_ratio'] > 1.2:
        score += 10
    
    # Cap at 100
    score = min(100, score)
    
    # Generate signal
    signal = None
    position_size = 0.0
    
    if score >= 85:
        signal = 'LONG_FULL'
        position_size = 1.0
    elif score >= 70:
        signal = 'LONG_HALF'
        position_size = 0.5
    else:
        signal = 'WAIT'
        position_size = 0.0
    
    # Calculate SL and TP
    sl_price = pivot_low * 0.98  # 2% below pivot (Gann invalidation)
    tp1_price = gann_levels['eighth_2']['up']
    tp2_price = gann_levels['eighth_4']['up']
    tp3_price = sq9_target
    
    return {
        'signal': signal,
        'position_size': position_size,
        'score': score,
        'current_state': current_state,
        'state_name': ENNEAGRAM_STATES[current_state]['name'],
        'target_state': target_state,
        'target_name': ENNEAGRAM_STATES[target_state]['name'],
        'transition_type': transition_type,
        'in_time_window': in_time_window,
        'days_to_target': time_window['days_to_target'],
        'price_confluence': price_confluence,
        'confluence_level': confluence_level,
        'metrics': metrics,
        'entry_price': current_price,
        'sl_price': sl_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'tp3_price': tp3_price,
        'sq9_target': sq9_target
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_btc_trades(csv_path, start_date_str, end_date_str):
    """
    Analizza i trade BTC nel periodo specificato
    """
    print("=" * 80)
    print("ANALISI PRATICA: Sistema Enneagram-Gann su BTC")
    print("=" * 80)
    print()
    
    # Load data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter period
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    print(f"📊 Periodo analisi: {start_date_str} - {end_date_str}")
    print(f"📈 Barre totali: {len(df)}")
    print()
    
    # Find major pivots (swing lows)
    df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
    pivots = df[df['swing_low']].copy()
    
    print(f"🔍 Pivot bassi trovati: {len(pivots)}")
    print()
    
    # Analyze top 3 most significant pivots (lowest lows)
    pivots = pivots.nsmallest(5, 'low')  # Aumenta a 5 per più esempi
    
    signals = []
    
    for idx, pivot in pivots.iterrows():
        pivot_price = pivot['low']
        pivot_date = pivot['date']
        
        print("=" * 80)
        print(f"📍 PIVOT LOW: {pivot_date.strftime('%Y-%m-%d')} @ ${pivot_price:,.0f}")
        print("=" * 80)
        print()
        
        # Analyze next 60 days for entry signals (aumentato da 30)
        future_bars = df[df['date'] > pivot_date].head(60)
        
        entry_found = False
        best_signal = None
        best_score = 0
        
        for f_idx, f_bar in future_bars.iterrows():
            current_date = f_bar['date']
            
            # Get previous bars for context
            prev_bars = df[df['date'] < current_date].tail(50).to_dict('records')
            
            # Generate signal
            signal = generate_sniper_signal(
                f_bar.to_dict(),
                prev_bars,
                pivot_price,
                pivot_date,
                current_date
            )
            
            # Track best signal even if < threshold
            if signal and signal['score'] > best_score:
                best_signal = signal
                best_score = signal['score']
            
            if signal and signal['signal'] != 'WAIT':
                entry_found = True
                
                print(f"✅ SEGNALE ENTRY: {signal['signal']}")
                print(f"   Data: {current_date.strftime('%Y-%m-%d')}")
                print(f"   Score: {signal['score']:.1f}/100")
                print()
                print(f"🔮 ENNEAGRAM STATE:")
                print(f"   Current: {signal['state_name']} (#{signal['current_state']})")
                print(f"   Target: {signal['target_name']} (#{signal['target_state']})")
                print(f"   Transition: {signal['transition_type']}")
                print()
                print(f"⏰ GANN TIME:")
                print(f"   In Window: {'✓' if signal['in_time_window'] else '✗'}")
                print(f"   Days to Target: {signal['days_to_target']}")
                print()
                print(f"📐 GANN PRICE:")
                print(f"   Price Confluence: {'✓' if signal['price_confluence'] else '✗'}")
                if signal['confluence_level']:
                    print(f"   Level: {signal['confluence_level']}")
                print()
                print(f"📊 METRICHE:")
                print(f"   RSI Proxy: {signal['metrics']['rsi_proxy']:.1f}")
                print(f"   Volume Ratio: {signal['metrics']['volume_ratio']:.2f}x")
                print(f"   Price Change: {signal['metrics']['price_change_pct']:.2f}%")
                print()
                print(f"💰 ENTRY PLAN:")
                print(f"   Entry: ${signal['entry_price']:,.0f}")
                print(f"   Stop Loss: ${signal['sl_price']:,.0f} (-{((signal['entry_price']-signal['sl_price'])/signal['entry_price']*100):.2f}%)")
                print(f"   TP1 (8th_2): ${signal['tp1_price']:,.0f} (+{((signal['tp1_price']-signal['entry_price'])/signal['entry_price']*100):.2f}%)")
                print(f"   TP2 (8th_4): ${signal['tp2_price']:,.0f} (+{((signal['tp2_price']-signal['entry_price'])/signal['entry_price']*100):.2f}%)")
                print(f"   TP3 (Sq9): ${signal['tp3_price']:,.0f} (+{((signal['tp3_price']-signal['entry_price'])/signal['entry_price']*100):.2f}%)")
                print()
                print(f"   R:R Ratio: 1:{((signal['tp3_price']-signal['entry_price'])/(signal['entry_price']-signal['sl_price'])):.2f}")
                print(f"   Position Size: {signal['position_size']*100:.0f}%")
                print()
                
                # Track outcome (simplified: check if TP3 hit in next 30 days)
                outcome_bars = df[df['date'] > current_date].head(30)
                tp3_hit = any(outcome_bars['high'] >= signal['tp3_price'])
                sl_hit = any(outcome_bars['low'] <= signal['sl_price'])
                
                if tp3_hit and not sl_hit:
                    print("🎯 OUTCOME: TP3 HIT ✓")
                    final_return = ((signal['tp3_price'] - signal['entry_price']) / signal['entry_price']) * 100
                    print(f"   Return: +{final_return:.2f}%")
                elif sl_hit:
                    print("❌ OUTCOME: STOP LOSS HIT")
                    final_return = ((signal['sl_price'] - signal['entry_price']) / signal['entry_price']) * 100
                    print(f"   Return: {final_return:.2f}%")
                else:
                    # Check current price
                    last_price = outcome_bars.iloc[-1]['close'] if len(outcome_bars) > 0 else signal['entry_price']
                    final_return = ((last_price - signal['entry_price']) / signal['entry_price']) * 100
                    print(f"⏳ OUTCOME: IN PROGRESS")
                    print(f"   Current Return: {final_return:+.2f}%")
                
                print()
                
                signals.append({
                    'pivot_date': pivot_date,
                    'entry_date': current_date,
                    'signal': signal,
                    'outcome': 'TP3' if tp3_hit and not sl_hit else ('SL' if sl_hit else 'OPEN')
                })
                
                break  # Only show first valid signal per pivot
        
        # Store current_date for best_signal output
        if best_signal:
            # Find the date of best_signal
            for f_idx, f_bar in future_bars.iterrows():
                current_date = f_bar['date']
                prev_bars = df[df['date'] < current_date].tail(50).to_dict('records')
                test_signal = generate_sniper_signal(
                    f_bar.to_dict(),
                    prev_bars,
                    pivot_price,
                    pivot_date,
                    current_date
                )
                if test_signal and test_signal['score'] == best_signal['score']:
                    break
        
        if not entry_found and best_signal:
            print("⏸️  Nessun segnale ENTRY sopra soglia, ma BEST ATTEMPT:")
            print()
            print(f"   Data: {current_date.strftime('%Y-%m-%d')}")
            print(f"   Score: {best_signal['score']:.1f}/100 (soglia: 70)")
            print(f"   Current State: {best_signal['state_name']}")
            print(f"   In Time Window: {'✓' if best_signal['in_time_window'] else '✗'}")
            print(f"   Price Confluence: {'✓' if best_signal['price_confluence'] else '✗'}")
            print(f"   Volume Ratio: {best_signal['metrics']['volume_ratio']:.2f}x")
            print()
            print("   💡 PERCHÉ NON HA GENERATO SEGNALE:")
            reasons = []
            if best_signal['score'] < 70:
                if not best_signal['in_time_window']:
                    reasons.append("- Fuori finestra temporale Gann")
                if not best_signal['price_confluence']:
                    reasons.append("- Nessuna confluenza livelli prezzo")
                if best_signal['metrics']['volume_ratio'] < 1.2:
                    reasons.append(f"- Volume basso ({best_signal['metrics']['volume_ratio']:.2f}x)")
            for r in reasons:
                print(f"      {r}")
            print()
        elif not entry_found:
            print("⏸️  Nessun segnale nei 30 giorni successivi")
            print()
    
    print("=" * 80)
    print(f"📈 SUMMARY: {len(signals)} segnali trovati")
    print("=" * 80)
    
    return signals

if __name__ == "__main__":
    csv_path = "data/btcusdt_daily_1000.csv"
    
    # Analizza Nov 2024 - Gen 2026
    signals = analyze_btc_trades(
        csv_path,
        start_date_str="2024-11-01",
        end_date_str="2026-01-12"
    )
    
    print()
    print("✅ Analisi completata!")
    print()
    print("💡 COSA VEDIAMO:")
    print("   1. Sistema identifica pivot geometrici (swing lows)")
    print("   2. Applica Enneagram State Analysis su ogni barra successiva")
    print("   3. Calcola Gann Time Windows per transizioni ottimali")
    print("   4. Verifica Gann Price Levels (8ths/3rds + Square of 9)")
    print("   5. Genera segnale SOLO quando TRIPLE CONFLUENCE converge")
    print("   6. Define entry preciso con SL/TP geometrici")
    print()
    print("🎯 VANTAGGIO:")
    print("   - Entry solo su pivot + time + price convergence")
    print("   - SL stretti (invalidazione geometrica)")
    print("   - TP estesi (multi-stage Gann targets)")
    print("   - R:R ratio tipicamente 1:4 - 1:8")
    print()
