#!/usr/bin/env python3
"""
ESEMPIO COMPLETO DI UN TRADE - GANN-ENNEAGRAM v5.2.0
Mostra PASSO-PASSO tutti i 7 layer e come lavorano insieme
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("🎯 ESEMPIO TRADE COMPLETO - GANN-ENNEAGRAM v5.2.0")
print("="*80)

# Carica dati BTC
df = pd.read_csv('data/btcusdt_daily_1000.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Focus periodo Nov 2024 - Gen 2026
df_focus = df[(df['date'] >= '2024-11-01') & (df['date'] <= '2026-01-12')].copy()
df_focus = df_focus.reset_index(drop=True)

print(f"\n📅 Periodo analisi: {df_focus['date'].min().date()} → {df_focus['date'].max().date()}")
print(f"📊 Totale barre: {len(df_focus)}")

# ============================================================================
# LAYER 1: MAJOR PIVOT DETECTION (Multi-timeframe)
# ============================================================================
print("\n" + "="*80)
print("LAYER 1: MAJOR PIVOT DETECTION")
print("="*80)

def detect_major_pivots(df, lookback_left=5, lookback_right=5):
    """Rileva major highs/lows con conferma bilaterale"""
    pivots = []
    
    for i in range(lookback_left, len(df) - lookback_right):
        # Pivot LOW
        is_low = True
        for j in range(i - lookback_left, i + lookback_right + 1):
            if j != i and df.loc[j, 'low'] <= df.loc[i, 'low']:
                is_low = False
                break
        
        if is_low:
            pivots.append({
                'date': df.loc[i, 'date'],
                'index': i,
                'type': 'LOW',
                'price': df.loc[i, 'low'],
                'close': df.loc[i, 'close']
            })
        
        # Pivot HIGH
        is_high = True
        for j in range(i - lookback_left, i + lookback_right + 1):
            if j != i and df.loc[j, 'high'] >= df.loc[i, 'high']:
                is_high = False
                break
        
        if is_high:
            pivots.append({
                'date': df.loc[i, 'date'],
                'index': i,
                'type': 'HIGH',
                'price': df.loc[i, 'high'],
                'close': df.loc[i, 'close']
            })
    
    return pd.DataFrame(pivots)

# Rileva tutti i pivot
pivots_df = detect_major_pivots(df_focus, lookback_left=7, lookback_right=7)
print(f"\n✓ Rilevati {len(pivots_df)} pivot totali")

# Seleziona UN PIVOT LOW significativo per l'esempio
# Cerca un pivot low attorno a Novembre 2024
pivot_lows = pivots_df[pivots_df['type'] == 'LOW']
pivot_candidates = pivot_lows[(pivot_lows['date'] >= '2024-11-01') & 
                               (pivot_lows['date'] <= '2024-11-30')]

if len(pivot_candidates) == 0:
    # Se non ci sono pivot in Nov, usa il primo pivot low disponibile
    example_pivot = pivot_lows.iloc[0]
    print(f"\n⚠️ Nessun pivot LOW in Nov 2024, uso primo disponibile")
else:
    example_pivot = pivot_candidates.iloc[0]

pivot_date = example_pivot['date']
pivot_price = example_pivot['price']
pivot_idx = example_pivot['index']

print(f"\n🎯 MAJOR PIVOT SELEZIONATO:")
print(f"   📅 Data: {pivot_date.date()}")
print(f"   💰 Prezzo: ${pivot_price:,.2f}")
print(f"   📊 Bar Index: {pivot_idx}")
print(f"   ✓ Conferma: Swing low con 7 barre left/right")

# ============================================================================
# LAYER 2: GANN TIME SEQUENCE (Bar counting from pivot)
# ============================================================================
print("\n" + "="*80)
print("LAYER 2: GANN TIME SEQUENCE")
print("="*80)

gann_sequence = [0, 7, 9, 13, 14, 18, 20, 21, 26, 28]
print(f"\n📈 Sequenza GANN sacra: {gann_sequence}")
print(f"   ⚠️ Il 7 e i suoi multipli sono CRITICI")
print(f"   ⚠️ 0-7 barre = continuazione trend")

# Calcola le date chiave dalla sequenza GANN
print(f"\n🎯 DATE CHIAVE dalla sequenza GANN (da pivot {pivot_date.date()}):")
bar_0_date = pivot_date

gann_dates = {}
for bar_num in gann_sequence:
    if pivot_idx + bar_num < len(df_focus):
        target_date = df_focus.loc[pivot_idx + bar_num, 'date']
        target_price = df_focus.loc[pivot_idx + bar_num, 'close']
        gann_dates[bar_num] = {
            'date': target_date,
            'price': target_price,
            'idx': pivot_idx + bar_num
        }
        print(f"   Bar {bar_num:2d}: {target_date.date()} @ ${target_price:,.2f}")

# Scelgo la Bar 14 come punto di analisi (zona critica per entry)
entry_bar = 14
entry_analysis_idx = pivot_idx + entry_bar
entry_date = df_focus.loc[entry_analysis_idx, 'date']
entry_price_current = df_focus.loc[entry_analysis_idx, 'close']

print(f"\n✓ ANALISI FOCALIZZATA su Bar {entry_bar}")
print(f"   📅 Data: {entry_date.date()}")
print(f"   💰 Prezzo corrente: ${entry_price_current:,.2f}")
print(f"   📊 Stato temporale: TRANSIZIONE CRITICA (Bar 13-14)")

time_score = 40  # Bar 14 è nella sequenza GANN → peso massimo
print(f"   🎯 TIME SCORE: {time_score}/40 (nella sequenza GANN)")

# ============================================================================
# LAYER 3: GANN RULE OF EIGHTHS (Price Geometry)
# ============================================================================
print("\n" + "="*80)
print("LAYER 3: GANN RULE OF EIGHTHS (Price Geometry)")
print("="*80)

# Calcola range dal pivot low all'high successivo
pivot_low = pivot_price
# Trova il prossimo major high dopo il pivot
next_highs = pivots_df[(pivots_df['type'] == 'HIGH') & 
                        (pivots_df['date'] > pivot_date)].head(1)
if len(next_highs) > 0:
    pivot_high = next_highs.iloc[0]['price']
else:
    # Usa il massimo del periodo se non c'è major high
    pivot_high = df_focus[df_focus['date'] > pivot_date]['high'].max()

range_total = pivot_high - pivot_low
eighths = {}
for i in range(9):
    eighths[i] = pivot_low + (range_total * i / 8)

print(f"\n📏 RANGE GANN 8ths:")
print(f"   LOW:  ${pivot_low:,.2f}")
print(f"   HIGH: ${pivot_high:,.2f}")
print(f"   Range: ${range_total:,.2f}")

print(f"\n🎯 Livelli GANN 8ths:")
for i in range(9):
    marker = ""
    if abs(entry_price_current - eighths[i]) < range_total * 0.05:  # Entro 5% del livello
        marker = " ← PREZZO CORRENTE QUI"
    print(f"   {i}/8: ${eighths[i]:,.2f}{marker}")

# Determina il livello più vicino
closest_eighth = min(range(9), key=lambda x: abs(eighths[x] - entry_price_current))
distance_pct = abs(entry_price_current - eighths[closest_eighth]) / range_total * 100

print(f"\n✓ Prezzo corrente ${entry_price_current:,.2f}")
print(f"   Livello più vicino: {closest_eighth}/8 (${eighths[closest_eighth]:,.2f})")
print(f"   Distanza: {distance_pct:.1f}% del range")

# Score price: massimo se vicino a 2/8, 3/8 o 4/8 (zone di entry ideali)
if closest_eighth in [2, 3, 4]:
    price_score = 40 - int(distance_pct * 2)  # Penalizza distanza
    price_score = max(25, min(40, price_score))
    print(f"   🎯 PRICE SCORE: {price_score}/40 (vicino a livello key {closest_eighth}/8)")
else:
    price_score = 20
    print(f"   ⚠️ PRICE SCORE: {price_score}/40 (non in zona entry ottimale)")

# ============================================================================
# LAYER 4: ENNEAGRAM STATES (9 Market Phases)
# ============================================================================
print("\n" + "="*80)
print("LAYER 4: ENNEAGRAM STATES (9 Market Phases)")
print("="*80)

def calculate_enneagram_state(df, idx, pivot_low_price):
    """Calcola lo stato Enneagram basato su geometria prezzo e momentum"""
    current_price = df.loc[idx, 'close']
    
    # Calcola % gain dal pivot
    gain_pct = ((current_price - pivot_low_price) / pivot_low_price) * 100
    
    # Calcola momentum (slope ultimi 7 giorni)
    if idx >= 7:
        price_7d_ago = df.loc[idx - 7, 'close']
        momentum = ((current_price - price_7d_ago) / price_7d_ago) * 100
    else:
        momentum = 0
    
    # Mappa a stato Enneagram 1-9
    if gain_pct < 2:
        state = 1  # INITIATION
    elif gain_pct < 5:
        state = 2  # EXPANSION
    elif gain_pct < 10:
        state = 3  # ACCELERATION
    elif gain_pct < 15:
        state = 4 if momentum > 0 else 5  # GROWTH / CONSOLIDATION
    elif gain_pct < 20:
        state = 6  # MATURITY
    elif gain_pct < 25:
        state = 7  # EXPANSION_2
    elif gain_pct < 30:
        state = 8  # STRONG_MARKUP
    else:
        state = 9  # CLIMAX
    
    return state, gain_pct, momentum

state, gain_pct, momentum = calculate_enneagram_state(df_focus, entry_analysis_idx, pivot_low)

state_names = {
    1: "INITIATION", 2: "EXPANSION", 3: "ACCELERATION",
    4: "GROWTH", 5: "CONSOLIDATION", 6: "MATURITY",
    7: "EXPANSION_2", 8: "STRONG_MARKUP", 9: "CLIMAX"
}

print(f"\n🔮 ENNEAGRAM STATE ANALYSIS:")
print(f"   Stato corrente: #{state} {state_names[state]}")
print(f"   Gain dal pivot: +{gain_pct:.2f}%")
print(f"   Momentum 7d: {momentum:+.2f}%")

# Transizione ideale per LONG: stati 1-4 (early phase)
if state <= 4:
    state_score = 20
    transition = "BULLISH - Early Phase"
    print(f"   ✓ Transizione: {transition}")
    print(f"   🎯 STATE SCORE: {state_score}/20 (fase iniziale ideale)")
elif state <= 6:
    state_score = 15
    transition = "NEUTRAL - Mid Phase"
    print(f"   ⚠️ Transizione: {transition}")
    print(f"   🎯 STATE SCORE: {state_score}/20 (fase intermedia)")
else:
    state_score = 5
    transition = "CAUTION - Late Phase"
    print(f"   ⚠️ Transizione: {transition}")
    print(f"   🎯 STATE SCORE: {state_score}/20 (fase avanzata, cautela)")

# ============================================================================
# LAYER 5: SQUARE OF 9 (Price Squaring Time)
# ============================================================================
print("\n" + "="*80)
print("LAYER 5: SQUARE OF 9 (Price Squaring Time)")
print("="*80)

def calculate_square_of_9_targets(pivot_price, angles=[90, 135, 180, 225, 270]):
    """Calcola target Square of 9 da pivot"""
    targets = {}
    sqrt_pivot = np.sqrt(pivot_price)
    
    for angle in angles:
        # Converti angolo in "units" (360° = 2 units)
        units = angle / 180
        new_sqrt = sqrt_pivot + units
        target_price = new_sqrt ** 2
        targets[angle] = target_price
    
    return targets

sq9_targets = calculate_square_of_9_targets(pivot_low)

print(f"\n🎲 SQUARE OF 9 TARGETS da pivot ${pivot_low:,.2f}:")
for angle, target in sq9_targets.items():
    distance = ((target - entry_price_current) / entry_price_current) * 100
    marker = ""
    if abs(distance) < 5:  # Entro 5% del target
        marker = " ← PREZZO CORRENTE VICINO"
    print(f"   {angle:3d}°: ${target:,.2f} ({distance:+.1f}%){marker}")

# Identifica target più vicino per confluence
closest_sq9_angle = min(sq9_targets.keys(), 
                        key=lambda x: abs(sq9_targets[x] - entry_price_current))
closest_sq9_target = sq9_targets[closest_sq9_angle]

print(f"\n✓ Target Sq9 più vicino: {closest_sq9_angle}° @ ${closest_sq9_target:,.2f}")

# Check confluence con Gann 8ths
confluence_sq9_gann = False
for i in range(9):
    if abs(closest_sq9_target - eighths[i]) / eighths[i] < 0.03:  # Entro 3%
        confluence_sq9_gann = True
        print(f"   ✓✓ CONFLUENCE: Sq9 {closest_sq9_angle}° ≈ Gann {i}/8")
        break

if not confluence_sq9_gann:
    print(f"   ⚠️ No confluence stretta con Gann 8ths")

# ============================================================================
# LAYER 6: ICHIMOKU GEOMETRICO (Structure)
# ============================================================================
print("\n" + "="*80)
print("LAYER 6: ICHIMOKU GEOMETRICO")
print("="*80)

def calculate_ichimoku(df, idx):
    """Calcola Ichimoku: Tenkan, Kijun, Senkou A/B, Chikou"""
    # Tenkan-sen (9-period high-low avg)
    tenkan_period = 9
    if idx >= tenkan_period:
        tenkan = (df.loc[idx-tenkan_period+1:idx, 'high'].max() + 
                  df.loc[idx-tenkan_period+1:idx, 'low'].min()) / 2
    else:
        tenkan = df.loc[idx, 'close']
    
    # Kijun-sen (26-period high-low avg)
    kijun_period = 26
    if idx >= kijun_period:
        kijun = (df.loc[idx-kijun_period+1:idx, 'high'].max() + 
                 df.loc[idx-kijun_period+1:idx, 'low'].min()) / 2
    else:
        kijun = df.loc[idx, 'close']
    
    # Senkou Span A (Tenkan+Kijun)/2 proiettato 26 avanti
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (52-period high-low avg proiettato 26 avanti)
    senkou_b_period = 52
    if idx >= senkou_b_period:
        senkou_b = (df.loc[idx-senkou_b_period+1:idx, 'high'].max() + 
                    df.loc[idx-senkou_b_period+1:idx, 'low'].min()) / 2
    else:
        senkou_b = df.loc[idx, 'close']
    
    return {
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b
    }

ichimoku = calculate_ichimoku(df_focus, entry_analysis_idx)
current_price = entry_price_current

print(f"\n☁️ ICHIMOKU LEVELS:")
print(f"   Tenkan-sen:  ${ichimoku['tenkan']:,.2f}")
print(f"   Kijun-sen:   ${ichimoku['kijun']:,.2f}")
print(f"   Senkou A:    ${ichimoku['senkou_a']:,.2f}")
print(f"   Senkou B:    ${ichimoku['senkou_b']:,.2f}")
print(f"   Prezzo:      ${current_price:,.2f}")

# Determina Kumo (cloud)
kumo_top = max(ichimoku['senkou_a'], ichimoku['senkou_b'])
kumo_bottom = min(ichimoku['senkou_a'], ichimoku['senkou_b'])

print(f"\n☁️ KUMO (Cloud): ${kumo_bottom:,.2f} - ${kumo_top:,.2f}")

# Segnali Ichimoku
signals = []
if current_price > kumo_top:
    signals.append("✓ Prezzo ABOVE Kumo (Bullish)")
elif current_price < kumo_bottom:
    signals.append("✗ Prezzo BELOW Kumo (Bearish)")
else:
    signals.append("⚠️ Prezzo INSIDE Kumo (Neutral)")

if ichimoku['tenkan'] > ichimoku['kijun']:
    signals.append("✓ TK Cross Bullish")
else:
    signals.append("✗ TK Cross Bearish")

print(f"\n🎯 ICHIMOKU SIGNALS:")
for sig in signals:
    print(f"   {sig}")

# Score Ichimoku
ichimoku_bullish = (current_price > kumo_top and 
                     ichimoku['tenkan'] > ichimoku['kijun'])
ichimoku_score = 20 if ichimoku_bullish else 10

print(f"   🎯 ICHIMOKU contribuisce a confluence generale")

# ============================================================================
# LAYER 7: GANN ANGLES (Dynamic Support/Resistance)
# ============================================================================
print("\n" + "="*80)
print("LAYER 7: GANN ANGLES (Dynamic Support/Resistance)")
print("="*80)

def calculate_gann_angles(pivot_price, pivot_idx, current_idx, atr):
    """Calcola Gann angles 1x1, 2x1, 1x2 dal pivot"""
    bars_elapsed = current_idx - pivot_idx
    
    # 1x1: 1 unità prezzo per 1 unità tempo (usa ATR come unità)
    angle_1x1 = pivot_price + (bars_elapsed * 1.0 * atr)
    
    # 2x1: 2 unità prezzo per 1 unità tempo
    angle_2x1 = pivot_price + (bars_elapsed * 2.0 * atr)
    
    # 1x2: 1 unità prezzo per 2 unità tempo
    angle_1x2 = pivot_price + (bars_elapsed * 0.5 * atr)
    
    # 4x1: aggressive
    angle_4x1 = pivot_price + (bars_elapsed * 4.0 * atr)
    
    # 1x4: conservative
    angle_1x4 = pivot_price + (bars_elapsed * 0.25 * atr)
    
    return {
        '1x1': angle_1x1,
        '2x1': angle_2x1,
        '1x2': angle_1x2,
        '4x1': angle_4x1,
        '1x4': angle_1x4
    }

# Calcola ATR (14-period)
atr_period = 14
if entry_analysis_idx >= atr_period:
    tr_list = []
    for i in range(entry_analysis_idx - atr_period + 1, entry_analysis_idx + 1):
        high = df_focus.loc[i, 'high']
        low = df_focus.loc[i, 'low']
        close_prev = df_focus.loc[i-1, 'close'] if i > 0 else low
        tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
        tr_list.append(tr)
    atr = np.mean(tr_list)
else:
    atr = (df_focus.loc[entry_analysis_idx, 'high'] - 
           df_focus.loc[entry_analysis_idx, 'low'])

gann_angles = calculate_gann_angles(pivot_low, pivot_idx, entry_analysis_idx, atr)

print(f"\n📐 GANN ANGLES da pivot (Bar {entry_bar}):")
print(f"   ATR (14): ${atr:,.2f}")
print(f"   Bars elapsed: {entry_bar}")

for angle_name, angle_price in sorted(gann_angles.items()):
    distance = ((current_price - angle_price) / angle_price) * 100
    marker = ""
    if abs(distance) < 3:
        marker = " ← PREZZO CORRENTE QUI"
    elif current_price > angle_price:
        marker = " (prezzo above - support)"
    else:
        marker = " (prezzo below - resistance)"
    
    print(f"   Gann {angle_name}: ${angle_price:,.2f} ({distance:+.1f}%){marker}")

# Determina posizione rispetto a 1x1 (key angle)
if current_price > gann_angles['1x1']:
    gann_signal = "✓ Prezzo ABOVE 1x1 angle (Bullish trend)"
else:
    gann_signal = "✗ Prezzo BELOW 1x1 angle (Bearish trend)"

print(f"\n🎯 GANN ANGLE SIGNAL:")
print(f"   {gann_signal}")

# ============================================================================
# TRIPLE CONFLUENCE SCORING
# ============================================================================
print("\n" + "="*80)
print("TRIPLE CONFLUENCE SCORING")
print("="*80)

total_score = time_score + price_score + state_score
max_score = 100

print(f"\n📊 SCORE BREAKDOWN:")
print(f"   TIME (Layer 2):  {time_score}/40  - Bar nella sequenza GANN")
print(f"   PRICE (Layer 3): {price_score}/40  - Vicinanza a livelli 8ths")
print(f"   STATE (Layer 4): {state_score}/20  - Fase Enneagram")
print(f"   " + "-"*50)
print(f"   TOTAL SCORE:     {total_score}/100")

# Bonus/penalità
bonus_points = 0

if confluence_sq9_gann:
    bonus_points += 5
    print(f"   + BONUS Square of 9 confluence: +5")

if ichimoku_bullish:
    bonus_points += 5
    print(f"   + BONUS Ichimoku bullish: +5")

if current_price > gann_angles['1x1']:
    bonus_points += 5
    print(f"   + BONUS Above Gann 1x1: +5")

total_score_final = min(100, total_score + bonus_points)

print(f"\n   🎯 FINAL SCORE: {total_score_final}/100")

# Decision
if total_score_final >= 85:
    decision = "LONG_FULL"
    position_size = 100
elif total_score_final >= 70:
    decision = "LONG_HALF"
    position_size = 50
else:
    decision = "WAIT"
    position_size = 0

print(f"\n🎯 DECISION: {decision}")
print(f"   Position size: {position_size}%")

# ============================================================================
# ENTRY/EXIT PLAN GEOMETRICO
# ============================================================================
print("\n" + "="*80)
print("ENTRY/EXIT PLAN GEOMETRICO")
print("="*80)

if decision != "WAIT":
    # Entry: prezzo corrente o pullback a livello inferiore
    entry_price = entry_price_current
    
    # Stop Loss: below pivot low o below 1/8 level
    sl_price = min(pivot_low * 0.98, eighths[1] * 0.99)  # 2% below pivot o 1% below 1/8
    sl_pct = ((sl_price - entry_price) / entry_price) * 100
    
    # Take Profit targets
    tp1_price = eighths[4]  # 4/8
    tp2_price = eighths[6]  # 6/8
    tp3_price = eighths[8]  # 8/8 (top)
    
    # TP4: Square of 9 target oltre 8/8
    tp4_angle = 225 if 225 in sq9_targets else 270
    tp4_price = sq9_targets[tp4_angle]
    
    tp1_pct = ((tp1_price - entry_price) / entry_price) * 100
    tp2_pct = ((tp2_price - entry_price) / entry_price) * 100
    tp3_pct = ((tp3_price - entry_price) / entry_price) * 100
    tp4_pct = ((tp4_price - entry_price) / entry_price) * 100
    
    # Risk:Reward
    risk = abs(sl_pct)
    reward = tp3_pct  # Base R:R su TP3
    rr_ratio = reward / risk if risk > 0 else 0
    
    print(f"\n💼 TRADE SETUP:")
    print(f"   📅 Data Entry: {entry_date.date()}")
    print(f"   💰 Entry Price: ${entry_price:,.2f}")
    print(f"   🛑 Stop Loss:   ${sl_price:,.2f} ({sl_pct:.2f}%)")
    print(f"   🎯 TP1 (4/8):   ${tp1_price:,.2f} (+{tp1_pct:.2f}%)")
    print(f"   🎯 TP2 (6/8):   ${tp2_price:,.2f} (+{tp2_pct:.2f}%)")
    print(f"   🎯 TP3 (8/8):   ${tp3_price:,.2f} (+{tp3_pct:.2f}%)")
    print(f"   🎯 TP4 (Sq9):   ${tp4_price:,.2f} (+{tp4_pct:.2f}%)")
    print(f"   📊 Risk:Reward: 1:{rr_ratio:.1f}")
    print(f"   💪 Position:    {position_size}%")
    
    print(f"\n📋 GESTIONE POSIZIONE:")
    print(f"   1. Entry a ${entry_price:,.2f}")
    print(f"   2. SL iniziale ${sl_price:,.2f}")
    print(f"   3. A TP1 (+{tp1_pct:.1f}%): move SL to breakeven")
    print(f"   4. A TP2 (+{tp2_pct:.1f}%): close 50%, trail lungo 1x1 Gann")
    print(f"   5. Trail SL sopra Gann 1x1 angle + 2.0×ATR")
    print(f"   6. TP3/TP4: exit finale o trail aggressivo")
    
    # Simula outcome se abbiamo dati futuri
    future_bars = 30
    if entry_analysis_idx + future_bars < len(df_focus):
        future_df = df_focus.iloc[entry_analysis_idx:entry_analysis_idx+future_bars+1]
        
        # Check se hit SL o TP
        hit_sl = (future_df['low'] <= sl_price).any()
        hit_tp1 = (future_df['high'] >= tp1_price).any()
        hit_tp2 = (future_df['high'] >= tp2_price).any()
        hit_tp3 = (future_df['high'] >= tp3_price).any()
        
        print(f"\n📊 OUTCOME SIMULATO (prossimi {future_bars} giorni):")
        if hit_sl:
            sl_date = future_df[future_df['low'] <= sl_price].iloc[0]['date']
            print(f"   🛑 STOP LOSS HIT il {sl_date.date()}")
            print(f"   📉 Perdita: {sl_pct:.2f}%")
        elif hit_tp3:
            tp3_date = future_df[future_df['high'] >= tp3_price].iloc[0]['date']
            print(f"   🎯✓✓✓ TP3 RAGGIUNTO il {tp3_date.date()}")
            print(f"   📈 Guadagno: +{tp3_pct:.2f}%")
        elif hit_tp2:
            tp2_date = future_df[future_df['high'] >= tp2_price].iloc[0]['date']
            print(f"   🎯✓✓ TP2 RAGGIUNTO il {tp2_date.date()}")
            print(f"   📈 Guadagno: +{tp2_pct:.2f}%")
        elif hit_tp1:
            tp1_date = future_df[future_df['high'] >= tp1_price].iloc[0]['date']
            print(f"   🎯✓ TP1 RAGGIUNTO il {tp1_date.date()}")
            print(f"   📈 Guadagno: +{tp1_pct:.2f}%")
        else:
            print(f"   ⏳ Trade ancora in corso dopo {future_bars} giorni")

else:
    print(f"\n⏳ WAIT: Score {total_score_final} < 70")
    print(f"   Motivo: Mancano condizioni ottimali per entry")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
print("\n" + "="*80)
print("RIEPILOGO COME I 7 LAYER LAVORANO INSIEME")
print("="*80)

print(f"""
🎯 QUESTO TRADE DIMOSTRA:

1️⃣ MAJOR PIVOT (Layer 1): Identifica il punto di partenza geometrico
   → Pivot Low @ ${pivot_low:,.2f} il {pivot_date.date()}

2️⃣ GANN TIME (Layer 2): Conta le barre dalla sequenza sacra
   → Bar {entry_bar} è nella sequenza [0,7,9,13,14,18,20,21,26,28]
   → TIME SCORE: {time_score}/40

3️⃣ GANN 8ths (Layer 3): Divide il range in 8 parti uguali
   → Prezzo @ {closest_eighth}/8 (${eighths[closest_eighth]:,.2f})
   → PRICE SCORE: {price_score}/40

4️⃣ ENNEAGRAM (Layer 4): Identifica la fase di mercato (1-9)
   → Stato #{state} {state_names[state]} ({transition})
   → STATE SCORE: {state_score}/20

5️⃣ SQUARE OF 9 (Layer 5): Calcola target geometrici dal pivot
   → Target {closest_sq9_angle}° @ ${closest_sq9_target:,.2f}
   → Confluence con Gann 8ths: {'YES' if confluence_sq9_gann else 'NO'}

6️⃣ ICHIMOKU (Layer 6): Conferma struttura e trend
   → {'Bullish' if ichimoku_bullish else 'Neutral/Bearish'} (Prezzo vs Kumo, TK cross)

7️⃣ GANN ANGLES (Layer 7): Dynamic support/resistance
   → Prezzo vs 1x1 angle: {'Above (bullish)' if current_price > gann_angles['1x1'] else 'Below (bearish)'}

🎯 TRIPLE CONFLUENCE:
   TIME + PRICE + STATE = {total_score}/100
   + Bonus Confluence: +{bonus_points}
   = FINAL SCORE: {total_score_final}/100

✅ DECISION: {decision}
   → {'TRADE ESEGUITO' if decision != 'WAIT' else 'ASPETTA SETUP MIGLIORE'}

📊 PRICE & TIME LAVORANO INSIEME:
   • TIME ci dice QUANDO guardare (Bar nella sequenza GANN)
   • PRICE ci dice DOVE entrare (Livelli 8ths e Sq9)
   • STATE ci dice PERCHÉ (Fase di mercato favorevole)
   • ICHIMOKU + GANN ANGLES confermano il trend

🎯 QUESTO È IL VERO GANN:
   Geometria del prezzo + Cicli temporali + Stati di mercato
   = CONFLUENCE TRADING ad alta probabilità
""")

print("="*80)
print("✅ ESEMPIO COMPLETO TERMINATO")
print("="*80)
