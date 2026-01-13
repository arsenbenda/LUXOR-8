"""
🚀 LUXOR v7.1 AGGRESSIVE MAXIMUM PROFIT
=========================================
OBIETTIVO: **MASSIMIZZARE P&L ASSOLUTO** - ZERO COMPROMESSI

FILOSOFIA:
- **FILTRI MINIMI**: Lascia entrare TUTTI i setup validi
- **PARTIAL EXITS AGGRESSIVI**: Proteggi profitti velocemente  
- **TRAILING STOP DINAMICO**: Massimizza big winners
- **POSITION SIZING AGGRESSIVO**: Carica sui setup ad alta probabilità
"""

import pandas as pd
import numpy as np

# ================== AGGRESSIVE CONFIG ==================
CONFIG = {
    # MINIMAL FILTERS (massima libertà)
    'min_score': 60,           # Era 75 → abbassato
    'max_score': 100,          # Era 95 → rimosso cap
    'ideal_volatility_min': 0.5,   # Era 1.5 → molto più basso
    'ideal_volatility_max': 8.0,   # Era 4.0 → molto più alto
    
    # ENTRY OPTIMIZATION (rilassato)
    'min_rsi': 25,             # Era 40 → molto più basso
    'max_rsi': 80,             # Era 70 → più alto
    'require_momentum': False,  # Era True → DISABILITATO
    'require_volume': False,    # Era True → DISABILITATO
    
    # EXIT OPTIMIZATION (più aggressivo)
    'trailing_atr_multiplier': 1.5,  # Era 2.0 → più stretto
    'breakeven_at': 0.3,       # Era 0.5 → più veloce
    'partial_exit_1': 0.50,    # Era 0.33 → esci 50% a +0.5R
    'partial_exit_2': 0.30,    # Era 0.33 → esci 30% a +1R
    'trail_remainder': True,   # Trail ultimo 20%
    
    # POSITION SIZING (super aggressivo)
    'base_size': 1.0,
    'high_confidence_multiplier': 2.0,  # Era 1.5 → raddoppia!
    
    # MARKET REGIME
    'bull_market_only': False,  # Era True → PERMETTI TUTTO!
}

# ================== INDICATORS (same as v7.0) ==================

def calculate_rsi(df, current_idx, period=14):
    if current_idx < period + 1:
        return 50.0
    closes = df['close'].iloc[current_idx-period:current_idx+1]
    deltas = closes.diff()
    gain = deltas.where(deltas > 0, 0).mean()
    loss = -deltas.where(deltas < 0, 0).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(df, current_idx, period=20):
    if current_idx < period:
        return df['close'].iloc[:current_idx+1].mean()
    closes = df['close'].iloc[current_idx-period+1:current_idx+1]
    return closes.ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_volume_ratio(df, current_idx, period=20):
    if current_idx < period:
        return 1.0
    current_vol = df['volume'].iloc[current_idx]
    avg_vol = df['volume'].iloc[current_idx-period:current_idx].mean()
    return current_vol / avg_vol if avg_vol > 0 else 1.0

def calculate_atr(df, current_idx, period=14):
    if current_idx < period + 1:
        high_low = df['high'].iloc[current_idx] - df['low'].iloc[current_idx]
        return high_low
    start = max(0, current_idx - period)
    high = df['high'].iloc[start:current_idx+1]
    low = df['low'].iloc[start:current_idx+1]
    close = df['close'].iloc[start:current_idx+1]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.iloc[-period:].mean()

def calculate_volatility(df, current_idx, window=30):
    if current_idx < window:
        window = current_idx
    if window < 2:
        return 0.0
    closes = df['close'].iloc[current_idx-window:current_idx+1]
    returns = closes.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return returns.std() * 100

def calculate_advanced_features(df, current_idx):
    features = {}
    current_price = df['close'].iloc[current_idx]
    features['price'] = current_price
    
    ema20 = calculate_ema(df, current_idx, 20)
    features['ema20'] = ema20
    features['price_vs_ema20'] = (current_price / ema20 - 1) * 100
    
    rsi = calculate_rsi(df, current_idx, 14)
    features['rsi'] = rsi
    features['rsi_normalized'] = (rsi - 50) / 50
    
    volatility = calculate_volatility(df, current_idx, 30)
    features['volatility'] = volatility
    
    vol_ratio = calculate_volume_ratio(df, current_idx, 20)
    features['volume_ratio'] = vol_ratio
    
    atr = calculate_atr(df, current_idx, 14)
    features['atr'] = atr
    features['atr_pct'] = (atr / current_price) * 100
    
    if current_idx >= 5:
        price_5d_ago = df['close'].iloc[current_idx - 5]
        features['return_5d'] = (current_price / price_5d_ago - 1) * 100
    else:
        features['return_5d'] = 0.0
    
    return features

# ================== AGGRESSIVE SCORING ==================

def score_entry_setup(features, pivot_type):
    """Score aggressivo - bonus per TUTTI i setup"""
    score = 60.0  # Base più alto
    
    # MOMENTUM
    if features['price_vs_ema20'] > 0:
        score += 15
    else:
        score += 5  # Bonus anche se negativo!
    
    # RSI
    rsi = features['rsi']
    if 40 <= rsi <= 70:
        score += 15
    elif 30 <= rsi <= 80:
        score += 10  # Bonus anche fuori sweet spot
    
    # VOLATILITY (più permissivo)
    vol = features['volatility']
    if 1.5 <= vol <= 5.0:
        score += 20
    elif vol < 1.0:
        score += 10  # Bassa volatilità OK
    elif vol > 6.0:
        score += 5   # Alta volatilità OK
    
    # VOLUME
    if features['volume_ratio'] > 1.0:
        score += 10
    
    # TREND
    if features['return_5d'] > 0:
        score += 10
    elif features['return_5d'] > -2:
        score += 5  # Leggero downtrend OK
    
    # Direction bonus
    if pivot_type == 'LOW':
        score += 10  # LONG sempre favorito
    
    return max(0, min(100, score))

# ================== MINIMAL FILTERS ==================

def check_entry_filters(features, score, pivot_type):
    """Filtri MINIMI - lascia passare quasi tutto!"""
    
    # Score minimo/massimo
    if score < CONFIG['min_score']:
        return False, f"Score {score:.1f} < {CONFIG['min_score']}"
    
    # Volatilità (range AMPIO)
    vol = features['volatility']
    if vol < CONFIG['ideal_volatility_min'] or vol > CONFIG['ideal_volatility_max']:
        return False, f"Volatility {vol:.2f}% fuori range estremo"
    
    # RSI (range MOLTO ampio)
    rsi = features['rsi']
    if rsi < CONFIG['min_rsi'] or rsi > CONFIG['max_rsi']:
        return False, f"RSI {rsi:.1f} troppo estremo"
    
    # TUTTO IL RESTO È OK!
    return True, "PASS"

# ================== AGGRESSIVE EXITS ==================

def calculate_optimal_exits(entry_price, sl_price, atr, direction):
    """Exit aggressivi per proteggere profitti"""
    risk = abs(entry_price - sl_price)
    
    exits = {
        'sl': sl_price,
        'tp1_price': None,
        'tp1_pct': CONFIG['partial_exit_1'],  # 50%
        'tp2_price': None,
        'tp2_pct': CONFIG['partial_exit_2'],  # 30%
        'trailing_start': None,
        'trailing_multiplier': CONFIG['trailing_atr_multiplier'],
    }
    
    if direction == 'LONG':
        exits['tp1_price'] = entry_price + risk * 0.5  # +0.5R (veloce!)
        exits['tp2_price'] = entry_price + risk * 1.0  # +1R
        exits['trailing_start'] = entry_price + risk * CONFIG['breakeven_at']
    else:  # SHORT
        exits['tp1_price'] = entry_price - risk * 0.5
        exits['tp2_price'] = entry_price - risk * 1.0
        exits['trailing_start'] = entry_price - risk * CONFIG['breakeven_at']
    
    return exits

def simulate_exit_with_partials(df, entry_idx, entry_price, exits, direction, atr):
    """Simula exit con partial exits aggressivi"""
    remaining_size = 1.0
    total_pnl = 0.0
    exit_details = []
    
    trailing_stop = exits['sl']
    hit_tp1 = False
    hit_tp2 = False
    
    # Simula 90 barre future (3 mesi - più lungo)
    for i in range(1, min(90, len(df) - entry_idx)):
        bar_idx = entry_idx + i
        high = df['high'].iloc[bar_idx]
        low = df['low'].iloc[bar_idx]
        close = df['close'].iloc[bar_idx]
        
        if direction == 'LONG':
            # Check SL PRIMA (priorità!)
            if low <= trailing_stop:
                pnl = ((trailing_stop - entry_price) / entry_price) * 100 * remaining_size
                total_pnl += pnl
                exit_details.append(f"SL @{trailing_stop:.2f} ({remaining_size:.1%}): {pnl:+.2f}%")
                return total_pnl, 'SL', df['date'].iloc[bar_idx], trailing_stop, exit_details
            
            # Check TP1
            if not hit_tp1 and high >= exits['tp1_price']:
                exit_size = exits['tp1_pct']
                pnl = ((exits['tp1_price'] - entry_price) / entry_price) * 100 * exit_size
                total_pnl += pnl
                remaining_size -= exit_size
                hit_tp1 = True
                exit_details.append(f"TP1 @{exits['tp1_price']:.2f} ({exit_size:.0%}): +{pnl:.2f}%")
                trailing_stop = max(trailing_stop, entry_price)  # Breakeven
            
            # Check TP2
            if not hit_tp2 and high >= exits['tp2_price']:
                exit_size = exits['tp2_pct']
                pnl = ((exits['tp2_price'] - entry_price) / entry_price) * 100 * exit_size
                total_pnl += pnl
                remaining_size -= exit_size
                hit_tp2 = True
                exit_details.append(f"TP2 @{exits['tp2_price']:.2f} ({exit_size:.0%}): +{pnl:.2f}%")
            
            # Update trailing stop
            if close > exits['trailing_start']:
                new_stop = close - atr * exits['trailing_multiplier']
                trailing_stop = max(trailing_stop, new_stop)
        
        else:  # SHORT
            # Check SL PRIMA
            if high >= trailing_stop:
                pnl = ((entry_price - trailing_stop) / entry_price) * 100 * remaining_size
                total_pnl += pnl
                exit_details.append(f"SL @{trailing_stop:.2f} ({remaining_size:.1%}): {pnl:+.2f}%")
                return total_pnl, 'SL', df['date'].iloc[bar_idx], trailing_stop, exit_details
            
            # Check TP1
            if not hit_tp1 and low <= exits['tp1_price']:
                exit_size = exits['tp1_pct']
                pnl = ((entry_price - exits['tp1_price']) / entry_price) * 100 * exit_size
                total_pnl += pnl
                remaining_size -= exit_size
                hit_tp1 = True
                exit_details.append(f"TP1 @{exits['tp1_price']:.2f} ({exit_size:.0%}): +{pnl:.2f}%")
                trailing_stop = min(trailing_stop, entry_price)
            
            # Check TP2
            if not hit_tp2 and low <= exits['tp2_price']:
                exit_size = exits['tp2_pct']
                pnl = ((entry_price - exits['tp2_price']) / entry_price) * 100 * exit_size
                total_pnl += pnl
                remaining_size -= exit_size
                hit_tp2 = True
                exit_details.append(f"TP2 @{exits['tp2_price']:.2f} ({exit_size:.0%}): +{pnl:.2f}%")
            
            # Update trailing stop
            if close < exits['trailing_start']:
                new_stop = close + atr * exits['trailing_multiplier']
                trailing_stop = min(trailing_stop, new_stop)
    
    # Time exit
    if remaining_size > 0:
        final_close = df['close'].iloc[min(entry_idx + 89, len(df) - 1)]
        if direction == 'LONG':
            pnl = ((final_close - entry_price) / entry_price) * 100 * remaining_size
        else:
            pnl = ((entry_price - final_close) / entry_price) * 100 * remaining_size
        total_pnl += pnl
        exit_details.append(f"TIME @{final_close:.2f} ({remaining_size:.0%}): {pnl:+.2f}%")
    
    return total_pnl, 'TIME', df['date'].iloc[min(entry_idx + 89, len(df) - 1)], final_close, exit_details

# ================== MAIN BACKTEST ==================

def run_backtest_v710(df, start_date, end_date):
    """Backtest v7.1 AGGRESSIVE"""
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_period = df[mask].reset_index(drop=True)
    
    results = []
    
    print("\n" + "="*70)
    print("🎯 SCANNING FOR **ALL** PROFITABLE SETUPS...")
    print("="*70)
    
    # Scan OGNI barra
    for idx in range(30, len(df_period) - 90):
        current_date = df_period['date'].iloc[idx]
        
        # Calcola features
        features = calculate_advanced_features(df_period, idx)
        
        # Identifica swing
        lookback = 5  # Era 7 → più sensibile
        if idx < lookback or idx >= len(df_period) - lookback:
            continue
        
        is_swing_low = (df_period['low'].iloc[idx] == df_period['low'].iloc[idx-lookback:idx+lookback+1].min())
        is_swing_high = (df_period['high'].iloc[idx] == df_period['high'].iloc[idx-lookback:idx+lookback+1].max())
        
        if not (is_swing_low or is_swing_high):
            continue
        
        pivot_type = 'LOW' if is_swing_low else 'HIGH'
        direction = 'LONG' if pivot_type == 'LOW' else 'SHORT'
        
        # Score
        score = score_entry_setup(features, pivot_type)
        
        # Check filters
        pass_filters, filter_msg = check_entry_filters(features, score, pivot_type)
        
        if not pass_filters:
            continue
        
        # ENTRY!
        entry_price = df_period['close'].iloc[idx]
        atr = features['atr']
        
        # Calculate SL (più stretto!)
        if direction == 'LONG':
            sl_distance = atr * 1.2  # Era 1.5 → più aggressivo
            sl_price = entry_price - sl_distance
        else:
            sl_distance = atr * 1.2
            sl_price = entry_price + sl_distance
        
        # Calculate exits
        exits = calculate_optimal_exits(entry_price, sl_price, atr, direction)
        
        # Simulate trade
        total_pnl, outcome, exit_date, exit_price, exit_details = simulate_exit_with_partials(
            df_period, idx, entry_price, exits, direction, atr
        )
        
        # Position sizing
        confidence_score = score / 100.0
        position_size = CONFIG['base_size']
        if confidence_score > 0.85:  # Era 0.90 → più facile
            position_size *= CONFIG['high_confidence_multiplier']
        
        # Adjusted P&L
        adjusted_pnl = total_pnl * position_size
        
        # Store result
        trade = {
            'entry_date': current_date,
            'exit_date': exit_date,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'sl_price': sl_price,
            'score': score,
            'volatility': features['volatility'],
            'rsi': features['rsi'],
            'outcome': outcome,
            'pnl': adjusted_pnl,
            'position_size': position_size,
        }
        results.append(trade)
        
        if len(results) <= 10 or len(results) % 5 == 0:  # Print primi 10 + ogni 5
            print(f"\n✅ TRADE #{len(results)}: {current_date.strftime('%Y-%m-%d')} | {direction} | Score {score:.0f} | P&L {adjusted_pnl:+.2f}% | {outcome}")
    
    return pd.DataFrame(results)

# ================== MAIN ==================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LUXOR v7.1 AGGRESSIVE MAXIMUM PROFIT")
    print("="*70)
    
    df = pd.read_csv('data/btcusdt_daily_1000.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    results = run_backtest_v710(
        df,
        start_date='2024-01-01',
        end_date='2026-01-12'
    )
    
    results.to_csv('/mnt/user-data/outputs/backtest_v710_results.csv', index=False)
    
    print("\n" + "="*70)
    print("📊 RISULTATI BACKTEST v7.1 AGGRESSIVE")
    print("="*70)
    
    if len(results) == 0:
        print("\n⚠️ NESSUN TRADE ESEGUITO")
    else:
        total_trades = len(results)
        winners = results[results['pnl'] > 0]
        losers = results[results['pnl'] <= 0]
        
        total_pnl = results['pnl'].sum()
        win_rate = len(winners) / total_trades * 100
        
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winners: {len(winners)} | Losers: {len(losers)}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   *** TOTAL P&L: {total_pnl:+.2f}% ***")
        print(f"   Avg Win: {avg_win:+.2f}%")
        print(f"   Avg Loss: {avg_loss:+.2f}%")
        
        long_trades = results[results['direction'] == 'LONG']
        short_trades = results[results['direction'] == 'SHORT']
        
        print(f"\n📊 BY DIRECTION:")
        print(f"   LONG: {len(long_trades)} | P&L: {long_trades['pnl'].sum():+.2f}%")
        print(f"   SHORT: {len(short_trades)} | P&L: {short_trades['pnl'].sum():+.2f}%")
        
        print(f"\n🏆 TOP 5 WINNERS:")
        for i, row in winners.nlargest(5, 'pnl').iterrows():
            print(f"   {row['entry_date'].strftime('%Y-%m-%d')} | {row['direction']} | {row['pnl']:+.2f}% | {row['outcome']}")
        
        print(f"\n💀 TOP 5 LOSERS:")
        for i, row in losers.nsmallest(5, 'pnl').iterrows():
            print(f"   {row['entry_date'].strftime('%Y-%m-%d')} | {row['direction']} | {row['pnl']:+.2f}% | {row['outcome']}")
    
    print("\n" + "="*70)
    print("✅ v7.1 AGGRESSIVE COMPLETATO!")
    print("="*70)

