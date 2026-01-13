#!/usr/bin/env python3
"""
LUXOR v5.2.0 GANN-ENNEAGRAM OPTIMIZED
Con ottimizzazioni critiche:
1. Volatility Adjustment (SL dinamico + score boost)
2. R:R Minimum Filter (1.5:1)
3. Momentum Confluence Check
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class GannEnneagramSystem:
    def __init__(self):
        self.gann_sequence = [0, 7, 9, 13, 14, 18, 20, 21, 26, 28]
        self.state_names = {
            1: "INITIATION", 2: "EXPANSION", 3: "ACCELERATION",
            4: "GROWTH", 5: "CONSOLIDATION", 6: "MATURITY",
            7: "EXPANSION_2", 8: "STRONG_MARKUP", 9: "CLIMAX"
        }
        
        # OTTIMIZZAZIONI CRITICHE
        self.high_volatility_threshold = 3.5  # %
        self.min_rr_ratio = 1.5
        self.volatility_sl_multiplier = 1.5
        self.volatility_min_score = 85
    
    def detect_pivots(self, df, lookback_left=7, lookback_right=7):
        """Rileva major pivots con conferma bilaterale"""
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
    
    def calculate_volatility(self, df, idx, period=30):
        """Calcola volatilità media come % range giornaliero"""
        if idx < period:
            period = idx
        
        if period == 0:
            return 0
        
        vol_list = []
        for i in range(idx - period + 1, idx + 1):
            daily_range = (df.loc[i, 'high'] - df.loc[i, 'low']) / df.loc[i, 'low'] * 100
            vol_list.append(daily_range)
        
        return np.mean(vol_list)
    
    def calculate_enneagram_state(self, df, idx, pivot_price, pivot_type='LOW'):
        """Calcola stato Enneagram e momentum"""
        current_price = df.loc[idx, 'close']
        
        if pivot_type == 'LOW':
            gain_pct = ((current_price - pivot_price) / pivot_price) * 100
        else:  # HIGH
            gain_pct = ((pivot_price - current_price) / pivot_price) * 100
        
        # Momentum ultimi 7 giorni
        if idx >= 7:
            price_7d_ago = df.loc[idx - 7, 'close']
            momentum = ((current_price - price_7d_ago) / price_7d_ago) * 100
        else:
            momentum = 0
        
        # Mappa stato
        if gain_pct < 2:
            state = 1
        elif gain_pct < 5:
            state = 2
        elif gain_pct < 10:
            state = 3
        elif gain_pct < 15:
            state = 4 if momentum > 0 else 5
        elif gain_pct < 20:
            state = 6
        elif gain_pct < 25:
            state = 7
        elif gain_pct < 30:
            state = 8
        else:
            state = 9
        
        return state, gain_pct, momentum
    
    def calculate_gann_eighths(self, pivot_price, pivot_type, df, pivot_date):
        """Calcola livelli Gann 8ths"""
        if pivot_type == 'LOW':
            # Cerca HIGH successivo
            future_df = df[df['date'] > pivot_date]
            if len(future_df) > 0:
                range_high = future_df['high'].max()
            else:
                range_high = pivot_price * 1.2
            range_low = pivot_price
        else:  # HIGH
            # Cerca LOW successivo
            future_df = df[df['date'] > pivot_date]
            if len(future_df) > 0:
                range_low = future_df['low'].min()
            else:
                range_low = pivot_price * 0.8
            range_high = pivot_price
        
        range_total = abs(range_high - range_low)
        eighths = {}
        for i in range(9):
            eighths[i] = range_low + (range_total * i / 8)
        
        return eighths, range_total
    
    def score_trade(self, df, pivot, entry_bar):
        """Score completo con ottimizzazioni"""
        
        pivot_idx = pivot['index']
        pivot_price = pivot['price']
        pivot_type = pivot['type']
        pivot_date = pivot['date']
        
        entry_idx = pivot_idx + entry_bar
        
        if entry_idx >= len(df):
            return None
        
        entry_date = df.loc[entry_idx, 'date']
        entry_price = df.loc[entry_idx, 'close']
        
        # Calcola volatilità
        volatility = self.calculate_volatility(df, entry_idx, period=30)
        is_high_vol = volatility > self.high_volatility_threshold
        
        # LAYER 1: TIME SCORE
        in_gann_sequence = entry_bar in self.gann_sequence
        time_score = 40 if in_gann_sequence else 20
        
        # LAYER 2: PRICE SCORE
        eighths, range_total = self.calculate_gann_eighths(
            pivot_price, pivot_type, df, pivot_date
        )
        
        closest_eighth = min(range(9), key=lambda x: abs(eighths[x] - entry_price))
        distance_pct = abs(entry_price - eighths[closest_eighth]) / range_total * 100
        
        # Entry zone ideali
        if pivot_type == 'LOW':
            ideal_levels = [2, 3, 4]
        else:
            ideal_levels = [4, 5, 6]
        
        if closest_eighth in ideal_levels:
            price_score = max(25, 40 - int(distance_pct * 2))
        else:
            price_score = 20
        
        # LAYER 3: STATE SCORE
        state, gain_pct, momentum = self.calculate_enneagram_state(
            df, entry_idx, pivot_price, pivot_type
        )
        
        # Validazione stato
        if pivot_type == 'LOW':
            state_valid = state <= 4
        else:
            state_valid = state >= 6
        
        state_score = 20 if state_valid else 10
        
        # OTTIMIZZAZIONE: Momentum Confluence Check
        if pivot_type == 'LOW':
            momentum_aligned = (gain_pct > 0 and momentum > 0) or (gain_pct < 0 and momentum < 0)
        else:
            momentum_aligned = (gain_pct > 0 and momentum < 0) or (gain_pct < 0 and momentum > 0)
        
        if not momentum_aligned and abs(momentum) > 3:
            state_score -= 5  # Penalità per momentum contrario
        
        # SCORE BASE
        total_score = time_score + price_score + state_score
        
        # Bonus
        bonus = 0
        if distance_pct < 2:
            bonus += 5
        if abs(momentum) < 3:
            bonus += 5
        
        # OTTIMIZZAZIONE: Volatility Adjustment
        if is_high_vol:
            # In alta volatilità serve score più alto
            bonus -= 5
        
        total_score_final = min(100, total_score + bonus)
        
        # OTTIMIZZAZIONE: Volatility minimum score
        min_score_threshold = self.volatility_min_score if is_high_vol else 70
        
        if total_score_final < min_score_threshold:
            return None
        
        # CALCOLA ENTRY/EXIT
        if pivot_type == 'LOW':
            # OTTIMIZZAZIONE: SL dinamico basato su volatilità
            sl_multiplier = self.volatility_sl_multiplier if is_high_vol else 1.0
            sl_price = eighths[1] * (1 - 0.01 * sl_multiplier)
            
            tp1_price = eighths[4]
            tp2_price = eighths[6]
            tp3_price = eighths[8]
            direction = 'LONG'
        else:  # SHORT
            sl_multiplier = self.volatility_sl_multiplier if is_high_vol else 1.0
            sl_price = eighths[7] * (1 + 0.01 * sl_multiplier)
            
            tp1_price = eighths[4]
            tp2_price = eighths[2]
            tp3_price = eighths[0]
            direction = 'SHORT'
        
        sl_pct = ((sl_price - entry_price) / entry_price) * 100
        tp1_pct = ((tp1_price - entry_price) / entry_price) * 100
        tp2_pct = ((tp2_price - entry_price) / entry_price) * 100
        tp3_pct = ((tp3_price - entry_price) / entry_price) * 100
        
        risk = abs(sl_pct)
        reward = abs(tp2_pct)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # OTTIMIZZAZIONE: R:R Filter
        if rr_ratio < self.min_rr_ratio:
            return None
        
        # Position size basato su score
        if total_score_final >= 90:
            position_size = 100
        elif total_score_final >= 80:
            position_size = 75
        elif total_score_final >= 70:
            position_size = 50
        else:
            position_size = 0
        
        # OTTIMIZZAZIONE: Riduce position in alta volatilità
        if is_high_vol:
            position_size = min(50, position_size)
        
        if position_size == 0:
            return None
        
        return {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'tp3_price': tp3_price,
            'sl_pct': sl_pct,
            'tp1_pct': tp1_pct,
            'tp2_pct': tp2_pct,
            'tp3_pct': tp3_pct,
            'rr_ratio': rr_ratio,
            'score': total_score_final,
            'position_size': position_size,
            'direction': direction,
            'volatility': volatility,
            'is_high_vol': is_high_vol,
            'pivot_date': pivot_date,
            'pivot_price': pivot_price,
            'pivot_type': pivot_type,
            'state': state,
            'gain_pct': gain_pct,
            'momentum': momentum,
            'closest_eighth': closest_eighth
        }
    
    def simulate_trade(self, df, trade, entry_idx):
        """Simula outcome del trade"""
        
        future_bars = 30
        if entry_idx + future_bars >= len(df):
            future_bars = len(df) - entry_idx - 1
        
        if future_bars <= 0:
            return None
        
        future_df = df.iloc[entry_idx:entry_idx+future_bars+1]
        
        sl_price = trade['sl_price']
        tp1_price = trade['tp1_price']
        tp2_price = trade['tp2_price']
        tp3_price = trade['tp3_price']
        direction = trade['direction']
        
        if direction == 'LONG':
            hit_sl = (future_df['low'] <= sl_price).any()
            hit_tp1 = (future_df['high'] >= tp1_price).any()
            hit_tp2 = (future_df['high'] >= tp2_price).any()
            hit_tp3 = (future_df['high'] >= tp3_price).any()
        else:  # SHORT
            hit_sl = (future_df['high'] >= sl_price).any()
            hit_tp1 = (future_df['low'] <= tp1_price).any()
            hit_tp2 = (future_df['low'] <= tp2_price).any()
            hit_tp3 = (future_df['low'] <= tp3_price).any()
        
        # Determina outcome
        if hit_sl:
            if direction == 'LONG':
                exit_date = future_df[future_df['low'] <= sl_price].iloc[0]['date']
            else:
                exit_date = future_df[future_df['high'] >= sl_price].iloc[0]['date']
            
            return {
                'outcome': 'SL',
                'exit_date': exit_date,
                'pnl_pct': trade['sl_pct'],
                'exit_price': sl_price,
                'bars_held': (exit_date - trade['entry_date']).days
            }
        
        elif hit_tp3:
            if direction == 'LONG':
                exit_date = future_df[future_df['high'] >= tp3_price].iloc[0]['date']
            else:
                exit_date = future_df[future_df['low'] <= tp3_price].iloc[0]['date']
            
            return {
                'outcome': 'TP3',
                'exit_date': exit_date,
                'pnl_pct': trade['tp3_pct'],
                'exit_price': tp3_price,
                'bars_held': (exit_date - trade['entry_date']).days
            }
        
        elif hit_tp2:
            if direction == 'LONG':
                exit_date = future_df[future_df['high'] >= tp2_price].iloc[0]['date']
            else:
                exit_date = future_df[future_df['low'] <= tp2_price].iloc[0]['date']
            
            return {
                'outcome': 'TP2',
                'exit_date': exit_date,
                'pnl_pct': trade['tp2_pct'],
                'exit_price': tp2_price,
                'bars_held': (exit_date - trade['entry_date']).days
            }
        
        elif hit_tp1:
            if direction == 'LONG':
                exit_date = future_df[future_df['high'] >= tp1_price].iloc[0]['date']
            else:
                exit_date = future_df[future_df['low'] <= tp1_price].iloc[0]['date']
            
            return {
                'outcome': 'TP1',
                'exit_date': exit_date,
                'pnl_pct': trade['tp1_pct'],
                'exit_price': tp1_price,
                'bars_held': (exit_date - trade['entry_date']).days
            }
        
        else:
            return {
                'outcome': 'OPEN',
                'exit_date': future_df.iloc[-1]['date'],
                'pnl_pct': 0,
                'exit_price': future_df.iloc[-1]['close'],
                'bars_held': future_bars
            }

def run_backtest(df, system, start_date, end_date):
    """Esegue backtest completo"""
    
    print(f"\n{'='*80}")
    print(f"🚀 BACKTEST v5.2.0 OPTIMIZED: {start_date} → {end_date}")
    print(f"{'='*80}")
    
    df_period = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    df_period = df_period.reset_index(drop=True)
    
    print(f"\n📊 Periodo: {len(df_period)} giorni")
    
    # Rileva pivots
    pivots = system.detect_pivots(df_period)
    print(f"📍 Pivot rilevati: {len(pivots)} (LOW: {len(pivots[pivots['type']=='LOW'])}, HIGH: {len(pivots[pivots['type']=='HIGH'])})")
    
    # Analizza ogni pivot
    trades = []
    
    for _, pivot in pivots.iterrows():
        pivot_idx = pivot['index']
        
        # Testa diverse entry bar dalla sequenza GANN
        for entry_bar in system.gann_sequence:
            if entry_bar == 0:
                continue
            
            entry_idx = pivot_idx + entry_bar
            if entry_idx >= len(df_period):
                continue
            
            # Score trade
            trade_setup = system.score_trade(df_period, pivot, entry_bar)
            
            if trade_setup is None:
                continue
            
            # Simula trade
            outcome = system.simulate_trade(df_period, trade_setup, entry_idx)
            
            if outcome is None:
                continue
            
            # Combina setup + outcome
            trade_complete = {**trade_setup, **outcome}
            trades.append(trade_complete)
            
            # Exit dopo primo segnale valido per questo pivot
            break
    
    return trades, df_period

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Carica dati
    df = pd.read_csv('data/btcusdt_daily_1000.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Inizializza sistema
    system = GannEnneagramSystem()
    
    # Backtest
    trades, df_backtest = run_backtest(
        df, 
        system,
        start_date='2024-01-01',
        end_date='2026-01-12'
    )
    
    print(f"\n{'='*80}")
    print(f"📊 RISULTATI BACKTEST")
    print(f"{'='*80}")
    
    if len(trades) == 0:
        print("\n⚠️ Nessun trade eseguito")
    else:
        # Converti in DataFrame
        trades_df = pd.DataFrame(trades)
        
        print(f"\n✅ Trade eseguiti: {len(trades_df)}")
        print(f"\n📊 BREAKDOWN:")
        print(f"   LONG:  {len(trades_df[trades_df['direction']=='LONG'])}")
        print(f"   SHORT: {len(trades_df[trades_df['direction']=='SHORT'])}")
        
        # Outcome distribution
        outcome_counts = trades_df['outcome'].value_counts()
        print(f"\n🎯 OUTCOME:")
        for outcome, count in outcome_counts.items():
            pct = count / len(trades_df) * 100
            print(f"   {outcome:5s}: {count:3d} ({pct:5.1f}%)")
        
        # Win rate
        winners = trades_df[trades_df['pnl_pct'] > 0]
        losers = trades_df[trades_df['pnl_pct'] < 0]
        win_rate = len(winners) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        print(f"\n💰 PERFORMANCE:")
        print(f"   Win Rate:    {win_rate:.1f}%")
        print(f"   Winners:     {len(winners)}")
        print(f"   Losers:      {len(losers)}")
        
        # P&L stats
        total_pnl = trades_df['pnl_pct'].sum()
        avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
        avg_rr = trades_df['rr_ratio'].mean()
        
        print(f"   Total P&L:   {total_pnl:+.2f}%")
        print(f"   Avg Win:     {avg_win:+.2f}%")
        print(f"   Avg Loss:    {avg_loss:+.2f}%")
        print(f"   Avg R:R:     1:{avg_rr:.2f}")
        
        # Score stats
        avg_score = trades_df['score'].mean()
        high_score_trades = trades_df[trades_df['score'] >= 85]
        high_score_win_rate = len(high_score_trades[high_score_trades['pnl_pct'] > 0]) / len(high_score_trades) * 100 if len(high_score_trades) > 0 else 0
        
        print(f"\n🎯 SCORE STATS:")
        print(f"   Avg Score:        {avg_score:.1f}/100")
        print(f"   High Score (≥85): {len(high_score_trades)} trades")
        print(f"   High Score WR:    {high_score_win_rate:.1f}%")
        
        # Volatility stats
        high_vol_trades = trades_df[trades_df['is_high_vol'] == True]
        if len(high_vol_trades) > 0:
            high_vol_win_rate = len(high_vol_trades[high_vol_trades['pnl_pct'] > 0]) / len(high_vol_trades) * 100
            print(f"\n📊 VOLATILITY STATS:")
            print(f"   High Vol Trades: {len(high_vol_trades)}")
            print(f"   High Vol WR:     {high_vol_win_rate:.1f}%")
            print(f"   Avg Vol:         {high_vol_trades['volatility'].mean():.2f}%")
        
        # Top 5 best trades
        print(f"\n🏆 TOP 5 BEST TRADES:")
        top_trades = trades_df.nlargest(5, 'pnl_pct')
        for idx, trade in top_trades.iterrows():
            print(f"   {trade['entry_date'].date()} | {trade['direction']:5s} | Score {trade['score']:3.0f} | {trade['outcome']:3s} | {trade['pnl_pct']:+6.2f}% | R:R 1:{trade['rr_ratio']:.1f}")
        
        # Top 5 worst trades
        print(f"\n📉 TOP 5 WORST TRADES:")
        worst_trades = trades_df.nsmallest(5, 'pnl_pct')
        for idx, trade in worst_trades.iterrows():
            print(f"   {trade['entry_date'].date()} | {trade['direction']:5s} | Score {trade['score']:3.0f} | {trade['outcome']:3s} | {trade['pnl_pct']:+6.2f}% | R:R 1:{trade['rr_ratio']:.1f}")
        
        # Salva risultati
        trades_df.to_csv('/mnt/user-data/outputs/backtest_v520_results.csv', index=False)
        print(f"\n💾 Risultati salvati: /mnt/user-data/outputs/backtest_v520_results.csv")
        
        print(f"\n{'='*80}")
        print(f"✅ BACKTEST COMPLETATO")
        print(f"{'='*80}")
