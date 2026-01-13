"""
LUXOR v7.1 AGGRESSIVE Strategy
Core trading logic and signal generation
"""

import numpy as np
import pandas as pd
from .indicators import calculate_indicators

class LuxorStrategy:
    """LUXOR v7.1 AGGRESSIVE Trading Strategy"""
    
    def __init__(self, config):
        self.config = config
        self.params = config['parameters']
        self.scoring = self.params['scoring']
        self.entry = self.params['entry']
        self.risk = self.params['risk_management']
        self.sizing = self.params['position_sizing']
    
    def calculate_score(self, row):
        """
        Calculate multi-layer technical score (0-100)
        
        Scoring components:
        - RSI: mean reversion signals
        - EMA: trend alignment
        - Volume: confirmation
        - Volatility: ideal range bonus
        """
        score = 50  # Base score
        
        # RSI scoring
        rsi = row['rsi']
        if rsi < 30:
            score += 20  # Oversold
        elif rsi > 70:
            score += 15  # Overbought (for shorts)
        elif 40 <= rsi <= 60:
            score += 5  # Neutral zone
        
        # EMA trend
        if row['ema_fast'] > row['ema_slow']:
            score += 10  # Bullish trend
        else:
            score += 5  # Bearish trend (still tradeable)
        
        # Volume confirmation
        if row['volume'] > row['volume_ma']:
            score += 10
        
        # Volatility bonus (ideal range)
        vol = row['volatility']
        if self.scoring['ideal_volatility_min'] <= vol <= self.scoring['ideal_volatility_max']:
            score += 10
        elif vol > self.scoring['ideal_volatility_max']:
            score -= 5  # Penalty for extreme volatility
        
        # Swing pivot bonus
        if row['pivot_high'] or row['pivot_low']:
            score += 15
        
        return min(100, max(0, score))
    
    def generate_signals(self, df):
        """
        Generate trading signals
        
        Returns:
            DataFrame with 'signal' column ('LONG', 'SHORT', or None)
        """
        df = df.copy()
        df['score'] = df.apply(self.calculate_score, axis=1)
        
        # Initialize signal column
        df['signal'] = None
        
        min_score = self.scoring['min_score']
        max_score = self.scoring['max_score']
        
        for i in range(len(df)):
            score = df.iloc[i]['score']
            
            # Score filter
            if score < min_score or score > max_score:
                continue
            
            # RSI filter (optional)
            rsi = df.iloc[i]['rsi']
            if rsi < self.scoring['min_rsi'] or rsi > self.scoring['max_rsi']:
                continue
            
            # Determine direction
            if df.iloc[i]['pivot_low'] and rsi < 40:
                df.iloc[i, df.columns.get_loc('signal')] = 'LONG'
            elif df.iloc[i]['pivot_high'] and rsi > 60:
                df.iloc[i, df.columns.get_loc('signal')] = 'SHORT'
            elif df.iloc[i]['ema_fast'] > df.iloc[i]['ema_slow'] and rsi < 50:
                df.iloc[i, df.columns.get_loc('signal')] = 'LONG'
            elif df.iloc[i]['ema_fast'] < df.iloc[i]['ema_slow'] and rsi > 50:
                df.iloc[i, df.columns.get_loc('signal')] = 'SHORT'
        
        return df
    
    def calculate_position_size(self, score):
        """Calculate position size based on score confidence"""
        base = self.sizing['base_size']
        multiplier = self.sizing['high_confidence_multiplier']
        threshold = self.sizing['high_confidence_threshold']
        
        if score >= threshold:
            return base * multiplier
        return base
    
    def calculate_stop_loss(self, entry_price, direction, atr):
        """Calculate initial stop loss"""
        multiplier = self.risk['sl_atr_multiplier']
        
        if direction == 'LONG':
            return entry_price - (atr * multiplier)
        else:  # SHORT
            return entry_price + (atr * multiplier)
    
    def calculate_trailing_stop(self, entry_price, current_price, direction, atr, profit_r):
        """
        Calculate trailing stop based on profit milestones
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            direction: 'LONG' or 'SHORT'
            atr: Current ATR value
            profit_r: Current profit in R multiples
        
        Returns:
            New trailing stop price or None
        """
        breakeven = self.risk['breakeven_at']
        trailing_mult = self.risk['trailing_atr_multiplier']
        
        # Move to breakeven at 0.3R profit
        if profit_r >= breakeven and profit_r < 0.5:
            return entry_price
        
        # Trailing stop after breakeven
        if profit_r >= 0.5:
            if direction == 'LONG':
                return current_price - (atr * trailing_mult)
            else:  # SHORT
                return current_price + (atr * trailing_mult)
        
        return None
