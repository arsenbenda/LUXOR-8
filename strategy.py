#!/usr/bin/env python3
"""
LUXOR v7.1 AGGRESSIVE STRATEGY
Bitcoin Trading Strategy with Gann High/Low Integration
Author: LUXOR Team
Version: 7.1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import logging

# Import indicators
from indicators import RSI, EMA, ATR, calculate_volatility

# Import Gann High/Low module
from gann_high_low import (
    find_gann_high_low,
    GannLevels,
    get_multi_timeframe_gann
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class TradeSignal:
    """Trade signal with complete metadata"""
    signal: str  # "LONG", "SHORT", "NEUTRAL"
    score: float  # 0-100
    price: float
    timestamp: datetime
    
    # Indicators
    rsi: float
    ema_fast: float
    ema_slow: float
    ema_200: float
    atr: float
    volatility: float
    
    # Gann Levels
    gann_high: float
    gann_low: float
    gann_confidence: float
    gann_volatility_regime: str
    
    # Entry/Exit Levels
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Risk Management
    risk_usd: float
    risk_pct: float
    position_size: float
    risk_reward_ratio: float
    
    # Metadata
    timeframe: str
    market: str
    exchange: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API/JSON export"""
        return {
            "signal": self.signal,
            "score": round(self.score, 2),
            "price": round(self.price, 2),
            "timestamp": self.timestamp.isoformat() + "Z",
            "indicators": {
                "rsi": round(self.rsi, 2),
                "ema_fast": round(self.ema_fast, 2),
                "ema_slow": round(self.ema_slow, 2),
                "ema_200": round(self.ema_200, 2),
                "atr": round(self.atr, 2),
                "volatility": round(self.volatility, 4)
            },
            "gann": {
                "high": round(self.gann_high, 2),
                "low": round(self.gann_low, 2),
                "confidence": round(self.gann_confidence, 1),
                "volatility_regime": self.gann_volatility_regime
            },
            "levels": {
                "entry": round(self.entry_price, 2),
                "stop_loss": round(self.stop_loss, 2),
                "take_profit": round(self.take_profit, 2),
                "support_1": round(self.gann_low, 2),
                "resistance_1": round(self.gann_high, 2)
            },
            "risk": {
                "risk_usd": round(self.risk_usd, 2),
                "risk_pct": round(self.risk_pct, 2),
                "position_size": round(self.position_size, 6),
                "risk_reward": round(self.risk_reward_ratio, 2)
            },
            "metadata": {
                "timeframe": self.timeframe,
                "market": self.market,
                "exchange": self.exchange,
                "strategy": "LUXOR v7.1 AGGRESSIVE",
                "version": "7.1.0"
            }
        }

# ============================================
# STRATEGY CLASS
# ============================================

class LuxorV7AggressiveStrategy:
    """
    LUXOR v7.1 AGGRESSIVE Strategy
    
    Features:
    - Multi-timeframe EMA alignment
    - RSI for momentum
    - ATR-based stops
    - Gann High/Low dynamic levels
    - Adaptive lookback based on volatility
    - Confidence scoring system
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,
        timeframe: str = "1D",
        # EMA Parameters
        ema_fast_period: int = 21,
        ema_slow_period: int = 50,
        ema_trend_period: int = 200,
        # RSI Parameters
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        # ATR Parameters
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        # Gann Parameters
        use_adaptive_gann: bool = True,
        gann_min_confidence: float = 50.0,
        # Scoring Thresholds
        min_long_score: float = 60,
        min_short_score: float = 85,
        # Market/Exchange
        market: str = "BTC/USDT",
        exchange: str = "Binance"
    ):
        # Capital & Risk
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.timeframe = timeframe
        
        # Indicator Parameters
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.ema_trend_period = ema_trend_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        
        # Gann Parameters
        self.use_adaptive_gann = use_adaptive_gann
        self.gann_min_confidence = gann_min_confidence
        
        # Scoring
        self.min_long_score = min_long_score
        self.min_short_score = min_short_score
        
        # Market Info
        self.market = market
        self.exchange = exchange
        
        logger.info(f"LUXOR v7.1 AGGRESSIVE initialized for {market} on {timeframe}")
    
    # ============================================
    # INDICATOR CALCULATIONS
    # ============================================
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with indicators added
        """
        logger.debug("Calculating indicators...")
        
        # EMAs
        df['ema_fast'] = EMA(df['close'], period=self.ema_fast_period)
        df['ema_slow'] = EMA(df['close'], period=self.ema_slow_period)
        df['ema_200'] = EMA(df['close'], period=self.ema_trend_period)
        
        # RSI
        df['rsi'] = RSI(df['close'], period=self.rsi_period)
        
        # ATR
        df['atr'] = ATR(df['high'], df['low'], df['close'], period=self.atr_period)
        
        # Volatility
        df['volatility'] = calculate_volatility(df['close'], period=20)
        
        return df
    
    # ============================================
    # SIGNAL GENERATION
    # ============================================
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_capital: Optional[float] = None
    ) -> TradeSignal:
        """
        Generate trading signal with Gann levels
        
        Args:
            df: DataFrame with OHLCV data
            current_capital: Current account capital (optional)
        
        Returns:
            TradeSignal object with complete trade setup
        """
        # Use initial capital if not provided
        if current_capital is None:
            current_capital = self.initial_capital
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest values
        idx = -1
        current_price = df.iloc[idx]['close']
        current_high = df.iloc[idx]['high']
        current_low = df.iloc[idx]['low']
        
        rsi = df.iloc[idx]['rsi']
        ema_fast = df.iloc[idx]['ema_fast']
        ema_slow = df.iloc[idx]['ema_slow']
        ema_200 = df.iloc[idx]['ema_200']
        atr = df.iloc[idx]['atr']
        volatility = df.iloc[idx]['volatility']
        
        # ============================================
        # GANN HIGH/LOW CALCULATION
        # ============================================
        
        gann = find_gann_high_low(
            df,
            timeframe=self.timeframe,
            use_adaptive=self.use_adaptive_gann
        )
        
        logger.info(f"Gann Levels - High: ${gann.high:.2f}, Low: ${gann.low:.2f}")
        logger.info(f"Gann Confidence: {gann.confidence:.1f}%, Regime: {gann.volatility_regime}")
        
        # ============================================
        # SCORING SYSTEM (0-100)
        # ============================================
        
        score = 0.0
        signal = "NEUTRAL"
        
        # 1. TREND ALIGNMENT (30 points)
        if current_price > ema_200:
            score += 30  # Bullish trend
        elif current_price < ema_200:
            score -= 30  # Bearish trend (for SHORT)
        
        # 2. MOMENTUM (25 points)
        if ema_fast > ema_slow:
            score += 25  # Bullish momentum
        elif ema_fast < ema_slow:
            score -= 25  # Bearish momentum
        
        # 3. RSI (20 points)
        if rsi < self.rsi_oversold:
            score += 20  # Oversold (bullish)
        elif rsi > self.rsi_overbought:
            score -= 20  # Overbought (bearish for SHORT)
        elif 40 <= rsi <= 60:
            score += 10  # Neutral zone bonus
        
        # 4. GANN POSITION (15 points)
        gann_range = gann.high - gann.low
        price_position = (current_price - gann.low) / gann_range if gann_range > 0 else 0.5
        
        if price_position < 0.3:
            score += 15  # Near Gann low (bullish)
        elif price_position > 0.7:
            score -= 15  # Near Gann high (bearish)
        
        # 5. GANN CONFIDENCE BONUS (10 points)
        if gann.confidence >= self.gann_min_confidence:
            score += (gann.confidence / 100) * 10
        
        # Normalize score to 0-100
        score = max(0, min(100, score))
        
        # ============================================
        # SIGNAL DETERMINATION
        # ============================================
        
        if score >= self.min_long_score and gann.confidence >= self.gann_min_confidence:
            signal = "LONG"
            logger.info(f"🟢 LONG signal generated (score: {score:.1f})")
        elif score <= (100 - self.min_short_score) and gann.confidence >= self.gann_min_confidence:
            signal = "SHORT"
            score = 100 - score  # Invert score for SHORT display
            logger.info(f"🔴 SHORT signal generated (score: {score:.1f})")
        else:
            signal = "NEUTRAL"
            logger.info(f"⚪ NEUTRAL signal (score: {score:.1f})")
        
        # ============================================
        # ENTRY/STOP/TARGET CALCULATION
        # ============================================
        
        if signal == "LONG":
            entry_price = current_price
            stop_loss = max(gann.low, current_price - (atr * self.atr_stop_multiplier))
            take_profit = min(gann.high, current_price + (atr * self.atr_stop_multiplier * 2))
            
        elif signal == "SHORT":
            entry_price = current_price
            stop_loss = min(gann.high, current_price + (atr * self.atr_stop_multiplier))
            take_profit = max(gann.low, current_price - (atr * self.atr_stop_multiplier * 2))
            
        else:  # NEUTRAL
            entry_price = current_price
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.03
        
        # ============================================
        # POSITION SIZING
        # ============================================
        
        risk_amount = current_capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
            # Convert to BTC (assuming USDT pair)
            position_size_btc = position_size / current_price
        else:
            position_size_btc = 0
            risk_amount = 0
        
        # Risk/Reward Ratio
        profit_distance = abs(take_profit - entry_price)
        rr_ratio = profit_distance / stop_distance if stop_distance > 0 else 0
        
        # ============================================
        # CREATE TRADE SIGNAL
        # ============================================
        
        trade_signal = TradeSignal(
            signal=signal,
            score=score,
            price=current_price,
            timestamp=datetime.utcnow(),
            # Indicators
            rsi=rsi,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_200=ema_200,
            atr=atr,
            volatility=volatility,
            # Gann
            gann_high=gann.high,
            gann_low=gann.low,
            gann_confidence=gann.confidence,
            gann_volatility_regime=gann.volatility_regime,
            # Levels
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            # Risk
            risk_usd=risk_amount,
            risk_pct=self.risk_per_trade * 100,
            position_size=position_size_btc,
            risk_reward_ratio=rr_ratio,
            # Metadata
            timeframe=self.timeframe,
            market=self.market,
            exchange=self.exchange
        )
        
        return trade_signal
    
    # ============================================
    # MULTI-TIMEFRAME ANALYSIS
    # ============================================
    
    def analyze_multi_timeframe(
        self,
        dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, TradeSignal]:
        """
        Analyze multiple timeframes and generate signals
        
        Args:
            dfs: Dictionary mapping timeframe to DataFrame
                 e.g., {"1D": df_daily, "1W": df_weekly}
        
        Returns:
            Dictionary mapping timeframe to TradeSignal
        
        Example:
            >>> strategy = LuxorV7AggressiveStrategy()
            >>> dfs = {"1D": daily_df, "1W": weekly_df}
            >>> signals = strategy.analyze_multi_timeframe(dfs)
            >>> print(signals["1D"].signal, signals["1W"].signal)
        """
        signals = {}
        
        for tf, df in dfs.items():
            logger.info(f"Analyzing {tf} timeframe...")
            
            # Temporarily change timeframe
            original_tf = self.timeframe
            self.timeframe = tf
            
            try:
                signal = self.generate_signal(df)
                signals[tf] = signal
            except Exception as e:
                logger.error(f"Error analyzing {tf}: {e}")
            
            # Restore original timeframe
            self.timeframe = original_tf
        
        return signals
    
    # ============================================
    # SIGNAL VALIDATION
    # ============================================
    
    def validate_signal(self, signal: TradeSignal) -> Tuple[bool, str]:
        """
        Validate if signal meets minimum requirements
        
        Args:
            signal: TradeSignal to validate
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check Gann confidence
        if signal.gann_confidence < self.gann_min_confidence:
            return False, f"Gann confidence too low: {signal.gann_confidence:.1f}%"
        
        # Check risk/reward
        if signal.risk_reward_ratio < 1.5:
            return False, f"R/R too low: {signal.risk_reward_ratio:.2f}"
        
        # Check volatility
        if signal.volatility > 0.05:  # 5% extreme volatility
            return False, f"Volatility too high: {signal.volatility * 100:.2f}%"
        
        # Check score
        if signal.signal == "LONG" and signal.score < self.min_long_score:
            return False, f"Score too low for LONG: {signal.score:.1f}"
        
        if signal.signal == "SHORT" and signal.score < self.min_short_score:
            return False, f"Score too low for SHORT: {signal.score:.1f}"
        
        return True, "Signal valid"

# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("LUXOR v7.1 AGGRESSIVE STRATEGY - Example Usage")
    print("=" * 70)
    
    # Create synthetic data
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    np.random.seed(42)
    price = 40000 + np.cumsum(np.random.randn(300) * 500)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(300) * 100,
        'high': price + abs(np.random.randn(300) * 200),
        'low': price - abs(np.random.randn(300) * 200),
        'close': price + np.random.randn(300) * 100,
        'volume': np.random.uniform(1000, 5000, 300)
    })
    
    # Initialize strategy
    strategy = LuxorV7AggressiveStrategy(
        initial_capital=10000,
        risk_per_trade=0.02,
        timeframe="1D",
        use_adaptive_gann=True
    )
    
    # Generate signal
    signal = strategy.generate_signal(df)
    
    # Print results
    print(f"\n{'='*70}")
    print("SIGNAL GENERATED")
    print(f"{'='*70}")
    print(f"Signal:     {signal.signal}")
    print(f"Score:      {signal.score:.1f}/100")
    print(f"Price:      ${signal.price:,.2f}")
    print(f"\n{'='*70}")
    print("GANN LEVELS")
    print(f"{'='*70}")
    print(f"High:       ${signal.gann_high:,.2f}")
    print(f"Low:        ${signal.gann_low:,.2f}")
    print(f"Confidence: {signal.gann_confidence:.1f}%")
    print(f"Regime:     {signal.gann_volatility_regime}")
    print(f"\n{'='*70}")
    print("TRADE SETUP")
    print(f"{'='*70}")
    print(f"Entry:      ${signal.entry_price:,.2f}")
    print(f"Stop Loss:  ${signal.stop_loss:,.2f} ({((signal.entry_price - signal.stop_loss) / signal.entry_price * 100):.2f}%)")
    print(f"Take Profit:${signal.take_profit:,.2f} ({((signal.take_profit - signal.entry_price) / signal.entry_price * 100):.2f}%)")
    print(f"R/R:        {signal.risk_reward_ratio:.2f}:1")
    print(f"\n{'='*70}")
    print("RISK MANAGEMENT")
    print(f"{'='*70}")
    print(f"Position:   {signal.position_size:.6f} BTC")
    print(f"Risk:       ${signal.risk_usd:.2f} ({signal.risk_pct:.2f}%)")
    print(f"\n{'='*70}")
    print("INDICATORS")
    print(f"{'='*70}")
    print(f"RSI:        {signal.rsi:.2f}")
    print(f"EMA Fast:   ${signal.ema_fast:,.2f}")
    print(f"EMA Slow:   ${signal.ema_slow:,.2f}")
    print(f"EMA 200:    ${signal.ema_200:,.2f}")
    print(f"ATR:        ${signal.atr:.2f}")
    print(f"Volatility: {signal.volatility * 100:.2f}%")
    
    # Validate signal
    is_valid, reason = strategy.validate_signal(signal)
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")
    print(f"Valid:      {'✅ YES' if is_valid else '❌ NO'}")
    print(f"Reason:     {reason}")
    
    # Export to dict
    print(f"\n{'='*70}")
    print("JSON EXPORT (for API)")
    print(f"{'='*70}")
    import json
    print(json.dumps(signal.to_dict(), indent=2))
    
    print(f"\n{'='*70}")
    print("✅ Example complete!")
    print(f"{'='*70}\n")
