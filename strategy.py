"""
LUXOR Trading Strategy v7.2.4
Kraken Compatible - Full Analysis Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TradingStrategy:
    """
    LUXOR Trading Strategy - Analisi completa multi-timeframe
    con regime detection, Gann analysis, consensus e trade setups
    """
    
    def __init__(self, symbol: str = "BTC/USD"):
        self.symbol = symbol
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "3d", "1w"]
        
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calcola RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calcola MACD"""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': float(macd.iloc[-1]) if len(macd) > 0 else 0.0,
            'signal': float(signal_line.iloc[-1]) if len(signal_line) > 0 else 0.0,
            'histogram': float(histogram.iloc[-1]) if len(histogram) > 0 else 0.0
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """Calcola Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return {
            'upper': float(upper.iloc[-1]) if len(upper) > 0 else 0.0,
            'middle': float(sma.iloc[-1]) if len(sma) > 0 else 0.0,
            'lower': float(lower.iloc[-1]) if len(lower) > 0 else 0.0
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calcola Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if len(atr) > 0 else 0.0
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict:
        """Calcola Average Directional Index"""
        # +DM e -DM
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean()
        
        # +DI e -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX e ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        adx_value = float(adx.iloc[-1]) if len(adx) > 0 else 0.0
        
        # Label
        if adx_value > 50:
            label = "Very Strong Trend"
        elif adx_value > 25:
            label = "Strong Trend"
        elif adx_value > 20:
            label = "Trending"
        else:
            label = "Weak/Ranging"
        
        return {
            'adx': adx_value,
            'plus_di': float(plus_di.iloc[-1]) if len(plus_di) > 0 else 0.0,
            'minus_di': float(minus_di.iloc[-1]) if len(minus_di) > 0 else 0.0,
            'label': label
        }
    
    def calculate_gann_levels(self, high: pd.Series, low: pd.Series, lookback: int = 20) -> Dict:
        """Calcola Gann Square of 9 levels"""
        recent_high = float(high.tail(lookback).max())
        recent_low = float(low.tail(lookback).min())
        gann_range = recent_high - recent_low
        
        # Gann 50% (pivot)
        gann_50 = recent_low + (gann_range * 0.5)
        
        # Livelli Gann 8ths
        levels = {}
        for i in range(9):
            level = recent_low + (gann_range * (i / 8))
            levels[f"{i}_8"] = float(level)
        
        # Supporti e Resistenze principali
        resistance_2 = levels["7_8"]
        resistance_1 = levels["6_8"]
        pivot = levels["4_8"]
        support_1 = levels["2_8"]
        support_2 = levels["1_8"]
        
        return {
            'resistance_2': resistance_2,
            'resistance_1': resistance_1,
            'pivot': pivot,
            'support_1': support_1,
            'support_2': support_2,
            'range': gann_range,
            'lookback_periods': lookback,
            'levels': levels
        }
    
    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """Rileva il regime di mercato"""
        close = df['close']
        
        # SMA 50 e 200
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        
        current_price = float(close.iloc[-1])
        sma_50_value = float(sma_50.iloc[-1]) if len(sma_50) > 0 else 0.0
        sma_200_value = float(sma_200.iloc[-1]) if len(sma_200) > 0 else 0.0
        
        # Posizione prezzo vs SMA
        price_vs_sma_50 = "ABOVE" if current_price > sma_50_value else "BELOW"
        price_vs_sma_200 = "ABOVE" if current_price > sma_200_value else "BELOW"
        
        # ADX per forza trend
        adx_data = self.calculate_adx(df['high'], df['low'], df['close'])
        adx = adx_data['adx']
        
        # Determina regime
        if price_vs_sma_200 == "ABOVE" and adx > 25:
            regime = "TRENDING_BULL"
            strength = 0.8
            allows_long = True
            allows_short = False
        elif price_vs_sma_200 == "BELOW" and adx > 25:
            regime = "TRENDING_BEAR"
            strength = 0.8
            allows_long = False
            allows_short = True
        elif adx < 20:
            regime = "RANGING"
            strength = 0.3
            allows_long = True
            allows_short = True
        else:
            regime = "CHOPPY"
            strength = 0.5
            allows_long = True
            allows_short = True
        
        return {
            'current': regime,
            'strength': strength,
            'strength_label': adx_data['label'],
            'sma_50': sma_50_value,
            'sma_200': sma_200_value,
            'price_vs_sma_50': price_vs_sma_50,
            'price_vs_sma_200': price_vs_sma_200,
            'allows_long': allows_long,
            'allows_short': allows_short,
            'warnings': []
        }
    
    def calculate_consensus(self, timeframes_data: Dict) -> Dict:
        """Calcola consensus multi-timeframe"""
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        tf_signals = {}
        
        for tf, data in timeframes_data.items():
            rsi = data.get('rsi', 50)
            macd = data.get('macd', {}).get('histogram', 0)
            
            # Segnale per timeframe
            if rsi > 55 and macd > 0:
                signal = "BULLISH"
                bullish_count += 1
            elif rsi < 45 and macd < 0:
                signal = "BEARISH"
                bearish_count += 1
            else:
                signal = "NEUTRAL"
                neutral_count += 1
            
            tf_signals[f"tf_{tf}"] = signal
        
        # Direzione primaria
        if bullish_count > bearish_count and bullish_count > neutral_count:
            primary_direction = "BULLISH"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            primary_direction = "BEARISH"
        else:
            primary_direction = "NEUTRAL"
        
        # Punteggio ponderato
        weighted_score = (bullish_count * 2) - (bearish_count * 2)
        
        # Allineamento
        total_tf = len(timeframes_data)
        dominant_count = max(bullish_count, bearish_count, neutral_count)
        alignment_pct = (dominant_count / total_tf) * 100 if total_tf > 0 else 0
        
        if alignment_pct >= 75:
            alignment = "STRONG"
        elif alignment_pct >= 60:
            alignment = "MODERATE"
        else:
            alignment = "WEAK"
        
        # Verdict
        if primary_direction == "BULLISH" and alignment in ["STRONG", "MODERATE"]:
            verdict = "BULLISH CONFIRMED"
        elif primary_direction == "BEARISH" and alignment in ["STRONG", "MODERATE"]:
            verdict = "BEARISH CONFIRMED"
        elif alignment == "WEAK":
            verdict = "CONFLICTED"
        else:
            verdict = "NEUTRAL"
        
        return {
            'primary_direction': primary_direction,
            'weighted_score': weighted_score,
            'alignment': alignment,
            'verdict': verdict,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'has_conflicts': alignment == "WEAK",
            **tf_signals
        }
    
    def generate_trade_setup(self, df: pd.DataFrame, signal: str, confidence: float, gann: Dict) -> Dict:
        """Genera trade setup con entry, stop, targets"""
        current_price = float(df['close'].iloc[-1])
        atr = self.calculate_atr(df['high'], df['low'], df['close'])
        
        if signal == "BULLISH":
            entry = current_price
            stop_loss = gann['support_1']
            tp1 = gann['resistance_1']
            tp2 = gann['resistance_2']
            tp3 = gann['resistance_2'] + (atr * 2)
            
            risk = entry - stop_loss
            reward = tp1 - entry
            
        elif signal == "BEARISH":
            entry = current_price
            stop_loss = gann['resistance_1']
            tp1 = gann['support_1']
            tp2 = gann['support_2']
            tp3 = gann['support_2'] - (atr * 2)
            
            risk = stop_loss - entry
            reward = entry - tp1
            
        else:
            return {
                'valid': False,
                'entry': current_price,
                'stop_loss': 0,
                'tp1': 0,
                'tp2': 0,
                'tp3': 0,
                'risk': 0,
                'risk_pct': 0,
                'rr_ratio': 0,
                'position_size': 0,
                'confidence': 'N/A',
                'rationale': 'No valid setup - WAIT signal'
            }
        
        rr_ratio = reward / risk if risk > 0 else 0
        risk_pct = (risk / entry) * 100
        
        # Position size (assumendo 2% risk)
        account_risk_pct = 2.0
        position_size = account_risk_pct / risk_pct if risk_pct > 0 else 0
        
        # Validità setup
        valid = (
            (signal == "BULLISH" and confidence >= 60 and rr_ratio >= 1.5) or
            (signal == "BEARISH" and confidence >= 85 and rr_ratio >= 2.0)
        )
        
        return {
            'valid': valid,
            'entry': float(entry),
            'stop_loss': float(stop_loss),
            'tp1': float(tp1),
            'tp2': float(tp2),
            'tp3': float(tp3),
            'risk': float(risk),
            'risk_pct': float(risk_pct),
            'rr_ratio': float(rr_ratio),
            'position_size': float(position_size),
            'confidence': f"{confidence:.0f}%",
            'rationale': f"{signal} setup with {confidence:.0f}% confidence, R/R {rr_ratio:.2f}:1"
        }
    
    def analyze(self, ohlcv_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analisi completa multi-timeframe
        Input: Dict con key = timeframe, value = DataFrame OHLCV
        Output: JSON completo per n8n formatter
        """
        try:
            # Timeframe principale: 1D
            df_1d = ohlcv_data.get('1d')
            if df_1d is None or len(df_1d) < 50:
                raise ValueError("Insufficient data for 1D timeframe")
            
            current_price = float(df_1d['close'].iloc[-1])
            timestamp = datetime.utcnow().isoformat() + 'Z'
            
            # =====================================
            # INDICATORI 1D (DAILY)
            # =====================================
            rsi_1d = self.calculate_rsi(df_1d['close'])
            macd_1d = self.calculate_macd(df_1d['close'])
            bb_1d = self.calculate_bollinger_bands(df_1d['close'])
            gann_1d = self.calculate_gann_levels(df_1d['high'], df_1d['low'])
            atr_1d = self.calculate_atr(df_1d['high'], df_1d['low'], df_1d['close'])
            adx_1d = self.calculate_adx(df_1d['high'], df_1d['low'], df_1d['close'])
            
            atr_pct = (atr_1d / current_price) * 100
            
            tf_1d_data = {
                'rsi': float(rsi_1d.iloc[-1]) if len(rsi_1d) > 0 else 0.0,
                'macd': macd_1d,
                'bollinger': bb_1d,
                'gann': gann_1d,
                'atr': atr_1d,
                'atr_pct': atr_pct,
                'adx': adx_1d['adx'],
                'adx_label': adx_1d['label']
            }
            
            # =====================================
            # INDICATORI ALTRI TIMEFRAMES
            # =====================================
            timeframes_analysis = {'1d': tf_1d_data}
            
            for tf in ['1w', '3d']:
                df_tf = ohlcv_data.get(tf)
                if df_tf is not None and len(df_tf) >= 20:
                    rsi_tf = self.calculate_rsi(df_tf['close'])
                    macd_tf = self.calculate_macd(df_tf['close'])
                    
                    timeframes_analysis[tf] = {
                        'rsi': float(rsi_tf.iloc[-1]) if len(rsi_tf) > 0 else 0.0,
                        'macd': macd_tf
                    }
            
            # =====================================
            # REGIME DETECTION
            # =====================================
            regime = self.detect_regime(df_1d)
            
            # =====================================
            # CONSENSUS MULTI-TIMEFRAME
            # =====================================
            consensus = self.calculate_consensus(timeframes_analysis)
            
            # =====================================
            # PRIMARY BIAS
            # =====================================
            rsi_value = tf_1d_data['rsi']
            macd_hist = macd_1d['histogram']
            
            # Conteggio segnali
            signals = []
            
            if rsi_value > 55:
                signals.append("RSI_BULLISH")
            elif rsi_value < 45:
                signals.append("RSI_BEARISH")
            
            if macd_hist > 50:
                signals.append("MACD_BULLISH")
            elif macd_hist < -50:
                signals.append("MACD_BEARISH")
            
            if consensus['primary_direction'] == "BULLISH":
                signals.append("CONSENSUS_BULLISH")
            elif consensus['primary_direction'] == "BEARISH":
                signals.append("CONSENSUS_BEARISH")
            
            # Bias finale
            bullish_signals = len([s for s in signals if 'BULLISH' in s])
            bearish_signals = len([s for s in signals if 'BEARISH' in s])
            
            if bullish_signals >= 2:
                primary_bias = "BULLISH"
                confidence = 0.6 + (bullish_signals * 0.15)
            elif bearish_signals >= 2:
                primary_bias = "BEARISH"
                confidence = 0.6 + (bearish_signals * 0.15)
            else:
                primary_bias = "NEUTRAL"
                confidence = 0.3
            
            confidence = min(confidence, 0.95)
            
            # Invalidation
            if primary_bias == "BULLISH":
                invalidation_price = gann_1d['support_1']
                invalidation_desc = f"Below Gann S1 (${invalidation_price:.2f})"
            elif primary_bias == "BEARISH":
                invalidation_price = gann_1d['resistance_1']
                invalidation_desc = f"Above Gann R1 (${invalidation_price:.2f})"
            else:
                invalidation_price = 0
                invalidation_desc = "No clear invalidation level"
            
            # =====================================
            # GANN CONTEXT
            # =====================================
            gann_50 = gann_1d['pivot']
            distance_pct = ((current_price - gann_50) / gann_50) * 100
            
            if current_price > gann_1d['resistance_1']:
                gann_position = "ABOVE_R1"
                gann_bias = "BULLISH"
            elif current_price > gann_50:
                gann_position = "ABOVE_PIVOT"
                gann_bias = "BULLISH"
            elif current_price > gann_1d['support_1']:
                gann_position = "BELOW_PIVOT"
                gann_bias = "BEARISH"
            else:
                gann_position = "BELOW_S1"
                gann_bias = "BEARISH"
            
            # =====================================
            # TRADE SETUPS
            # =====================================
            trade_setup = self.generate_trade_setup(
                df_1d, 
                primary_bias, 
                confidence * 100,
                gann_1d
            )
            
            # =====================================
            # OUTPUT JSON COMPLETO
            # =====================================
            output = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'current_price': current_price,
                
                'primary_bias': {
                    'primary_bias': primary_bias,
                    'confidence': confidence,
                    'source': 'Multi-Indicator Analysis',
                    'conflict': len(signals) < 2,
                    'invalidation': {
                        'price': invalidation_price,
                        'description': invalidation_desc
                    }
                },
                
                'signals': signals,
                
                'timeframes': {
                    '1D': tf_1d_data,
                    '1W': timeframes_analysis.get('1w', {}),
                    '3D': timeframes_analysis.get('3d', {})
                },
                
                'regime': regime,
                
                'consensus': consensus,
                
                'gann_context': {
                    'gann_50': gann_50,
                    'position': gann_position,
                    'distance_pct': distance_pct,
                    'bias': gann_bias
                },
                
                'trade_setups': [trade_setup] if trade_setup['valid'] else [],
                
                'capitulation': {
                    'is_capitulation': False,
                    'score': 0,
                    'max_score': 12
                },
                
                'time_forecast': {
                    'cycle_origin': {},
                    'primary_forecast': {}
                }
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Strategy analysis error: {str(e)}")
            raise
