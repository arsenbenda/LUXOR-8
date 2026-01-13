"""
LUXOR-8 Trading Bot - FastAPI Application v7.2 STABLE
Pure Python Implementation - Zero Native Dependencies
Exchange: Bybit (no geo-restrictions)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Optional
import ccxt
import pandas as pd
from strategy import TradingStrategy
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI(
    title="LUXOR-8 Trading Bot",
    version="7.2.0",
    description="Multi-Indicator Trading Strategy API (Pure Python)"
)

# Initialize trading strategy
strategy = TradingStrategy(
    rsi_period=14,
    rsi_oversold=30.0,
    rsi_overbought=70.0,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_period=20,
    bb_std=2.0
)

# Initialize exchange - BYBIT (no geo-blocking)
try:
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    })
except Exception as e:
    print(f"Warning: Exchange initialization failed: {e}")
    exchange = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class SignalResponse(BaseModel):
    signal: str
    reason: str
    price: float
    timestamp: str
    indicators: Dict
    signals: list


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "LUXOR-8 Trading Bot",
        "version": "7.2.0",
        "status": "running",
        "exchange": "Bybit",
        "endpoints": {
            "health": "/health",
            "test_signal": "/signal/test",
            "live_signal": "/signal/daily"
        },
        "implementation": "Pure Python (NO TA-Lib)",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="7.2.0"
    )


@app.get("/signal/test", response_model=SignalResponse, tags=["Signals"])
async def get_test_signal():
    """
    Test signal endpoint with mock data
    Returns a signal based on synthetic OHLCV data
    """
    try:
        # Generate mock OHLCV data (50 periods for indicators)
        mock_data = []
        base_price = 45000.0
        
        for i in range(50):
            # Simulate price movement
            price = base_price + (i * 100) + ((-1) ** i * 50)
            mock_data.append({
                'timestamp': datetime.utcnow().timestamp() - (3600 * (50 - i)),
                'open': price - 50,
                'high': price + 100,
                'low': price - 100,
                'close': price,
                'volume': 1000 + (i * 10)
            })
        
        df = pd.DataFrame(mock_data)
        
        # Analyze with strategy
        result = strategy.analyze(df)
        result['timestamp'] = datetime.utcnow().isoformat() + "Z"
        result['mock'] = True
        
        return SignalResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Test signal generation failed",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.get("/signal/daily", response_model=SignalResponse, tags=["Signals"])
async def get_daily_signal(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 100
):
    """
    Live signal endpoint - fetches real market data from Bybit
    
    Args:
        symbol: Trading pair (default: BTC/USDT)
        timeframe: Candle timeframe (default: 1d)
        limit: Number of candles to fetch (default: 100)
    
    Returns:
        Trading signal with indicators
    """
    try:
        if exchange is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Exchange not available",
                    "message": "CCXT exchange initialization failed",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        
        # Fetch OHLCV data from Bybit
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Analyze with strategy
        result = strategy.analyze(df)
        result['timestamp'] = datetime.utcnow().isoformat() + "Z"
        result['symbol'] = symbol
        result['timeframe'] = timeframe
        result['exchange'] = 'Bybit'
        result['mock'] = False
        
        return SignalResponse(**result)
        
    except ccxt.NetworkError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Network error",
                "message": f"Failed to connect to exchange: {str(e)}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    except ccxt.ExchangeError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Exchange error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Strategy execution failed",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
