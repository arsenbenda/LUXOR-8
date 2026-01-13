# LUXOR v7.1 AGGRESSIVE - FastAPI Application
# Main API endpoint for trading signals

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime
import logging

# Import strategy (make sure strategy.py is in the same directory)
try:
    from strategy import LuxorStrategy
except ImportError:
    # Fallback: return mock data if strategy not available
    LuxorStrategy = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LUXOR v7.1 AGGRESSIVE API",
    description="Trading signal API with Gann High/Low integration",
    version="7.1.0"
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class SignalResponse(BaseModel):
    signal: str
    score: float
    price: float
    timestamp: str
    indicators: Dict[str, Any]
    gann_levels: Optional[Dict[str, Any]] = None
    multi_timeframe_gann: Optional[Dict[str, Any]] = None
    levels: Dict[str, float]
    risk: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "7.1.0",
        "strategy_loaded": LuxorStrategy is not None
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LUXOR v7.1 AGGRESSIVE Trading API",
        "version": "7.1.0",
        "endpoints": {
            "health": "/health",
            "signal": "/signal/daily",
            "test": "/signal/test"
        },
        "docs": "/docs",
        "timestamp": datetime.utcnow().isoformat()
    }

# Main signal endpoint
@app.get("/signal/daily", response_model=SignalResponse)
async def get_daily_signal(
    symbol: str = Query(default="BTCUSDT", description="Trading pair symbol"),
    timeframe: str = Query(default="1D", description="Timeframe (1D, 3D, 1W, 1M)"),
    use_gann: bool = Query(default=True, description="Enable Gann High/Low analysis"),
    use_multi_timeframe: bool = Query(default=True, description="Enable multi-timeframe consensus")
):
    """
    Get daily trading signal with Gann High/Low integration
    
    Parameters:
    - symbol: Trading pair (default: BTCUSDT)
    - timeframe: Chart timeframe (default: 1D)
    - use_gann: Enable Gann analysis (default: True)
    - use_multi_timeframe: Enable MTF consensus (default: True)
    
    Returns:
    - Complete trading signal with entry/stop/target levels
    """
    try:
        logger.info(f"Generating signal for {symbol} on {timeframe}")
        
        if LuxorStrategy is None:
            # Return mock data if strategy not loaded
            logger.warning("Strategy not loaded, returning mock data")
            return get_mock_signal()
        
        # Initialize strategy
        strategy = LuxorStrategy(
            symbol=symbol,
            timeframe=timeframe,
            use_gann=use_gann,
            use_multi_timeframe=use_multi_timeframe
        )
        
        # Generate signal
        signal_data = strategy.generate_signal()
        
        logger.info(f"Signal generated: {signal_data['signal']} with score {signal_data['score']}")
        
        return signal_data
        
    except Exception as e:
        logger.error(f"Error generating signal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating signal: {str(e)}")

# Test endpoint with mock data
@app.get("/signal/test", response_model=SignalResponse)
async def get_test_signal():
    """
    Get test signal with mock data (for testing without live data)
    """
    return get_mock_signal()

def get_mock_signal() -> dict:
    """Generate mock signal data for testing"""
    return {
        "signal": "LONG",
        "score": 85.3,
        "price": 45230.00,
        "timestamp": datetime.utcnow().isoformat(),
        "indicators": {
            "rsi": 58.5,
            "ema_fast": 44900.00,
            "ema_slow": 44200.00,
            "ema_200": 42800.00,
            "atr": 920.00,
            "volatility": 0.028
        },
        "gann_levels": {
            "timeframe": "1D",
            "gann_high": 46800.00,
            "gann_low": 43200.00,
            "range_pct": 8.33,
            "confidence": 85,
            "volatility_regime": "MEDIUM",
            "lookback_used": 108
        },
        "multi_timeframe_gann": {
            "consensus_high": 46500.00,
            "consensus_low": 43500.00,
            "avg_confidence": 82.5,
            "dominant_trend": "BULLISH",
            "timeframes": {
                "1D": {"high": 46800.00, "low": 43200.00, "conf": 85},
                "1W": {"high": 46400.00, "low": 43600.00, "conf": 80},
                "1M": {"high": 46200.00, "low": 43800.00, "conf": 78}
            }
        },
        "levels": {
            "entry": 45200.00,
            "stop_loss": 44400.00,
            "take_profit": 46800.00,
            "support_1": 44000.00,
            "support_2": 43200.00,
            "resistance_1": 46000.00,
            "resistance_2": 47500.00
        },
        "risk": {
            "risk_usd": 400.00,
            "risk_pct": 1.8,
            "position_size": 0.015,
            "reward_risk_ratio": 2.0
        },
        "metadata": {
            "timeframe": "1D",
            "market": "BTC/USDT",
            "exchange": "Binance",
            "strategy_version": "7.1.0",
            "mock_data": True
        }
    }

# Run server (for local testing)
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
