"""
LUXOR Trading Bot v7.2.4
FastAPI backend with Kraken integration
Full strategy analysis with multi-timeframe consensus
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import ccxt
import pandas as pd
import logging
from typing import Dict, Optional
from strategy import TradingStrategy

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="LUXOR Trading Bot",
    description="Advanced crypto trading signals with Gann analysis",
    version="7.2.4"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# KRAKEN EXCHANGE INITIALIZATION
# ========================================
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
})

# ========================================
# MODELS
# ========================================
class HealthCheck(BaseModel):
    status: str
    version: str
    timestamp: str
    exchange: str

# ========================================
# HELPER FUNCTIONS
# ========================================
def fetch_ohlcv_multi_timeframe(symbol: str = "BTC/USD") -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple timeframes from Kraken
    """
    timeframes = {
        '1d': '1d',
        '3d': '3d',
        '1w': '1w'
    }
    
    data = {}
    
    for tf_key, tf_value in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf_value, limit=200)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data[tf_key] = df
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {tf_value}")
            
        except Exception as e:
            logger.error(f"Error fetching {tf_value}: {str(e)}")
            data[tf_key] = pd.DataFrame()
    
    return data

# ========================================
# ENDPOINTS
# ========================================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "LUXOR Trading Bot",
        "version": "7.2.4",
        "description": "Advanced crypto trading signals with Kraken",
        "endpoints": {
            "health": "/health",
            "signal_test": "/signal/test",
            "signal_daily": "/signal/daily"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="7.2.4",
        timestamp=datetime.utcnow().isoformat() + 'Z',
        exchange="Kraken"
    )

@app.get("/signal/test")
async def signal_test():
    """Test endpoint with mock data"""
    try:
        # Mock OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        mock_df = pd.DataFrame({
            'timestamp': dates,
            'open': [45000 + i*10 for i in range(200)],
            'high': [45100 + i*10 for i in range(200)],
            'low': [44900 + i*10 for i in range(200)],
            'close': [45000 + i*10 for i in range(200)],
            'volume': [1000 + i for i in range(200)]
        })
        
        ohlcv_data = {
            '1d': mock_df,
            '3d': mock_df.iloc[::3],
            '1w': mock_df.iloc[::7]
        }
        
        strategy = TradingStrategy(symbol="BTC/USD")
        result = strategy.analyze(ohlcv_data)
        result['mock'] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Test signal error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signal/daily")
async def signal_daily(symbol: str = "BTC/USD"):
    """
    Daily trading signal with full analysis
    """
    try:
        logger.info(f"Fetching signal for {symbol}")
        
        # Fetch real data from Kraken
        ohlcv_data = fetch_ohlcv_multi_timeframe(symbol)
        
        # Verify 1D data exists
        if '1d' not in ohlcv_data or len(ohlcv_data['1d']) < 50:
            raise HTTPException(
                status_code=500,
                detail="Insufficient data from Kraken"
            )
        
        # Run strategy
        strategy = TradingStrategy(symbol=symbol)
        result = strategy.analyze(ohlcv_data)
        
        return result
        
    except ccxt.NetworkError as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Network error",
                "message": f"Failed to connect to exchange: {str(e)}",
                "timestamp": datetime.utcnow().isoformat() + 'Z'
            }
        )
    except Exception as e:
        logger.error(f"Signal generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Strategy execution failed",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat() + 'Z'
            }
        )

# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
