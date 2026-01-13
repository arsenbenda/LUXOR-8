"""
LUXOR Trading Bot - Main Application
Version: 7.1.0
Author: arsenbenda
Date: 2026-01-13
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime
import os

# Lazy import strategy (import only when needed)
# from strategy import luxor_strategy

app = FastAPI(
    title="LUXOR Trading Bot",
    description="BTC Trading Strategy API with TA-Lib & Gann Indicators",
    version="7.1.0"
)

@app.get("/")
async def root():
    """API Info"""
    return {
        "name": "LUXOR Trading Bot",
        "version": "7.1.0",
        "status": "online",
        "endpoints": [
            "/health",
            "/signal/test",
            "/signal/daily"
        ]
    }

@app.get("/health")
async def health_check():
    """
    Healthcheck endpoint for Docker/Coolify
    Returns: status, timestamp, version, strategy loaded
    """
    try:
        # Verify strategy module can be imported
        from strategy import luxor_strategy
        strategy_loaded = True
    except Exception as e:
        strategy_loaded = False
        
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": "7.1.0",
            "strategy_loaded": strategy_loaded,
            "python_version": os.sys.version.split()[0]
        }
    )

@app.get("/signal/test")
async def get_test_signal():
    """
    Test endpoint - returns mock signal
    """
    return {
        "signal": "BUY",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mock": True,
        "data": {
            "price": 45000.0,
            "rsi": 45.0,
            "macd": 120.0
        }
    }

@app.get("/signal/daily")
async def get_daily_signal():
    """
    Production endpoint - returns real BTC signal
    Imports strategy module lazily to avoid startup failures
    """
    try:
        from strategy import luxor_strategy
        
        # Execute strategy (fetches real data from exchange)
        result = luxor_strategy()
        
        return JSONResponse(
            status_code=200,
            content={
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "symbol": "BTC/USDT",
                **result
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Strategy execution failed",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
