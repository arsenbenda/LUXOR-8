#!/usr/bin/env python3
"""
LUXOR v7.1 AGGRESSIVE - Configuration
Centralized configuration management
"""

import os
from typing import Dict, Any

# ============================================
# STRATEGY CONFIGURATION
# ============================================

STRATEGY_CONFIG = {
    # Capital & Risk
    "initial_capital": float(os.getenv("INITIAL_CAPITAL", "10000")),
    "risk_per_trade": float(os.getenv("RISK_PER_TRADE", "0.02")),  # 2%
    
    # Timeframe
    "default_timeframe": os.getenv("DEFAULT_TIMEFRAME", "1D"),
    
    # EMA Parameters
    "ema_fast_period": int(os.getenv("EMA_FAST", "21")),
    "ema_slow_period": int(os.getenv("EMA_SLOW", "50")),
    "ema_trend_period": int(os.getenv("EMA_TREND", "200")),
    
    # RSI Parameters
    "rsi_period": int(os.getenv("RSI_PERIOD", "14")),
    "rsi_oversold": float(os.getenv("RSI_OVERSOLD", "30")),
    "rsi_overbought": float(os.getenv("RSI_OVERBOUGHT", "70")),
    
    # ATR Parameters
    "atr_period": int(os.getenv("ATR_PERIOD", "14")),
    "atr_stop_multiplier": float(os.getenv("ATR_STOP_MULT", "2.0")),
    
    # Gann Parameters
    "use_adaptive_gann": os.getenv("USE_ADAPTIVE_GANN", "true").lower() == "true",
    "gann_min_confidence": float(os.getenv("GANN_MIN_CONFIDENCE", "50.0")),
    
    # Scoring Thresholds
    "min_long_score": float(os.getenv("MIN_LONG_SCORE", "60")),
    "min_short_score": float(os.getenv("MIN_SHORT_SCORE", "85")),
    
    # Market
    "market": os.getenv("MARKET", "BTC/USDT"),
    "exchange": os.getenv("EXCHANGE", "Binance"),
}

# ============================================
# API CONFIGURATION
# ============================================

API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "5000")),
    "debug": os.getenv("API_DEBUG", "false").lower() == "true",
    "cors_enabled": True,
}

# ============================================
# N8N WEBHOOK CONFIGURATION
# ============================================

N8N_CONFIG = {
    "webhook_url": os.getenv("N8N_WEBHOOK_URL", ""),
    "enabled": os.getenv("N8N_ENABLED", "false").lower() == "true",
}

# ============================================
# TELEGRAM CONFIGURATION
# ============================================

TELEGRAM_CONFIG = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "alert_chat_id": os.getenv("TELEGRAM_ALERT_CHAT_ID", ""),
    "enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
}

# ============================================
# LOGGING CONFIGURATION
# ============================================

LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.getenv("LOG_FILE", "luxor.log"),
    "console": os.getenv("LOG_CONSOLE", "true").lower() == "true",
}

# ============================================
# BACKTESTING CONFIGURATION
# ============================================

BACKTEST_CONFIG = {
    "start_date": os.getenv("BACKTEST_START", "2023-01-01"),
    "end_date": os.getenv("BACKTEST_END", "2024-12-31"),
    "commission": float(os.getenv("COMMISSION", "0.001")),  # 0.1%
    "slippage": float(os.getenv("SLIPPAGE", "0.0005")),    # 0.05%
}

# ============================================
# DATA SOURCE CONFIGURATION
# ============================================

DATA_CONFIG = {
    "source": os.getenv("DATA_SOURCE", "binance"),  # binance, csv, database
    "csv_path": os.getenv("CSV_PATH", "./data/btc_data.csv"),
    "database_url": os.getenv("DATABASE_URL", ""),
    "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_config(section: str = "strategy") -> Dict[str, Any]:
    """
    Get configuration for a specific section
    
    Args:
        section: Configuration section name
                 ("strategy", "api", "n8n", "telegram", "logging", "backtest", "data")
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = get_config("strategy")
        >>> print(config["initial_capital"])
    """
    configs = {
        "strategy": STRATEGY_CONFIG,
        "api": API_CONFIG,
        "n8n": N8N_CONFIG,
        "telegram": TELEGRAM_CONFIG,
        "logging": LOGGING_CONFIG,
        "backtest": BACKTEST_CONFIG,
        "data": DATA_CONFIG,
    }
    
    if section not in configs:
        raise ValueError(f"Invalid config section: {section}")
    
    return configs[section]

def validate_config() -> tuple:
    """
    Validate configuration
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    # Check required strategy parameters
    if STRATEGY_CONFIG["initial_capital"] <= 0:
        errors.append("initial_capital must be > 0")
    
    if not 0 < STRATEGY_CONFIG["risk_per_trade"] <= 1:
        errors.append("risk_per_trade must be between 0 and 1")
    
    # Check Telegram config if enabled
    if TELEGRAM_CONFIG["enabled"]:
        if not TELEGRAM_CONFIG["bot_token"]:
            errors.append("TELEGRAM_BOT_TOKEN required when Telegram enabled")
        if not TELEGRAM_CONFIG["chat_id"]:
            errors.append("TELEGRAM_CHAT_ID required when Telegram enabled")
    
    # Check n8n config if enabled
    if N8N_CONFIG["enabled"]:
        if not N8N_CONFIG["webhook_url"]:
            errors.append("N8N_WEBHOOK_URL required when n8n enabled")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def print_config():
    """Print all configuration values (hide sensitive data)"""
    print("=" * 70)
    print("LUXOR v7.1 AGGRESSIVE - Configuration")
    print("=" * 70)
    
    print("\n📊 STRATEGY")
    for key, value in STRATEGY_CONFIG.items():
        print(f"  {key:.<30} {value}")
    
    print("\n🔌 API")
    for key, value in API_CONFIG.items():
        print(f"  {key:.<30} {value}")
    
    print("\n📨 N8N")
    print(f"  enabled:.<30 {N8N_CONFIG['enabled']}")
    print(f"  webhook_url:.<30 {'***' if N8N_CONFIG['webhook_url'] else 'Not set'}")
    
    print("\n📱 TELEGRAM")
    print(f"  enabled:.<30 {TELEGRAM_CONFIG['enabled']}")
    print(f"  bot_token:.<30 {'***' if TELEGRAM_CONFIG['bot_token'] else 'Not set'}")
    print(f"  chat_id:.<30 {'***' if TELEGRAM_CONFIG['chat_id'] else 'Not set'}")
    
    print("\n📝 LOGGING")
    for key, value in LOGGING_CONFIG.items():
        print(f"  {key:.<30} {value}")
    
    print("\n" + "=" * 70)

# ============================================
# ENVIRONMENT FILE TEMPLATE
# ============================================

ENV_TEMPLATE = """# LUXOR v7.1 AGGRESSIVE - Environment Variables

# Strategy Configuration
INITIAL_CAPITAL=10000
RISK_PER_TRADE=0.02
DEFAULT_TIMEFRAME=1D

# Indicator Parameters
EMA_FAST=21
EMA_SLOW=50
EMA_TREND=200
RSI_PERIOD=14
RSI_OVERSOLD=30
RSI_OVERBOUGHT=70
ATR_PERIOD=14
ATR_STOP_MULT=2.0

# Gann Configuration
USE_ADAPTIVE_GANN=true
GANN_MIN_CONFIDENCE=50.0

# Scoring
MIN_LONG_SCORE=60
MIN_SHORT_SCORE=85

# Market
MARKET=BTC/USDT
EXCHANGE=Binance

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=false

# n8n Integration
N8N_ENABLED=false
N8N_WEBHOOK_URL=

# Telegram Integration
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_ALERT_CHAT_ID=

# Logging
LOG_LEVEL=INFO
LOG_FILE=luxor.log
LOG_CONSOLE=true

# Backtesting
BACKTEST_START=2023-01-01
BACKTEST_END=2024-12-31
COMMISSION=0.001
SLIPPAGE=0.0005

# Data Source
DATA_SOURCE=binance
CSV_PATH=./data/btc_data.csv
DATABASE_URL=
CACHE_ENABLED=true
"""

def generate_env_file(filename: str = ".env"):
    """Generate .env template file"""
    with open(filename, 'w') as f:
        f.write(ENV_TEMPLATE)
    print(f"✅ Generated {filename}")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print_config()
    
    # Validate
    is_valid, errors = validate_config()
    
    print("\n📋 VALIDATION")
    if is_valid:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Optionally generate .env template
    # generate_env_file()
