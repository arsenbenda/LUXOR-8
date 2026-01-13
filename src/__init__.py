"""
LUXOR v7.1 AGGRESSIVE Trading System
High-performance BTC/USDT trading strategy
"""

__version__ = "7.1.0"
__author__ = "LUXOR Trading Systems"

from .strategy import LuxorStrategy
from .backtester import run_backtest
from .indicators import calculate_indicators

__all__ = ['LuxorStrategy', 'run_backtest', 'calculate_indicators']
