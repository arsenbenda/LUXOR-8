"""
Utility Functions
Helper functions for data processing and analysis
"""

import pandas as pd
import numpy as np
import json

def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(trades_df, stats, output_dir='results/'):
    """Save backtest results"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trades
    trades_df.to_csv(f'{output_dir}trades.csv', index=False)
    
    # Save stats
    with open(f'{output_dir}performance_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Results saved to {output_dir}")

def print_performance_summary(stats):
    """Print formatted performance summary"""
    print("\n" + "="*60)
    print("LUXOR v7.1 AGGRESSIVE - PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Winners: {stats['winners']} | Losers: {stats['losers']}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Total P&L: {stats['total_pnl']:+.2f}%")
    print(f"Avg Win: {stats['avg_win']:+.2f}%")
    print(f"Avg Loss: {stats['avg_loss']:+.2f}%")
    print("="*60 + "\n")

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return drawdown.min() * 100
