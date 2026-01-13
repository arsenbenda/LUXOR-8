#!/usr/bin/env python3
"""
LUXOR v7.1 AGGRESSIVE - Backtest Runner
Command-line interface for running backtests
"""

import argparse
from src.backtester import run_backtest
from src.utils import load_config, save_results, print_performance_summary

def main():
    parser = argparse.ArgumentParser(description='Run LUXOR v7.1 AGGRESSIVE backtest')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override dates if provided
    if args.start_date:
        config['backtest']['start_date'] = args.start_date
    if args.end_date:
        config['backtest']['end_date'] = args.end_date
    
    print("🚀 Starting LUXOR v7.1 AGGRESSIVE backtest...")
    print(f"Period: {config['backtest']['start_date']} to {config['backtest']['end_date']}")
    
    # Run backtest
    trades_df, data_df, stats = run_backtest(config)
    
    # Print results
    print_performance_summary(stats)
    
    # Save results
    save_results(trades_df, stats, args.output_dir)
    
    print("✅ Backtest completed successfully!")

if __name__ == '__main__':
    main()
