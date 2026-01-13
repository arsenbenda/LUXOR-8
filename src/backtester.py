"""
Backtesting Engine
Executes strategy over historical data with position management
"""

import pandas as pd
import numpy as np
from .strategy import LuxorStrategy
from .indicators import calculate_indicators

class Trade:
    """Represents a single trade"""
    def __init__(self, entry_date, direction, entry_price, size, score, atr, sl_price):
        self.entry_date = entry_date
        self.direction = direction
        self.entry_price = entry_price
        self.size = size
        self.score = score
        self.atr = atr
        self.sl_price = sl_price
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.remaining_size = size
        self.partial_exits = []
    
    def close(self, exit_date, exit_price, reason):
        """Close the trade"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        
        if self.direction == 'LONG':
            self.pnl_percent = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            self.pnl_percent = ((self.entry_price - exit_price) / self.entry_price) * 100
        
        self.pnl = self.pnl_percent * self.remaining_size
    
    def partial_exit(self, exit_date, exit_price, percent, reason):
        """Execute partial exit"""
        exit_size = self.remaining_size * percent
        
        if self.direction == 'LONG':
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100
        
        self.partial_exits.append({
            'date': exit_date,
            'price': exit_price,
            'size': exit_size,
            'pnl_percent': pnl_pct,
            'reason': reason
        })
        
        self.remaining_size -= exit_size
        self.pnl += pnl_pct * exit_size

def run_backtest(config):
    """
    Execute backtest
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (trades_df, equity_curve, performance_stats)
    """
    # Load data
    data_file = config['backtest']['data_file']
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Date filter
    start_date = pd.to_datetime(config['backtest']['start_date'])
    end_date = pd.to_datetime(config['backtest']['end_date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop=True)
    
    # Calculate indicators
    df = calculate_indicators(df, config)
    
    # Generate signals
    strategy = LuxorStrategy(config)
    df = strategy.generate_signals(df)
    
    # Execute trades
    trades = []
    active_trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_date = row['date']
        current_price = row['close']
        current_atr = row['atr']
        
        # Manage active trades
        for trade in active_trades[:]:
            # Calculate profit R
            if trade.direction == 'LONG':
                profit_r = (current_price - trade.entry_price) / (trade.entry_price - trade.sl_price)
            else:
                profit_r = (trade.entry_price - current_price) / (trade.sl_price - trade.entry_price)
            
            # Stop loss check
            if trade.direction == 'LONG' and current_price <= trade.sl_price:
                trade.close(current_date, trade.sl_price, 'SL')
                trades.append(trade)
                active_trades.remove(trade)
                continue
            elif trade.direction == 'SHORT' and current_price >= trade.sl_price:
                trade.close(current_date, trade.sl_price, 'SL')
                trades.append(trade)
                active_trades.remove(trade)
                continue
            
            # Partial exits
            if profit_r >= 0.5 and len(trade.partial_exits) == 0:
                trade.partial_exit(current_date, current_price, 
                                 strategy.risk['partial_exit_1']['percent'], 
                                 'TP1 +0.5R')
            
            if profit_r >= 1.0 and len(trade.partial_exits) == 1:
                trade.partial_exit(current_date, current_price,
                                 strategy.risk['partial_exit_2']['percent'],
                                 'TP2 +1.0R')
            
            # Update trailing stop
            new_sl = strategy.calculate_trailing_stop(
                trade.entry_price, current_price, trade.direction, 
                current_atr, profit_r
            )
            if new_sl is not None:
                if trade.direction == 'LONG':
                    trade.sl_price = max(trade.sl_price, new_sl)
                else:
                    trade.sl_price = min(trade.sl_price, new_sl)
        
        # New signals
        if row['signal'] in ['LONG', 'SHORT']:
            size = strategy.calculate_position_size(row['score'])
            sl_price = strategy.calculate_stop_loss(
                current_price, row['signal'], current_atr
            )
            
            new_trade = Trade(
                entry_date=current_date,
                direction=row['signal'],
                entry_price=current_price,
                size=size,
                score=row['score'],
                atr=current_atr,
                sl_price=sl_price
            )
            active_trades.append(new_trade)
    
    # Close remaining trades at end
    final_price = df.iloc[-1]['close']
    final_date = df.iloc[-1]['date']
    for trade in active_trades:
        trade.close(final_date, final_price, 'END')
        trades.append(trade)
    
    # Convert to DataFrame
    if not trades:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    trades_data = []
    for t in trades:
        trades_data.append({
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'score': t.score,
            'size': t.size,
            'pnl_percent': t.pnl_percent,
            'pnl': t.pnl,
            'exit_reason': t.exit_reason,
            'partial_exits': len(t.partial_exits)
        })
    
    trades_df = pd.DataFrame(trades_data)
    
    # Performance stats
    stats = {
        'total_trades': len(trades_df),
        'winners': len(trades_df[trades_df['pnl_percent'] > 0]),
        'losers': len(trades_df[trades_df['pnl_percent'] < 0]),
        'win_rate': len(trades_df[trades_df['pnl_percent'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'total_pnl': trades_df['pnl_percent'].sum(),
        'avg_win': trades_df[trades_df['pnl_percent'] > 0]['pnl_percent'].mean() if len(trades_df[trades_df['pnl_percent'] > 0]) > 0 else 0,
        'avg_loss': trades_df[trades_df['pnl_percent'] < 0]['pnl_percent'].mean() if len(trades_df[trades_df['pnl_percent'] < 0]) > 0 else 0
    }
    
    return trades_df, df, stats
