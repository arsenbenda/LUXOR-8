#!/usr/bin/env python3
"""
Quick test script to verify Luxor V7 Prana installation
"""

import sys
import os

def test_installation():
    """Test if all required files and dependencies are present."""
    
    print("=" * 80)
    print("🧪 LUXOR V7 PRANA - Installation Test")
    print("=" * 80)
    
    # Test 1: Check Python version
    print("\n1️⃣ Checking Python version...")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Test 2: Check dependencies
    print("\n2️⃣ Checking dependencies...")
    required_modules = ['pandas', 'numpy', 'ccxt', 'ta']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} (missing)")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Install missing: pip install {' '.join(missing)}")
        return False
    
    # Test 3: Check core files
    print("\n3️⃣ Checking core files...")
    required_files = [
        'luxor_v7_prana.py',
        'backtest_v515.py',
        'data/btcusdt_daily_1000.csv',
        'requirements.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            return False
    
    # Test 4: Import core system
    print("\n4️⃣ Testing core system import...")
    try:
        from luxor_v7_prana import LuxorV7PranaSystem
        system = LuxorV7PranaSystem()
        print(f"✅ LuxorV7PranaSystem initialized (v{system.version})")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Test 5: Load test data
    print("\n5️⃣ Testing data loading...")
    try:
        import pandas as pd
        df = pd.read_csv('data/btcusdt_daily_1000.csv')
        print(f"✅ Loaded {len(df)} bars")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Test 6: Quick signal generation test
    print("\n6️⃣ Testing signal generation...")
    try:
        # Prepare data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Generate signal using last 100 bars
        test_df = df.tail(100)
        signals = system.generate_mtf_signal(
            symbol="BTCUSDT",
            df_historical=test_df
        )
        
        if signals and '1D' in signals:
            print(f"✅ Signal generated: {signals['1D'].get('primary_bias', 'N/A')}")
        else:
            print("⚠️  No signal generated (this is OK for testing)")
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False
    
    # Success
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\n🚀 You can now run:")
    print("   python backtest_v515.py")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
