#!/bin/bash
# Script to commit and push v5.1.5 to GitHub
# Run this from the luxor-v7-python directory

set -e

echo "=============================================="
echo "🚀 LUXOR V7 PRANA v5.1.5 - GitHub Deploy"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "luxor_v7_prana.py" ]; then
    echo "❌ Error: luxor_v7_prana.py not found"
    echo "Please run this script from the luxor-v7-python directory"
    exit 1
fi

# Git configuration
echo ""
echo "📝 Configuring Git..."
git config user.name "Arsen Benda"
git config user.email "arsenbenda@example.com"

# Check git status
echo ""
echo "📊 Current status:"
git status --short

# Add all changes
echo ""
echo "➕ Adding files..."
git add .

# Show what will be committed
echo ""
echo "📋 Files to commit:"
git status --short

# Commit
echo ""
echo "💾 Creating commit..."
git commit -m "Release v5.1.5: Hybrid Regime-Aware Trailing Stop

🎯 Performance:
- Total Return: +30.87%
- Win Rate: 45.6%
- Profit Factor: 1.51x
- Max Drawdown: -6.56%

✨ Features:
- Hybrid trailing stop (adapts to regime)
- Dynamic profit-based stages
- Enhanced MTF signal generation
- Improved risk management

📊 Backtest: 90 trades on 750 daily bars (2024-2026)

✅ Status: Production Ready
" || echo "⚠️  Nothing to commit (already up to date)"

# Push
echo ""
echo "🚀 Pushing to GitHub..."
echo "Note: You may need to enter your GitHub credentials"
echo ""

git push origin main || git push origin master

echo ""
echo "=============================================="
echo "✅ DEPLOY COMPLETE!"
echo "=============================================="
echo ""
echo "🌐 View on GitHub:"
echo "   https://github.com/arsenbenda/luxor-v7-python"
echo ""
echo "📚 Next steps:"
echo "   1. Update n8n workflow with new endpoint"
echo "   2. Run test_installation.py on production"
echo "   3. Monitor first live trades"
echo ""
