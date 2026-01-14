#!/usr/bin/env python3
"""
Liberal Trading Strategy Test Script
Runs the liberal trading logic outside of the notebook to test it.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("LIBERAL TRADING STRATEGY TEST SCRIPT")
print("="*80)

try:
    # Try to load the backtest data from the notebook's global variables
    # This is a simplified version - we'll need to recreate the data

    print("\n[INFO] This script needs to be run from within the notebook environment")
    print("       where 'backtest_data' variable is available.")
    print("       Please run the liberal trading cell directly in the notebook.")

    # For now, let's create a mock test to show the logic works
    print("\n[TEST] Creating mock backtest data for testing...")

    # Mock data structure
    n_bars = 1000
    mock_data = {
        'close': np.random.uniform(50000, 51000, n_bars),
        'weighted_signal': np.random.uniform(0.3, 0.7, n_bars),  # Around neutral
        'avg_confidence': np.random.uniform(0.1, 0.8, n_bars),
        'signal_strength': np.random.uniform(0.05, 0.4, n_bars),
        'agreement_pct': np.random.uniform(0.3, 0.8, n_bars),
        'magnitude_coherent': np.random.choice([True, False], n_bars, p=[0.3, 0.7]),
        'direction_aligned': np.random.choice([True, False], n_bars, p=[0.8, 0.2]),
        'delta_h1': np.random.normal(0, 50, n_bars)
    }

    backtest_data = pd.DataFrame(mock_data)

    print(f"✅ Mock data created with {len(backtest_data)} bars")

    # ================================
    # LIBERAL ENTRY PARAMETERS
    # ================================

    REQUIRE_MAGNITUDE_COHRENCE = False
    REQUIRE_DIRECTION_ALIGNMENT = True
    MIN_AGREEMENT_PCT = 0.40
    MIN_SIGNAL_STRENGTH = 0.10
    MIN_CONFIDENCE = 0.10
    BASE_ENTRY_THRESHOLD = 0.35
    HIGH_QUALITY_THRESHOLD_REDUCTION = 0.15
    LOW_QUALITY_THRESHOLD_INCREASE = 0.10
    HIGH_QUALITY_CONFIDENCE = 0.60
    HIGH_QUALITY_STRENGTH = 0.20
    STANDARD_QUALITY_AGREEMENT = 0.50

    print("\n🎛️  LIBERAL ENTRY TOLERANCE SETTINGS:")
    print("="*50)
    print(f"Quality Filters:")
    print(f"  Require magnitude coherence: {REQUIRE_MAGNITUDE_COHRENCE}")
    print(f"  Require direction alignment: {REQUIRE_DIRECTION_ALIGNMENT}")
    print(f"Agreement Tolerance:")
    print(f"  Min agreement %: {MIN_AGREEMENT_PCT:.0%}")
    print(f"Signal Strength Tolerance:")
    print(f"  Min signal strength: {MIN_SIGNAL_STRENGTH:.2f}")
    print(f"Confidence Tolerance:")
    print(f"  Min confidence: {MIN_CONFIDENCE:.2f}")
    print(f"Entry Threshold Tolerance:")
    print(f"  Base threshold: {BASE_ENTRY_THRESHOLD:.2f}")

    # ================================
    # EXECUTE LIBERAL TRADING
    # ================================

    trades = []
    position = None
    max_hold = 30

    for bar in range(n_bars - max_hold):
        row = backtest_data.iloc[bar]
        curr_price = row['close']

        # Extract signals
        weighted_dir = row['weighted_signal']
        avg_conf = row['avg_confidence']
        strength = row['signal_strength']
        agreement = row['agreement_pct']
        mag_coherent = row['magnitude_coherent']
        dir_aligned = row['direction_aligned']
        delta_h1 = row['delta_h1']

        # Quality checks
        quality_ok = (
            (not REQUIRE_MAGNITUDE_COHRENCE or mag_coherent) and
            (not REQUIRE_DIRECTION_ALIGNMENT or dir_aligned)
        )

        # Determine signal quality
        is_high_quality = (
            avg_conf >= HIGH_QUALITY_CONFIDENCE and
            strength >= HIGH_QUALITY_STRENGTH and
            agreement >= STANDARD_QUALITY_AGREEMENT
        )

        is_standard_quality = (
            avg_conf >= MIN_CONFIDENCE and
            strength >= MIN_SIGNAL_STRENGTH and
            agreement >= MIN_AGREEMENT_PCT
        )

        # Adaptive threshold
        if is_high_quality:
            entry_threshold = BASE_ENTRY_THRESHOLD - HIGH_QUALITY_THRESHOLD_REDUCTION
        elif is_standard_quality:
            entry_threshold = BASE_ENTRY_THRESHOLD
        else:
            entry_threshold = BASE_ENTRY_THRESHOLD + LOW_QUALITY_THRESHOLD_INCREASE

        # Entry logic
        if position is None and quality_ok:
            # LONG entry
            if (weighted_dir > entry_threshold and
                strength >= MIN_SIGNAL_STRENGTH and
                avg_conf >= MIN_CONFIDENCE and
                agreement >= MIN_AGREEMENT_PCT):

                tp1 = curr_price + abs(delta_h1) * 0.5
                tp2 = curr_price + abs(delta_h1) * 1.0
                sl = curr_price - abs(delta_h1) * 0.5

                position = {
                    'type': 'LONG',
                    'entry_bar': bar,
                    'entry_price': curr_price,
                    'tp1': tp1, 'tp2': tp2, 'sl': sl,
                    'entry_quality': 'high' if is_high_quality else 'standard' if is_standard_quality else 'low'
                }

            # SHORT entry
            elif (weighted_dir < (1.0 - entry_threshold) and
                  strength >= MIN_SIGNAL_STRENGTH and
                  avg_conf >= MIN_CONFIDENCE and
                  agreement >= MIN_AGREEMENT_PCT):

                tp1 = curr_price - abs(delta_h1) * 0.5
                tp2 = curr_price - abs(delta_h1) * 1.0
                sl = curr_price + abs(delta_h1) * 0.5

                position = {
                    'type': 'SHORT',
                    'entry_bar': bar,
                    'entry_price': curr_price,
                    'tp1': tp1, 'tp2': tp2, 'sl': sl,
                    'entry_quality': 'high' if is_high_quality else 'standard' if is_standard_quality else 'low'
                }

        # Exit logic
        elif position is not None:
            pos_type = position['type']
            entry_price = position['entry_price']
            bars_held = bar - position['entry_bar']

            should_exit = False
            exit_reason = 'unknown'

            if pos_type == 'LONG':
                if curr_price >= position['tp2']:
                    should_exit, exit_reason = True, 'TP2'
                elif curr_price >= position['tp1'] and bars_held >= 3:
                    should_exit, exit_reason = True, 'TP1'
                elif curr_price <= position['sl']:
                    should_exit, exit_reason = True, 'SL'
            else:  # SHORT
                if curr_price <= position['tp2']:
                    should_exit, exit_reason = True, 'TP2'
                elif curr_price <= position['tp1'] and bars_held >= 3:
                    should_exit, exit_reason = True, 'TP1'
                elif curr_price >= position['sl']:
                    should_exit, exit_reason = True, 'SL'

            if not should_exit and bars_held >= max_hold:
                should_exit, exit_reason = True, 'TIME'

            if should_exit:
                profit = (curr_price - entry_price) if pos_type == 'LONG' else (entry_price - curr_price)
                trade = {
                    'entry_bar': position['entry_bar'],
                    'exit_bar': bar,
                    'entry_price': entry_price,
                    'exit_price': curr_price,
                    'profit': profit,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason,
                    'entry_quality': position['entry_quality']
                }
                trades.append(trade)
                position = None

    print(f"✅ LIBERAL STRATEGY EXECUTED: {len(trades)} trades generated")

    # Results analysis
    if len(trades) > 0:
        profits = [t['profit'] for t in trades]
        win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100

        print(f"\n📊 LIBERAL STRATEGY RESULTS:")
        print(f"  Total trades: {len(trades)}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total profit: ${sum(profits):+.2f}")
        print(f"  Avg profit: ${sum(profits)/len(trades):+.2f}")

        # Quality breakdown
        quality_counts = {}
        for t in trades:
            quality = t['entry_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        print(f"  Entry quality breakdown:")
        for quality, count in quality_counts.items():
            print(f"    {quality}: {count} trades")
    else:
        print("\n❌ No trades generated with current settings")
        print("   Try adjusting tolerances lower in the notebook")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Please run the liberal trading cell directly in the inference.ipynb notebook")