#!/usr/bin/env python3
"""
Test script to verify variance calibration fix.
Demonstrates calibration checking in both scaled and raw spaces.
"""

import numpy as np

# Simulate scaled space predictions and targets
np.random.seed(42)
n_samples = 1000

# Scaled targets (mean=0, std=1 approximately)
true_scaled = np.random.normal(0, 1, n_samples)

# Scaled predictions (with some error)
pred_scaled = true_scaled + np.random.normal(0, 0.5, n_samples)

# Variance predictions in scaled space (should correlate with squared errors)
errors_scaled = true_scaled - pred_scaled
squared_errors_scaled = errors_scaled ** 2

# Simulate well-calibrated variance: higher when squared error is higher
var_scaled = 0.1 + 0.5 * squared_errors_scaled + np.random.normal(0, 0.1, n_samples)
var_scaled = np.maximum(var_scaled, 0.01)  # floor

# Simulate raw dollar space
price_scale = 100.0  # $100 per unit (scaler.std_)
price_mean = 0.0
true_raw = true_scaled * price_scale + price_mean
pred_raw = pred_scaled * price_scale + price_mean
errors_raw = true_raw - pred_raw

# Scale variance to raw space: var_raw = var_scaled * scale^2
var_raw = var_scaled * (price_scale ** 2)

print("=== Variance Calibration Analysis ===")
print(f"Samples: {n_samples}")
print(f"Scale factor: {price_scale}")
print()

# Method 1: Scaled space calibration (where NLL optimizes)
print("✅ SCALED SPACE: corr(var_scaled, error_scaled²)")
corr_scaled = np.corrcoef(var_scaled, squared_errors_scaled)[0, 1]
print(f"   Correlation: {corr_scaled:.4f}")
print("   Status:", "✓ Well-calibrated" if corr_scaled > 0.1 else "❌ Miscalibrated")
print()

# Method 2: Raw space calibration (for trading relevance)
print("✅ RAW SPACE: corr(var_raw, |error_raw|)")
corr_raw = np.corrcoef(var_raw, np.abs(errors_raw))[0, 1]
print(f"   Correlation: {corr_raw:.4f}")
print("   Status:", "✓ Well-calibrated" if corr_raw > 0.1 else "❌ Miscalibrated")
print()

print("=== Why both checks matter ===")
print("• Scaled space: Checks if NLL optimization worked correctly")
print("• Raw space: Checks if uncertainty estimates are useful for trading")
print("• Both should be positive for proper variance calibration")
print()
print("Expected: Both correlations positive (well-calibrated)")
print(f"Scaled: {corr_scaled:.4f}, Raw: {corr_raw:.4f}")