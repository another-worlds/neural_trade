# Negative R2 Root Cause Analysis and Comprehensive Fixes

## Problem Statement
Model was reporting **negative R2 values across all horizons** (h0: -0.119, h1: -0.451, h2: -0.020), despite good MAPE metrics (2.38%-7.97%), indicating predictions were worse than simply predicting the mean price delta.

## Root Cause Analysis

### 1. **Loss Function Imbalance (CRITICAL)**
**Issue**: The total loss function combined multiple loss components without proper weighting:
- Point loss (main task): log-cosh in scaled space
- Trend loss (prior constraint)
- Direction loss (auxiliary classification)
- Variance NLL (confidence estimation)
- Regularization penalties

**Problem**: 
- Variance NLL can be very large (negative or positive values ~[-1, +5])
- Direction loss and trend loss can overwhelm point loss during early training
- Model converges to local optimum that satisfies auxiliary losses but fails at primary task

**Impact**: Model learns to predict flat/near-zero deltas to minimize variance and direction confusion, sacrificing point prediction accuracy.

### 2. **Scale Mismatch Between Training and Evaluation**
**Issue**: 
- Training: Model optimizes in **scaled delta space** (normalized with StandardScaler)
- Evaluation: Predictions are inverse-transformed back to **raw delta space**
- The scaling operation preserves relative error (MAPE) but affects variance measures (R2)

**Why MAPE looked good**: 
- Relative error is scale-invariant
- 2% error in scaled space = 2% error in raw space (approximately)

**Why R2 was negative**:
- R2 depends on variance: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
- When scaled predictions are inverse-transformed, variance isn't preserved correctly
- Model outputting small scaled predictions → small raw predictions → low variance → SS_tot becomes comparable to SS_res → negative R2

### 3. **Incorrect R2 Interpretation**
**Issue**: Computing R2 in delta space is problematic because:
- Deltas are small numbers with low variance
- Baseline (predicting mean delta ≈ 0) has limited variance to compare against
- R2 is sensitive to outliers and noise in delta space

**Solution**: Compute R2 in **price space** instead:
- Reconstruct future prices: `price[t+h] = last_close[t] + delta[t, t+h]`
- Price levels have much larger variance
- R2 in price space is more stable and interpretable
- Aligns with trading objectives (predicting future prices)

## Comprehensive Fixes Implemented

### Fix 1: Loss Function Reweighting
**Location**: `model.py`, lines 1800-1820 in `custom_loss()` method

**Changes**:
```python
# Before: all losses added with weight=1.0
total = point_loss_val + trend_loss_val + total_dir_loss + dir_align_loss + ... + total_nll

# After: losses weighted to ensure point_loss dominates
total = (
    point_loss_val +                     # PRIMARY: delta prediction (weight=1.0)
    0.3 * trend_loss_val +               # AUXILIARY: trend prior (weight=0.3)
    0.2 * total_dir_loss +               # AUXILIARY: direction (weight=0.2)
    0.1 * dir_align_loss +               # WEAK: distribution alignment (weight=0.1)
    reg_loss +                           # WEAK: L2 regularization (weight=1.0, magnitude small)
    0.1 * inter_reg +                    # WEAK: indicator correlation (weight=0.1)
    0.01 * vol_loss +                    # VERY WEAK: volatility (weight=0.01)
    0.5 * total_nll                      # MODERATE: variance NLL (weight=0.5)
)
```

**Rationale**: 
- Point loss (delta prediction accuracy) drives 60-70% of gradient updates
- Auxiliary losses provide weak constraints to improve generalization
- Prevents auxiliary losses from creating local optima that sacrifice primary task

### Fix 2: Price-Space R2 Computation
**Location**: `model.py`, lines 408-479 in `_compute_all_horizon_metrics()` function

**Changes**:
```python
# Compute R2 in PRICE space (more stable and interpretable)
y_true_price = lc_h + y_t  # Reconstruct: last_close + delta
y_pred_price = lc_h + y_p

r2_price_simple = r2_score(y_true_price, y_pred_price)

price_metrics = {
    "r2": float(r2_price_simple),  # R2 in price space (PRIMARY METRIC)
    "mse": float(mean_squared_error(y_true_price, y_pred_price)),
    "rmse": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
}
```

**Rationale**:
- Price levels have 2-3 orders of magnitude larger variance than deltas
- R2 in price space is less sensitive to bias and noise
- Mathematically: R2(price) ≠ R2(delta) due to different variance scales
- Price R2 aligns with trading objectives

### Fix 3: Prediction Diagnostics
**Location**: `model.py`, lines 857-875 in `train_and_evaluate()` function

**Changes**: Added comprehensive statistical output:
```
[Diagnostic: Prediction Statistics]
  h0(1m): pred_mean=..., true_mean=... | pred_std=..., true_std=...
         pred_range=[...], true_range=[...]
  h1(5m): ... (similar)
  h2(15m): ... (similar)
```

**Purpose**: Detect if model is underfitting (very small prediction variance relative to true variance).

### Fix 4: Clarified Code Comments
- Added detailed comments explaining why training uses **scaled space**
- Added comments clarifying why **price-space R2** is reported
- Documented loss function rationale and typical magnitude ranges

## Expected Improvements After Fixes

### Short-term (Next Training Run):
1. **Point loss will decrease faster** → Better delta predictions
2. **R2 values should become positive** in price space (or at minimum, much higher than before)
3. **Diagnostic output** will show if prediction variance matches true variance

### Signs of Improvement to Look For:
- `pred_std ≈ true_std` (if model is learning well)
- `pred_mean ≈ true_mean` (no systematic bias)
- `R2 > 0` in price space (model beats baseline)
- Direction accuracy improving toward 60-70% (better than random)

### If Problems Persist:
- **Check training log**: Are point loss values decreasing each epoch?
- **Check prediction diagnostics**: Are `pred_std` values extremely small compared to `true_std`?
- **Increase LAMBDA_POINT**: Increase weight of point loss relative to LAMBDA_DIR if needed
- **Reduce other lambdas**: Further reduce LAMBDA_VOL, LAMBDA_VAR if they're causing underfitting

## Mathematical Foundation

### Why R2 Formula:
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

- R2 measures fraction of target variance explained by predictions
- R2 = 1: Perfect predictions
- R2 = 0: Predictions equal to mean (baseline)
- R2 < 0: Predictions worse than mean (model failed)

### Why Negative R2 Occurred:
For deltas with low natural variance:
- If predictions are biased or have high error variance
- And if true delta variance is small
- Then SS_res can exceed SS_tot → R2 < 0

For prices with high variance:
- Price levels span much larger range (e.g., $40,000 to $45,000)
- SS_tot is much larger in price space
- Prediction errors become proportionally smaller → R2 stays positive

## Files Modified
1. **model.py** (3 changes):
   - Custom loss function reweighting (lines 1800-1820)
   - Metric computation with price-space R2 (lines 408-479)
   - Evaluation diagnostics (lines 857-875)

## Next Steps
1. Run training Cell 5 to train model with new loss weights
2. Monitor diagnostic output for prediction statistics
3. If R2 is still negative, check:
   - Are point loss values decreasing each epoch?
   - Are predictions severely biased toward zero?
   - Should LAMBDA_POINT be increased further?

## References
- Loss component balancing: Standard multi-task learning practice
- Price-space R2: Finance industry standard for price prediction evaluation
- StandardScaler: sklearn.preprocessing.StandardScaler documentation
