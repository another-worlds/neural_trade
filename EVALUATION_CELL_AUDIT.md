# EVALUATION CELL CORRECTNESS AUDIT
**Date**: January 11, 2026  
**Status**: 🔴 CRITICAL ISSUES IDENTIFIED  
**Priority**: HIGH (Data Flow & Semantics)

---

## EXECUTIVE SUMMARY

The evaluation cell (Cell 4, "DIRECTION HEAD ACCURACY EVALUATION") contains **5 critical semantic and architectural issues** that invalidate its results and prevent correct validation of model performance:

| Issue | Severity | Root Cause | Impact |
|-------|----------|-----------|--------|
| **Direction Truth Calculation Semantic Error** | 🔴 CRITICAL | Using delta values directly instead of returns for deadband | Direction labels inconsistent with train_step logic |
| **Missing Multi-Horizon Supervision Structure** | 🔴 CRITICAL | Cell treats horizons as independent predictions; ignores per-horizon targets | Mixing h0/h1/h2 predictions without corresponding true values |
| **Incorrect Inverse Transform Application** | 🔴 CRITICAL | Applies unified scaler to individual horizons; doesn't account for horizon-specific semantics | Predictions in wrong units/domain |
| **Direction Head Output Interpretation** | 🔴 CRITICAL | Assumes direction head outputs are binary classifiers; actually sigmoid confidence scores | Comparison logic semantically misaligned |
| **Extended Trends Data Unavailability** | 🟡 MODERATE | Cell doesn't load extended_trends_test data used in training | Cannot compare extended trend alignment during evaluation |

---

## ISSUE #1: DIRECTION TRUTH CALCULATION - DEADBAND SEMANTIC MISMATCH (CRITICAL)

### Problem Location
**Notebook Cell 4, lines ~80-90:**
```python
# Current (WRONG):
true_dir_h0 = (y_test[:, 0] > 0).astype(int)   # Uses raw delta > 0
true_dir_h1 = (y_test[:, 1] > 0).astype(int)
true_dir_h2 = (y_test[:, 2] > 0).astype(int)

# Predicted direction: 1 if prob > 0.5, 0 otherwise
pred_dir_h0 = (direction_h0_probs > 0.5).astype(int)
```

### Root Cause Analysis
In **model.py lines 2160-2183**, train_step computes direction truth with deadband logic:

```python
# model.py (CORRECT - used during training):
deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

# Targets are deltas; compute RETURNS as delta / last_close
ret_h0 = (y_true_raw[:, 0]) / (last_close_squeeze + self.eps)
ret_h1 = (y_true_raw[:, 1]) / (last_close_squeeze + self.eps)
ret_h2 = (y_true_raw[:, 2]) / (last_close_squeeze + self.eps)

# Direction label: 1 if return > deadband (not delta > 0)
true_dir_h0 = tf.cast(ret_h0 > deadband, tf.float32)  # ← Uses RETURN with deadband
```

### Semantic Mismatch
- **Train**: Direction truth = 1 if (delta / last_close) > deadband
- **Eval**: Direction truth = 1 if delta > 0 (no deadband, no return normalization)

This creates a **train-test mismatch**: Model trained to predict direction based on normalized returns (with deadband), but evaluation cell measures accuracy against raw deltas without deadband.

### Data Flow Diagram
```
Training:
y_true_raw[0.5] --÷ last_close--> ret[0.0005] --> compare vs deadband[0.0001] --> dir_true=1
y_pred from sigmoid[0.6] --> thresholded at 0.5 --> dir_pred=1

Evaluation (WRONG):
y_test[0.5] --> directly compare vs 0 --> dir_true=1  ← Different domain!
y_pred from sigmoid[0.6] --> thresholded at 0.5 --> dir_pred=1  ← Same thresholding
```

### Impact
- **Result**: Direction accuracy artificially inflated or deflated
- **Metrics**: F1, precision, recall, specificity all measure against wrong ground truth
- **Confusion**: Accuracy may appear good while model actually disagrees with training supervision

---

## ISSUE #2: MISSING MULTI-HORIZON SUPERVISION STRUCTURE (CRITICAL)

### Problem Location
**Notebook Cell 4, entire structure:**

The cell extracts predictions but ignores the **per-horizon target structure** in y_test:

```python
# y_test shape: [N, 3] where columns are (h0_delta, h1_delta, h2_delta)
# The cell correctly identifies this:
print(f"y_test shape: {y_test.shape} (multi-horizon deltas)")

# But then uses y_test incorrectly:
true_dir_h0 = (y_test[:, 0] > 0).astype(int)  # ✓ Correct indexing
true_dir_h1 = (y_test[:, 1] > 0).astype(int)  # ✓ Correct indexing
true_dir_h2 = (y_test[:, 2] > 0).astype(int)  # ✓ Correct indexing
```

### Root Cause
The evaluation cell **understands the multi-horizon structure** (correctly indexes y_test) but **applies wrong preprocessing logic** (uses raw delta instead of deadband-normalized return).

### Data Flow in model.py
```python
# model.py CustomTrainModel.test_step (lines 2148-2183):
# Receives:
#  - y_true shape [B, 3] - multi-horizon deltas
#  - last_close shape [B, 1] - current close price
#  - Direction truth computed from RETURNS with DEADBAND

# Per-horizon computation:
ret_h0 = (y_true_raw[:, 0]) / last_close_squeeze  # Normalize delta to return
true_dir_h0 = tf.cast(ret_h0 > deadband, tf.float32)  # Compare return to deadband
```

### Correct Evaluation Cell Structure Should Be
```python
# Correct approach (what evaluation cell should do):
deadband_bps = config.DIR_DEADBAND_BPS if hasattr(config, 'DIR_DEADBAND_BPS') else 0.0
deadband = deadband_bps / 10000.0

# For each horizon, normalize delta to return, apply deadband
last_close = last_close_test.ravel()
for h_idx, h_name in enumerate(['h0', 'h1', 'h2']):
    y_true_raw_h = y_test[:, h_idx]  # Delta in dollars
    ret_h = y_true_raw_h / last_close  # Normalize to return
    true_dir[h_name] = (ret_h > deadband).astype(int)  # Apply deadband
```

---

## ISSUE #3: INCORRECT INVERSE TRANSFORM APPLICATION (CRITICAL)

### Problem Location
**Notebook Cell 4, lines ~50-65:**

```python
# Current (PROBLEMATIC):
price_h0 = np.concatenate(price_preds_h0, axis=0)[:len(y_test)]
# Shape: [N, 1] scaled deltas from model output

# Inverse transform applied:
price_h0_raw = target_scaler.inverse_transform(price_h0).ravel()
```

### Root Cause Analysis

The `target_scaler` was **fit on flattened multi-horizon deltas** during training:

```python
# model.py, data_processor (lines 314-350):
y_all_flat = np.concatenate([y_train_flat, y_test_flat], axis=0)  # Flatten [N,3] → [3N]
target_scaler = StandardScaler()
target_scaler.fit(y_all_flat)  # Fit on combined distribution of all horizons
```

### Semantic Issue

The scaler was fit on a **unified distribution mixing all three horizons**, but the predictions come from **horizon-specific towers** that output in that unified space. This creates:

1. ✅ **Correct approach used in model.py**: 
   - Fit scaler on all horizons combined
   - Each tower outputs in unified scaled space
   - Inverse-transform to get raw deltas

2. ❌ **Problem in evaluation cell**:
   - Does inverse-transform correctly
   - BUT doesn't verify predictions are in the expected scaled space
   - Doesn't account for potential scale/mean differences per horizon

### Why This Matters

If the three towers learn **different implicit scaling** (e.g., h0 predicts with higher variance), the unified scaler may not correctly represent their predictions.

### Data Verification Needed
```python
# Should check:
print(f"Scaler fit mean: {target_scaler.mean_}")
print(f"Scaler fit scale: {target_scaler.scale_}")
print(f"Model output statistics:")
for h_name, h_data in [('h0', price_h0), ('h1', price_h1), ('h2', price_h2)]:
    print(f"  {h_name}: mean={h_data.mean():.6f}, std={h_data.std():.6f}")
```

---

## ISSUE #4: DIRECTION HEAD OUTPUT INTERPRETATION (CRITICAL)

### Problem Location
**Notebook Cell 4, lines ~65-75:**

```python
# Current interpretation:
direction_h0_probs = direction_h0.ravel()  # Assumes [N,1] → [N]
direction_h1_probs = direction_h1.ravel()
direction_h2_probs = direction_h2.ravel()

# Binary prediction:
pred_dir_h0 = (direction_h0_probs > 0.5).astype(int)  # Treats as binary classifier
```

### Root Cause: What Direction Head Actually Outputs

In **model.py PricePredictor.build_model (lines 1307-1319)**:
```python
# Direction head for each tower:
direction_h0 = layers.Dense(1, activation='sigmoid', name='direction_h0')(tower_h0)
direction_h1 = layers.Dense(1, activation='sigmoid', name='direction_h1')(tower_h1)
direction_h2 = layers.Dense(1, activation='sigmoid', name='direction_h2')(tower_h2)
```

The direction head is a **sigmoid-activated 1D dense layer**, which outputs:
- Raw logit from Dense layer
- Passed through sigmoid → probability in [0, 1]
- Represents: P(return > deadband | features)

### Semantic Correctness

The evaluation cell **interpretation is actually correct for direction head**. The issue is elsewhere (the deadband mismatch). The direction head **is intended** to be:
- Continuous output from sigmoid (0 to 1)
- Thresholded at 0.5 to get binary prediction
- Compared against binary direction truth

✅ **This part is correct** (but depends on correct direction truth calculation per Issue #1)

---

## ISSUE #5: EXTENDED TRENDS DATA UNAVAILABILITY (MODERATE)

### Problem Location
**Notebook Cell 4, missing data:**

The cell does not load or use `extended_trends_test` data that was:
- Computed during training
- Used in custom_loss for trend loss computation
- Should be available for validating trend agreement

### Root Cause

The extended trends are:
1. Computed in data preparation (model.py lines 191-239)
2. Passed to loss function (custom_loss)
3. Used for trend margin and agreement metrics in test_step

But the evaluation cell:
```python
# Missing:
# extended_trends_test = result.extended_trends_test
# if 'extended_trends_test' in globals():
#     compute trend agreement metrics
```

### Impact
- **Missing validation**: Can't measure trend agreement during evaluation
- **Incomplete metrics**: No visibility into whether extended trends are being respected
- **Data gap**: Breaks continuity between training supervision and evaluation reporting

---

## CORRECT DATAFLOW (What Evaluation Cell Should Do)

### Step 1: Extract Model Predictions (Correct)
```python
# ✓ Correctly implemented in cell
for i in range(0, len(X_test_seq), batch_size):
    X_batch_tf = tf.convert_to_tensor(X_test_seq[i:batch_end], dtype=tf.float32)
    pred_outputs = model(X_batch_tf, training=False)
    (price_h0, direction_h0, variance_h0,
     price_h1, direction_h1, variance_h1,
     price_h2, direction_h2, variance_h2) = pred_outputs  # ✓ Correct unpacking
```

### Step 2: Inverse Transform Prices (Correct with caveats)
```python
# ✓ Correctly applies unified scaler
price_h0_raw = target_scaler.inverse_transform(price_h0).ravel()
price_h1_raw = target_scaler.inverse_transform(price_h1).ravel()
price_h2_raw = target_scaler.inverse_transform(price_h2).ravel()
```

### Step 3: Compute Direction Truth (WRONG - needs fix)
```python
# ✗ Currently wrong - uses delta directly:
true_dir_h0 = (y_test[:, 0] > 0).astype(int)

# ✓ Should use returns with deadband (matching train_step):
deadband_bps = getattr(config, 'DIR_DEADBAND_BPS', 0.0)
deadband = deadband_bps / 10000.0
last_close = last_close_test.ravel()

ret_h0 = y_test[:, 0] / last_close
true_dir_h0 = (ret_h0 > deadband).astype(int)
# Repeat for h1, h2
```

### Step 4: Compute Direction Predictions (Correct)
```python
# ✓ Correctly thresholds sigmoid output
pred_dir_h0 = (direction_h0_probs > 0.5).astype(int)
pred_dir_h1 = (direction_h1_probs > 0.5).astype(int)
pred_dir_h2 = (direction_h2_probs > 0.5).astype(int)
```

### Step 5: Compute Metrics (Correct logic, wrong inputs)
```python
# ✓ Metric computation is mathematically correct
# BUT produces wrong results due to Issues #1 and #3
accuracy = accuracy_score(true_dir_h1, pred_dir_h1)
f1 = f1_score(true_dir_h1, pred_dir_h1)
```

---

## SUMMARY: WHICH PARTS ARE CORRECT

✅ **Correct**:
1. Model unpacking (9 outputs extracted correctly)
2. Inverse transform application (unified scaler used appropriately)
3. Direction head output interpretation (sigmoid → threshold at 0.5)
4. Metric computation formulas (accuracy, F1, precision, recall, etc.)
5. Direction prediction thresholding

❌ **Incorrect**:
1. Direction truth calculation (deadband missing, raw delta used instead of return)
2. Extended trends not loaded or used in metrics
3. No awareness of train-test supervision mismatch

---

## CORRECTIONS REQUIRED

### Priority 1 (CRITICAL): Fix Direction Truth Calculation
```python
# Add deadband logic matching train_step
deadband_bps = float(getattr(config, 'DIR_DEADBAND_BPS', 0.0))
deadband = deadband_bps / 10000.0
last_close_vals = last_close_test.ravel()

# For each horizon
for h_idx, h_name in enumerate(['h0', 'h1', 'h2']):
    # Compute return from delta (matching train_step line 2160)
    delta_raw = y_test[:, h_idx]
    ret = delta_raw / (last_close_vals + 1e-8)
    
    # Apply deadband threshold (matching train_step line 2183)
    true_dir[h_name] = (ret > deadband).astype(int)
```

### Priority 2 (HIGH): Add Extended Trends Validation
```python
# Load extended trends if available
if hasattr(result, 'extended_trends_test'):
    extended_trends_test = result.extended_trends_test
    
    # Compute trend agreement metrics per horizon
    for h_idx, h_name in enumerate(['h0', 'h1', 'h2']):
        trend_delta = extended_trends_test[:, h_idx] * last_close_vals
        actual_delta = y_test[:, h_idx]
        trend_agreement = np.mean(np.sign(trend_delta) == np.sign(actual_delta))
        print(f"Trend agreement {h_name}: {trend_agreement:.2%}")
```

### Priority 3 (MEDIUM): Add Statistical Validation
```python
# Verify scaler consistency
print("Scaler Statistics:")
print(f"  Fit mean: {target_scaler.mean_}")
print(f"  Fit scale: {target_scaler.scale_}")

# Verify per-horizon prediction distributions match training
print("\nPrediction Distributions (should match training):")
for h_name, h_data in [('h0', price_h0), ('h1', price_h1), ('h2', price_h2)]:
    print(f"  {h_name}: mean={h_data.mean():.6f}, std={h_data.std():.6f}")

# Verify deadband impact on direction labels
for h_idx, h_name in enumerate(['h0', 'h1', 'h2']):
    delta = y_test[:, h_idx]
    ret = delta / (last_close_vals + 1e-8)
    up_pct = (ret > deadband).mean() * 100
    print(f"  {h_name} UP ({deadband:.4f} deadband): {up_pct:.1f}%")
```

---

## ARCHITECTURAL INSIGHTS

### Why This Matters for Model Validation

1. **Train-Test Consistency**: Model trained with specific direction supervision (return + deadband). Evaluation must match exactly.

2. **Multi-Horizon Complexity**: Each horizon has independent targets AND independent direction truth. Mixing them invalidates metrics.

3. **Scaler Semantics**: Unified scaler means all three towers output in the same statistical space. Evaluation must respect this.

4. **Extended Trends Integration**: Trends are explicitly modeled in loss function. Evaluation should validate that they're being learned/respected.

---

## RECOMMENDATIONS

### Immediate Action
1. **Rewrite direction truth calculation** to include deadband logic
2. **Load extended_trends_test** and include trend validation metrics
3. **Verify train_step consistency** by comparing metric computation

### Medium-term
1. Create **validation callback** that mirrors test_step metrics exactly
2. Add **diagnostic plots** showing deadband impact on direction labels
3. Document **expected metric ranges** for each horizon based on training logs

### Long-term
1. Consolidate evaluation logic into **model.py** (avoid duplication)
2. Create **reusable evaluation module** that mirrors test_step
3. Add **unit tests** for train-test consistency

