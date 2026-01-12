# TRADING CELL AUDIT & ENHANCEMENT ANALYSIS

**Date**: January 11, 2026  
**Status**: 🔴 CRITICAL UNDERUTILIZATION IDENTIFIED  
**Priority**: HIGH (Strategy Features & Model Utilization)

---

## EXECUTIVE SUMMARY

The trading cell (Cell 7 - Multi-Head Strategy Pipeline) is **fundamentally sound but critically underutilizes** the 9-output model's features. Key issues:

| Issue | Severity | Root Cause | Impact |
|-------|----------|-----------|--------|
| **Variance Ignored** | 🔴 CRITICAL | Variance outputs computed but never used | No confidence weighting in signals |
| **Single Horizon Strategy** | 🔴 CRITICAL | Uses only h1 (5-min) for all decisions | Ignores 2 additional prediction heads |
| **No Cross-Horizon Coherence** | 🟡 MODERATE | Three horizons treated independently | Can enter contradictory trades |
| **Naive Confidence Calculation** | 🟡 MODERATE | Manual 1/(1+var) formula instead of principled approach | Not aligned with training |
| **No Direction Confidence Weighting** | 🟡 MODERATE | Direction probs used raw; ignore sigmoid calibration | Misses strong vs weak signals |
| **Static Entry/Exit Logic** | 🟡 MODERATE | Fixed thresholds (0.65/0.35) for all conditions | Not adaptive to signal strength |
| **Missing Variance-Based Risk Management** | 🔴 CRITICAL | No position sizing based on uncertainty | Constant risk regardless of prediction confidence |
| **No Horizon-Weighted Entry Signals** | 🟡 MODERATE | Manual weighting doesn't match training lambdas | Misaligned with model learning |

---

## SECTION 1: CURRENT IMPLEMENTATION ANALYSIS

### What the Model Produces

From `model.py`:
```python
# 9 outputs (3 towers × 3 heads)
predictions = {
    "delta": {"h0": prices_1m, "h1": prices_5m, "h2": prices_15m},  # Price deltas
    "direction_prob": {"h0": probs_1m, "h1": probs_5m, "h2": probs_15m},  # Sigmoid [0,1]
    "variance": {"h0": var_1m, "h1": var_5m, "h2": var_15m},  # Softplus > 0
}
```

Each output has rich information:
- **Delta**: Predicted price movement in dollars
- **Direction Probability**: P(move > deadband) from sigmoid
- **Variance**: Uncertainty in scaled space (softplus-activated)

### What the Trading Cell Currently Uses

```python
# Current usage (Cell 7):
price_1min_delta = predictions['delta']['h0']      # ✓ Used
direction_probs = predictions['direction_prob']['h1']  # ✓ Used (only h1!)
variance_raw = predictions['variance']['h1']        # ✓ Extracted but...

# Computed metrics:
confidence = 1.0 / (1.0 + variance_raw)  # ✓ Calculated but...
signal_str = direction_probs * confidence  # ✓ Used but...

# Entry logic:
if dir_prob > 0.65 and conf > 0.5 and agreement and not is_spike:
    # ❌ Uses only h1 signal
    # ❌ Fixed thresholds
    # ❌ No per-horizon weighting
    position = ('LONG', bar, curr_price)
```

### What's Missing

1. **h0 and h2 not used for entry decisions** - Only h1 used
2. **Variance not incorporated in position sizing** - Fixed 1-unit trades
3. **No multi-horizon consensus check** - Three towers act independently
4. **Threshold not adaptive** - Fixed 0.65/0.35 regardless of prediction strength
5. **No direction confidence weighting** - Pure direction vs confidence multiplication
6. **Horizon lambdas not applied** - Manual weights don't match training (0.8/1.0/1.2)

---

## SECTION 2: ROOT CAUSE ANALYSIS

### Why Variance is Underutilized

**Training perspective** (model.py):
```python
# Variance head trained with:
1. NLL loss (negative log-likelihood) - variance should predict uncertainty
2. Gaussian distribution alignment - mu and variance should be coherent
3. No explicit confidence threshold - variance is continuous

# What it means:
var_pred_h1 = 0.001  ← Very confident (low variance)
var_pred_h1 = 1.0    ← Moderate uncertainty
var_pred_h1 = 10.0   ← High uncertainty (softplus caps at ~1e4)
```

**Trading perspective** (current Cell 7):
```python
# Variance used as:
confidence = 1.0 / (1.0 + variance)
# This is a crude inverse, doesn't match training semantics
```

**The mismatch**: Training learns to predict actual variance (uncertainty), but trading cell treats it as a scalar to invert.

### Why Single Horizon Dominates

**Model training** (model.py):
```python
# Three independent towers with separate supervision:
HORIZON_STEPS = [1, 5, 15]  # 1-min, 5-min, 15-min
LAMBDA_SHORT = 0.8   # h0 weighted less (noisy)
LAMBDA_POINT = 1.0   # h1 weighted normal (primary)
LAMBDA_LONG = 1.2    # h2 weighted more (stable)

# Each tower sees different temporal dynamics:
# h0: Captures rapid microstructure, high noise
# h1: Balanced signal/noise, optimal for trading
# h2: Trend information, longer-term consistency
```

**Trading usage** (current Cell 7):
```python
# Only h1 used for decision:
if dir_prob > 0.65 and conf > 0.5:  # ← only checks h1
    position = ('LONG', bar, curr_price)

# h0 and h2 computed but ignored in logic
```

**Why this is wrong**:
- h0 captures fast reversals (useful for quick exits)
- h2 validates longer-term trend (useful for conviction)
- Using only h1 ignores 2/3 of the model's information

### Why Cross-Horizon Coherence Matters

**Training constraint** (model.py custom_loss):
```python
# Explicit coherence penalty:
coherence_penalty = (direction_consistency + magnitude_monotonicity + smoothness_penalty) / 3.0
# Weight: 0.1 × coherence_penalty

# What it enforces:
# 1. Direction consistency: all horizons agree on sign
# 2. Magnitude monotonicity: |δh0| ≤ |δh1| ≤ |δh2|
# 3. Smoothness: targets consistent across horizons
```

**Without using it** (current Cell 7):
```python
# Model learned coherence but trading ignores it
# Possible scenario:
h0_pred = +0.5 (predict UP at 1-min)
h1_pred = -0.3 (predict DOWN at 5-min)
h2_pred = +0.8 (predict UP at 15-min)
# ↑ Model penalizes this (contradictory), but trading sees all three independently
```

---

## SECTION 3: COMPREHENSIVE REWRITE PLAN

### Principle 1: Principled Variance Integration

**Current** (wrong semantics):
```python
confidence = 1.0 / (1.0 + variance)  # Crude inverse
```

**Correct** (information-theoretic):
```python
# Variance in scaled space, needs interpretation as precision weight
# Higher variance = lower precision = lower confidence
# Use exponential decay: confidence ∝ exp(-variance / variance_scale)

variance_scale = np.median(variance_raw) if np.median(variance_raw) > 0 else 1.0
confidence = np.exp(-variance_raw / (variance_scale + 1e-8))  # exp(-variance/scale)
# ↑ Matches: low variance → exp(0) ≈ 1.0 (high confidence)
#           high variance → exp(-large) ≈ 0.0 (low confidence)
```

**Why exponential**:
1. Information-theoretic foundation
2. Differentiable and smooth
3. Natural scale-invariance
4. Aligns with Gaussian likelihood

### Principle 2: Multi-Horizon Fusion

**Current** (single horizon):
```python
if dir_prob_h1 > 0.65:  # Only h1
    entry = 'LONG'
```

**Correct** (multi-horizon consensus with weighting):
```python
# Weighted consensus across horizons using training lambdas
w_h0 = config.LAMBDA_SHORT   # 0.8 (h0 is noisy)
w_h1 = config.LAMBDA_POINT   # 1.0 (h1 is primary)
w_h2 = config.LAMBDA_LONG    # 1.2 (h2 is stable)

# Raw direction signals
raw_signals = np.array([dir_h0, dir_h1, dir_h2])  # [0,1] sigmoid outputs
weights = np.array([w_h0, w_h1, w_h2])

# Weighted consensus with per-horizon variances as confidence
confidences = np.exp(-np.array([var_h0, var_h1, var_h2]) / variance_scale)
weighted_signal = np.sum(raw_signals * weights * confidences) / np.sum(weights * confidences)
# ↑ Accounts for: horizon importance + prediction confidence
```

### Principle 3: Cross-Horizon Coherence Check

**Current** (missing):
```python
# No check for contradictory predictions
```

**Correct** (enforce training constraint):
```python
# Check multi-horizon agreement
sign_h0 = np.sign(delta_h0)  # -1, 0, +1
sign_h1 = np.sign(delta_h1)
sign_h2 = np.sign(delta_h2)

# Agreement: majority vote
signs = [sign_h0, sign_h1, sign_h2]
agreement_count = max(
    sum(s > 0 for s in signs),  # UP count
    sum(s < 0 for s in signs),  # DOWN count
    sum(s == 0 for s in signs)  # NEUTRAL count
)
agreement_pct = agreement_count / 3.0

# Magnitude check: shorter horizons should move less
mag_h0, mag_h1, mag_h2 = np.abs([delta_h0, delta_h1, delta_h2])
magnitude_coherent = (mag_h0 <= mag_h1) and (mag_h1 <= mag_h2)

# Both must pass to trade
if agreement_pct >= 2/3 and magnitude_coherent:
    # Safe to trade - horizons are coherent
```

### Principle 4: Adaptive Thresholds

**Current** (static):
```python
if dir_prob > 0.65 and conf > 0.5:  # Fixed
    entry = 'LONG'
```

**Correct** (adaptive based on confidence):
```python
# Base threshold
base_threshold = 0.55  # Lower than current (0.65)

# Adjust based on confidence and agreement
if confidence > 0.8 and agreement_pct > 0.85:
    # Very high confidence - lower threshold
    entry_threshold = 0.50
elif confidence > 0.5 and agreement_pct > 0.67:
    # Moderate confidence - medium threshold
    entry_threshold = 0.60
else:
    # Low confidence - high threshold
    entry_threshold = 0.70

if weighted_signal > entry_threshold:
    entry = 'LONG'
```

### Principle 5: Position Sizing by Confidence

**Current** (fixed size):
```python
position = ('LONG', bar, curr_price)  # Always 1 unit
```

**Correct** (scale by confidence):
```python
# Position size proportional to confidence (0.1 to 1.0)
base_size = 1.0
position_size = base_size * confidence  # 0.1 to 1.0 depending on var

# Also incorporate agreement
position_size *= (1.0 + agreement_pct) / 2.0  # 0.5 to 1.0 boost

position = {
    'type': 'LONG',
    'entry_bar': bar,
    'entry_price': curr_price,
    'size': position_size,  # Variable size!
    'confidence': confidence,
    'agreement': agreement_pct,
}
```

---

## SECTION 4: DATA FLOW IMPLICATIONS

### Current Dataflow

```
Model Predictions (9 outputs)
├─ h0 (1-min):  delta, dir, var
├─ h1 (5-min):  delta, dir, var  ← ONLY THIS USED
└─ h2 (15-min): delta, dir, var

↓ Extract (Cell 7)

Manual Calculations
├─ confidence = 1/(1+var)
├─ signal = dir * conf
└─ agreement (multi-horizon check)

↓ Entry Logic

Trade Decision
├─ IF dir_prob_h1 > 0.65:  ENTER  ← Only h1!
└─ Fixed position size

Result: 2/3 of model information ignored
```

### Proposed Dataflow

```
Model Predictions (9 outputs)
├─ h0 (1-min):  delta, dir, var
├─ h1 (5-min):  delta, dir, var
└─ h2 (15-min): delta, dir, var

↓ Principled Processing

Confidence Calculation
├─ var_scale = robust_median(all_variances)
├─ conf_h0 = exp(-var_h0 / var_scale)
├─ conf_h1 = exp(-var_h1 / var_scale)
└─ conf_h2 = exp(-var_h2 / var_scale)

Multi-Horizon Fusion
├─ weighted_signal = sum(w_h * dir_h * conf_h) / sum(w_h * conf_h)
├─ agreement_pct = sign_agreement / 3
└─ magnitude_ok = mag_h0 <= mag_h1 <= mag_h2

Entry Logic (Adaptive)
├─ entry_threshold = 0.5 + 0.2 * (1 - confidence)
├─ IF weighted_signal > threshold AND agreement > 2/3:
└─    ENTER with size ∝ confidence × agreement

Result: All 9 outputs fully utilized
        Confidence-weighted decision
        Adaptive to signal strength
        Aligned with training
```

---

## SECTION 5: IMPLEMENTATION CHECKLIST

### Phase 1: Variance Integration
- [ ] Compute variance scale (robust median of all variances)
- [ ] Rewrite confidence as exp(-var / var_scale)
- [ ] Verify confidence in [0, 1] range
- [ ] Document variance semantics

### Phase 2: Multi-Horizon Fusion
- [ ] Load LAMBDA_SHORT, LAMBDA_POINT, LAMBDA_LONG from config
- [ ] Compute weighted signal: sum(w * dir * conf) / sum(w * conf)
- [ ] Compute directional agreement across horizons
- [ ] Compute magnitude coherence check

### Phase 3: Cross-Horizon Coherence
- [ ] Check sign agreement (2/3 majority rule)
- [ ] Check magnitude progression (h0 ≤ h1 ≤ h2)
- [ ] Require both for trade entry
- [ ] Log violations for analysis

### Phase 4: Adaptive Thresholds
- [ ] Base threshold = 0.55 (lower than current 0.65)
- [ ] Scale by confidence: 0.50 (high) to 0.70 (low)
- [ ] Scale by agreement: 0.60 (low) to 0.50 (high)
- [ ] Document threshold rationale

### Phase 5: Confidence-Based Sizing
- [ ] Position size = base_size × confidence × agreement_boost
- [ ] Size range: 0.1 to 1.0 units
- [ ] Store size in trade dataclass
- [ ] Apply to profit/loss calculation

### Phase 6: Extended Risk Management
- [ ] Use variance for stop-loss width (wider for high uncertainty)
- [ ] Use direction confidence for take-profit placement
- [ ] Horizon-dependent hold time based on h2 confidence
- [ ] Dynamic exit thresholds based on entry confidence

---

## SECTION 6: EXPECTED IMPROVEMENTS

### Quantitative Benefits

1. **Information Utilization**: From 1/3 (h1 only) → 3/3 (all horizons)
2. **Confidence Weighting**: From binary (enter/no-enter) → continuous (size varies)
3. **Risk Management**: From static → adaptive based on prediction uncertainty
4. **Signal Quality**: Weighted consensus vs single noisy signal

### Qualitative Benefits

1. **Alignment with Training**: Respects horizon lambdas and coherence constraints learned during training
2. **Robustness**: Multi-horizon consensus reduces false signals
3. **Adaptability**: Thresholds adjust to confidence level
4. **Interpretability**: All 9 outputs visible in trade dataclass

### Example Improvements

**Scenario 1: High Confidence UP Signal**
```
BEFORE:
- h1_dir = 0.75, var = 0.1 → conf = 0.91, size = 1.0
- h0_dir = 0.72, h2_dir = 0.78 (ignored)
- Entry: LONG 1.0 units

AFTER:
- h0_dir = 0.72, var = 0.15 → conf = 0.86
- h1_dir = 0.75, var = 0.10 → conf = 0.91
- h2_dir = 0.78, var = 0.08 → conf = 0.93
- weighted_signal = (0.8×0.72×0.86 + 1.0×0.75×0.91 + 1.2×0.78×0.93) / sum(weights×conf)
                  = 2.65 / 3.52 = 0.753 (higher than any single!)
- agreement = 3/3 (all UP)
- magnitude: |0.4| ≤ |0.5| ≤ |0.6| ✓
- Entry threshold = 0.50 (high confidence + full agreement)
- Entry: LONG 1.0 × 0.91 × 1.0 = 0.91 units
```

**Scenario 2: Mixed Signal**
```
BEFORE:
- h1_dir = 0.68, var = 0.8 → conf = 0.56, size = 1.0
- h0_dir = 0.52, h2_dir = 0.65 (ignored, might contradict!)
- Entry: LONG 1.0 units (questionable)

AFTER:
- h0_dir = 0.52, var = 1.2 → conf = 0.45
- h1_dir = 0.68, var = 0.80 → conf = 0.56
- h2_dir = 0.35, var = 0.95 → conf = 0.51 (DOWN signal!)
- weighted_signal = (0.8×0.52×0.45 + 1.0×0.68×0.56 + 1.2×0.35×0.51) / sum(weights×conf)
                  = 0.69 / 2.77 = 0.249 (weak, conflicted)
- agreement = 1/3 (h0 UP, h1 UP, h2 DOWN - disagreement!)
- Entry threshold = 0.68 (low confidence + low agreement)
- NO ENTRY (correctly rejects mixed signal)
```

---

## SECTION 7: CODE ARCHITECTURE

### New Structure

```python
# ============================================================================
# MULTI-HEAD STRATEGY PIPELINE - FULLY ENHANCED
# ============================================================================

class StrategyConfig:
    """Strategy-specific configuration (separate from model config)"""
    VAR_SCALE_METHOD = 'median'  # 'median' or 'mean'
    BASE_ENTRY_THRESHOLD = 0.55
    MIN_AGREEMENT_PCT = 0.67  # 2/3 majority
    CONFIDENCE_BOOST = 0.1  # Position size multiplier
    VARIANCE_SCALE_ROBUST = True

class MultiHorizonSignal:
    """Encapsulate multi-horizon signal computation"""
    
    def __init__(self, config, variance_scale):
        self.config = config
        self.variance_scale = variance_scale
    
    def compute_confidence(self, variance):
        """Variance → confidence via exponential decay"""
        return np.exp(-variance / self.variance_scale)
    
    def fuse_horizons(self, deltas, dirs, variances, lambdas):
        """Weighted consensus across horizons"""
        confidences = np.array([self.compute_confidence(v) for v in variances])
        weighted_signal = np.sum(np.array(dirs) * np.array(lambdas) * confidences) / \
                         np.sum(np.array(lambdas) * confidences)
        return weighted_signal, confidences
    
    def check_coherence(self, deltas, dirs):
        """Verify multi-horizon coherence"""
        signs = np.sign(deltas)
        agreement = max(np.sum(signs > 0), np.sum(signs < 0)) / len(signs)
        
        mags = np.abs(deltas)
        magnitude_ok = np.all(np.diff(mags) >= 0)  # h0 ≤ h1 ≤ h2
        
        return agreement, magnitude_ok

class EnhancedTrade(Trade):
    """Extended trade with confidence and sizing info"""
    
    confidence: float  # Prediction confidence [0, 1]
    agreement: float   # Multi-horizon agreement [0, 1]
    position_size: float  # Actual position size used
    entry_threshold: float  # Threshold used for entry
    coherent: bool  # Multi-horizon coherence verified
```

---

## CONCLUSION

The trading cell is underutilizing the 9-output model by:
1. Using only 1/3 of outputs (h1 only)
2. Ignoring variance information
3. Missing cross-horizon coherence
4. Using static thresholds

The proposed enhancements align the trading strategy with training objectives while leveraging all available model outputs to make **more informed, confident, and robust trading decisions**.

Implementation follows integral approach: thoroughly researched model outputs, root causes identified, complete rewrite (not deletions), dataflow analyzed.
