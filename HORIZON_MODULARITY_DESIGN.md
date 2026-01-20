# Horizon Modularity Refactoring Design

## Current State Analysis

### Problems
1. **Hardcoded horizon count**: Model assumes exactly 3 horizons (h0, h1, h2)
2. **Hardcoded horizon labels**: String literals "h0", "h1", "h2" throughout codebase
3. **Fixed model architecture**: 3 towers explicitly coded, cannot scale to N horizons
4. **Tuple-based loss returns**: 22-component tuple prevents extensibility
5. **Hardcoded coherence constraints**: Pairwise comparisons (h01, h12) only
6. **No validation**: EXTENDED_TREND_PERIODS and HORIZON_STEPS can diverge

### What Works Well
- Centralized config registry (Config.HORIZON_STEPS)
- Data processing reads from config dynamically
- Metrics computation generates labels dynamically

---

## Target Architecture

### Design Principles
1. **Single Source of Truth**: Config.HORIZON_STEPS defines all horizon behavior
2. **Dynamic Generation**: All horizon-dependent code uses loops/comprehensions
3. **Validated Coupling**: Assert EXTENDED_TREND_PERIODS == HORIZON_STEPS
4. **Dict-based APIs**: Replace tuples with dicts for extensibility
5. **Zero Hardcoding**: No "h0", "h1", "h2" literals in code

---

## Implementation Plan

### 1. Config Class Enhancements

```python
class Config:
    # Existing
    HORIZON_STEPS = [1, 5, 15]
    EXTENDED_TREND_PERIODS = [1, 5, 15]

    # NEW: Dynamic properties
    @property
    def num_horizons(self):
        return len(self.HORIZON_STEPS)

    @property
    def horizon_keys(self):
        """Generate horizon keys dynamically: ('h0', 'h1', 'h2', ...)"""
        return tuple(f"h{i}" for i in range(self.num_horizons))

    # NEW: Validation at initialization
    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate horizon configuration consistency."""
        if not self.HORIZON_STEPS:
            raise ValueError("HORIZON_STEPS must be non-empty")
        if self.EXTENDED_TREND_PERIODS != self.HORIZON_STEPS:
            raise ValueError(
                f"EXTENDED_TREND_PERIODS {self.EXTENDED_TREND_PERIODS} "
                f"must match HORIZON_STEPS {self.HORIZON_STEPS}"
            )
        if not all(h > 0 for h in self.HORIZON_STEPS):
            raise ValueError("All HORIZON_STEPS must be positive")
        if self.HORIZON_STEPS != sorted(self.HORIZON_STEPS):
            raise ValueError("HORIZON_STEPS must be in ascending order")
```

### 2. Model Architecture: Dynamic Tower Generation

**Current (hardcoded):**
```python
tower_h0 = layers.Dense(16, ...)
price_h0 = layers.Dense(1, name='price_h0')(tower_h0)
direction_h0 = layers.Dense(1, name='direction_h0')(tower_h0)
variance_h0 = layers.Dense(1, name='variance_h0')(tower_h0)
# Repeat for h1, h2
outputs = [price_h0, direction_h0, variance_h0,
           price_h1, direction_h1, variance_h1,
           price_h2, direction_h2, variance_h2]
```

**Refactored (dynamic):**
```python
outputs = []
for i, horizon_step in enumerate(self.config.HORIZON_STEPS):
    h_key = f"h{i}"
    tower = layers.Dense(
        16,
        activation='gelu',
        name=f'tower_{h_key}',
        kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2)
    )(shared_dense)

    price = layers.Dense(1, name=f'price_{h_key}')(tower)
    direction = layers.Dense(
        1,
        activation='sigmoid',
        name=f'direction_{h_key}',
        bias_initializer=dir_bias_init
    )(tower)
    variance = layers.Dense(
        1,
        activation='softplus',
        name=f'variance_{h_key}',
        bias_initializer=var_bias_init
    )(tower)

    outputs.extend([price, direction, variance])

return models.Model(inputs=inp, outputs=outputs)
```

### 3. Loss Function: Dict-Based Returns

**Current (22-tuple):**
```python
return (
    total,
    point_loss_h0_val, point_loss_h1_val, point_loss_h2_val,
    local_trend_h0, global_trend_h0, extended_trend_h0,
    local_trend_h1, global_trend_h1, extended_trend_h1,
    local_trend_h2, global_trend_h2, extended_trend_h2,
    dir_loss_h0, dir_loss_h1, dir_loss_h2,
    nll_h0_val, nll_h1_val, nll_h2_val,
    reg_loss, inter_reg, vol_loss
)
```

**Refactored (dict):**
```python
# Flat dict for easy Keras metric logging
components = {'total': total}

for i in range(num_horizons):
    h_key = f"h{i}"
    components[f'point_loss_{h_key}'] = point_losses[i]
    components[f'local_trend_{h_key}'] = local_trends[i]
    components[f'global_trend_{h_key}'] = global_trends[i]
    components[f'extended_trend_{h_key}'] = extended_trends[i]
    components[f'dir_loss_{h_key}'] = dir_losses[i]
    components[f'nll_{h_key}'] = nlls[i]

components['reg_loss'] = reg_loss
components['inter_reg'] = inter_reg
components['vol_loss'] = vol_loss

return components
```

### 4. Loss Computation: Dynamic Horizon Processing

**Current (hardcoded unpacking):**
```python
y_true_h0 = y_true[:, 0:1]
y_true_h1 = y_true[:, 1:2]
y_true_h2 = y_true[:, 2:3]

price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2 = y_pred

extended_trends_scaled_h0 = extended_trends[:, 0:1] / pred_scale
extended_trends_scaled_h1 = extended_trends[:, 1:2] / pred_scale
extended_trends_scaled_h2 = extended_trends[:, 2:3] / pred_scale
```

**Refactored (loop-based):**
```python
num_horizons = len(model.config.HORIZON_STEPS)

# Unpack predictions (3 outputs per horizon: price, direction, variance)
predictions = []
for i in range(num_horizons):
    base_idx = i * 3
    predictions.append({
        'price': y_pred[base_idx],
        'direction': y_pred[base_idx + 1],
        'variance': y_pred[base_idx + 2],
    })

# Process each horizon
point_losses = []
dir_losses = []
nlls = []
extended_trends_list = []

for i in range(num_horizons):
    y_true_h = y_true[:, i:i+1]
    extended_trend_h = extended_trends[:, i:i+1] / pred_scale

    price_pred = predictions[i]['price']
    dir_pred = predictions[i]['direction']
    var_pred = predictions[i]['variance']

    # Compute losses
    point_loss = compute_point_loss(y_true_h, price_pred, lambda_weights[i])
    dir_loss = compute_direction_loss(...)
    nll = compute_nll(...)

    point_losses.append(point_loss)
    dir_losses.append(dir_loss)
    nlls.append(nll)
    extended_trends_list.append(extended_trend_h)
```

### 5. Coherence Constraints: Generalized N-Horizon

**Current (pairwise hardcoded):**
```python
sign_pred_h0 = tf.sign(price_h0)
sign_pred_h1 = tf.sign(price_h1)
sign_pred_h2 = tf.sign(price_h2)

dir_agree_h01 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h0, sign_pred_h1), tf.float32))
dir_agree_h12 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h1, sign_pred_h2), tf.float32))
dir_disagree_loss = 1.0 - (dir_agree_h01 + dir_agree_h12) / 2.0

magnitude_h01_violation = tf.nn.relu(abs_pred_h0 - abs_pred_h1)
magnitude_h12_violation = tf.nn.relu(abs_pred_h1 - abs_pred_h2)
magnitude_loss = tf.reduce_mean(magnitude_h01_violation + magnitude_h12_violation)
```

**Refactored (loop over consecutive pairs):**
```python
# Extract signs and magnitudes for all horizons
sign_preds = [tf.sign(predictions[i]['price']) for i in range(num_horizons)]
abs_preds = [tf.abs(predictions[i]['price']) for i in range(num_horizons)]
sign_targets = [tf.sign(y_true_raw[:, i]) for i in range(num_horizons)]

# Direction agreement across consecutive horizons
dir_agreements = []
for i in range(num_horizons - 1):
    agreement = tf.reduce_mean(tf.cast(
        tf.equal(sign_preds[i], sign_preds[i+1]),
        tf.float32
    ))
    dir_agreements.append(agreement)

dir_disagree_loss = 1.0 - tf.reduce_mean(dir_agreements) if dir_agreements else 0.0

# Magnitude monotonicity: |h_i| <= |h_{i+1}| (longer horizons should have larger moves)
magnitude_violations = []
for i in range(num_horizons - 1):
    violation = tf.nn.relu(abs_preds[i] - abs_preds[i+1])
    magnitude_violations.append(violation)

magnitude_loss = tf.reduce_mean(magnitude_violations) if magnitude_violations else 0.0

# Target smoothness
smoothness_losses = []
for i in range(1, num_horizons - 1):
    # Middle horizon should agree with at least one neighbor
    loss = tf.reduce_mean(tf.cast(
        tf.math.logical_xor(
            sign_targets[i] == sign_targets[i-1],
            sign_targets[i] == sign_targets[i+1]
        ),
        tf.float32
    ))
    smoothness_losses.append(loss)

target_smoothness_loss = tf.reduce_mean(smoothness_losses) if smoothness_losses else 0.0

coherence_penalty = (dir_disagree_loss + magnitude_loss + target_smoothness_loss) / 3.0
```

### 6. Training Step: Dict Unpacking

**Current (tuple unpacking):**
```python
(total_loss_val,
 point_h0, point_h1, point_h2,
 local_h0, global_h0, extended_h0,
 local_h1, global_h1, extended_h1,
 local_h2, global_h2, extended_h2,
 dir_h0, dir_h1, dir_h2,
 nll_h0, nll_h1, nll_h2,
 reg_val, inter_reg, vol_loss) = loss_components
```

**Refactored (dict access):**
```python
loss_dict = self.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
total_loss_val = loss_dict['total']

# Build metrics dict for logging (Keras expects flat dict)
metrics = {'loss': total_loss_val}
metrics.update({k: v for k, v in loss_dict.items() if k != 'total'})
```

### 7. Metrics Computation: Remove Hardcoded Horizons

**Current:**
```python
horizons = ("h0", "h1", "h2")  # HARDCODED
if y_true_deltas.shape[1] != 3:  # HARDCODED
    raise ValueError(...)
```

**Refactored:**
```python
horizon_keys = config.horizon_keys  # Dynamic from config
num_horizons = config.num_horizons

if y_true_deltas.shape[1] != num_horizons:
    raise ValueError(
        f"Expected y_true_deltas shape (N,{num_horizons}), "
        f"got {y_true_deltas.shape}"
    )

for idx, (h_key, h_label) in enumerate(zip(horizon_keys, horizon_names)):
    # ... compute metrics
```

### 8. Data Processing: Validation

**Current:**
```python
ext_features = self.compute_extended_trend_features(
    close_array, int(i-1),
    self.config.EXTENDED_TREND_PERIODS
)
```

**Refactored (with assertion):**
```python
# At DataProcessor initialization
if self.config.EXTENDED_TREND_PERIODS != self.config.HORIZON_STEPS:
    raise ValueError(
        f"EXTENDED_TREND_PERIODS {self.config.EXTENDED_TREND_PERIODS} "
        f"must equal HORIZON_STEPS {self.config.HORIZON_STEPS} "
        f"for target-feature alignment in loss computation"
    )
```

---

## Migration Path

1. ✅ Add Config properties and validation
2. ✅ Refactor model architecture to use loops
3. ✅ Refactor loss function to dict-based returns
4. ✅ Update training/validation steps for dict handling
5. ✅ Refactor coherence constraints
6. ✅ Update metrics computation
7. ✅ Update tests to validate N horizons
8. ✅ End-to-end testing

---

## Backward Compatibility

**Breaking Changes:**
- Loss function now returns dict instead of tuple
- Model outputs still ordered the same way (price_h0, dir_h0, var_h0, ...)
- Config validation may reject previously "working" invalid configs

**Migration:**
- Any code unpacking the 22-tuple must switch to dict access
- Tests asserting `len(horizons) == 3` must be updated
- Plotting code using hardcoded "h1" should use `config.horizon_keys[1]`

---

## Testing Strategy

1. **Unit tests**: Validate each component works with 1, 2, 3, 4, 5 horizons
2. **Integration test**: Train model end-to-end with non-default horizon count
3. **Regression test**: Ensure 3-horizon config produces identical results
4. **Edge cases**: Empty horizons (error), single horizon, 10+ horizons

---

## Success Criteria

✅ **True Modularity**: Change `HORIZON_STEPS = [2, 10, 30]` → model retrains without code changes
✅ **No Hardcoding**: Zero occurrences of literal "h0", "h1", "h2" in core logic
✅ **Validated Coupling**: Config.validate() enforces EXTENDED_TREND_PERIODS == HORIZON_STEPS
✅ **Extensible APIs**: Dict-based returns allow adding new loss components
✅ **Dynamic Constraints**: Coherence penalties work for any N >= 2 horizons
