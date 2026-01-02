
"""Multi-Head Trading Strategy Helper Functions"""
import numpy as np

def calculate_confidence(variance, eps=1e-7):
    """Convert variance to confidence score [0, 1]"""
    return 1.0 / (1.0 + np.asarray(variance) + eps)

def calculate_signal_strength(direction_prob, confidence):
    """Combine direction and confidence into unified signal"""
    return np.asarray(direction_prob) * np.asarray(confidence)

def normalize_variance(variance, variance_rolling_mean, variance_rolling_std, eps=1e-7):
    """Normalize variance relative to rolling statistics"""
    variance_array = np.asarray(variance)
    mean_array = np.asarray(variance_rolling_mean)
    std_array = np.asarray(variance_rolling_std)
    result = np.where(
        std_array < eps,
        0.0,
        (variance_array - mean_array) / (std_array + eps)
    )
    return result

def calculate_profit_targets(entry_price, price_predictions):
    """Use price predictions as profit targets"""
    # Validate inputs
    if len(price_predictions) < 3:
        raise ValueError(f"Need at least 3 price predictions, got {len(price_predictions)}")

    entry_price = float(entry_price)
    if abs(entry_price) < 1e-10:
        raise ValueError(f"Invalid entry_price: {entry_price} (too close to zero)")

    tp1 = float(price_predictions[0])
    tp2 = float(price_predictions[1])
    tp3 = float(price_predictions[2])
    tp1_pct = (tp1 - entry_price) / entry_price * 100
    tp2_pct = (tp2 - entry_price) / entry_price * 100
    tp3_pct = (tp3 - entry_price) / entry_price * 100
    return {
        'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
        'tp1_pct': tp1_pct, 'tp2_pct': tp2_pct, 'tp3_pct': tp3_pct,
    }

def calculate_dynamic_stop_loss(entry_price, position_type, variance, variance_rolling_mean, 
                               base_stop_pct=0.02, max_variance_multiplier=2.0):
    """Calculate variance-adjusted stop loss"""
    entry_price = float(entry_price)
    variance = float(variance)
    variance_rolling_mean = float(variance_rolling_mean)
    eps = 1e-7
    variance_ratio = min(variance / (variance_rolling_mean + eps), max_variance_multiplier)
    adjustment_factor = 1.0 + variance_ratio
    stop_distance = entry_price * base_stop_pct * adjustment_factor
    if position_type == 'LONG':
        stop_loss = entry_price - stop_distance
    else:
        stop_loss = entry_price + stop_distance
    return stop_loss

def calculate_position_size_multiplier(confidence, size_high=1.2, size_normal=1.0, size_low=0.6,
                                      conf_high_thresh=0.7, conf_low_thresh=0.5):
    """Calculate position size based on confidence"""
    confidence = float(confidence)
    if confidence > conf_high_thresh:
        return size_high
    elif confidence > conf_low_thresh:
        return size_normal
    else:
        return size_low

def check_multi_horizon_agreement(price_predictions, current_price, agreement_threshold=0.67):
    """Check if multiple horizons agree on direction"""
    price_predictions = np.asarray(price_predictions)

    # Validate inputs
    if len(price_predictions) == 0:
        raise ValueError("price_predictions cannot be empty")

    current_price = float(current_price)
    up_count = np.sum(price_predictions > current_price)
    down_count = np.sum(price_predictions < current_price)
    agreement = max(up_count, down_count) / len(price_predictions)
    is_agreed = agreement >= agreement_threshold
    return is_agreed, agreement

def detect_variance_spike(variance, variance_rolling_mean, variance_rolling_std, 
                         spike_threshold=2.0, eps=1e-7):
    """Detect variance spikes indicating model uncertainty"""
    variance = float(variance)
    variance_rolling_mean = float(variance_rolling_mean)
    spike_level = spike_threshold * (variance_rolling_mean + eps)
    is_spike = variance > spike_level
    return is_spike
