# Software Requirements Specification (SRS)
## Neural Trade - Deep Learning Trading System

**Document Status:** Version 1.1 - Complete  
**Created:** 2026-01-02  
**Revised:** 2026-01-02  
**Author:** AI System  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Module Specifications](#3-module-specifications)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [System Architecture](#6-system-architecture)
7. [Data Models and Storage](#7-data-models-and-storage)
8. [User Interface](#8-user-interface)
9. [Integration Points](#9-integration-points)
10. [Assumptions, Dependencies, and Constraints](#10-assumptions-dependencies-and-constraints)
11. [File Structure and Documentation](#11-file-structure-and-documentation)
12. [Configuration Management](#12-configuration-management)
13. [Testing Requirements](#13-testing-requirements)
14. [Deployment and Operations](#14-deployment-and-operations)

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) document provides a complete description of the Neural Trade system, a deep learning-based cryptocurrency trading prediction system. It specifies all functional and non-functional requirements, system architecture, data models, and operational procedures.

### 1.2 Scope
Neural Trade is a Python-based machine learning system that:
- Fetches historical and live cryptocurrency market data
- Trains deep neural networks with learnable technical indicators
- Predicts multi-horizon price movements with uncertainty quantification
- Provides trading signals with confidence metrics
- Performs backtesting of trading strategies

### 1.3 Intended Audience
- Software Developers implementing and maintaining the system
- Data Scientists optimizing models and strategies
- System Architects planning deployments
- Quality Assurance engineers developing test plans
- Project Managers tracking development progress

### 1.4 Product Context
Neural Trade is a standalone research and development system for algorithmic trading strategy development. It currently supports BTC/USDT on Binance and is designed for research, backtesting, and model development purposes.

---

## 2. System Overview

### 2.1 System Description
Neural Trade implements a multi-horizon deep learning model that predicts cryptocurrency price movements at three time horizons (1 minute, 5 minutes, 15 minutes) with uncertainty quantification. The system uses learnable technical indicators and produces actionable trading signals with confidence metrics.

### 2.2 Key Features
- **Multi-Horizon Predictions**: Simultaneous predictions for h0 (1 min), h1 (5 min), h2 (15 min)
- **Learnable Indicators**: Technical indicators with trainable parameters (30+ learnable parameters)
- **Uncertainty Quantification**: Variance estimates for each prediction
- **Trading Signal Generation**: Confidence-based position sizing and risk management
- **Custom Backtesting**: Trade simulation with multi-target profit taking
- **Model Versioning**: Semantic versioning with rollback support

### 2.3 System Context
```
┌─────────────────────────────────────────────────────────────┐
│                      Neural Trade System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │  Data        │────▶│  Model       │────▶│  Trading    │ │
│  │  Acquisition │     │  Training    │     │  Signals    │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                     │                    │         │
│         ▼                     ▼                    ▼         │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │  Data        │     │  Model       │     │  Backtesting│ │
│  │  Validation  │     │  Inference   │     │  Engine     │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
  External APIs                              Performance Metrics
  (Binance/CCXT)                            (Training Logs, PnL)
```

---

## 3. Module Specifications

### 3.1 Market Data Acquisition Module

**File:** `dw_ccxt.py`  
**Purpose:** Fetch historical and live market data from cryptocurrency exchanges via CCXT library.

#### 3.1.1 Core Functionality
- Connect to Binance exchange via CCXT
- Fetch OHLCV (Open, High, Low, Close, Volume) data
- Handle pagination for large date ranges
- Implement retry logic with exponential backoff
- Validate data quality and integrity

#### 3.1.2 Detailed Implementation Specifications

**Connection Configuration:**
```python
exchange = ccxt.binance({
    'enableRateLimit': True,
    'rateLimit': 1200,  # milliseconds per request
})
```

**Pagination Logic:**
- Batch size: 500 candles per request (Binance supports up to 1000, but 500 is safer)
- Time-based pagination using `since` parameter
- Automatic advancement to next batch after last candle timestamp
- Increment: last_candle_time + 60000ms (1 minute) to avoid duplicates

**Retry Mechanism:**
- Network errors: 5-second delay before retry
- Exchange errors: 5-second delay before retry
- Maximum retries: Implicit (loop continues until success or date range complete)
- Exponential backoff: Not explicitly implemented (uses fixed 5-second delay)
- Rate limiting: Built-in via CCXT `enableRateLimit` + manual 0.1s sleep between requests

**Error Handling:**
```python
try:
    candles = exchange.fetch_ohlcv(symbol, timeframe, since=timestamp, limit=500)
except ccxt.NetworkError as e:
    # Handle network failures
    time.sleep(5)
    continue
except ccxt.ExchangeError as e:
    # Handle exchange-specific errors
    time.sleep(5)
    continue
except Exception as e:
    # Log unexpected errors and break
    break
```

**Data Validation:**
- Check for missing values (null/NaN)
- Verify OHLC logic: High ≥ max(Open, Close), Low ≤ min(Open, Close)
- Detect time gaps (missing 1-minute candles)
- Validate positive prices and non-negative volumes
- Calculate and display data quality metrics

**CSV Output Format:**
```csv
datetime,open,high,low,close,volume
2026-01-02 12:00:00+00:00,42150.5,42180.2,42145.0,42170.0,1.234567
2026-01-02 12:01:00+00:00,42170.0,42195.5,42165.0,42190.0,0.987654
```
- Index: datetime (UTC timezone-aware)
- Columns: open, high, low, close, volume (lowercase)
- File: `binance_btcusdt_1min_ccxt.csv`

#### 3.1.3 API Rate Limits
- Respect Binance rate limits (1200ms per request via CCXT)
- Implement 0.1s additional sleep between requests
- Gracefully handle 429 errors (rate limit exceeded)

#### 3.1.4 Data Quality Metrics
- Total candles fetched
- Date range coverage
- Number of missing value instances
- Number of OHLC logic violations
- Number of time gaps detected

---

### 3.2 Data Preprocessing Module

**File:** `model.py` (integrated)  
**Purpose:** Transform raw OHLCV data into model-ready features with learnable technical indicators.

#### 3.2.1 Windowing and Sampling
- Lookback window: 60 minutes (HOUR = 60)
- Window step: 1 minute (generate training sample every minute)
- Sequence limit: 2880 most recent sequences (144 batches × 20)

#### 3.2.2 Feature Engineering
**Learnable Technical Indicators:**
- Moving Averages (3 periods, learnable)
- MACD (3 settings × 3 parameters = 9 learnable params)
- Custom MACD pairs (3 pairs × 3 parameters = 9 learnable params)
- RSI (3 periods, learnable)
- Bollinger Bands (3 periods, learnable)
- Momentum (3 periods, learnable)
- **Total: 30+ learnable indicator parameters**

**Fixed Features:**
- Extended trend features: 1m, 5m, 15m percent changes
- Price returns and volatility
- Volume statistics

#### 3.2.3 Data Scaling
- StandardScaler for input features (`scaler_input.joblib`)
- StandardScaler for output targets (`scaler.joblib`)
- Fit on training set, applied to validation/test sets
- Inverse scaling for predictions

---

### 3.3 Trading Strategy Module

**File:** `helper_functions.py`  
**Purpose:** Provide utility functions for trading signal generation, risk management, and trade execution logic.

#### 3.3.1 Helper Function Specifications

**Function 1: calculate_confidence**
```python
def calculate_confidence(variance, eps=1e-7):
    """Convert variance to confidence score [0, 1]"""
    return 1.0 / (1.0 + np.asarray(variance) + eps)
```
- **Input:** variance (float or array), eps (float, default 1e-7)
- **Output:** confidence score [0, 1]
- **Purpose:** Convert model variance to confidence metric (inverse relationship)

**Function 2: calculate_signal_strength**
```python
def calculate_signal_strength(direction_prob, confidence):
    """Combine direction and confidence into unified signal"""
    return np.asarray(direction_prob) * np.asarray(confidence)
```
- **Input:** direction_prob (float or array), confidence (float or array)
- **Output:** signal strength value
- **Purpose:** Combine directional prediction and confidence into single signal

**Function 3: normalize_variance**
```python
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
```
- **Input:** variance, variance_rolling_mean, variance_rolling_std, eps=1e-7
- **Output:** normalized variance (z-score)
- **Purpose:** Normalize variance for anomaly detection

**Function 4: calculate_profit_targets**
```python
def calculate_profit_targets(entry_price, price_predictions):
    """Use price predictions as profit targets"""
    entry_price = float(entry_price)
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
```
- **Input:** entry_price (float), price_predictions (list of 3 floats)
- **Output:** dict with tp1, tp2, tp3 and their percentages
- **Purpose:** Calculate three-tier profit targets from multi-horizon predictions

**Function 5: calculate_dynamic_stop_loss**
```python
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
```
- **Input:** entry_price, position_type ('LONG'/'SHORT'), variance, variance_rolling_mean, base_stop_pct=0.02, max_variance_multiplier=2.0
- **Output:** stop_loss price level
- **Purpose:** Calculate dynamic stop loss adjusted for model uncertainty

**Function 6: calculate_position_size_multiplier**
```python
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
```
- **Input:** confidence, size_high=1.2, size_normal=1.0, size_low=0.6, conf_high_thresh=0.7, conf_low_thresh=0.5
- **Output:** position size multiplier
- **Purpose:** Adjust position size based on prediction confidence

**Function 7: check_multi_horizon_agreement**
```python
def check_multi_horizon_agreement(price_predictions, current_price, agreement_threshold=0.67):
    """Check if multiple horizons agree on direction"""
    price_predictions = np.asarray(price_predictions)
    current_price = float(current_price)
    up_count = np.sum(price_predictions > current_price)
    down_count = np.sum(price_predictions < current_price)
    agreement = max(up_count, down_count) / len(price_predictions)
    is_agreed = agreement >= agreement_threshold
    return is_agreed, agreement
```
- **Input:** price_predictions (array), current_price, agreement_threshold=0.67
- **Output:** tuple (is_agreed: bool, agreement: float)
- **Purpose:** Check if multiple prediction horizons agree on market direction

**Function 8: detect_variance_spike**
```python
def detect_variance_spike(variance, variance_rolling_mean, variance_rolling_std, 
                          spike_threshold=2.0, eps=1e-7):
    """Detect variance spikes indicating model uncertainty"""
    variance = float(variance)
    variance_rolling_mean = float(variance_rolling_mean)
    spike_level = spike_threshold * (variance_rolling_mean + eps)
    is_spike = variance > spike_level
    return is_spike
```
- **Input:** variance, variance_rolling_mean, variance_rolling_std, spike_threshold=2.0, eps=1e-7
- **Output:** is_spike (bool)
- **Purpose:** Detect variance spikes indicating high model uncertainty or regime changes

---

### 3.4 Custom Backtesting Module

**File:** `inference.ipynb` (integrated)  
**Purpose:** Simulate trading strategies using historical data and model predictions.

**Note:** Backtrader integration is not currently implemented. The system uses a custom backtesting implementation within the inference notebook.

#### 3.4.1 Backtest Implementation
- Uses Trade dataclass for trade tracking (see Section 7.1.4)
- Simulates entry signals based on multi-horizon agreement
- Implements three-tier profit taking (TP1, TP2, TP3)
- Tracks stop loss hits and winning/losing trades
- Calculates PnL and performance metrics

#### 3.4.2 Trade Simulation Logic
- Entry conditions: Multi-horizon agreement + confidence threshold
- Position sizing: Based on confidence multiplier
- Exit conditions: Hit TP1/TP2/TP3 or stop loss
- Partial profit taking: Close portions at each target
- Trade status tracking: OPEN, CLOSED_WIN, CLOSED_LOSS

#### 3.4.3 Performance Metrics
- Total PnL and PnL percentage
- Win rate and average win/loss
- Maximum drawdown
- Sharpe ratio (if applicable)
- Trade count and duration statistics

**Future Enhancement:** Full Backtrader integration planned (see Section 10.4)

---

### 3.5 Model Training Module

**File:** `model.py`  
**Purpose:** Define, train, and persist the deep learning model.

#### 3.5.1 Model Architecture
- **Input:** 60-minute OHLCV sequences + learnable indicators
- **Backbone:** Multi-layer LSTM/GRU with attention
- **Output Towers:** 3 independent towers for h0, h1, h2
- **Outputs per Tower:** 
  - Price prediction (regression)
  - Direction classification (binary with focal loss)
  - Variance/uncertainty (heteroscedastic)
- **Total Outputs:** 9 (3 towers × 3 outputs each)

#### 3.5.2 Loss Functions
- **Point Loss:** MSE/Huber for price predictions
- **Direction Loss:** Focal loss (α=0.7, γ=1.0) for directional classification
- **Variance Loss:** Negative log-likelihood for uncertainty
- **Trend Losses:** Local, global, and extended trend matching
- **Regularization:** L2 on learnable indicators, interconnection penalties

#### 3.5.3 Training Configuration
- Optimizer: Adam with learning rate 1e-3
- Batch size: 144
- Epochs: 40
- Early stopping: Patience = 40 (monitors val_dir_mcc_h1)
- Gradient clipping: Norm = 5.0

#### 3.5.4 Model Persistence
- Weights: `nn_learnable_indicators_v3.weights.h5` (~3.2 MB)
- Input scaler: `scaler_input.joblib`
- Output scaler: `scaler.joblib`
- Training log: `training_log.csv`
- Indicator history: `indicator_params_history.csv`

---

## 4. Functional Requirements

### FR-1: Data Acquisition
**FR-1.1:** The system shall fetch 1-minute OHLCV data from Binance via CCXT.  
**FR-1.2:** The system shall support date range specification for historical data.  
**FR-1.3:** The system shall validate data integrity (OHLC logic, missing values, gaps).  
**FR-1.4:** The system shall save fetched data to CSV format.  

### FR-2: Model Training
**FR-2.1:** The system shall train a multi-horizon neural network with 3 independent output towers.  
**FR-2.2:** The system shall implement learnable technical indicators with 30+ trainable parameters.  
**FR-2.3:** The system shall save model weights, scalers, and training logs.  
**FR-2.4:** The system shall track learnable indicator evolution over epochs.  

### FR-3: Prediction and Inference
**FR-3.1:** The system shall predict prices for 3 horizons: 1min, 5min, 15min.  
**FR-3.2:** The system shall provide direction classification for each horizon.  
**FR-3.3:** The system shall estimate prediction variance/uncertainty.  
**FR-3.4:** The system shall generate confidence scores from variance estimates.  

### FR-4: Trading Signal Generation
**FR-4.1:** The system shall calculate signal strength combining direction and confidence.  
**FR-4.2:** The system shall determine position size multipliers based on confidence.  
**FR-4.3:** The system shall calculate dynamic stop loss levels adjusted for variance.  
**FR-4.4:** The system shall set three-tier profit targets from multi-horizon predictions.  
**FR-4.5:** The system shall detect multi-horizon directional agreement.  
**FR-4.6:** The system shall detect variance spikes indicating high uncertainty.  

### FR-5: Backtesting
**FR-5.1:** The system shall simulate trades using historical data and model predictions.  
**FR-5.2:** The system shall track trade entries, exits, and PnL.  
**FR-5.3:** The system shall implement partial profit taking at multiple targets.  
**FR-5.4:** The system shall calculate performance metrics (win rate, total PnL, drawdown).  

---

## 5. Non-Functional Requirements

### NFR-1: Performance
**NFR-1.1:** Model inference shall complete in <1 second per prediction.  
**NFR-1.2:** Data fetching shall respect API rate limits (1200ms per request).  
**NFR-1.3:** Training shall complete within 4 hours on modern GPU hardware.  

### NFR-2: Reliability
**NFR-2.1:** The system shall handle network failures with retry logic.  
**NFR-2.2:** The system shall validate data quality before model training.  
**NFR-2.3:** The system shall maintain model version history for rollback.  

### NFR-3: Maintainability
**NFR-3.1:** Code shall follow PEP 8 style guidelines.  
**NFR-3.2:** Functions shall have descriptive docstrings.  
**NFR-3.3:** Configuration shall be externalized (see Section 12).  

### NFR-4: Scalability
**NFR-4.1:** The system shall support training on sequences up to 2880 samples.  
**NFR-4.2:** Memory usage shall remain within 16GB during training.  

### NFR-5: Security
**NFR-5.1:** API keys shall not be hardcoded in source files.  
**NFR-5.2:** Sensitive configuration shall be loaded from environment variables.  

---

## 6. System Architecture

### 6.1 Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                          Neural Trade System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────────┐ │
│  │   dw_ccxt   │─────▶│   model.py   │─────▶│ helper_funcs   │ │
│  │  (Data In)  │      │  (Training)  │      │   (Signals)    │ │
│  └─────────────┘      └──────────────┘      └────────────────┘ │
│         │                     │                      │           │
│         │                     ▼                      │           │
│         │              ┌──────────────┐             │           │
│         │              │  Model       │             │           │
│         │              │  Artifacts   │             │           │
│         │              │  (.h5, .pkl) │             │           │
│         │              └──────────────┘             │           │
│         │                     │                      │           │
│         ▼                     ▼                      ▼           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │             inference.ipynb (Jupyter Notebook)            │  │
│  │  - Load model and scalers                                 │  │
│  │  - Run inference on new data                              │  │
│  │  - Generate trading signals                               │  │
│  │  - Perform backtesting                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Component Interaction
1. **Data Acquisition:** `dw_ccxt.py` fetches data → CSV
2. **Model Training:** `model.py` reads CSV → trains model → saves artifacts
3. **Inference:** `inference.ipynb` loads model → predicts on new data
4. **Signal Generation:** `helper_functions.py` processes predictions → trading signals
5. **Backtesting:** `inference.ipynb` simulates trades → performance metrics

### 6.3 Data Flow
```
CCXT API → Raw OHLCV → CSV → Feature Engineering → 
→ Scaled Features → Model → Raw Predictions → 
→ Inverse Scaling → Helper Functions → Trading Signals
```

### 6.4 Technology Stack
- **Language:** Python 3.10+
- **ML Framework:** TensorFlow/Keras
- **Data Processing:** NumPy, Pandas
- **Exchange API:** CCXT
- **Persistence:** Joblib, HDF5
- **Notebook:** Jupyter

### 6.5 Model Prediction API (Future)

**Future Enhancement:** REST API for serving predictions.

**Endpoint:** `POST /api/v1/predict`

**Request Body:**
```json
{
  "symbol": "BTC/USDT",
  "recent_data": [...],
  "include_metadata": true
}
```

**Response:**
```json
{
  "timestamp": "2026-01-02T12:34:56Z",
  "symbol": "BTC/USDT",
  "predictions": {
    "h0": {
      "price": 42150.5,
      "direction": 1,
      "variance": 0.0023,
      "confidence": 0.85
    },
    "h1": {
      "price": 42200.3,
      "direction": 1,
      "variance": 0.0045,
      "confidence": 0.78
    },
    "h2": {
      "price": 42350.1,
      "direction": 1,
      "variance": 0.0098,
      "confidence": 0.65
    }
  },
  "signals": {
    "signal_strength": 0.72,
    "position_size_multiplier": 1.2,
    "multi_horizon_agreement": true,
    "variance_spike_detected": false
  },
  "model_version": "v3.0"
}
```

---

## 7. Data Models and Storage

### 7.1 Data Models

#### 7.1.1 OHLCV Data Model
| Field | Type | Description |
|-------|------|-------------|
| datetime | datetime | Timestamp (UTC, timezone-aware) |
| open | float | Opening price |
| high | float | Highest price in period |
| low | float | Lowest price in period |
| close | float | Closing price |
| volume | float | Trading volume (base currency) |

#### 7.1.2 Model Prediction Output
| Field | Type | Description |
|-------|------|-------------|
| price_h0 | float | Predicted price at horizon 0 (1 min) |
| price_h1 | float | Predicted price at horizon 1 (5 min) |
| price_h2 | float | Predicted price at horizon 2 (15 min) |
| direction_h0 | int | Direction classification (0=down, 1=up) |
| direction_h1 | int | Direction classification (0=down, 1=up) |
| direction_h2 | int | Direction classification (0=down, 1=up) |
| variance_h0 | float | Prediction variance for h0 |
| variance_h1 | float | Prediction variance for h1 |
| variance_h2 | float | Prediction variance for h2 |

#### 7.1.3 Training Metrics
| Field | Type | Description |
|-------|------|-------------|
| epoch | int | Training epoch number |
| loss | float | Training loss |
| val_loss | float | Validation loss |
| dir_acc_h0/h1/h2 | float | Direction accuracy per horizon |
| dir_f1_h0/h1/h2 | float | Direction F1 score per horizon |
| dir_mcc_h0/h1/h2 | float | Matthews Correlation Coefficient |
| price_mae_h0/h1/h2 | float | Price prediction MAE per horizon |

#### 7.1.4 Trade Dataclass
| Field | Type | Description |
|-------|------|-------------|
| entry_time | datetime | Trade entry timestamp |
| entry_price | float | Entry price |
| exit_time | datetime | Trade exit timestamp (optional) |
| exit_price | float | Exit price (optional) |
| position_type | str | 'LONG' or 'SHORT' |
| size | float | Position size |
| stop_loss | float | Stop loss level |
| take_profit_1 | float | First profit target |
| take_profit_2 | float | Second profit target |
| take_profit_3 | float | Third profit target |
| confidence | float | Confidence at entry [0, 1] |
| variance | float | Model variance at entry |
| status | str | 'OPEN', 'CLOSED_WIN', 'CLOSED_LOSS' |
| pnl | float | Profit/loss (optional) |
| pnl_pct | float | Profit/loss percentage (optional) |

### 7.2 Data Storage

#### 7.2.1 Market Data
- **Format:** CSV
- **Path:** `binance_btcusdt_1min_ccxt.csv`
- **Index:** datetime (UTC)
- **Update Frequency:** On-demand via `dw_ccxt.py`

#### 7.2.2 Model Weights
- **Format:** HDF5
- **Path:** `nn_learnable_indicators_v3.weights.h5`
- **Size:** ~3.2 MB
- **Version:** v3.0 (current)

#### 7.2.3 Data Scalers
- **Format:** Joblib pickle
- **Paths:** `scaler_input.joblib`, `scaler.joblib`
- **Purpose:** Feature normalization for consistent inference

#### 7.2.4 Learnable Indicator Parameters
- **Format:** CSV
- **Path:** `indicator_params_history.csv`
- **Update Frequency:** Per training epoch
- **Contents:**
  - Epoch number
  - MA periods (3 parameters)
  - MACD parameters (9 parameters: 3 settings × 3 params each)
  - Custom MACD pairs (9 parameters: 3 pairs × 3 params each)
  - RSI periods (3 parameters)
  - Bollinger Band periods (3 parameters)
  - Momentum periods (3 parameters)
  - **Total: 30+ learnable parameters tracked over training**

#### 7.2.5 Training Logs
- **Format:** CSV
- **Path:** `training_log.csv`
- **Contents:** Epoch-wise training and validation metrics
- **Use:** Model performance tracking, early stopping decisions

---

## 8. User Interface

### 8.1 Jupyter Notebook Interface
Primary user interface is `inference.ipynb` Jupyter notebook providing:
- Interactive model training controls
- Real-time training visualization
- Prediction generation interface
- Backtest configuration and execution
- Performance metric dashboards

### 8.2 Command Line Interface
Scripts can be executed via command line:
```bash
python dw_ccxt.py          # Fetch market data
python model.py            # Train model (if configured)
```

### 8.3 Future Web Dashboard (Planned)
- Real-time prediction monitoring
- Live trading signal display
- Performance analytics
- Model management interface
- See Section 10.4 for details

---

## 9. Integration Points

### 9.1 External APIs
**Binance Exchange via CCXT:**
- **Purpose:** Market data acquisition
- **Authentication:** None required for public data
- **Rate Limits:** 1200ms per request (enforced via CCXT)
- **Error Handling:** Network retry with exponential backoff

### 9.2 Internal Interfaces
**Model to Helper Functions:**
- Model outputs → numpy arrays
- Helper functions consume predictions → generate signals

**Data Pipeline:**
- CSV → Pandas DataFrame → NumPy arrays → TensorFlow tensors

### 9.3 Future Integrations (Planned)
- **Secrets Management:** AWS Secrets Manager / HashiCorp Vault
- **Order Execution:** Live trading via exchange APIs
- **Monitoring:** Prometheus + Grafana
- **Alerting:** PagerDuty / Slack webhooks

---

## 10. Assumptions, Dependencies, and Constraints

### 10.1 Assumptions
- Internet connectivity available for API access
- Historical data representative of future market behavior
- Model predictions have limited time horizon relevance
- User has Python 3.10+ environment with GPU support

### 10.2 Dependencies
- **Python Version:** 3.10+
- **Core Libraries:** TensorFlow, CCXT, NumPy, Pandas, Scikit-learn
- **Total Packages:** 217 packages in `reqirements.txt` (note: filename typo in repo)
- **Hardware:** GPU recommended for training (CUDA-compatible)

### 10.3 Known Limitations
- **Backtrader Integration Not Implemented:** Current system uses custom backtesting logic within `inference.ipynb` instead of Backtrader library
- **No Automated Retraining Pipeline:** Model retraining requires manual execution
- **Single Trading Pair Support:** Only BTC/USDT currently supported
- **No Live Trading Execution:** System provides signals only; no order execution
- **Manual API Key Management:** No secrets management system integration
- **No Real-time Data Streaming:** Relies on periodic batch fetching

### 10.4 Future Enhancements
- **Multi-Pair Support:** Extend to ETH/USDT and other altcoins
- **Automated Retraining Pipeline:** Scheduled model updates with performance monitoring
- **Live Trading Execution Module:** Order management and execution integration
- **Secrets Management Integration:** AWS Secrets Manager or HashiCorp Vault
- **Real-time Streaming Data:** WebSocket integration for live price feeds
- **Advanced Backtesting with Backtrader:** Full Backtrader library integration
- **Web-based Monitoring Dashboard:** Real-time visualization and control
- **Model Explainability Features:** SHAP values, attention visualization
- **Ensemble Models:** Combine multiple model architectures
- **Advanced Risk Management:** Portfolio-level risk controls

---

## 11. File Structure and Documentation

### 11.1 Repository Structure
```
neural_trade/
├── model.py                              # Neural network model (~2,162 lines)
├── dw_ccxt.py                            # Data fetching via CCXT (~297 lines)
├── helper_functions.py                   # Trading strategy helpers (8 functions, ~84 lines)
├── inference.ipynb                       # Training and inference notebook
├── reqirements.txt                       # Python dependencies (217 packages)
├── binance_btcusdt_1min_ccxt.csv         # Historical market data cache
├── nn_learnable_indicators.weights.h5    # Trained model weights (~3.2 MB)
├── scaler_input.joblib                   # Input data scaler
├── scaler.joblib                         # Output data scaler
├── training_log.csv                      # Epoch-wise training metrics
├── indicator_params_history.csv          # Learnable indicator evolution
└── SRS.md                                # This requirements document
```

### 11.2 Documentation Structure
- **SRS.md:** Complete system requirements (this document)
- **Inline Code Documentation:** Docstrings in all modules
- **Notebook Documentation:** Markdown cells in `inference.ipynb`

### 11.3 Document Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-01-02 | AI System | Initial SRS document creation |
| 1.1 | 2026-01-02 | AI System | Complete revision: Added missing modules, data models, testing specs, deployment guide, configuration management, and future enhancements |

---

## 12. Configuration Management

### 12.1 Configuration File

**Current Implementation:**
Configuration is defined as a Python class in `model.py`:
```python
class Config:
    CSV_PATH = "Bitcoin_BTCUSDT.csv"
    LOOKBACK = 60
    BATCH_SIZE = 144
    EPOCHS = 40
    LR = 1e-3
    # ... additional parameters
```

**Future Enhancement:**
Externalize configuration to YAML/JSON format:
```yaml
# config.yaml
data:
  csv_path: "binance_btcusdt_1min_ccxt.csv"
  lookback: 60
  window_step: 1
  
training:
  batch_size: 144
  epochs: 40
  learning_rate: 0.001
  patience: 40
  
model:
  ma_spans: [5, 15, 30]
  rsi_periods: [9, 21, 30]
  # ... additional parameters
```

**Environment Variable Overrides:**
```bash
export NEURAL_TRADE_EPOCHS=50
export NEURAL_TRADE_BATCH_SIZE=256
export NEURAL_TRADE_LR=0.0005
```

**Configuration Validation:**
- Validate required fields present
- Check value ranges (e.g., learning_rate > 0)
- Ensure file paths exist
- Validate array dimensions match

### 12.2 Model Versioning Strategy

**Semantic Versioning:**
- **v1.0:** Initial single-output model (deprecated)
- **v2.0:** Multi-horizon with shared heads (deprecated)
- **v3.0:** Independent towers with focal loss (current)

**Version Metadata:**
- Stored in model weights filename: `nn_learnable_indicators_v3.weights.h5`
- Tracked in training logs: `training_log.csv` header
- Embedded in model checkpoint metadata (future)

**Rollback Support:**
- Maintain last 3 model versions on disk
- Version comparison: `v3.0` → `v2.0` → `v1.0`
- Quick rollback command: `cp nn_learnable_indicators_v2.weights.h5 active_model.weights.h5`

**Version Compatibility:**
- Input/output schema documented per version
- Migration scripts for version upgrades
- Backward compatibility testing required

### 12.3 Hyperparameter Tuning

**Tunable Parameters:**

| Parameter | Current Value | Acceptable Range | Impact |
|-----------|---------------|------------------|--------|
| learning_rate | 1e-3 | [1e-5, 1e-2] | Training speed/stability |
| batch_size | 144 | [32, 256] | Memory usage/convergence |
| lookback | 60 | [30, 120] | Model context window |
| focal_alpha | 0.7 | [0.5, 0.9] | Class imbalance handling |
| focal_gamma | 1.0 | [0.0, 5.0] | Hard example focus |
| lambda_dir | 1.0 | [0.1, 10.0] | Direction loss weight |
| lambda_var | 1.0 | [0.1, 10.0] | Variance loss weight |

**Tuning Methodologies:**

1. **Manual Tuning (Current):**
   - Expert-driven parameter adjustment
   - Observe training curves and validation metrics
   - Iterative refinement

2. **Grid Search (Future):**
   - Exhaustive search over parameter grid
   - Best for 2-3 parameters
   - Computationally expensive

3. **Bayesian Optimization (Future):**
   - Sample-efficient search
   - Libraries: Optuna, Hyperopt
   - Recommended for production

**Tuning Workflow:**
1. Define parameter search space
2. Set evaluation metric (e.g., val_dir_mcc_h1)
3. Run training experiments
4. Select best configuration
5. Validate on hold-out test set
6. Document final hyperparameters

---

## 13. Testing Requirements

### 13.1 Unit Testing

**Test Coverage:**
- All helper functions in `helper_functions.py` shall have unit tests
- Data validation logic shall be tested with edge cases
- Model inference shall be tested with synthetic inputs

**Example Test Cases:**

**Test: calculate_confidence**
```python
def test_calculate_confidence():
    assert calculate_confidence(0.0) == pytest.approx(1.0)
    assert calculate_confidence(1.0) == pytest.approx(0.5)
    assert calculate_confidence(float('inf')) == pytest.approx(0.0)
```

**Test: detect_variance_spike**
```python
def test_detect_variance_spike():
    assert detect_variance_spike(10.0, 2.0, 1.0, spike_threshold=2.0) == True
    assert detect_variance_spike(3.0, 2.0, 1.0, spike_threshold=2.0) == False
```

**Test: calculate_profit_targets**
```python
def test_calculate_profit_targets():
    result = calculate_profit_targets(100.0, [102.0, 105.0, 110.0])
    assert result['tp1'] == 102.0
    assert result['tp1_pct'] == pytest.approx(2.0)
```

**Edge Cases:**
- Zero and negative values
- NaN and infinity handling
- Empty arrays
- Division by zero protection

### 13.2 Integration Testing

**Test Scenarios:**

**End-to-End Data Pipeline:**
1. Fetch sample data via `dw_ccxt.py`
2. Load and validate CSV
3. Preprocess and generate features
4. Load model and scalers
5. Generate predictions
6. Produce trading signals
7. Verify output format

**Backtesting Integration:**
1. Load historical data
2. Run model inference
3. Simulate trades with Trade dataclass
4. Calculate performance metrics
5. Verify PnL calculations

**Model Loading and Persistence:**
1. Train model on synthetic data
2. Save model weights and scalers
3. Clear memory
4. Load saved artifacts
5. Verify predictions match

### 13.3 Performance Testing

**Benchmarks:**

**Inference Latency:**
- **Requirement:** <1 second per prediction
- **Test:** Measure prediction time for single 60-minute sequence
- **Target:** Mean <500ms, P95 <1000ms

**Memory Usage:**
- **Requirement:** <16GB during training
- **Test:** Monitor memory with `psutil` during full training run
- **Target:** Peak usage <14GB (allow 2GB buffer)

**API Rate Limiting:**
- **Test:** Fetch 10,000 candles from CCXT
- **Verify:** No rate limit errors (429)
- **Measure:** Effective requests per minute <50

**Training Time:**
- **Target:** <4 hours for 40 epochs on GPU
- **Test:** Full training run on standard dataset
- **Measure:** Time per epoch, total training time

### 13.4 Test Coverage Requirements

**Coverage Targets:**
- **Critical Modules:** Minimum 70% code coverage
- **Helper Functions:** 100% coverage (all 8 functions)
- **Public APIs:** 100% coverage (all exported functions)
- **Data Validation:** 100% coverage

**Testing Tools:**
- **Framework:** pytest
- **Coverage:** pytest-cov
- **Mocking:** pytest-mock
- **Fixtures:** Shared test data

**Continuous Testing:**
- Run tests on every commit
- Block merge if tests fail
- Generate coverage reports
- Track coverage trends

---

## 14. Deployment and Operations

### 14.1 Container Deployment

**Docker Image Specification:**
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY reqirements.txt .
RUN pip install --no-cache-dir -r reqirements.txt

# Copy application code
COPY model.py dw_ccxt.py helper_functions.py ./
COPY inference.ipynb ./

# Download model weights or mount as volume
# COPY nn_learnable_indicators.weights.h5 ./
# or: VOLUME /app/models

# Environment variables for configuration
ENV NEURAL_TRADE_MODEL_PATH=/app/nn_learnable_indicators.weights.h5
ENV NEURAL_TRADE_SCALER_INPUT=/app/scaler_input.joblib
ENV NEURAL_TRADE_SCALER=/app/scaler.joblib

# Health check endpoint (future API)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#   CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "model.py"]
```

**Volume Mounts:**
- `/app/models` - Model weights and scalers
- `/app/data` - Historical data cache
- `/app/logs` - Training and inference logs

**Environment Variables:**
- `NEURAL_TRADE_MODEL_PATH` - Path to model weights
- `NEURAL_TRADE_DATA_PATH` - Path to market data CSV
- `BINANCE_API_KEY` - Exchange API key (future)
- `BINANCE_API_SECRET` - Exchange API secret (future)

### 14.2 Model Serving

**REST API Specification (Future):**

**Base URL:** `http://localhost:8000/api/v1`

**Endpoints:**

**POST /predict**
- **Description:** Generate predictions for recent market data
- **Input:** OHLCV data (last 60 minutes)
- **Output:** Multi-horizon predictions with metadata
- **Rate Limit:** 100 requests/minute

**GET /health**
- **Description:** Health check endpoint
- **Output:** `{"status": "ok", "model_version": "v3.0"}`

**GET /metrics**
- **Description:** Prometheus-compatible metrics
- **Output:** Prediction latency, request count, error rate

**Input Validation:**
- Verify 60 candles provided
- Check OHLC logic validity
- Validate timestamp continuity
- Reject requests with missing fields

**Response Format:**
```json
{
  "timestamp": "2026-01-02T12:34:56Z",
  "symbol": "BTC/USDT",
  "predictions": { ... },
  "signals": { ... },
  "model_version": "v3.0",
  "latency_ms": 245
}
```

### 14.3 Monitoring and Logging

**Prediction Logging:**
```json
{
  "timestamp": "2026-01-02T12:34:56Z",
  "model_version": "v3.0",
  "input_sequence": "<summary>",
  "predictions": { ... },
  "latency_ms": 245,
  "confidence": 0.85
}
```

**Performance Metrics:**
- Prediction latency (P50, P95, P99)
- Throughput (predictions per second)
- Model accuracy over time (online evaluation)
- Variance spike frequency

**Error Tracking:**
- Exception types and frequency
- API error rates (4xx, 5xx)
- Model loading failures
- Data validation failures

**Model Drift Detection:**
- Track prediction distribution over time
- Monitor variance statistics
- Alert on significant distribution shifts
- Compare against validation set statistics

**Alerting Rules:**
- Prediction latency P95 > 2 seconds
- Error rate > 5% over 10 minutes
- Variance spike rate > 20% of predictions
- Model drift score exceeds threshold

### 14.4 Update and Rollback Procedures

**Model Update Process:**
1. Train new model version (e.g., v3.1)
2. Validate on hold-out test set
3. Save new weights with versioned filename
4. Update model path in configuration
5. Gradual rollout (A/B testing)
6. Monitor performance metrics
7. Full deployment if successful

**A/B Testing Strategy:**
- Route 10% of traffic to new model (v3.1)
- Route 90% to current model (v3.0)
- Compare performance metrics over 24-48 hours
- Increase new model traffic if superior
- Rollback if performance degrades

**Rollback Criteria:**
- Prediction accuracy drops >5%
- Latency increases >50%
- Error rate exceeds 10%
- Variance spikes increase >30%
- Manual rollback requested

**Rollback Procedure:**
1. Identify issue with new model version
2. Update configuration to previous version path
3. Restart services (if necessary)
4. Verify rollback successful
5. Investigate root cause
6. Document incident

**Downtime Expectations:**
- Model update: <30 seconds (hot reload)
- Service restart: <2 minutes
- Rollback: <1 minute
- Zero-downtime deployment: Future goal

---

## Appendix A: Glossary

- **CCXT:** Cryptocurrency Exchange Trading Library
- **OHLCV:** Open, High, Low, Close, Volume
- **Horizon:** Future time point for prediction
- **Variance:** Model uncertainty estimate
- **Confidence:** Inverse of variance, [0, 1]
- **MCC:** Matthews Correlation Coefficient
- **Focal Loss:** Loss function focusing on hard examples
- **PnL:** Profit and Loss
- **TP:** Take Profit target
- **SL:** Stop Loss level

---

## Appendix B: References

1. TensorFlow Documentation: https://www.tensorflow.org/
2. CCXT Documentation: https://docs.ccxt.com/
3. Binance API Documentation: https://binance-docs.github.io/apidocs/
4. Focal Loss Paper: Lin et al., "Focal Loss for Dense Object Detection"
5. Time Series Cross-Validation: Bergmeir & Benítez (2012)

---

**End of Document**
