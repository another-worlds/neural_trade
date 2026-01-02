# Software Requirements Specification (SRS)
## Neural Trade - AI-Powered Cryptocurrency Trading System

---

### Document Information
- **Project Name:** Neural Trade
- **Version:** 1.0
- **Date:** January 2, 2026
- **Status:** Draft

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) document describes the functional and non-functional requirements for the Neural Trade system, an AI-powered cryptocurrency trading platform that leverages deep learning for price prediction and automated trading decisions on the Bitcoin/USDT trading pair.

### 1.2 Scope
Neural Trade is a machine learning-based trading system that:
- Fetches real-time and historical cryptocurrency market data from Binance exchange
- Processes market data with learnable technical indicators using neural networks
- Predicts price movements across multiple time horizons (1-minute, 5-minute, 15-minute)
- Provides trading signals with confidence metrics and variance estimates
- Supports backtesting of trading strategies
- Enables automated or semi-automated trading decisions

### 1.3 Intended Audience
This document is intended for:
- Software developers implementing the system
- Data scientists and ML engineers working on model improvements
- Quality assurance teams testing the system
- Project managers overseeing development
- Stakeholders evaluating system capabilities

### 1.4 Definitions, Acronyms, and Abbreviations
- **API**: Application Programming Interface
- **BTC**: Bitcoin
- **CCXT**: CryptoCurrency eXchange Trading library
- **CNN**: Convolutional Neural Network
- **LSTM**: Long Short-Term Memory network
- **MACD**: Moving Average Convergence Divergence
- **MCC**: Matthews Correlation Coefficient
- **ML**: Machine Learning
- **NLL**: Negative Log-Likelihood
- **OHLCV**: Open, High, Low, Close, Volume (market data format)
- **RSI**: Relative Strength Index
- **SRS**: Software Requirements Specification
- **USDT**: Tether (stablecoin pegged to USD)

### 1.5 References
- TensorFlow/Keras Documentation
- CCXT Library Documentation
- Binance API Documentation
- Backtrader Documentation

---

## 2. Overall Description

### 2.1 Product Perspective
Neural Trade is a standalone trading system that interfaces with:
- **Binance Exchange**: Via CCXT library for market data and order execution
- **TensorFlow/Keras**: For neural network model training and inference
- **Backtrader**: For historical strategy backtesting
- **File System**: For model weights, scalers, training logs, and market data storage

### 2.2 Product Features
The system provides the following core features:

1. **Market Data Acquisition**
   - Real-time and historical BTC/USDT 1-minute candlestick data
   - Automatic data validation and quality checks
   - Pagination handling for large historical datasets

2. **Neural Network Model**
   - Multi-horizon price prediction (1min, 5min, 15min)
   - Direction classification (UP/DOWN/NEUTRAL)
   - Uncertainty estimation via variance prediction
   - Learnable technical indicators
   - Multi-head architecture with independent output towers

3. **Trading Strategy Helpers**
   - Confidence calculation from variance
   - Signal strength computation
   - Multi-horizon directional agreement detection
   - Dynamic stop-loss calculation
   - Position sizing based on confidence
   - Profit target calculation from price predictions
   - Variance spike detection for uncertainty management

4. **Backtesting and Validation**
   - Historical strategy performance evaluation
   - Multiple performance metrics tracking
   - Time-series cross-validation support

### 2.3 User Classes and Characteristics

**Quantitative Traders**
- Technical expertise: High
- Domain knowledge: Trading and finance
- Usage: Strategy development and backtesting

**Data Scientists**
- Technical expertise: High
- Domain knowledge: Machine learning
- Usage: Model training, evaluation, and improvement

**Automated Trading Systems**
- Technical expertise: N/A (programmatic)
- Usage: Consuming model predictions for automated execution

### 2.4 Operating Environment
- **Platform**: Linux, Windows, macOS
- **Python Version**: 3.8+
- **Primary Dependencies**:
  - TensorFlow 2.20.0
  - Keras 3.11.3
  - CCXT 4.3.98
  - Pandas 2.2.3
  - NumPy 2.2.6
  - Scikit-learn 1.7.1
  - Backtrader 1.9.78.123

### 2.5 Design and Implementation Constraints
- Must respect Binance API rate limits (1200ms per request minimum)
- Model inference must complete within reasonable time for live trading (<1 second)
- Memory usage must accommodate model weights (~3.2MB) and historical data
- Must handle network interruptions gracefully
- All predictions must include uncertainty estimates

### 2.6 Assumptions and Dependencies
- Stable internet connection for market data access
- Binance exchange availability and API accessibility
- Sufficient computational resources for model training (GPU recommended)
- Historical data quality from Binance is accurate
- Python environment with all required dependencies installed

---

## 3. Functional Requirements

### 3.1 Market Data Acquisition Module

#### FR-1.1: Historical Data Fetching
**Description**: The system shall fetch historical BTC/USDT 1-minute OHLCV data from Binance.

**Inputs**:
- Start date (datetime)
- End date (datetime)
- Symbol (default: BTC/USDT)
- Timeframe (default: 1m)

**Processing**:
- Initialize CCXT Binance connection with rate limiting
- Paginate requests to fetch data in batches of 500 candles
- Handle API rate limits with appropriate delays
- Retry on network errors with exponential backoff

**Outputs**:
- Pandas DataFrame with columns: open, high, low, close, volume
- DateTime index in UTC
- CSV file saved to disk

**Validation**:
- Verify OHLC logic (high ≥ open, high ≥ close, low ≤ open, low ≤ close)
- Check for missing values
- Detect time-series gaps
- Validate positive prices and non-negative volumes

#### FR-1.2: Real-time Data Streaming
**Description**: The system shall support real-time data updates for live trading.

**Requirements**:
- Fetch latest candles at regular intervals
- Append new data to existing dataset
- Maintain data continuity and consistency

#### FR-1.3: Data Quality Validation
**Description**: The system shall validate all fetched market data.

**Validation Checks**:
- No missing values in OHLCV columns
- OHLC relationships are logically consistent
- All prices are positive
- Volumes are non-negative
- Time-series has no unexpected gaps
- Data falls within expected statistical ranges

### 3.2 Neural Network Model Module

#### FR-2.1: Model Architecture
**Description**: The system shall implement a multi-horizon, multi-output neural network.

**Architecture Components**:
- Input layer accepting LOOKBACK=60 timesteps (1 hour of 1-minute data)
- Learnable technical indicator layers
- Feature extraction layers (LSTM/CNN)
- Three independent output towers for horizons: h0=1min, h1=5min, h2=15min
- Nine total outputs (3 price + 3 direction + 3 variance)

**Specifications**:
- Batch size: 144
- Lookback window: 60 minutes
- Window step: 1 minute
- Training epochs: 40
- Learning rate: 1e-3
- Optimizer: Adam

#### FR-2.2: Learnable Technical Indicators
**Description**: The system shall compute technical indicators with learnable parameters.

**Indicators**:
- Moving Averages (MA) with spans [5, 15, 30]
- MACD with multiple settings
- Momentum over periods [1, 2, 3]
- RSI with periods [9, 21, 30]
- Bollinger Bands with periods [10, 20, 50]

**Requirements**:
- Indicator parameters shall be trainable neural network weights
- Gradient flow shall be enabled through indicator calculations
- Parameters shall be constrained to valid ranges

#### FR-2.3: Price Prediction
**Description**: The system shall predict future prices at multiple horizons.

**Outputs**:
- Price prediction for 1 minute ahead (h0)
- Price prediction for 5 minutes ahead (h1)
- Price prediction for 15 minutes ahead (h2)

**Loss Function Components**:
- Point prediction loss (MSE/Huber)
- Local trend loss
- Global trend loss
- Extended trend loss over periods [1, 5, 15]
- Quantile loss for uncertainty

#### FR-2.4: Direction Classification
**Description**: The system shall classify price movement direction for each horizon.

**Classifications**:
- UP: Price expected to increase
- DOWN: Price expected to decrease
- NEUTRAL: Price expected to remain stable (within deadband)

**Requirements**:
- Use focal loss with α=0.7, γ=1.0 for class imbalance
- Compute per-horizon metrics: accuracy, F1, sensitivity, specificity, MCC
- Support direction deadband threshold for neutral classification

#### FR-2.5: Uncertainty Estimation
**Description**: The system shall estimate prediction uncertainty via variance.

**Outputs**:
- Variance estimate for each horizon (h0, h1, h2)

**Loss Function**:
- Negative log-likelihood (NLL) for variance
- Variance penalty to prevent overconfidence

**Post-processing**:
- Convert variance to confidence scores [0, 1]
- Normalize variance relative to rolling statistics
- Detect variance spikes indicating model uncertainty

#### FR-2.6: Model Training
**Description**: The system shall support training on historical data.

**Training Process**:
- Time-series cross-validation with TimeSeriesSplit
- Early stopping based on validation MCC for h1 horizon
- Model checkpoint saving
- Training metrics logging (CSV format)
- Gradient clipping (norm=5.0)

**Data Preprocessing**:
- StandardScaler for input features
- Scaler persistence (joblib)
- Sequence generation with sliding windows

#### FR-2.7: Model Inference
**Description**: The system shall perform fast inference for live trading.

**Requirements**:
- Load pre-trained model weights from disk
- Load fitted scalers for preprocessing
- Accept recent OHLCV data (60 timesteps)
- Return predictions within <1 second
- Provide all 9 outputs (3 price, 3 direction, 3 variance)

### 3.3 Trading Strategy Module

#### FR-3.1: Confidence Calculation
**Description**: The system shall convert variance to confidence scores.

**Formula**:
```
confidence = 1.0 / (1.0 + variance + epsilon)
```

**Requirements**:
- Return values in range [0, 1]
- Higher confidence for lower variance
- Numerical stability with epsilon=1e-7

#### FR-3.2: Signal Strength Computation
**Description**: The system shall combine direction probability and confidence.

**Formula**:
```
signal_strength = direction_probability × confidence
```

**Usage**:
- Unified signal for trade execution decisions
- Higher values indicate stronger trading signals

#### FR-3.3: Multi-Horizon Agreement
**Description**: The system shall detect agreement across prediction horizons.

**Inputs**:
- Price predictions for all horizons [h0, h1, h2]
- Current price
- Agreement threshold (default: 0.67)

**Processing**:
- Count how many horizons predict price increase
- Count how many horizons predict price decrease
- Calculate agreement ratio

**Outputs**:
- Boolean: is_agreed (True if agreement ≥ threshold)
- Float: agreement ratio [0, 1]

#### FR-3.4: Dynamic Stop-Loss Calculation
**Description**: The system shall calculate variance-adjusted stop-loss levels.

**Inputs**:
- Entry price
- Position type (LONG/SHORT)
- Current variance
- Variance rolling mean
- Base stop percentage (default: 2%)
- Max variance multiplier (default: 2.0)

**Formula**:
```
variance_ratio = min(variance / variance_rolling_mean, max_multiplier)
adjustment_factor = 1.0 + variance_ratio
stop_distance = entry_price × base_stop_pct × adjustment_factor
```

**Outputs**:
- Stop-loss price level
- Wider stops for higher variance (higher uncertainty)
- Tighter stops for lower variance (higher confidence)

#### FR-3.5: Position Sizing
**Description**: The system shall recommend position sizes based on confidence.

**Inputs**:
- Confidence score [0, 1]
- Size multipliers: high=1.2, normal=1.0, low=0.6
- Confidence thresholds: high=0.7, low=0.5

**Logic**:
- If confidence > 0.7: use high size (1.2x)
- Else if confidence > 0.5: use normal size (1.0x)
- Else: use low size (0.6x)

**Output**:
- Position size multiplier

#### FR-3.6: Profit Target Calculation
**Description**: The system shall use price predictions as profit targets.

**Inputs**:
- Entry price
- Price predictions [h0, h1, h2]

**Outputs**:
- Three profit targets (TP1, TP2, TP3)
- Percentage gains for each target
- Dictionary with all target information

#### FR-3.7: Variance Spike Detection
**Description**: The system shall detect unusual uncertainty spikes.

**Inputs**:
- Current variance
- Variance rolling mean
- Variance rolling standard deviation
- Spike threshold (default: 2.0)

**Logic**:
```
spike_level = spike_threshold × variance_rolling_mean
is_spike = variance > spike_level
```

**Output**:
- Boolean indicating variance spike
- Used to avoid trading during high uncertainty

### 3.4 Backtesting Module

#### FR-4.1: Strategy Backtesting
**Description**: The system shall support historical strategy evaluation using Backtrader.

**Requirements**:
- Load historical OHLCV data
- Execute trading strategy with model predictions
- Track all trades and positions
- Calculate performance metrics

#### FR-4.2: Performance Metrics
**Description**: The system shall compute comprehensive performance metrics.

**Metrics**:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Number of trades
- Average trade duration

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements

#### NFR-1.1: Model Inference Latency
- Model inference shall complete within 1 second for live trading scenarios
- Batch prediction for backtesting shall process ≥100 samples/second

#### NFR-1.2: Data Fetching Efficiency
- API requests shall respect Binance rate limits (≤1 request per 1.2 seconds)
- Data fetching shall handle pagination efficiently
- Network errors shall trigger retry with exponential backoff (max 4 retries)

#### NFR-1.3: Memory Usage
- Model weights shall not exceed 50MB in memory
- Historical data shall be limited to recent windows to prevent memory overflow
- Maximum sequence count: 2880 (144 × 20)

### 4.2 Reliability Requirements

#### NFR-2.1: Data Validation
- All market data shall pass validation before use in predictions
- Invalid data shall be logged and rejected
- System shall continue operation with fallback mechanisms

#### NFR-2.2: Model Robustness
- Model shall provide uncertainty estimates for all predictions
- System shall detect and flag high-uncertainty conditions
- Gradient clipping shall prevent training instability

#### NFR-2.3: Error Handling
- Network errors shall be caught and logged
- API errors shall trigger appropriate retry logic
- System shall gracefully degrade in case of partial failures

### 4.3 Usability Requirements

#### NFR-3.1: Code Readability
- Code shall follow PEP 8 style guidelines
- Functions shall have clear docstrings
- Configuration shall be centralized in Config class

#### NFR-3.2: Logging and Monitoring
- Training progress shall be logged with metrics
- Data fetching shall provide progress indicators
- Model predictions shall include metadata (timestamp, confidence, etc.)

### 4.4 Security Requirements

#### NFR-4.1: API Key Management
- API keys shall not be hardcoded in source files
- Credentials shall be stored in environment variables or secure vaults
- API keys shall have minimal required permissions

#### NFR-4.2: Data Integrity
- All data fetched from external sources shall be validated
- Model weights shall be verified before loading
- Training logs shall be tamper-evident

### 4.5 Maintainability Requirements

#### NFR-5.1: Modularity
- System shall be organized into distinct modules: data, model, strategy, backtest
- Each module shall have clear interfaces and responsibilities
- Code dependencies shall be minimized

#### NFR-5.2: Version Control
- Model weights shall be versioned (e.g., v1, v2, v3)
- Configuration changes shall be tracked
- Backward compatibility shall be maintained where possible

#### NFR-5.3: Testing
- Critical functions shall have unit tests
- Model outputs shall be validated against known scenarios
- Backtesting results shall be reproducible

### 4.6 Portability Requirements

#### NFR-6.1: Platform Independence
- System shall run on Linux, Windows, and macOS
- Dependencies shall be cross-platform compatible
- File paths shall use platform-agnostic methods

#### NFR-6.2: Deployment
- System shall support containerization (Docker)
- Dependencies shall be specified in requirements.txt
- Installation shall be documented with clear steps

---

## 5. System Features

### 5.1 Learnable Technical Indicators
**Priority**: High
**Description**: Neural network layers that compute traditional technical indicators with trainable parameters.

**Functional Requirements**:
- FR-2.2: Learnable Technical Indicators

**Benefits**:
- Indicators adapt to market conditions during training
- Reduces manual parameter tuning
- Potentially discovers optimal indicator settings

### 5.2 Multi-Horizon Prediction
**Priority**: High
**Description**: Independent prediction towers for short (1min), medium (5min), and long (15min) horizons.

**Functional Requirements**:
- FR-2.3: Price Prediction
- FR-2.4: Direction Classification
- FR-2.5: Uncertainty Estimation

**Benefits**:
- Captures different timeframe dynamics
- Enables multi-timeframe trading strategies
- Provides richer signal information

### 5.3 Uncertainty-Aware Trading
**Priority**: High
**Description**: All predictions include confidence/variance estimates for risk management.

**Functional Requirements**:
- FR-2.5: Uncertainty Estimation
- FR-3.1: Confidence Calculation
- FR-3.4: Dynamic Stop-Loss Calculation
- FR-3.5: Position Sizing
- FR-3.7: Variance Spike Detection

**Benefits**:
- Avoid trading during high uncertainty
- Size positions according to confidence
- Dynamically adjust risk parameters

### 5.4 Automated Data Pipeline
**Priority**: Medium
**Description**: Automated fetching, validation, and preprocessing of market data.

**Functional Requirements**:
- FR-1.1: Historical Data Fetching
- FR-1.2: Real-time Data Streaming
- FR-1.3: Data Quality Validation

**Benefits**:
- Reduces manual data management
- Ensures data quality
- Enables continuous operation

### 5.5 Strategy Backtesting
**Priority**: Medium
**Description**: Comprehensive backtesting framework for strategy validation.

**Functional Requirements**:
- FR-4.1: Strategy Backtesting
- FR-4.2: Performance Metrics

**Benefits**:
- Validate strategies before live deployment
- Optimize parameters
- Understand historical performance

---

## 6. External Interface Requirements

### 6.1 User Interfaces
The system primarily operates via:
- **Jupyter Notebook**: For interactive exploration, training, and analysis
- **Python Scripts**: For automated execution and deployment
- **Command Line**: For model training and backtesting

### 6.2 Hardware Interfaces
- **GPU (Optional)**: NVIDIA GPU with CUDA support for faster training
- **CPU**: Multi-core processor for parallel data processing
- **Storage**: Minimum 10GB for data, models, and logs

### 6.3 Software Interfaces

#### 6.3.1 Binance Exchange API (via CCXT)
- **Interface Type**: REST API
- **Protocol**: HTTPS
- **Data Format**: JSON
- **Purpose**: Market data retrieval
- **Functions Used**:
  - `fetch_ohlcv()`: Retrieve candlestick data
- **Rate Limits**: 1200ms per request

#### 6.3.2 TensorFlow/Keras
- **Interface Type**: Python Library
- **Version**: TensorFlow 2.20.0, Keras 3.11.3
- **Purpose**: Neural network model implementation
- **Functions Used**:
  - Model definition and training
  - Custom layers and loss functions
  - Model persistence (save/load weights)

#### 6.3.3 Backtrader
- **Interface Type**: Python Library
- **Version**: 1.9.78.123
- **Purpose**: Strategy backtesting
- **Functions Used**:
  - Strategy execution simulation
  - Performance analytics

### 6.4 Communication Interfaces
- **Network**: Internet connection required for API access
- **Protocol**: HTTPS for secure API communication
- **Data Transfer**: JSON format for API requests/responses

---

## 7. Data Requirements

### 7.1 Data Models

#### 7.1.1 Market Data (OHLCV)
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| datetime | datetime64[ns, UTC] | Candle timestamp | Index, unique |
| open | float64 | Opening price | > 0 |
| high | float64 | Highest price | ≥ open, ≥ close |
| low | float64 | Lowest price | ≤ open, ≤ close |
| close | float64 | Closing price | > 0 |
| volume | float64 | Trading volume | ≥ 0 |

#### 7.1.2 Model Predictions
| Field | Type | Description | Range |
|-------|------|-------------|-------|
| price_h0 | float32 | Price prediction 1min ahead | > 0 |
| price_h1 | float32 | Price prediction 5min ahead | > 0 |
| price_h2 | float32 | Price prediction 15min ahead | > 0 |
| direction_h0 | int | Direction 1min (0=DOWN, 1=UP) | {0, 1} |
| direction_h1 | int | Direction 5min (0=DOWN, 1=UP) | {0, 1} |
| direction_h2 | int | Direction 15min (0=DOWN, 1=UP) | {0, 1} |
| variance_h0 | float32 | Variance 1min | ≥ 0 |
| variance_h1 | float32 | Variance 5min | ≥ 0 |
| variance_h2 | float32 | Variance 15min | ≥ 0 |

#### 7.1.3 Training Logs
| Field | Type | Description |
|-------|------|-------------|
| epoch | int | Training epoch number |
| loss | float | Training loss |
| val_loss | float | Validation loss |
| dir_acc_h0 | float | Direction accuracy h0 |
| dir_acc_h1 | float | Direction accuracy h1 |
| dir_acc_h2 | float | Direction accuracy h2 |
| dir_mcc_h1 | float | Matthews Correlation Coefficient h1 |
| lr | float | Learning rate |

### 7.2 Data Storage

#### 7.2.1 Model Artifacts
- **Model Weights**: HDF5 format (`.h5`)
  - Path: `nn_learnable_indicators_v3.weights.h5`
  - Size: ~3.2 MB

- **Scalers**: Joblib format (`.joblib`)
  - Input scaler: `scaler_input.joblib`
  - Output scaler: `scaler.joblib`

#### 7.2.2 Market Data
- **Format**: CSV
- **Example**: `binance_btcusdt_1min_ccxt.csv`
- **Size**: ~3.2 MB for 30 days of 1-minute data

#### 7.2.3 Training Logs
- **Format**: CSV
- **Path**: `training_log.csv`
- **Contents**: Epoch-wise metrics

#### 7.2.4 Indicator Parameters History
- **Format**: CSV
- **Path**: `indicator_params_history.csv`
- **Contents**: Learnable indicator parameters over training

### 7.3 Data Retention
- **Training Data**: Keep latest 30-60 days for model updates
- **Model Weights**: Keep last 3 versions for rollback capability
- **Training Logs**: Retain indefinitely for analysis
- **Market Data**: Archive older data to compressed formats

---

## 8. System Architecture

### 8.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Neural Trade System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐       ┌──────────────────┐            │
│  │  Data Module    │       │  Model Module    │            │
│  │  (dw_ccxt.py)   │──────▶│  (model.py)      │            │
│  │                 │       │                  │            │
│  │ - CCXT client   │       │ - Architecture   │            │
│  │ - Data fetch    │       │ - Training       │            │
│  │ - Validation    │       │ - Inference      │            │
│  └─────────────────┘       │ - Learnable TAs  │            │
│           │                └──────────────────┘            │
│           │                         │                      │
│           ▼                         ▼                      │
│  ┌─────────────────────────────────────────┐              │
│  │       Strategy Module                    │              │
│  │       (helper_functions.py)              │              │
│  │                                           │              │
│  │ - Confidence calc                        │              │
│  │ - Signal strength                        │              │
│  │ - Stop-loss calc                         │              │
│  │ - Position sizing                        │              │
│  │ - Multi-horizon agreement                │              │
│  └─────────────────────────────────────────┘              │
│                         │                                  │
│                         ▼                                  │
│  ┌─────────────────────────────────────────┐              │
│  │       Backtesting Module                 │              │
│  │       (Backtrader)                       │              │
│  │                                           │              │
│  │ - Strategy execution                     │              │
│  │ - Performance metrics                    │              │
│  │ - Visualization                          │              │
│  └─────────────────────────────────────────┘              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│  Binance API     │                  │  File System     │
│  (via CCXT)      │                  │                  │
│                  │                  │ - Model weights  │
│ - Market data    │                  │ - Scalers        │
│ - Order exec     │                  │ - Training logs  │
└──────────────────┘                  │ - Market data    │
                                       └──────────────────┘
```

### 8.2 Data Flow Diagram

```
1. Data Acquisition Flow:
   Binance API → CCXT → Validation → DataFrame → CSV Storage

2. Training Flow:
   CSV Data → Preprocessing → Scaler → Sequences → Model Training
   → Weights (.h5) + Scalers (.joblib) + Training Log (.csv)

3. Inference Flow:
   Recent Data → Scaler Transform → Model Inference
   → [Price, Direction, Variance] × 3 horizons

4. Strategy Flow:
   Model Predictions → Confidence Calc → Signal Strength
   → Trading Decision → Stop-Loss/TP Levels → Position Size

5. Backtesting Flow:
   Historical Data → Model Predictions → Strategy Execution
   → Backtrader → Performance Metrics
```

---

## 9. Quality Attributes

### 9.1 Accuracy
- Direction prediction accuracy target: >55% (h1 horizon)
- Price prediction MAPE target: <2% (h1 horizon)
- Uncertainty calibration: variance should correlate with actual error

### 9.2 Robustness
- Handle missing data gracefully
- Detect and flag anomalous market conditions
- Provide fallback predictions in edge cases

### 9.3 Scalability
- Support multiple trading pairs (future enhancement)
- Handle increasing historical data volumes
- Parallelize backtesting across multiple strategies

### 9.4 Transparency
- Model predictions include explainable components (indicators)
- All decisions logged with rationale
- Training metrics fully tracked

---

## 10. Constraints and Limitations

### 10.1 Technical Constraints
- Limited to markets with sufficient liquidity and data availability
- Model performance depends on market regime (may degrade in unprecedented conditions)
- GPU recommended but not required (CPU training is slower)

### 10.2 Regulatory Constraints
- Trading subject to local regulations
- Must comply with exchange terms of service
- Not intended as financial advice

### 10.3 Known Limitations
- Model trained on historical data; past performance ≠ future results
- Predictions are probabilistic, not deterministic
- High volatility periods may exceed model's uncertainty estimates
- Slippage and execution delays not fully modeled in backtesting

---

## 11. Appendices

### 11.1 Model Configuration Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| LOOKBACK | 60 | Input window (1 hour of 1-min data) |
| BATCH_SIZE | 144 | Training batch size |
| EPOCHS | 40 | Maximum training epochs |
| LR | 1e-3 | Learning rate |
| HORIZON_STEPS | [1, 5, 15] | Prediction horizons (minutes) |
| FOCAL_ALPHA | 0.7 | Focal loss class weight |
| FOCAL_GAMMA | 1.0 | Focal loss focus parameter |
| GRAD_CLIP_NORM | 5.0 | Gradient clipping threshold |

### 11.2 File Structure

```
neural_trade/
├── model.py                              # Neural network model definition & training
├── dw_ccxt.py                            # Data fetching from Binance via CCXT
├── helper_functions.py                   # Trading strategy helper functions
├── inference.ipynb                       # Inference and backtesting notebook
├── requirements.txt                      # Python dependencies
├── nn_learnable_indicators_v3.weights.h5 # Trained model weights
├── scaler_input.joblib                   # Input data scaler
├── scaler.joblib                         # Output data scaler
├── training_log.csv                      # Training metrics history
├── indicator_params_history.csv          # Indicator parameter evolution
└── binance_btcusdt_1min_ccxt.csv        # Market data cache
```

### 11.3 Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.20.0 | Deep learning framework |
| keras | 3.11.3 | High-level neural network API |
| ccxt | 4.3.98 | Cryptocurrency exchange API |
| pandas | 2.2.3 | Data manipulation |
| numpy | 2.2.6 | Numerical computing |
| scikit-learn | 1.7.1 | ML utilities and metrics |
| backtrader | 1.9.78.123 | Backtesting framework |
| plotly | 6.3.0 | Visualization |
| joblib | 1.5.1 | Model persistence |

---

## Document Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-01-02 | AI System | Initial SRS document creation |

---

**End of Document**
