import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

print("="*80)
print("FETCHING LIVE MARKET DATA FROM CCXT (BINANCE)")
print("="*80)

try:
    # Initialize Binance exchange via CCXT
    print("\nInitializing CCXT Binance connection...")
    binance = ccxt.binance({
        'enableRateLimit': True,
        'rateLimit': 1200,  # 1200ms per request to respect rate limits
    })
    
    # Verify exchange capabilities
    print(f"Exchange: {binance.name}")
    print(f"Available timeframes: {binance.timeframes}")
    print(f"Rate limit: {binance.rateLimit}ms")
    
    # ============================================================================
    # CONFIGURATION: Set date range for historical data
    # ============================================================================
    
    # For testing, we'll fetch the same date range as our model's test set
    # Adjust these dates based on your Bitcoin_BTCUSDT.csv date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)  # Fetch 30 days of 1-min data
    
    print(f"\nData Fetch Configuration:")
    print(f"  Pair: BTC/USDT")
    print(f"  Timeframe: 1m (1-minute candles)")
    print(f"  Start Date: {start_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  End Date:   {end_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Duration:   {(end_date - start_date).days} days")
    
    # Note: 1-minute data for 30 days = ~43,200 candles (30*24*60)
    # CCXT API limits: typically 500-1000 candles per request
    # So we need multiple requests with pagination
    
    # ============================================================================
    # FETCH DATA WITH PAGINATION
    # ============================================================================
    
    symbol = 'BTC/USDT'
    timeframe = '1m'
    batch_size = 500  # Candles per request (Binance limit is 1000, but safer with 500)
    
    print(f"\nFetching {symbol} 1-minute candles with pagination...")
    print(f"  Batch size: {batch_size} candles per request")
    
    all_candles = []
    current_timestamp = int(start_date.timestamp() * 1000)  # Convert to milliseconds
    end_timestamp = int(end_date.timestamp() * 1000)
    
    request_count = 0
    last_candle_time = None
    
    while current_timestamp < end_timestamp:
        try:
            # Fetch batch of candles
            candles = binance.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=batch_size)
            
            if not candles:
                print(f"  ⚠️  No more candles available (stopped at {datetime.fromtimestamp(current_timestamp/1000)})")
                break
            
            all_candles.extend(candles)
            request_count += 1
            
            # Update timestamp for next batch (start from last candle's time)
            last_candle_time = candles[-1][0]
            current_timestamp = last_candle_time + 60000  # Add 1 minute (60000ms) to avoid duplicates
            
            # Print progress
            batch_time = datetime.fromtimestamp(last_candle_time / 1000)
            print(f"  Request {request_count:3d}: Fetched {len(candles):3d} candles | "
                  f"Last candle: {batch_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Total: {len(all_candles):6d} candles")
            
            # Rate limiting - be nice to the API
            time.sleep(0.1)
            
        except ccxt.NetworkError as e:
            print(f"  ❌ Network error: {e}")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"  ❌ Exchange error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            break
    
    print(f"\n✅ Data fetch complete!")
    print(f"  Total requests: {request_count}")
    print(f"  Total candles fetched: {len(all_candles):,}")
    print(f"  Date range: {datetime.fromtimestamp(all_candles[0][0]/1000)} to {datetime.fromtimestamp(all_candles[-1][0]/1000)}")
    
    # ============================================================================
    # CONVERT TO PANDAS DATAFRAME
    # ============================================================================
    
    print("\nConverting to pandas DataFrame...")
    
    df_ccxt = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp to datetime
    df_ccxt['datetime'] = pd.to_datetime(df_ccxt['timestamp'], unit='ms', utc=True)
    df_ccxt = df_ccxt.set_index('datetime')
    df_ccxt = df_ccxt.drop('timestamp', axis=1)
    
    # Sort by datetime (should be already sorted, but let's be safe)
    df_ccxt = df_ccxt.sort_index()
    
    # Rename columns to lowercase for consistency
    df_ccxt.columns = [col.lower() for col in df_ccxt.columns]
    
    print(f"\nDataFrame created:")
    print(f"  Shape: {df_ccxt.shape}")
    print(f"  Columns: {list(df_ccxt.columns)}")
    print(f"  Data types:\n{df_ccxt.dtypes}")
    
    # ============================================================================
    # DATA VALIDATION & QUALITY CHECKS
    # ============================================================================
    
    print("\n" + "="*80)
    print("DATA VALIDATION & QUALITY CHECKS")
    print("="*80)
    
    print(f"\nPrice Statistics (USDT):")
    print(f"  Close - Min: ${df_ccxt['close'].min():.2f}, Max: ${df_ccxt['close'].max():.2f}, "
          f"Mean: ${df_ccxt['close'].mean():.2f}")
    print(f"  High  - Min: ${df_ccxt['high'].min():.2f}, Max: ${df_ccxt['high'].max():.2f}")
    print(f"  Low   - Min: ${df_ccxt['low'].min():.2f}, Max: ${df_ccxt['low'].max():.2f}")
    print(f"  Open  - Min: ${df_ccxt['open'].min():.2f}, Max: ${df_ccxt['open'].max():.2f}")
    
    print(f"\nVolume Statistics (BTC):")
    print(f"  Min: {df_ccxt['volume'].min():.6f}, Max: {df_ccxt['volume'].max():.6f}, "
          f"Mean: {df_ccxt['volume'].mean():.6f}")
    
    # Check for missing values
    print(f"\nMissing Values:")
    missing = df_ccxt.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("  ✅ No missing values")
    
    # Check for data gaps (missing 1-minute candles)
    time_diffs = df_ccxt.index.to_series().diff().dt.total_seconds() / 60
    gaps = time_diffs[time_diffs > 1.5]  # Allow 1.5 minute tolerance for gaps
    
    if len(gaps) > 0:
        print(f"\n⚠️  Found {len(gaps)} potential gaps in time series:")
        for idx, gap in gaps.items():
            print(f"    Gap of {gap:.1f} minutes before {idx}")
    else:
        print(f"\n✅ No gaps detected in time series")
    
    # Check OHLC logic (High >= max(Open,Close), Low <= min(Open,Close), etc.)
    ohlc_valid = (
        (df_ccxt['high'] >= df_ccxt['open']) &
        (df_ccxt['high'] >= df_ccxt['close']) &
        (df_ccxt['low'] <= df_ccxt['open']) &
        (df_ccxt['low'] <= df_ccxt['close']) &
        (df_ccxt['high'] >= df_ccxt['low'])
    )
    
    if ohlc_valid.all():
        print(f"✅ OHLC logic valid for all {len(df_ccxt)} candles")
    else:
        invalid_count = (~ohlc_valid).sum()
        print(f"❌ OHLC logic violated in {invalid_count} candles!")
    
    # Check for negative or zero prices/volumes
    if (df_ccxt[['open', 'high', 'low', 'close']] > 0).all().all():
        print(f"✅ All prices are positive")
    else:
        print(f"❌ Found non-positive prices!")
    
    if (df_ccxt['volume'] >= 0).all():
        print(f"✅ All volumes are non-negative")
    else:
        print(f"❌ Found negative volumes!")
    
    # ============================================================================
    # SAMPLE DATA PREVIEW
    # ============================================================================
    
    print("\n" + "="*80)
    print("DATA PREVIEW (First & Last 5 Rows)")
    print("="*80)
    
    print("\nFirst 5 candles:")
    print(df_ccxt.head().to_string())
    
    print("\n\nLast 5 candles:")
    print(df_ccxt.tail().to_string())
    
    # ============================================================================
    # STORE FOR BACKTRADER USE
    # ============================================================================
    
    print("\n" + "="*80)
    print("PREPARING DATA FOR BACKTRADER")
    print("="*80)
    
    # This DataFrame is ready for backtrader
    # Store it globally so it can be used in the backtrader cell
    ccxt_backtest_data = df_ccxt.copy()
    
    # Also save to CSV for future reference
    csv_save_path = 'binance_btcusdt_1min_ccxt.csv'
    ccxt_backtest_data.to_csv(csv_save_path)
    print(f"\n✅ Data saved to: {csv_save_path}")
    
    print(f"\n✅ CCXT data is ready for backtrader!")
    print(f"  DataFrame name: ccxt_backtest_data")
    print(f"  Shape: {ccxt_backtest_data.shape}")
    print(f"  Date range: {ccxt_backtest_data.index[0]} to {ccxt_backtest_data.index[-1]}")
    print(f"  Ready to use in backtrader strategy!")
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    print("\n" + "="*80)
    print("Creating data visualization...")
    print("="*80)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('BTC/USDT Price (1-min candles)', 'Volume (1-min candles)'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Plot close prices (last 500 candles for clarity)
    sample_data = ccxt_backtest_data.tail(500)
    
    fig.add_trace(
        go.Scatter(x=sample_data.index, y=sample_data['close'],
                   mode='lines', name='Close Price',
                   line=dict(color='blue', width=1.5)),
        row=1, col=1
    )
    
    # Add candlestick pattern (simplified with close line + high/low range)
    fig.add_trace(
        go.Scatter(x=sample_data.index, y=sample_data['high'],
                   fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                   showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=sample_data.index, y=sample_data['low'],
                   fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                   fillcolor='rgba(0,100,200,0.2)', name='High-Low Range',
                   showlegend=False),
        row=1, col=1
    )
    
    # Plot volume
    colors = ['green' if sample_data['close'].iloc[i] >= sample_data['open'].iloc[i] else 'red' 
              for i in range(len(sample_data))]
    
    fig.add_trace(
        go.Bar(x=sample_data.index, y=sample_data['volume'],
               marker_color=colors, name='Volume', showlegend=False),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (BTC)", row=2, col=1)
    
    fig.update_layout(
        height=700, width=1400,
        template='plotly_dark',
        title_text=f"BTC/USDT 1-Minute Candles from CCXT Binance ({len(ccxt_backtest_data):,} candles)"
    )
    
    display(fig)
    
    print("\n✅ Data fetch and validation complete!")
    
except Exception as e:
    print(f"\n❌ CCXT data fetch failed: {e}")
    import traceback
    traceback.print_exc()
