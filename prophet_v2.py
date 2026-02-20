# IMPORTANT: Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web servers


import base64
from io import BytesIO
# Import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Try to import plotly early so Prophet doesn't print its default warning.
try:
    import plotly
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False
    print("Optional: plotly not installed — interactive Prophet plots will be disabled.\nInstall with: pip install plotly")

from prophet import Prophet
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import datetime
import os
import traceback
import numpy as np
try:
    from textblob import TextBlob
    def get_text_sentiment(text):
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return 0.0
except Exception:
    TextBlob = None
    def get_text_sentiment(text):
        # TextBlob not available; return neutral sentiment
        return 0.0
import warnings
warnings.filterwarnings('ignore')

# Optional Gaff pattern recognition module (user-provided). Import dynamically if available.
import importlib
import os
import pkgutil

# Try a list of common candidate module names first, then scan workspace for any
# .py files containing 'gaff' in their name so custom modules like
# 'gaff_pattern_reconiton.py' are picked up.
gaff_pattern_module = None
_HAS_GAFF = False
candidate_names = ['gaff_pattern', 'gaff_pattern_reconiton', 'gaff_pattern_recognition', 'gaff_pattern_reconition', 'gaff']
for name in candidate_names:
    try:
        gaff_pattern_module = importlib.import_module(name)
        _HAS_GAFF = True
        break
    except Exception:
        gaff_pattern_module = None

if not _HAS_GAFF:
    # Scan current working directory for any Python files with 'gaff' in the filename
    try:
        cwd_files = os.listdir('.')
        for fname in cwd_files:
            if fname.lower().endswith('.py') and 'gaff' in fname.lower():
                mod_name = os.path.splitext(fname)[0]
                try:
                    # prefer already-imported modules
                    if mod_name in globals():
                        gaff_pattern_module = globals()[mod_name]
                        _HAS_GAFF = True
                        break
                    gaff_pattern_module = importlib.import_module(mod_name)
                    _HAS_GAFF = True
                    break
                except Exception:
                    gaff_pattern_module = None
    except Exception:
        gaff_pattern_module = None
        _HAS_GAFF = False

# Disable interactive plotting
plt.ioff()

def clean_static_dir_folder():
    """Clean old files from static_dir folder to prevent accumulation"""
    try:
        static_dir = 'static'
        if os.path.exists(static_dir):
            for filename in os.listdir(static_dir):
                if filename.endswith(('.png', '.txt')):
                    file_path = os.path.join(static_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed old file: {file_path}")
                    except Exception as e:
                        print(f"Could not remove {file_path}: {e}")
    except Exception as e:
        print(f"Error cleaning static_dir folder: {e}")

def safe_extract_value(series_value):
    """Safely extract scalar value from pandas Series or scalar"""
    try:
        if isinstance(series_value, (int, float, np.integer, np.floating)):
            return float(series_value) if not pd.isna(series_value) else 0.0
        
        if isinstance(series_value, pd.Series):
            if series_value.empty:
                return 0.0
            val = series_value.iloc[0]
            return float(val) if not pd.isna(val) else 0.0
        
        if hasattr(series_value, 'item'):
            val = series_value.item()
            return float(val) if not pd.isna(val) else 0.0
        
        val = float(series_value)
        return val if not pd.isna(val) else 0.0
        
    except (ValueError, TypeError, AttributeError):
        return 0.0

def generate_sentiment_scores(data):
    """
    Generate synthetic sentiment scores based on price action and volatility
    In production, replace with actual news/social media sentiment analysis
    """
    print("Generating sentiment analysis...")
    
    # Calculate price momentum
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=14).std()
    
    # Create sentiment score based on multiple factors
    sentiment_scores = []
    
    for i in range(len(data)):
        if i < 14:
            sentiment_scores.append(0.0)  # Neutral for initial period
            continue
            
        # Factor 1: Price momentum (30% weight)
        returns_14d = data['Returns'].iloc[i-14:i].mean()
        momentum_score = np.tanh(returns_14d * 100) * 0.3
        
        # Factor 2: Volume trend (20% weight)
        vol_ratio = data['Volume'].iloc[i] / data['Volume'].iloc[i-14:i].mean()
        volume_score = (np.tanh((vol_ratio - 1) * 2) * 0.2)
        
        # Factor 3: Volatility (negative indicator, 20% weight)
        vol_norm = data['Volatility'].iloc[i] / data['Volatility'].iloc[i-30:i].mean()
        volatility_score = -np.tanh((vol_norm - 1) * 2) * 0.2
        
        # Factor 4: Price vs MA (30% weight)
        ma_20 = data['Close'].iloc[i-20:i].mean()
        price_ma_ratio = (data['Close'].iloc[i] - ma_20) / ma_20
        ma_score = np.tanh(price_ma_ratio * 10) * 0.3
        
        # Combined sentiment (-1 to 1)
        total_sentiment = momentum_score + volume_score + volatility_score + ma_score
        sentiment_scores.append(total_sentiment)
    
    data['sentiment'] = sentiment_scores
    
    # Smooth sentiment
    data['sentiment_smooth'] = data['sentiment'].rolling(window=5, min_periods=1).mean()
    # Ensure no NaNs remain in the smoothed sentiment (fill and clip)
    data['sentiment_smooth'] = data['sentiment_smooth'].fillna(method='ffill').fillna(0.0)
    data['sentiment_smooth'] = data['sentiment_smooth'].clip(-1.0, 1.0)

    return data

def analyze_stock_with_sentiment(ticker="PGEL.NS"):
    try:
        clean_static_dir_folder()
        
        output_static_dir = 'static'
        os.makedirs(output_static_dir, exist_ok=True)

        # Download historical data
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, period="3y", auto_adjust=True)
        
        if data.empty:
            return {
                'success': False,
                'error': f"No data found for ticker {ticker}",
                'ticker': ticker
            }

        data = data.reset_index()
        if 'Date' not in data.columns and 'index' in data.columns:
            data = data.rename(columns={'index': 'Date'})

        # Generate sentiment scores
        data = generate_sentiment_scores(data)

        # Calculate technical indicators early so they can be used as regressors
        # Ensure series are 1-D pandas Series
        def as_1d_series_local(x):
            if isinstance(x, pd.DataFrame):
                s = x.iloc[:, 0]
            else:
                s = x
            if not isinstance(s, pd.Series):
                s = pd.Series(s)
            try:
                s = s.astype(float)
            except Exception:
                s = s.apply(pd.to_numeric, errors='coerce')
            s.index = data.index
            return s

        close_s = as_1d_series_local(data['Close'])
        high_s = as_1d_series_local(data['High'])
        low_s = as_1d_series_local(data['Low'])
        vol_s = as_1d_series_local(data['Volume'])

        # Indicators used as regressors
        data['RSI_14'] = RSIIndicator(close_s, window=14).rsi()
        macd = MACD(close_s)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()

        bb = BollingerBands(close_s)
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Middle'] = bb.bollinger_mavg()
        try:
            vwap_series = VolumeWeightedAveragePrice(
                high=high_s,
                low=low_s,
                close=close_s,
                volume=vol_s,
                window=20
            ).volume_weighted_average_price()
            data['VWAP'] = vwap_series
        except Exception:
            data['VWAP'] = close_s

        # Weekly support/resistance levels (week starting date)
        data['WeekStart'] = pd.to_datetime(data['Date']).dt.to_period('W').apply(lambda r: r.start_time)
        data['weekly_resistance'] = data.groupby('WeekStart')['Close'].transform('max')
        data['weekly_support'] = data.groupby('WeekStart')['Close'].transform('min')

        # Prepare data for Prophet WITHOUT sentiment
        df_no_sentiment = data[['Date', 'Close']].copy()
        df_no_sentiment.columns = ['ds', 'y']

        # Prepare data for Prophet WITH sentiment + regressors
        df_with_sentiment = data[['Date', 'Close', 'sentiment_smooth', 'VWAP', 'RSI_14', 'weekly_resistance', 'weekly_support']].copy()
        df_with_sentiment.columns = ['ds', 'y', 'sentiment', 'VWAP', 'RSI_14', 'weekly_resistance', 'weekly_support']

        # Fill any remaining NaNs in the sentiment regressor and ensure numeric
        df_with_sentiment['sentiment'] = df_with_sentiment['sentiment'].astype(float)
        if df_with_sentiment['sentiment'].isna().any():
            mean_sent = float(np.nan_to_num(df_with_sentiment['sentiment'].mean(), nan=0.0))
            df_with_sentiment['sentiment'] = df_with_sentiment['sentiment'].fillna(mean_sent)
        # Drop rows that still have NaNs in critical columns
        df_with_sentiment = df_with_sentiment.dropna(subset=['ds', 'y', 'sentiment'])

        # Train Prophet model WITHOUT sentiment
        print("Training Prophet model without sentiment...")
        model_no_sentiment = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.5,
            seasonality_mode='multiplicative'
        )
        model_no_sentiment.fit(df_no_sentiment)

        # Train Prophet model WITH sentiment
        print("Training Prophet model with sentiment...")
        model_with_sentiment = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.5,
            seasonality_mode='multiplicative'
        )
        # Add regressors: sentiment, VWAP, RSI, weekly support/resistance
        model_with_sentiment.add_regressor('sentiment')
        model_with_sentiment.add_regressor('VWAP')
        model_with_sentiment.add_regressor('RSI_14')
        model_with_sentiment.add_regressor('weekly_resistance')
        model_with_sentiment.add_regressor('weekly_support')

        # Ensure regressors have no NaNs
        for col in ['sentiment', 'VWAP', 'RSI_14', 'weekly_resistance', 'weekly_support']:
            if col in df_with_sentiment.columns:
                df_with_sentiment[col] = df_with_sentiment[col].fillna(method='ffill').fillna(0.0)

        model_with_sentiment.fit(df_with_sentiment)

        # Create future dataframe
        future = model_no_sentiment.make_future_dataframe(periods=66, freq='B')
        
        # For sentiment+regressor model, extend regressors with recent averages
        future_sentiment = future.copy()
        recent_sentiment = float(np.nan_to_num(data['sentiment_smooth'].iloc[-30:].mean(), nan=0.0))
        recent_vwap = float(np.nan_to_num(data['VWAP'].iloc[-30:].mean(), nan=0.0))
        recent_rsi = float(np.nan_to_num(data['RSI_14'].iloc[-30:].mean(), nan=0.0))
        recent_resistance = float(np.nan_to_num(data['weekly_resistance'].iloc[-4:].mean(), nan=0.0))
        recent_support = float(np.nan_to_num(data['weekly_support'].iloc[-4:].mean(), nan=0.0))

        future_sentiment['sentiment'] = recent_sentiment
        future_sentiment['VWAP'] = recent_vwap
        future_sentiment['RSI_14'] = recent_rsi
        future_sentiment['weekly_resistance'] = recent_resistance
        future_sentiment['weekly_support'] = recent_support

        # Make predictions
        forecast_no_sentiment = model_no_sentiment.predict(future)
        forecast_with_sentiment = model_with_sentiment.predict(future_sentiment)

        # Calculate accuracy metrics on historical data by aligning forecasts to actual dates
        comparison_df = pd.DataFrame()
        try:
            # Normalize ds columns and create prediction columns
            hist_no = forecast_no_sentiment[['ds', 'yhat']].copy().rename(columns={'yhat': 'pred_no_sent'})
            hist_no['ds'] = pd.to_datetime(hist_no['ds']).dt.normalize()

            hist_with = forecast_with_sentiment[['ds', 'yhat']].copy().rename(columns={'yhat': 'pred_with_sent'})
            hist_with['ds'] = pd.to_datetime(hist_with['ds']).dt.normalize()

            # Ensure hist predictions are scalar values (handle array-like entries)
            import numpy as _np
            def _scalarize_val(v):
                try:
                    if isinstance(v, (_np.ndarray, list, tuple)):
                        arr = _np.asarray(v)
                        if arr.size == 0:
                            return _np.nan
                        return float(arr.flatten()[0])
                    return float(v)
                except Exception:
                    return _np.nan

            hist_no['pred_no_sent'] = hist_no['pred_no_sent'].apply(_scalarize_val)
            hist_with['pred_with_sent'] = hist_with['pred_with_sent'].apply(_scalarize_val)

            # Ensure we extract 1-D Close series even when DataFrame has MultiIndex columns
            try:
                date_series = pd.to_datetime(data['Date']).dt.normalize()
            except Exception:
                date_series = pd.to_datetime(data.iloc[:, 0]).dt.normalize()

            try:
                close_series = as_1d_series_local(data['Close'])
            except Exception:
                # Fallback: attempt to get first numeric column as close
                close_series = data.select_dtypes(include=['number']).iloc[:, 0]

            actual_df = pd.DataFrame({
                'ds': date_series,
                'Close': close_series.values
            })

            merged = pd.merge(actual_df, hist_no[['ds', 'pred_no_sent']], on='ds', how='inner')
            merged = pd.merge(merged, hist_with[['ds', 'pred_with_sent']], on='ds', how='inner')

            # Coerce any array-like prediction entries to scalars (handle nested arrays)
            import numpy as _np
            def _scalarize(v):
                try:
                    if isinstance(v, (_np.ndarray, list, tuple)):
                        arr = _np.asarray(v)
                        if arr.size == 0:
                            return _np.nan
                        return float(arr.flatten()[0])
                    return v
                except Exception:
                    return _np.nan

            merged['pred_no_sent'] = merged['pred_no_sent'].apply(_scalarize)
            merged['pred_with_sent'] = merged['pred_with_sent'].apply(_scalarize)

            # Ensure numeric and drop invalid rows
            comp = merged[['Close', 'pred_no_sent', 'pred_with_sent']].apply(pd.to_numeric, errors='coerce')
            comp = comp.replace([np.inf, -np.inf], _np.nan).dropna()

            # Keep a copy with dates for plotting and CSV output
            comparison_df = merged.copy()
            try:
                comparison_df['Date'] = pd.to_datetime(comparison_df['ds'])
            except Exception:
                comparison_df['Date'] = comparison_df['ds']

            def compute_mape(df, pred_col):
                if df.empty:
                    return float('nan')
                mask = df['Close'] != 0
                if not mask.any():
                    return float('nan')
                return float(np.mean(np.abs((df.loc[mask, 'Close'] - df.loc[mask, pred_col]) / df.loc[mask, 'Close'])) * 100.0)

            mape_no_sent = compute_mape(comp, 'pred_no_sent')
            mape_with_sent = compute_mape(comp, 'pred_with_sent')
            improvement = (mape_no_sent - mape_with_sent) if (np.isfinite(mape_no_sent) and np.isfinite(mape_with_sent)) else float('nan')

        except Exception as e:
            print(f"MAPE calculation error: {e}")
            mape_no_sent = float('nan')
            mape_with_sent = float('nan')
            improvement = float('nan')
            # Ensure comparison_df exists for downstream plotting even if MAPE failed
            try:
                comparison_df = actual_df.copy()
                comparison_df['Date'] = comparison_df['ds']
            except Exception:
                comparison_df = pd.DataFrame({'ds': pd.to_datetime(data['Date']).dt.normalize(), 'Close': data['Close'].values})
                comparison_df['Date'] = comparison_df['ds']
        
        print(f"MAPE without sentiment: {mape_no_sent:.2f}%")
        print(f"MAPE with sentiment: {mape_with_sent:.2f}%")
        print(f"Improvement: {improvement:.2f}%")

        # Prepare the actual vs predicted dataframe used to compute MAPE
        mape_df = None
        try:
            if 'comp' in locals() and not comp.empty:
                # Use the aligned numeric comparison if available; attach dates when possible
                try:
                    mape_df = comparison_df[['Date', 'Close', 'pred_no_sent', 'pred_with_sent']].copy()
                except Exception:
                    mape_df = comp.copy()
            elif 'merged_try' in locals() and not merged_try.empty:
                # merged_try contains Close, pred_no_sent, pred_with_sent
                mape_df = merged_try.reset_index(drop=True).copy()
            else:
                # fallback to best-available columns from comparison_df
                tmp = comparison_df.copy()
                if 'Date' in tmp.columns and 'pred_no_sent' in tmp.columns and 'pred_with_sent' in tmp.columns:
                    mape_df = tmp[['Date', 'Close', 'pred_no_sent', 'pred_with_sent']].copy()

            # Clean and format mape_df
            if mape_df is not None and not mape_df.empty:
                mape_df = mape_df.replace([np.inf, -np.inf], np.nan).dropna()
                # Ensure Date formatting
                if 'Date' in mape_df.columns:
                    mape_df['Date'] = pd.to_datetime(mape_df['Date']).dt.strftime('%Y-%m-%d')

        except Exception as e:
            print(f"Preparing MAPE dataframe failed: {e}")
            mape_df = None

        # Save MAPE comparison CSV for inspection
        mape_csv_name = None
        mape_sample = []
        try:
            if mape_df is not None and not mape_df.empty:
                mape_csv_name = f"{ticker}_mape_comparison.csv"
                mape_csv_path = os.path.join(output_static_dir, mape_csv_name)
                
                csv_data = mape_df.to_csv(index=False)

                # Provide a small sample for quick inspection
                mape_sample = mape_df.head(20).to_dict(orient='records')
                print(f"Saved MAPE comparison data to: {mape_csv_path}")
        except Exception as e:
            print(f"Failed to save MAPE CSV: {e}")

        # Plotting
        print("Creating enhanced charts...")
        plt.clf()
        plt.close('all')
        
        fig = plt.figure(figsize=(16, 22))

        # 1. Sentiment Score Over Time
        plt.subplot(5, 1, 1)
        plt.plot(data['Date'], data['sentiment_smooth'], label='Sentiment Score', color='purple', linewidth=2)
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.fill_between(data['Date'], data['sentiment_smooth'], 0, 
                         where=(data['sentiment_smooth'] > 0), alpha=0.3, color='green', label='Positive')
        plt.fill_between(data['Date'], data['sentiment_smooth'], 0, 
                         where=(data['sentiment_smooth'] <= 0), alpha=0.3, color='red', label='Negative')
        plt.title(f'{ticker} - Sentiment Analysis (Market Mood)', fontsize=14, fontweight='bold')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Actual vs Predicted (Last 6 months)
        plt.subplot(5, 1, 2)
        recent_comparison = comparison_df.tail(126)  # ~6 months
        plt.plot(recent_comparison['Date'], recent_comparison['Close'], 
                label='Actual Price', color='black', linewidth=2.5)
        if 'pred_no_sent' in recent_comparison.columns:
            plt.plot(recent_comparison['Date'], recent_comparison['pred_no_sent'], 
                label=f'Predicted (No Sentiment) - MAPE: {mape_no_sent:.2f}%', 
                color='orange', linewidth=2, linestyle='--', alpha=0.7)
        if 'pred_with_sent' in recent_comparison.columns:
            plt.plot(recent_comparison['Date'], recent_comparison['pred_with_sent'], 
                label=f'Predicted (With Sentiment) - MAPE: {mape_with_sent:.2f}%', 
                color='green', linewidth=2, linestyle='--', alpha=0.7)
        plt.title('Model Accuracy Comparison (Last 6 Months)', fontsize=14, fontweight='bold')
        plt.ylabel('Price (Rs.)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Prophet Forecast Comparison
        plt.subplot(5, 1, 3)
        plt.plot(data['Date'], data['Close'], label='Historical Price', linewidth=2, color='black')
        plt.plot(forecast_no_sentiment['ds'], forecast_no_sentiment['yhat'], 
                label='Forecast (No Sentiment)', color='orange', linewidth=2, alpha=0.7)
        plt.plot(forecast_with_sentiment['ds'], forecast_with_sentiment['yhat'], 
                label='Forecast (With Sentiment)', color='green', linewidth=2, alpha=0.7)
        plt.fill_between(forecast_with_sentiment['ds'], 
                        forecast_with_sentiment['yhat_lower'], 
                        forecast_with_sentiment['yhat_upper'], 
                        alpha=0.2, color='green')
        # Plot adjusted forecast median (support-aware) as a horizontal guide and annotate
        try:
            if adjusted_forecast_median is not None:
                # Use the last date in the forecast range to place a marker
                last_forecast_date = pd.to_datetime(forecast_with_sentiment['ds'].iloc[-1])
                plt.axhline(adjusted_forecast_median, color='purple', linestyle='--', linewidth=1.5, label='Adjusted Median (support-aware)')
                plt.scatter([last_forecast_date], [adjusted_forecast_median], color='purple', s=40)
                plt.text(last_forecast_date, adjusted_forecast_median, f"  Adjusted: Rs.{adjusted_forecast_median:.2f}", va='center', color='purple')
            # Optionally plot next lower support
            if next_lower_support is not None:
                plt.axhline(next_lower_support, color='red', linestyle=':', linewidth=1, label='Next Lower Support')
        except Exception:
            pass
        # Plot latest weekly support/resistance levels
        try:
            latest_week_resistance = safe_extract_value(data['weekly_resistance'].iloc[-1])
            latest_week_support = safe_extract_value(data['weekly_support'].iloc[-1])
            plt.axhline(latest_week_resistance, color='red', linestyle='--', alpha=0.7, label='Weekly Resistance')
            plt.axhline(latest_week_support, color='green', linestyle='--', alpha=0.7, label='Weekly Support')
        except Exception:
            latest_week_resistance = None
            latest_week_support = None
        plt.title(f'{ticker} - 3-Month Forecast Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Price (Rs.)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. MACD
        plt.subplot(5, 1, 4)
        plt.plot(data['Date'], data['MACD'], label='MACD', color='blue', linewidth=2)
        plt.plot(data['Date'], data['MACD_Signal'], label='Signal', color='orange', linewidth=2)
        plt.bar(data['Date'], data['MACD_Hist'], label='Histogram', color='gray', alpha=0.5, width=1)
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.title('MACD Indicator', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. RSI
        plt.subplot(5, 1, 5)
        plt.plot(data['Date'], data['RSI_14'], label='RSI (14)', color='purple', linewidth=2)
        plt.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        plt.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        plt.axhline(50, color='gray', linestyle='-', alpha=0.3)
        plt.ylim(0, 100)
        plt.title('Relative Strength Index', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_filename = f"{ticker}_sentiment_analysis.png"
        full_plot_path = os.path.join(output_static_dir, plot_filename)
        
        if os.path.exists(full_plot_path):
            os.remove(full_plot_path)
            




        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

        
        plt.close('all')
        print(f"Plot saved to: {full_plot_path}")

        # Get latest values
        latest_close = safe_extract_value(data['Close'].iloc[-1])
        latest_sentiment = safe_extract_value(data['sentiment_smooth'].iloc[-1])
        
        forecast_no_sent_high = safe_extract_value(forecast_no_sentiment['yhat_upper'].iloc[-1])
        forecast_no_sent_low = safe_extract_value(forecast_no_sentiment['yhat_lower'].iloc[-1])
        forecast_no_sent_median = safe_extract_value(forecast_no_sentiment['yhat'].iloc[-1])
        
        forecast_with_sent_high = safe_extract_value(forecast_with_sentiment['yhat_upper'].iloc[-1])
        forecast_with_sent_low = safe_extract_value(forecast_with_sentiment['yhat_lower'].iloc[-1])
        forecast_with_sent_median = safe_extract_value(forecast_with_sentiment['yhat'].iloc[-1])
        
        latest_rsi = safe_extract_value(data['RSI_14'].iloc[-1])
        latest_vwap = safe_extract_value(data['VWAP'].iloc[-1])
        latest_week_resistance = safe_extract_value(data.get('weekly_resistance', pd.Series([0.0])).iloc[-1])
        latest_week_support = safe_extract_value(data.get('weekly_support', pd.Series([0.0])).iloc[-1])
        if latest_vwap == 0.0:
            latest_vwap = latest_close
        
        bb_upper = safe_extract_value(data['BB_Upper'].iloc[-1])
        bb_lower = safe_extract_value(data['BB_Lower'].iloc[-1])
        if bb_upper == 0.0:
            bb_upper = latest_close
        if bb_lower == 0.0:
            bb_lower = latest_close
        
        latest_macd = safe_extract_value(data['MACD'].iloc[-1])
        latest_macd_signal = safe_extract_value(data['MACD_Signal'].iloc[-1])
        
        macd_status = 'Bullish' if latest_macd > latest_macd_signal else 'Bearish'
        
        if latest_close > bb_upper:
            bb_status = 'Above Upper Band (Potentially Overbought)'
        elif latest_close < bb_lower:
            bb_status = 'Below Lower Band (Potentially Oversold)'
        else:
            bb_status = 'Within Bands (Normal Range)'
            
        vwap_status = 'above' if latest_close > latest_vwap else 'below'
        
        sentiment_label = 'Positive' if latest_sentiment > 0.1 else 'Negative' if latest_sentiment < -0.1 else 'Neutral'

        # Determine adjusted forecast if prediction falls below current weekly support
        adjusted_forecast_median = forecast_with_sent_median
        fell_to_next_support = False
        next_lower_support = None
        try:
            # Compute distinct weekly support levels from weekly minima (one value per week)
            try:
                week_mins = data.groupby('WeekStart')['Close'].min().dropna().values
            except Exception:
                # Fallback: try grouping by week derived from Date
                week_mins = data.groupby(pd.to_datetime(data['Date']).dt.to_period('W'))['Close'].min().dropna().values

            # Round supports to 2 decimals and get unique sorted list
            supports = sorted(np.unique(np.round(week_mins.astype(float), 2)).tolist()) if len(week_mins) > 0 else []
            print(f"Detected weekly supports (sorted): {supports}")

            if forecast_with_sent_median is not None and latest_week_support is not None and supports:
                if forecast_with_sent_median < latest_week_support:
                    fell_to_next_support = True
                    # Prefer the highest support level that is <= forecast (the level where price may stop)
                    lower_or_equal = [s for s in supports if s <= forecast_with_sent_median]
                    if lower_or_equal:
                        adjusted_forecast_median = float(max(lower_or_equal))
                    else:
                        # If forecast is below all known supports, choose the lowest known support
                        adjusted_forecast_median = float(min(supports))

                    # For reporting, determine the immediate next lower support below current weekly support
                    lower_levels = [s for s in supports if s < latest_week_support]
                    if lower_levels:
                        next_lower_support = float(max(lower_levels))
        except Exception as e:
            print(f"Support adjustment failed: {e}")

        # Attempt to detect Gaff pattern if module available. Be flexible with function name.
        pattern_info = {'name': None, 'confidence': None, 'breakout': None}
        if _HAS_GAFF and gaff_pattern_module is not None:
            detector = None
            for fname in ('detect_pattern', 'detect_patterns', 'find_pattern', 'analyze_pattern', 'analyze', 'detect'):
                if hasattr(gaff_pattern_module, fname):
                    detector = getattr(gaff_pattern_module, fname)
                    break

            if detector is not None:
                try:
                        pat_res = detector(data)
                        # Handle several possible return shapes from user modules:
                        # - dict with keys 'pattern'/'name'/'type' + 'confidence'/'score'
                        # - dict with key 'patterns' -> list of pattern dicts
                        # - list of pattern dicts
                        # - simple string/result
                        if isinstance(pat_res, dict):
                            # Prefer a top-level concise result
                            if 'pattern' in pat_res or 'name' in pat_res or 'type' in pat_res:
                                pattern_info['name'] = pat_res.get('pattern') or pat_res.get('name') or pat_res.get('type')
                                pattern_info['confidence'] = pat_res.get('confidence') or pat_res.get('score')
                                pattern_info['breakout'] = pat_res.get('breakout') if 'breakout' in pat_res else None
                            elif 'patterns' in pat_res and isinstance(pat_res['patterns'], (list, tuple)):
                                pats = pat_res['patterns']
                                if len(pats) > 0 and isinstance(pats[0], dict):
                                    # take first detected pattern as representative
                                    p = pats[0]
                                    pattern_info['name'] = p.get('pattern') or p.get('name') or p.get('type')
                                    pattern_info['confidence'] = p.get('confidence') or p.get('score')
                                    pattern_info['breakout'] = p.get('breakout') if 'breakout' in p else None
                                else:
                                    # store summary string of patterns
                                    pattern_info['name'] = ','.join([str(x) for x in pats])
                        elif isinstance(pat_res, (list, tuple)):
                            if len(pat_res) > 0 and isinstance(pat_res[0], dict):
                                p = pat_res[0]
                                pattern_info['name'] = p.get('pattern') or p.get('name') or p.get('type')
                                pattern_info['confidence'] = p.get('confidence') or p.get('score')
                                pattern_info['breakout'] = p.get('breakout') if 'breakout' in p else None
                            else:
                                pattern_info['name'] = ','.join([str(x) for x in pat_res])
                        else:
                            pattern_info['name'] = str(pat_res)
                except Exception as e:
                    print(f"Gaff pattern detection failed: {e}")

            # Normalize confidence if present (try to coerce arrays/strings to float)
            def _parse_confidence(val):
                try:
                    import numpy as _np
                    if val is None:
                        return None
                    # numpy arrays / lists
                    if isinstance(val, (list, tuple)):
                        if len(val) == 0:
                            return None
                        return float(val[0])
                    if 'numpy' in str(type(val)).lower() or isinstance(val, _np.ndarray):
                        try:
                            return float(val.flatten()[0])
                        except Exception:
                            return None
                    # string like "[0.9866]" or "0.9866"
                    if isinstance(val, str):
                        s = val.strip()
                        # remove brackets
                        s = s.strip('[]()')
                        # take first token
                        parts = s.replace(',', ' ').split()
                        if len(parts) == 0:
                            return None
                        try:
                            return float(parts[0])
                        except Exception:
                            return None
                    # numeric
                    if isinstance(val, (int, float)):
                        return float(val)
                except Exception:
                    return None

            try:
                conf = pattern_info.get('confidence')
                parsed = _parse_confidence(conf)
                pattern_info['confidence'] = parsed
            except Exception:
                pattern_info['confidence'] = None

        # Derive a simple breakout flag from forecast vs weekly levels (if not provided by pattern detector)
        breakout_flag = pattern_info.get('breakout')
        try:
            if breakout_flag is None:
                if forecast_with_sent_median is not None and latest_week_resistance is not None and latest_week_support is not None:
                    if forecast_with_sent_median > latest_week_resistance * 1.005:
                        breakout_flag = True
                    elif forecast_with_sent_median < latest_week_support * 0.995:
                        breakout_flag = True
                    else:
                        breakout_flag = False
        except Exception:
            breakout_flag = pattern_info.get('breakout')

        # Create enhanced analysis summary
        analysis_summary = f"""{ticker} Advanced Analysis with Sentiment - {datetime.date.today().strftime('%Y-%m-%d')}
========================================================

CURRENT STATUS:
• Price: Rs.{latest_close:.2f}
• Market Sentiment: {sentiment_label} ({latest_sentiment:.3f})
• Sentiment Impact: {'Bullish momentum detected' if latest_sentiment > 0 else 'Bearish pressure detected' if latest_sentiment < 0 else 'Neutral market mood'}

MODEL PERFORMANCE:
• Standard Prophet MAPE: {mape_no_sent:.2f}%
• Sentiment-Enhanced MAPE: {mape_with_sent:.2f}%
• Accuracy Improvement: {improvement:.2f}% {'✓' if improvement > 0 else '✗'}
• Detected Pattern: {pattern_info.get('name') or 'N/A'}
• Pattern Confidence: {('%.2f%%' % (pattern_info.get('confidence')*100)) if (pattern_info.get('confidence') is not None and isinstance(pattern_info.get('confidence'), (int,float))) else (pattern_info.get('confidence') or 'N/A')}
• Pattern Breakout: {('Yes' if breakout_flag else 'No') if breakout_flag is not None else 'N/A'}

3-MONTH FORECAST (WITHOUT SENTIMENT):
• Range: Rs.{forecast_no_sent_low:.2f} - Rs.{forecast_no_sent_high:.2f}
• Median: Rs.{forecast_no_sent_median:.2f}
• Change: {((forecast_no_sent_median - latest_close) / latest_close * 100):+.2f}%

3-MONTH FORECAST (WITH SENTIMENT):
• Range: Rs.{forecast_with_sent_low:.2f} - Rs.{forecast_with_sent_high:.2f}
• Median: Rs.{forecast_with_sent_median:.2f}
• Change: {((forecast_with_sent_median - latest_close) / latest_close * 100):+.2f}%
• Sentiment Adjustment: {((forecast_with_sent_median - forecast_no_sent_median) / forecast_no_sent_median * 100):+.2f}%
• Adjusted Median (support-aware): {('Rs.{:.2f}'.format(adjusted_forecast_median) + (' (fell to next support)' if fell_to_next_support else '')) if adjusted_forecast_median is not None else 'N/A'}

TECHNICAL INDICATORS:
• RSI (14): {latest_rsi:.2f} {'(Overbought)' if latest_rsi > 70 else '(Oversold)' if latest_rsi < 30 else '(Neutral)'}
• MACD: {macd_status}
• Bollinger Bands: {bb_status}
• VWAP: Rs.{latest_vwap:.2f} (Price is {vwap_status})

TRADING SIGNALS:
• RSI Signal: {'SELL' if latest_rsi > 70 else 'BUY' if latest_rsi < 30 else 'HOLD'}
• MACD Signal: {'BUY' if macd_status == 'Bullish' else 'SELL'}
• Sentiment Signal: {'BULLISH' if latest_sentiment > 0.1 else 'BEARISH' if latest_sentiment < -0.1 else 'NEUTRAL'}
• Combined Outlook: {'STRONG BUY' if (latest_sentiment > 0.1 and forecast_with_sent_median > latest_close and latest_rsi < 50) else 'BUY' if forecast_with_sent_median > latest_close else 'SELL' if forecast_with_sent_median < latest_close * 0.95 else 'HOLD'}

RECOMMENDATION:
{f"The sentiment-enhanced model suggests {'upward' if forecast_with_sent_median > latest_close else 'downward'} movement. " +
 f"Current market sentiment is {sentiment_label.lower()}, which is " +
 f"{'supporting' if (latest_sentiment > 0 and forecast_with_sent_median > latest_close) or (latest_sentiment < 0 and forecast_with_sent_median < latest_close) else 'contradicting'} " +
 "the price forecast."}
"""

        text_filename = f"{ticker}_sentiment_summary.txt"
        full_text_path = os.path.join(output_static_dir, text_filename)
        
        if os.path.exists(full_text_path):
            os.remove(full_text_path)
            
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write(analysis_summary)
        print(f"Analysis summary saved to: {full_text_path}")

        return {
            'success': True,
            'ticker': ticker,
            'summary': analysis_summary,
            'plot_base64': plot_base64,
            'current_price': f"Rs.{latest_close:.2f}",
            'sentiment': {
                'score': f"{latest_sentiment:.3f}",
                'label': sentiment_label
            },
            'forecast_no_sentiment': {
                'high': f"Rs.{forecast_no_sent_high:.2f}",
                'median': f"Rs.{forecast_no_sent_median:.2f}",
                'low': f"Rs.{forecast_no_sent_low:.2f}"
            },
            'forecast_with_sentiment': {
                'high': f"Rs.{forecast_with_sent_high:.2f}",
                'median': f"Rs.{forecast_with_sent_median:.2f}",
                'low': f"Rs.{forecast_with_sent_low:.2f}",
                'adjusted_median': (f"Rs.{adjusted_forecast_median:.2f}" if adjusted_forecast_median is not None else 'N/A'),
                'fell_to_next_support': fell_to_next_support,
                'next_lower_support': (f"Rs.{next_lower_support:.2f}" if next_lower_support is not None else None)
            },
            'patterns': pattern_info,
            'model_performance': {
                'mape_no_sentiment': f"{mape_no_sent:.2f}%" if np.isfinite(mape_no_sent) else 'N/A',
                'mape_with_sentiment': f"{mape_with_sent:.2f}%" if np.isfinite(mape_with_sent) else 'N/A',
                'improvement': f"{improvement:.2f}%" if np.isfinite(improvement) else 'N/A'
            },
            'mape_comparison_file': mape_csv_name,
            'mape_sample': mape_sample,
            'indicators': {
                'rsi': f"{latest_rsi:.2f}",
                'macd_status': macd_status,
                'bb_status': bb_status,
                'vwap': f"Rs.{latest_vwap:.2f}",
                'weekly_resistance': f"Rs.{latest_week_resistance:.2f}",
                'weekly_support': f"Rs.{latest_week_support:.2f}",
                'vwap_status': vwap_status
            }
        }

    except Exception as e:
        error_msg = f"Error analyzing {ticker}: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {
            'success': False,
            'error': error_msg,
            'ticker': ticker
        }


def analyze_stock(ticker="PGEL.NS"):
    """Compatibility wrapper expected by app.py. Returns the same structure
    as analyze_stock_with_sentiment but ensures key names match and adds
    a simple `direction` field for integration with LSTM (up/down/neutral).
    """
    result = analyze_stock_with_sentiment(ticker)
    if not result.get('success'):
        return result

    try:
        # extract numeric median forecast from sentiment-enhanced model
        median_str = result.get('forecast_with_sentiment', {}).get('median', '')
        # median_str is like 'Rs.123.45' — try to parse
        median_price = None
        if isinstance(median_str, str) and median_str.startswith('Rs.'):
            try:
                median_price = float(median_str.replace('Rs.', '').replace(',', ''))
            except Exception:
                median_price = None

        # extract current price numeric
        current_price_str = result.get('current_price', '')
        current_price = None
        if isinstance(current_price_str, str) and current_price_str.startswith('Rs.'):
            try:
                current_price = float(current_price_str.replace('Rs.', '').replace(',', ''))
            except Exception:
                current_price = None

        direction = 'Neutral'
        if median_price is not None and current_price is not None:
            if median_price > current_price * 1.005:
                direction = 'Up'
            elif median_price < current_price * 0.995:
                direction = 'Down'
            else:
                direction = 'Neutral'

        # Attach numeric values for downstream use
        result['direction'] = direction
        result['numeric'] = {
            'current_price': current_price,
            'forecast_median': median_price
        }

        # Provide a `forecast` key expected by the Flask app/template.
        # Use the existing formatted forecast fields when available.
        result['forecast'] = {
            'with_sentiment': result.get('forecast_with_sentiment', {}),
            'no_sentiment': result.get('forecast_no_sentiment', {}),
            'median_numeric': median_price,
            'current_price_numeric': current_price
        }

    except Exception:
        # If anything fails, still return original result
        pass

    return result

# For standalone execution
if __name__ == "__main__":
    result = analyze_stock_with_sentiment("PGEL.NS")
    if result['success']:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        try:
            print(result['summary'])
        except UnicodeEncodeError:
            # Fallback for consoles that don't support some unicode characters
            import sys
            try:
                sys.stdout.buffer.write(result['summary'].encode('utf-8'))
                sys.stdout.buffer.write(b"\n")
            except Exception:
                # Last resort: replace non-encodable characters
                print(result['summary'].encode('ascii', 'replace').decode('ascii'))
    else:
        print(f"Analysis failed: {result['error']}")