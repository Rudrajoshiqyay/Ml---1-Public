# IMPORTANT: Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web servers

# Import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import datetime
import os
import traceback
import numpy as np

# Disable interactive plotting
plt.ioff()

def clean_static_dir_folder():
    """Clean old files from static_dir folder to prevent accumulation"""
    try:
        static_dir = 'static'
        if os.path.exists(static_dir):
            for filename in os.listdir(static_dir):  # Fixed: was 'liststatic_dir'
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
        # If it's already a scalar (int, float, etc.), return it
        if isinstance(series_value, (int, float, np.integer, np.floating)):
            return float(series_value) if not pd.isna(series_value) else 0.0
        
        # If it's a pandas Series, extract the first/only value
        if isinstance(series_value, pd.Series):
            if series_value.empty:
                return 0.0
            # Get the first value and check if it's NaN
            val = series_value.iloc[0]
            return float(val) if not pd.isna(val) else 0.0
        
        # If it's some other pandas object, try to get the item
        if hasattr(series_value, 'item'):
            val = series_value.item()
            return float(val) if not pd.isna(val) else 0.0
        
        # Try to convert to float directly
        val = float(series_value)
        return val if not pd.isna(val) else 0.0
        
    except (ValueError, TypeError, AttributeError):
        return 0.0

def analyze_stock(ticker="PGEL.NS"):
    try:
        # Clean old files first
        clean_static_dir_folder()
        
        # Use your existing static_dir folder
        output_static_dir = 'static'
        os.makedirs(output_static_dir, exist_ok=True)  # Fixed: was 'makestatic_dirs'

        # Download historical data
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, period="3y", auto_adjust=True)
        
        if data.empty:
            return {
                'success': False,
                'error': f"No data found for ticker {ticker}. Please check if the ticker symbol is correct.",
                'ticker': ticker
            }

        # Reset index and ensure we have a 'Date' column
        data = data.reset_index()
        if 'Date' not in data.columns and 'index' in data.columns:
            data = data.rename(columns={'index': 'Date'})

        # Prepare data for Prophet
        df = data[['Date', 'Close']].copy()
        df.columns = ['ds', 'y']

        # Create and fit Prophet model
        print("Training Prophet model...")
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.5,
            seasonality_mode='multiplicative'
        )
        model.fit(df)

        # Create future dataframe (3 months = ~66 trading days)
        future = model.make_future_dataframe(periods=66, freq='B')
        forecast = model.predict(future)

        # Calculate technical indicators
        print("Calculating technical indicators...")
        close_prices = data['Close'].squeeze()
        
        # RSI
        data['RSI_14'] = RSIIndicator(close_prices, window=14).rsi()
        
        # MACD
        macd = MACD(close_prices)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()

        # Bollinger Bands
        bb = BollingerBands(close_prices)
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Middle'] = bb.bollinger_mavg()
        
        # VWAP
        data['VWAP'] = VolumeWeightedAveragePrice(
            high=data['High'].squeeze(),
            low=data['Low'].squeeze(),
            close=close_prices,
            volume=data['Volume'].squeeze(),
            window=20
        ).volume_weighted_average_price()

        # Plotting
        print("Creating charts...")
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create new figure
        fig = plt.figure(figsize=(15, 20))

        # Prophet forecast plot
        plt.subplot(4, 1, 1)
        plt.plot(data['Date'], data['Close'], label='Actual Price', linewidth=2)
        plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', color='orange', linewidth=2)
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='orange')
        plt.title(f'{ticker} - Prophet Forecast (Next 3 Months)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Price with Bollinger Bands
        plt.subplot(4, 1, 2)
        plt.plot(data['Date'], data['Close'], label='Price', linewidth=2)
        plt.plot(data['Date'], data['BB_Upper'], label='Upper Band', linestyle='--', color='red', alpha=0.7)
        plt.plot(data['Date'], data['BB_Lower'], label='Lower Band', linestyle='--', color='green', alpha=0.7)
        plt.plot(data['Date'], data['BB_Middle'], label='Middle Band', linestyle='--', color='blue', alpha=0.7)
        plt.fill_between(data['Date'], data['BB_Upper'], data['BB_Lower'], alpha=0.1)
        plt.title('Bollinger Bands', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # MACD
        plt.subplot(4, 1, 3)
        plt.plot(data['Date'], data['MACD'], label='MACD', color='blue', linewidth=2)
        plt.plot(data['Date'], data['MACD_Signal'], label='Signal Line', color='orange', linewidth=2)
        plt.bar(data['Date'], data['MACD_Hist'], label='Histogram', color='gray', alpha=0.5, width=1)
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.title('MACD Indicator', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # RSI
        plt.subplot(4, 1, 4)
        plt.plot(data['Date'], data['RSI_14'], label='RSI (14)', color='purple', linewidth=2)
        plt.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        plt.axhline(50, color='gray', linestyle='-', alpha=0.3)
        plt.ylim(0, 100)
        plt.title('Relative Strength Index', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot to static_dir folder - ensure overwrite
        plot_filename = f"{ticker}_analysis.png"
        full_plot_path = os.path.join(output_static_dir, plot_filename)
        
        # Remove existing file if it exists
        if os.path.exists(full_plot_path):
            os.remove(full_plot_path)
            
        plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
        plt.close('all')  # Close all figures completely
        print(f"Plot saved to: {full_plot_path}")

        # Get latest values and predictions - Fixed to handle Series properly
        latest_close = safe_extract_value(data['Close'].iloc[-1])
        forecast_high = safe_extract_value(forecast['yhat_upper'].iloc[-1])
        forecast_low = safe_extract_value(forecast['yhat_lower'].iloc[-1])
        forecast_median = safe_extract_value(forecast['yhat'].iloc[-1])
        
        # Handle potential NaN values in indicators
        latest_rsi = safe_extract_value(data['RSI_14'].iloc[-1])
        latest_vwap = safe_extract_value(data['VWAP'].iloc[-1])
        if latest_vwap == 0.0:
            latest_vwap = latest_close
        
        # Get Bollinger Band values as scalars
        bb_upper = safe_extract_value(data['BB_Upper'].iloc[-1])
        bb_lower = safe_extract_value(data['BB_Lower'].iloc[-1])
        if bb_upper == 0.0:
            bb_upper = latest_close
        if bb_lower == 0.0:
            bb_lower = latest_close
        
        # Get MACD values as scalars
        latest_macd = safe_extract_value(data['MACD'].iloc[-1])
        latest_macd_signal = safe_extract_value(data['MACD_Signal'].iloc[-1])
        
        # Status calculations - Fixed comparisons
        macd_status = 'Bullish' if latest_macd > latest_macd_signal else 'Bearish'
        
        # Fixed Bollinger Band comparison
        if latest_close > bb_upper:
            bb_status = 'Above Upper Band (Potentially Overbought)'
        elif latest_close < bb_lower:
            bb_status = 'Below Lower Band (Potentially Oversold)'
        else:
            bb_status = 'Within Bands (Normal Range)'
            
        vwap_status = 'above' if latest_close > latest_vwap else 'below'

        # Debug print statements
        print(f"Debug - latest_close: {latest_close} (type: {type(latest_close)})")
        print(f"Debug - bb_upper: {bb_upper} (type: {type(bb_upper)})")
        print(f"Debug - bb_lower: {bb_lower} (type: {type(bb_lower)})")

        # Create analysis summary
        analysis_summary = f"""{ticker} Technical Analysis Summary - {datetime.date.today().strftime('%Y-%m-%d')}
--------------------------------------------------------
Current Price: Rs.{latest_close:.2f}
3-Month Forecast Range: Rs.{forecast_low:.2f} - Rs.{forecast_high:.2f}
Median Forecast: Rs.{forecast_median:.2f}

Technical Indicators:
• RSI (14): {latest_rsi:.2f} {'(Overbought)' if latest_rsi > 70 else '(Oversold)' if latest_rsi < 30 else '(Neutral)'}
• MACD Status: {macd_status}
• Bollinger Bands: {bb_status}
• VWAP: Rs.{latest_vwap:.2f} | Price is {vwap_status} VWAP

Trading Signals:
• RSI Signal: {'SELL' if latest_rsi > 70 else 'BUY' if latest_rsi < 30 else 'HOLD'}
• MACD Signal: {'BUY' if macd_status == 'Bullish' else 'SELL'}
• Overall Trend: {'BULLISH' if forecast_median > latest_close else 'BEARISH'}"""

        # Save to text file in static_dir folder with UTF-8 encoding - ensure overwrite
        text_filename = f"{ticker}_summary.txt"
        full_text_path = os.path.join(output_static_dir, text_filename)
        
        # Remove existing file if it exists
        if os.path.exists(full_text_path):
            os.remove(full_text_path)
            
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write(analysis_summary)
        print(f"Analysis summary saved to: {full_text_path}")

        # Return structured data for Flask app
        return {
            'success': True,
            'ticker': ticker,
            'summary': analysis_summary,
            'image_filename': plot_filename,
            'current_price': f"Rs.{latest_close:.2f}",
            'forecast': {
                'high': f"Rs.{forecast_high:.2f}",
                'median': f"Rs.{forecast_median:.2f}",
                'low': f"Rs.{forecast_low:.2f}"
            },
            'indicators': {
                'rsi': f"{latest_rsi:.2f}",
                'macd_status': macd_status,
                'bb_status': bb_status,
                'vwap': f"Rs.{latest_vwap:.2f}",
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

# For standalone execution
if __name__ == "__main__":
    result = analyze_stock("PGEL.NS")
    if result['success']:
        print("Analysis completed successfully!")
        print(result['summary'])
    else:
        print(f"Analysis failed: {result['error']}")