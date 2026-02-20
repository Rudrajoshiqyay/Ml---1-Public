from flask import Flask, render_template, request, flash
from prophet_v2 import analyze_stock
from lstm_predictor import predict_lstm_change
from collections import Counter
import os
import sys
import json
from threading import Thread
import time

# Import stock recommender
try:
    from stockforcating.stock_recommender import run_recommender
    _HAS_RECOMMENDER = True
except Exception as e:
    print(f"Warning: Could not import stock recommender: {e}")
    _HAS_RECOMMENDER = False
    def run_recommender(*args, **kwargs):
        return {}

try:
    from gaff_pattern_reconiton import detect as detect_gaff
except Exception:
    def detect_gaff(*args, **kwargs):
        return {'patterns': [], 'forecast': None, 'error': 'GAFF module unavailable'}

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global cache for recommendations
_recommendations_cache = {}
_cache_timestamp = 0
_CACHE_DURATION = 3600  # Cache for 1 hour

def get_recommendations_cached():
    """Get cached recommendations or generate new ones"""
    global _recommendations_cache, _cache_timestamp
    
    current_time = time.time()
    # Use cache if it's fresh
    if _recommendations_cache and (current_time - _cache_timestamp) < _CACHE_DURATION:
        return _recommendations_cache
    
    # Generate new recommendations
    if _HAS_RECOMMENDER:
        try:
            print("📊 Generating stock recommendations (this may take a minute)...")
            _recommendations_cache = run_recommender(period='1y')
            _cache_timestamp = current_time
            print("✓ Stock recommendations ready!")
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            _recommendations_cache = {}
    
    return _recommendations_cache

# Pre-generate recommendations on app startup (background thread)
def _init_recommendations():
    """Initialize recommendations in background"""
    try:
        get_recommendations_cached()
    except Exception as e:
        print(f"Background recommendation init error: {e}")

if _HAS_RECOMMENDER:
    init_thread = Thread(target=_init_recommendations, daemon=True)
    init_thread.start()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get cached recommendations
    recommendations = get_recommendations_cached()
    
    if request.method == 'GET':
        # Show the form with default ticker and recommendations
        return render_template('index.html', 
                             ticker="PGEL.NS",
                             recommendations=recommendations)
    
    elif request.method == 'POST':
        # Get ticker from form
        ticker = request.form.get('ticker', 'PGEL.NS').strip().upper()
        
        if not ticker:
            flash('Please enter a valid ticker symbol', 'error')
            return render_template('index.html', ticker="PGEL.NS", recommendations=recommendations)
        
        print(f"Analyzing ticker: {ticker}")
        
        # Perform analysis
        analysis_result = analyze_stock(ticker)
        
        if not analysis_result['success']:
            flash(f"Error analyzing {ticker}: {analysis_result['error']}", 'error')
            return render_template('index.html', 
                                 ticker=ticker,
                                 error=analysis_result['error'],
                                 recommendations=recommendations)
        
        # Success - return analysis results
        # Run LSTM predictor to estimate magnitude of change
        lstm_result = predict_lstm_change(analysis_result.get('ticker', ticker))
        lstm_info = {}
        if lstm_result.get('success'):
            lstm_info = {
                'predicted_price': lstm_result.get('predicted_price'),
                'predicted_change_pct': lstm_result.get('predicted_change_pct'),
                'train_mape': lstm_result.get('train_mape')
            }
        else:
            lstm_info = {'error': lstm_result.get('error')}

        # Get GAFF pattern detection data (best-effort, non-blocking)
        import io, contextlib
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                gaff_result = detect_gaff(ticker)
        except Exception as e:
            gaff_result = {'patterns': [], 'forecast': None, 'error': str(e)}

        # Synthesize a short summary for all detected GAFF patterns
        patterns_list = gaff_result.get('patterns') if isinstance(gaff_result, dict) else []
        if patterns_list is None:
            patterns_list = []
        # Ensure list-like
        try:
            patterns_list = list(patterns_list)
        except Exception:
            patterns_list = []

        total_patterns = len(patterns_list)
        counts = Counter([p.get('pattern', p.get('name', 'unknown')) for p in patterns_list])
        top_pattern, top_count = (counts.most_common(1)[0] if counts else ('None', 0))

        conf_vals = []
        for p in patterns_list:
            c = p.get('confidence')
            if c is None:
                continue
            try:
                c = float(c)
                if c <= 1:
                    c = c * 100.0
                conf_vals.append(c)
            except Exception:
                continue

        avg_confidence = round(sum(conf_vals) / len(conf_vals), 2) if conf_vals else None
        breakout_count = sum(1 for p in patterns_list if p.get('breakout'))

        # Try to compute support/resistance and suggested buy levels
        suggested = {}
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period='6mo')
            if df is not None and len(df) > 0:
                latest = df['Close'].iloc[-1]
                support = float(df['Low'].rolling(window=20, min_periods=1).min().iloc[-1])
                resistance = float(df['High'].rolling(window=20, min_periods=1).max().iloc[-1])

                # Default suggestions
                suggested['current_price'] = round(latest, 2)
                suggested['support'] = round(support, 2)
                suggested['resistance'] = round(resistance, 2)

                # Pattern-based suggestions
                if top_pattern and top_pattern != 'None':
                    if 'inverted' in top_pattern.lower() or 'double' in top_pattern.lower():
                        suggested['buy'] = round(support * 1.01, 2)
                        suggested['stop_loss'] = round(support * 0.96, 2)
                        suggested['note'] = 'Buy near support on pullback; set tight stop below recent low.'
                    elif 'engulfing' in top_pattern.lower():
                        suggested['buy'] = round(latest * 0.995, 2)
                        suggested['stop_loss'] = round(latest * 0.97, 2)
                        suggested['note'] = 'Consider buy on confirmation candle or minor pullback.'
                    elif 'ascending' in top_pattern.lower() or 'triangle' in top_pattern.lower():
                        suggested['buy'] = round(resistance * 1.01, 2)
                        suggested['stop_loss'] = round(support * 0.98, 2)
                        suggested['note'] = 'Buy on breakout above resistance with volume confirmation.'
                    else:
                        suggested['buy'] = round(latest * 0.995, 2)
                        suggested['stop_loss'] = round(latest * 0.97, 2)
                        suggested['note'] = 'Generic suggestion: buy on pullback or breakout confirmation.'
                else:
                    suggested['buy'] = round(support * 1.01, 2)
                    suggested['stop_loss'] = round(support * 0.96, 2)
                    suggested['note'] = 'No clear pattern: consider buying near support or wait for breakout.'
            else:
                suggested = {}
        except Exception:
            suggested = {}

        gaff_summary = {
            'total': total_patterns,
            'counts': dict(counts),
            'top_pattern': top_pattern,
            'top_count': top_count,
            'avg_confidence': avg_confidence,
            'breakouts': int(breakout_count),
            'suggested': suggested
        }

        flash(f'Analysis completed successfully for {ticker}!', 'success')
        return render_template('index.html',
                     ticker=analysis_result['ticker'],
                     summary=analysis_result['summary'],
                     plot_base64=analysis_result.get('plot_base64'),
                     current_price=analysis_result['current_price'],
                     forecast=analysis_result.get('forecast', {}),
                     forecast_with_sentiment=analysis_result.get('forecast_with_sentiment', {}),
                     indicators=analysis_result.get('indicators', {}),
                     model_performance=analysis_result.get('model_performance', {}),
                     patterns=analysis_result.get('patterns', {}),
                     success=True,
                     direction=analysis_result.get('direction'),
                     lstm=lstm_info,
                     gaff=gaff_result,
                     gaff_summary=gaff_summary,
                     recommendations=recommendations)

@app.route('/index')
def index_check():

    return {}

@app.errorhandler(404)
def not_found_error(error):
    recommendations = {}
    if _HAS_RECOMMENDER:
        try:
            recommendations = run_recommender(period='1y')
        except Exception:
            pass
    return render_template('index.html', ticker="PGEL.NS", recommendations=recommendations), 404

@app.errorhandler(500)
def internal_error(error):
    flash('An internal error occurred. Please try again.', 'error')
    recommendations = {}
    if _HAS_RECOMMENDER:
        try:
            recommendations = run_recommender(period='1y')
        except Exception:
            pass
    return render_template('index.html', ticker="PGEL.NS", recommendations=recommendations), 500

if __name__ == '__main__':
    # Read port from environment variable (Hugging Face Spaces uses PORT=7860 by default)
    port = int(os.getenv('PORT', 7860))
    host = '0.0.0.0'  # Listen on all interfaces for containerized deployment
    
    print("🚀 Starting Stock Analysis App...")
    print(f"📡 Listening on {host}:{port}")
    print("🔗 Ensure templates/ and static/ folders exist in deployment")
    
    # Disable Flask dev server warnings in production
    app.run(host=host, port=port, threaded=True, debug=False)

