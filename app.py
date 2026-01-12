from flask import Flask, render_template, request, flash
from prophet_v2 import analyze_stock
from lstm_predictor import predict_lstm_change
from collections import Counter
import os
try:
    from gaff_pattern_reconiton import detect as detect_gaff
except Exception:
    def detect_gaff(*args, **kwargs):
        return {'patterns': [], 'forecast': None, 'error': 'GAFF module unavailable'}

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Ensure static directory exists
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Show the form with default ticker
        return render_template('index.html', ticker="PGEL.NS")
    
    elif request.method == 'POST':
        # Get ticker from form
        ticker = request.form.get('ticker', 'PGEL.NS').strip().upper()
        
        if not ticker:
            flash('Please enter a valid ticker symbol', 'error')
            return render_template('index.html', ticker="PGEL.NS")
        
        print(f"Analyzing ticker: {ticker}")
        
        # Perform analysis
        analysis_result = analyze_stock(ticker)
        
        if not analysis_result['success']:
            flash(f"Error analyzing {ticker}: {analysis_result['error']}", 'error')
            return render_template('index.html', 
                                 ticker=ticker,
                                 error=analysis_result['error'])
        
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
                     image_filename=analysis_result['image_filename'],
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
                     gaff_summary=gaff_summary)

@app.route('/index')
def index_check():

    return {}

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', ticker="PGEL.NS"), 404

@app.errorhandler(500)
def internal_error(error):
    flash('An internal error occurred. Please try again.', 'error')
    return render_template('index.html', ticker="PGEL.NS"), 500

if __name__ == '__main__':
    print("Starting Stock Analysis App...")
    print("Make sure you have the following dependencies installed:")
    print("pip install flask yfinance pandas matplotlib prophet ta")
    print("\nAccess the app at: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
