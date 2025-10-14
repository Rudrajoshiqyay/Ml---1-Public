from flask import Flask, render_template, request, flash
from prophet_v2 import analyze_stock
import os

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
        flash(f'Analysis completed successfully for {ticker}!', 'success')
        return render_template('index.html',
                             ticker=analysis_result['ticker'],
                             summary=analysis_result['summary'],
                             image_filename=analysis_result['image_filename'],
                             current_price=analysis_result['current_price'],
                             forecast=analysis_result['forecast'],
                             indicators=analysis_result['indicators'],
                             success=True)

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
    app.run(debug=True, host='127.0.0.1', port=5000)