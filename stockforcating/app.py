"""
Flask Application for Stock Analysis
HuggingFace-friendly web interface
"""

import os
import sys
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import json

# Add stockforcating to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stockforcating'))

from cointegration_dtw import StockPairAnalyzer, analyze_stock_pairs

app = Flask(__name__, template_folder='stockforcating/templates', static_folder='stockforcating/static')
app.config['JSON_SORT_KEYS'] = False

# Cache for analysis results
analysis_cache = {
    'data': None,
    'timestamp': None
}

# Predefined stock pairs for 3 sectors
STOCK_PAIRS = {
    'energy': [
        ('RELIANCE.NS', 'ONGC.NS'),
        ('RELIANCE.NS', 'IOC.NS'),
        ('BPCL.NS', 'IOC.NS'),
    ],
    'pharma': [
        ('DIVISLAB.NS', 'LUPIN.NS'),
        ('CIPLA.NS', 'DRREDDY.NS'),
        ('SUNPHARMA.NS', 'CIPLA.NS'),
    ],
    'it': [
        ('TCS.NS', 'INFY.NS'),
        ('WIPRO.NS', 'INFY.NS'),
        ('HCLTECH.NS', 'TCS.NS'),
    ]
}


def run_all_analysis():
    """Run analysis for all sector pairs"""
    all_results = {}
    
    for sector, pairs in STOCK_PAIRS.items():
        print(f"\n🔍 Analyzing {sector.upper()} sector...")
        all_results[sector] = analyze_stock_pairs(pairs, period='2y')
    
    return all_results


def get_cached_results(force_refresh=False):
    """Get cached results or run new analysis"""
    if force_refresh or analysis_cache['data'] is None:
        print("📊 Running analysis...")
        analysis_cache['data'] = run_all_analysis()
        analysis_cache['timestamp'] = datetime.now().isoformat()
    
    return analysis_cache['data']


@app.route('/')
def index():
    """Main page with analysis results"""
    # Template will fetch data via /api/analysis endpoint
    return render_template('analysis.html')


@app.route('/api/analysis')
def api_analysis():
    """API endpoint for analysis results"""
    results = get_cached_results()
    return jsonify({
        'status': 'success',
        'timestamp': analysis_cache['timestamp'],
        'data': results
    })


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Force refresh analysis"""
    results = get_cached_results(force_refresh=True)
    return jsonify({
        'status': 'success',
        'message': 'Analysis refreshed',
        'timestamp': analysis_cache['timestamp'],
        'data': results
    })


@app.route('/api/analysis/<sector>')
def api_sector_analysis(sector):
    """Get analysis for specific sector"""
    results = get_cached_results()
    
    if sector in results:
        return jsonify({
            'status': 'success',
            'sector': sector,
            'data': results[sector]
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Sector {sector} not found'
        }), 404


@app.route('/api/pair/<ticker1>/<ticker2>')
def api_pair_analysis(ticker1, ticker2):
    """Analyze specific stock pair"""
    try:
        analyzer = StockPairAnalyzer(f"{ticker1}.NS", f"{ticker2}.NS")
        analyzer.fetch_data(period='2y')
        result = analyzer.run_analysis()
        
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Not found'
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Server error'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STOCK ANALYSIS APPLICATION - COINTEGRATION & DTW")
    print("="*70)
    print("📍 Open browser: http://127.0.0.1:7860")
    print("🔄 API Endpoints:")
    print("   - GET  /api/analysis          (all results)")
    print("   - GET  /api/analysis/<sector> (sector results)")
    print("   - POST /api/refresh           (force refresh)")
    print("   - GET  /api/pair/<t1>/<t2>    (pair analysis)")
    print("="*70 + "\n")
    
    app.run(host='127.0.0.1', port=7860, debug=False, use_reloader=False)

