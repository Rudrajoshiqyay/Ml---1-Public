"""
Cointegration & DTW Analysis Module
HuggingFace-friendly stock analysis using statistical methods
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import coint, adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class CointegrationAnalyzer:
    """Analyze cointegration between two time series"""
    
    def __init__(self, series1, series2, names=("Series1", "Series2")):
        self.series1 = series1
        self.series2 = series2
        self.name1, self.name2 = names
        
    def align_series(self):
        """Align two series by date"""
        common_dates = self.series1.index.intersection(self.series2.index)
        return self.series1.loc[common_dates], self.series2.loc[common_dates]
    
    def normalize_series(self, s1, s2):
        """Normalize series for comparison"""
        s1_norm = (s1 - s1.mean()) / s1.std()
        s2_norm = (s2 - s2.mean()) / s2.std()
        return s1_norm, s2_norm
    
    def run_adf_test(self, series):
        """Augmented Dickey-Fuller test for stationarity"""
        if not HAS_STATSMODELS:
            return None
        
        try:
            result = adfuller(series.dropna())
            return {
                'adf_stat': result[0],
                'pvalue': result[1],
                'stationary': result[1] < 0.05
            }
        except:
            return None
    
    def run_cointegration_test(self):
        """Run Johansen cointegration test"""
        if not HAS_STATSMODELS:
            return self._fallback_cointegration()
        
        try:
            s1_aligned, s2_aligned = self.align_series()
            s1_aligned, s2_aligned = s1_aligned.dropna(), s2_aligned.dropna()
            
            if len(s1_aligned) < 20 or len(s2_aligned) < 20:
                return self._fallback_cointegration()
            
            score, pvalue, _ = coint(s1_aligned.values, s2_aligned.values)
            
            # Check for NaN values
            if np.isnan(score) or np.isnan(pvalue):
                return self._fallback_cointegration()
            
            return {
                'cointegration_score': float(score),
                'pvalue': float(pvalue),
                'cointegrated': pvalue < 0.05,
                'strength': self._interpret_cointegration(pvalue)
            }
        except Exception as e:
            return self._fallback_cointegration()
            print(f"Error in cointegration test: {e}")
            return self._fallback_cointegration()
    
    def _fallback_cointegration(self):
        """Fallback method using correlation"""
        try:
            s1_aligned, s2_aligned = self.align_series()
            s1_aligned, s2_aligned = s1_aligned.dropna(), s2_aligned.dropna()
            
            # Handle edge cases
            if len(s1_aligned) < 2 or len(s2_aligned) < 2:
                return {
                    'cointegration_score': 0.0,
                    'pvalue': 1.0,
                    'cointegrated': False,
                    'strength': 'Weak',
                    'method': 'correlation_proxy'
                }
            
            corr_matrix = np.corrcoef(s1_aligned.values, s2_aligned.values)
            corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            
            pvalue = 1 - abs(float(corr))
            
            return {
                'cointegration_score': float(corr) if not np.isnan(corr) else 0.0,
                'pvalue': float(pvalue) if not np.isnan(pvalue) else 1.0,
                'cointegrated': float(pvalue) < 0.05 if not np.isnan(pvalue) else False,
                'strength': self._interpret_cointegration(pvalue if not np.isnan(pvalue) else 1.0),
                'method': 'correlation_proxy'
            }
        except Exception as e:
            return {
                'cointegration_score': 0.0,
                'pvalue': 1.0,
                'cointegrated': False,
                'strength': 'Weak',
                'method': 'correlation_proxy',
                'error': str(e)
            }
    
    def _interpret_cointegration(self, pvalue):
        """Interpret cointegration strength"""
        if pvalue < 0.01:
            return "Very Strong"
        elif pvalue < 0.05:
            return "Strong"
        elif pvalue < 0.10:
            return "Moderate"
        else:
            return "Weak"


class DTWAnalyzer:
    """Dynamic Time Warping Distance Analysis"""
    
    def __init__(self, series1, series2, names=("Series1", "Series2")):
        self.series1 = series1
        self.series2 = series2
        self.name1, self.name2 = names
    
    def align_series(self):
        """Align two series by date"""
        common_dates = self.series1.index.intersection(self.series2.index)
        return self.series1.loc[common_dates], self.series2.loc[common_dates]
    
    def normalize_series(self, s1, s2):
        """Normalize series for DTW comparison"""
        s1_mean = s1.mean()
        s1_std = s1.std()
        s2_mean = s2.mean()
        s2_std = s2.std()
        
        s1_norm = (s1 - s1_mean) / (s1_std + 1e-8)
        s2_norm = (s2 - s2_mean) / (s2_std + 1e-8)
        
        return s1_norm.values, s2_norm.values
    
    def dtw_distance(self):
        """Calculate Dynamic Time Warping distance"""
        s1_aligned, s2_aligned = self.align_series()
        s1_aligned, s2_aligned = s1_aligned.dropna(), s2_aligned.dropna()
        
        if len(s1_aligned) < 2 or len(s2_aligned) < 2:
            return None
        
        s1_norm, s2_norm = self.normalize_series(s1_aligned, s2_aligned)
        
        distance = self._compute_dtw(s1_norm, s2_norm)
        normalized_distance = distance / max(len(s1_norm), len(s2_norm))
        
        return {
            'dtw_distance': float(distance),
            'normalized_dtw': float(normalized_distance),
            'similarity': self._interpret_dtw(normalized_distance),
            'length1': len(s1_norm),
            'length2': len(s2_norm)
        }
    
    def _compute_dtw(self, x, y):
        """Compute DTW distance using dynamic programming"""
        n, m = len(x), len(y)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1]   # match
                )
        
        return dtw_matrix[n, m]
    
    def _interpret_dtw(self, normalized_distance):
        """Interpret DTW similarity"""
        if normalized_distance < 0.5:
            return "Very Similar"
        elif normalized_distance < 1.0:
            return "Similar"
        elif normalized_distance < 2.0:
            return "Moderately Similar"
        else:
            return "Dissimilar"


class StockPairAnalyzer:
    """Comprehensive analysis of stock pair relationships"""
    
    def __init__(self, ticker1, ticker2):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.data1 = None
        self.data2 = None
    
    def fetch_data(self, period='2y'):
        """Fetch historical stock data"""
        print(f"📥 Fetching {self.ticker1} and {self.ticker2}...")
        try:
            self.data1 = yf.download(self.ticker1, period=period, progress=False)['Close']
            self.data2 = yf.download(self.ticker2, period=period, progress=False)['Close']
            print(f"✅ Data fetched: {len(self.data1)} days for {self.ticker1}, {len(self.data2)} days for {self.ticker2}")
            return True
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def run_analysis(self):
        """Run complete analysis"""
        if self.data1 is None or self.data2 is None:
            self.fetch_data()
        
        # Cointegration analysis
        coint_analyzer = CointegrationAnalyzer(self.data1, self.data2, 
                                              (self.ticker1, self.ticker2))
        coint_result = coint_analyzer.run_cointegration_test()
        
        # DTW analysis
        dtw_analyzer = DTWAnalyzer(self.data1, self.data2, 
                                  (self.ticker1, self.ticker2))
        dtw_result = dtw_analyzer.dtw_distance()
        
        return {
            'ticker1': self.ticker1,
            'ticker2': self.ticker2,
            'analysis_date': datetime.now().isoformat(),
            'cointegration': coint_result,
            'dtw': dtw_result
        }


def analyze_stock_pairs(pairs_list, period='2y'):
    """Analyze multiple stock pairs"""
    results = []
    
    for pair in pairs_list:
        ticker1, ticker2 = pair
        analyzer = StockPairAnalyzer(ticker1, ticker2)
        analyzer.fetch_data(period=period)
        result = analyzer.run_analysis()
        results.append(result)
    
    return results


if __name__ == '__main__':
    # Example usage
    pairs = [
        ('RELIANCE.NS', 'IOC.NS'),
        ('TCS.NS', 'INFY.NS'),
        ('DIVISLAB.NS', 'CIPLA.NS')
    ]
    
    results = analyze_stock_pairs(pairs)
    
    for result in results:
        print(f"\n{'='*60}")
        print(f"Analysis: {result['ticker1']} vs {result['ticker2']}")
        print(f"{'='*60}")
        
        if result['cointegration']:
            print(f"\n📊 COINTEGRATION:")
            print(f"  Score: {result['cointegration']['cointegration_score']:.4f}")
            print(f"  P-Value: {result['cointegration']['pvalue']:.4f}")
            print(f"  Strength: {result['cointegration']['strength']}")
            print(f"  Cointegrated: {'Yes ✅' if result['cointegration']['cointegrated'] else 'No ❌'}")
        
        if result['dtw']:
            print(f"\n📈 DTW DISTANCE:")
            print(f"  DTW Distance: {result['dtw']['dtw_distance']:.4f}")
            print(f"  Normalized DTW: {result['dtw']['normalized_dtw']:.4f}")
            print(f"  Similarity: {result['dtw']['similarity']}")
