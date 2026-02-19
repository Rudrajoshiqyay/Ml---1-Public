import numpy as np
import pandas as pd
import yfinance as yf
import math

try:
    from statsmodels.tsa.stattools import coint, adfuller
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    _HAS_FASTDTW = True
except Exception:
    _HAS_FASTDTW = False


def _simple_dtw(a, b):
    # O(n*m) DTW distance fallback
    n, m = len(a), len(b)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return float(dtw[n, m])


def _cointegration_pvalue(x, y):
    # Try statsmodels coint, else use residual ADF if available, else use correlation proxy
    try:
        if _HAS_STATSMODELS:
            score, pvalue, _ = coint(x, y)
            return float(pvalue)
    except Exception:
        pass

    # Fallback: regress y ~ x, compute residuals and ADF on residuals
    try:
        coeffs = np.polyfit(x, y, 1)
        resid = y - (coeffs[0] * x + coeffs[1])
        if _HAS_STATSMODELS:
            adf_stat, pval, _, _, _, _ = adfuller(resid)
            # lower pval => stationary residual => cointegrated
            return float(pval)
    except Exception:
        pass

    # Last-resort proxy: use 1 - abs(correlation)
    try:
        r = np.corrcoef(x, y)[0, 1]
        pproxy = max(0.0, 1.0 - abs(float(r)))
        return float(pproxy)
    except Exception:
        return 1.0


def build_sector_index(df_dict):
    # df_dict: {ticker: df_close_series}
    frames = []
    for t, s in df_dict.items():
        frames.append(s.rename(t))
    if not frames:
        return None
    stacked = pd.concat(frames, axis=1)
    # use mean across available tickers (skip NaNs)
    return stacked.mean(axis=1)


def fetch_close(ticker, period='2y'):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df['Close']
    except Exception:
        return None


def run_recommender(period='2y'):
    """Build three sector indexes (from lists below), compute cointegration and DTW
    between each stock and its sector index. Returns ranked results.
    """
    # Predefined stock lists (NSE format). These are examples and may be adjusted.
    pharma = [
        'SUNPHARMA.NS','CIPLA.NS','DRREDDY.NS','DIVISLAB.NS','LUPIN.NS',
        'AUROPHARMA.NS','ALKEM.NS','TORNTPHARM.NS','GLAXO.NS','IPCALAB.NS'
    ]
    it = [
        'TCS.NS','INFY.NS','WIPRO.NS','HCLTECH.NS','TECHM.NS',
        'LTIM.NS','MINDTREE.NS','COFORGE.NS','LTI.NS','PERSISTENT.NS'
    ]
    energy = [
        'RELIANCE.NS','ONGC.NS','BPCL.NS','IOC.NS','TATAPOWER.NS',
        'NTPC.NS','GAIL.NS','PETRONET.NS','ADANIPOWER.NS','HINDPETRO.NS'
    ]

    sectors = {'Pharma': pharma, 'IT': it, 'Energy': energy}

    all_results = {}

    for sector_name, tickers in sectors.items():
        # fetch closes
        close_dict = {}
        for t in tickers:
            s = fetch_close(t, period=period)
            if s is not None and len(s) > 10:
                close_dict[t] = s

        if not close_dict:
            all_results[sector_name] = {'error': 'No data for tickers'}
            continue

        # align on intersection of dates
        df_all = pd.concat(close_dict.values(), axis=1, join='inner')
        df_all.columns = list(close_dict.keys())
        if df_all.empty:
            all_results[sector_name] = {'error': 'No overlapping dates'}
            continue

        # build sector index as mean close
        sector_index = df_all.mean(axis=1)

        # For each ticker compute cointegration pvalue and DTW distance
        records = []
        for t in df_all.columns:
            s = df_all[t].dropna()
            common = sector_index.loc[s.index].dropna()
            s = s.loc[common.index]
            if len(s) < 30:
                continue

            # Normalize series for DTW
            a = (s.values - np.mean(s.values)) / (np.std(s.values) + 1e-9)
            b = (common.values - np.mean(common.values)) / (np.std(common.values) + 1e-9)

            # cointegration pvalue
            try:
                pval = _cointegration_pvalue(s.values, common.values)
            except Exception:
                pval = 1.0

            # DTW distance
            try:
                if _HAS_FASTDTW:
                    dist, _ = fastdtw(a, b, dist=euclidean)
                else:
                    dist = _simple_dtw(a, b)
            except Exception:
                dist = float('inf')

            records.append({'ticker': t, 'pvalue': float(pval), 'dtw': float(dist)})

        if not records:
            all_results[sector_name] = {'error': 'No valid series for sector'}
            continue

        df_rec = pd.DataFrame(records)

        df_rec = df_rec.sort_values('pvalue')

        most_cointegrated = df_rec.nsmallest(3, 'pvalue').to_dict('records')
        least_cointegrated = df_rec.nlargest(3, 'pvalue').to_dict('records')

        most_similar_dtw = df_rec.nsmallest(3, 'dtw').to_dict('records')
        least_similar_dtw = df_rec.nlargest(3, 'dtw').to_dict('records')

        all_results[sector_name] = {
            'summary': {
                'count_analyzed': len(df_rec),
            },
            'rank_by_cointegration': {
                'most_cointegrated': most_cointegrated,
                'least_cointegrated': least_cointegrated,
            },
            'rank_by_dtw': {
                'most_similar': most_similar_dtw,
                'least_similar': least_similar_dtw,
            },
            'raw': df_rec.to_dict('records')
        }

    return all_results


if __name__ == '__main__':
    import json
    res = run_recommender()
    print(json.dumps(res, indent=2))
