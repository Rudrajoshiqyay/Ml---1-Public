import yfinance as yf
import numpy as np
import pandas as pd
import warnings
from typing import TYPE_CHECKING
warnings.filterwarnings('ignore')


# Help static analysers (Pylance/pyright) while keeping runtime guards.
if TYPE_CHECKING:
    # These imports are for type checkers only and won't execute at runtime.
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore
    from sklearn.preprocessing import MinMaxScaler  # type: ignore

# Optional heavy deps - handle missing installs gracefully at runtime
try:
    import tensorflow as tf  # type: ignore[import]
    from tensorflow.keras.models import Sequential  # type: ignore[import]
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore[import]
    _HAS_TF = True
except Exception:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    _HAS_TF = False

try:
    from sklearn.preprocessing import MinMaxScaler  # type: ignore[import]
    _HAS_SKLEARN = True
except Exception:
    MinMaxScaler = None
    _HAS_SKLEARN = False


def predict_lstm_change(ticker, lookback=60, epochs=8, verbose=0):
    """Train a small LSTM on recent history and predict next-day price change.
    Returns a dict: {'predicted_price': float, 'predicted_change_pct': float}
    This is a lightweight, on-the-fly predictor intended for prototyping.
    """
    # Check for required heavy dependencies
    if not _HAS_TF:
        return {'success': False, 'error': 'tensorflow not installed', 'predicted_price': None}
    if not _HAS_SKLEARN:
        return {'success': False, 'error': 'scikit-learn not installed', 'predicted_price': None}

    try:
        data = yf.download(ticker, period="3y", auto_adjust=True)
        if data.empty:
            return {'success': False, 'error': 'No data for LSTM', 'predicted_price': None}

        close = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close)

        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)
        if len(X) < 10:
            return {'success': False, 'error': 'Not enough data for LSTM', 'predicted_price': None}

        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build model
        model = Sequential([
            LSTM(32, input_shape=(X.shape[1], 1), return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train quickly
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=verbose)

        # Evaluate on training data (report MSE in original price units)
        try:
            train_pred_scaled = model.predict(X, verbose=0).reshape(-1,)
            train_pred_prices = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).reshape(-1,)
            true_prices = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1,)
            mask = true_prices != 0
            if np.any(mask):
                train_mape = float(np.mean(np.abs((train_pred_prices[mask] - true_prices[mask]) / true_prices[mask])) * 100.0)
            else:
                train_mape = None
        except Exception:
            train_mape = None

        # Prepare last sequence for prediction
        last_seq = scaled[-lookback:].reshape((1, lookback, 1))
        pred_scaled = model.predict(last_seq, verbose=0)[0, 0]
        pred_price = scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]

        latest_price = float(close[-1, 0])
        change_pct = (pred_price - latest_price) / latest_price * 100.0

        return {
            'success': True,
            'predicted_price': float(pred_price),
            'predicted_change_pct': float(change_pct),
            'train_mape': train_mape
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'predicted_price': None}


if __name__ == '__main__':
    print(predict_lstm_change('PGEL.NS', lookback=60, epochs=3, verbose=1))
