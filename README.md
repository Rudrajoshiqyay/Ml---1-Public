# Stock Analysis App

This app uses Prophet for direction/forecasting and a lightweight LSTM for magnitude estimation.

Quick setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

Notes

- `plotly` is optional but recommended for interactive plots. If not installed, static plots are generated.
- `tensorflow` is required for LSTM predictions. If you don't need LSTM, you can skip installing it.

Run

```bash
python app.py
```

Files changed by the assistant

- `prophet_v2.py` — defensive imports and fixes for NaNs/merges/TA indicators
- `lstm_predictor.py` — guarded imports and clear errors when TF/sklearn missing
- `requirements.txt` — list of packages used
- `README.md` — setup instructions
- `test_run.py` — helper to run `analyze_stock` directly

If you want, I can:

- Add a `templates/index.html` update to show `direction` and LSTM results.
- Persist a pretrained LSTM model to avoid training on every request.
- Create a `requirements-dev.txt` with pinned versions.
