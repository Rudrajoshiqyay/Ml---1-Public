---
title: Stock Forecasting Analytics
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
license: mit
short_description: AI-powered stock analysis with sentiment analysis and Prophet forecasting
---

# Stock Forecasting Analytics App

A stateless, containerized Flask application for stock market analysis using Prophet, LSTM, and sentiment analysis. Fully compliant with Hugging Face Spaces and cloud deployment requirements.

## 🚀 Features

- **Prophet-based Forecasting**: 3-month ahead price predictions
- **Sentiment Analysis**: Market mood integration with technical indicators
- **LSTM Predictions**: Deep learning-based price change predictions
- **Technical Indicators**: RSI, MACD, Bollinger Bands, VWAP
- **GAFF Pattern Recognition**: Advanced pattern detection
- **In-Memory Image Rendering**: Base64-encoded charts (no disk writes)
- **Stateless Architecture**: Container-friendly, no persistent storage
- **Production Ready**: Gunicorn deployment with health checks

## 📋 System Requirements

- Python 3.10+
- Docker (for Hugging Face Spaces deployment)
- 2GB RAM minimum
- 512MB disk space (no runtime artifacts)

## 🔧 Installation & Setup

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd ipd_vs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

### Docker Deployment

```bash
# Build image
docker build -t stock-forecasting .

# Run container
docker run -p 7860:7860 \
  -e SECRET_KEY=your-production-secret \
  stock-forecasting
```

### Hugging Face Spaces

Push to your Spaces repository:

```bash
git push huggingface main
```

The app will automatically build and deploy using the Dockerfile.

## 🌍 Environment Variables

- **PORT**: Server port (default: 7860)
- **SECRET_KEY**: Flask secret key (auto-generated if not provided)

## 📊 API Endpoints

### GET /

Home page with ticker input form.

### POST /

Analyze stock ticker (returns analysis with base64-encoded chart).

### GET /index

Pattern analysis endpoint.

## 🏗️ Architecture

### Stateless Design

- ✅ **No file I/O**: All images rendered to base64
- ✅ **No persistent storage**: No static/ or dir/ folders needed
- ✅ **Container-friendly**: Runs on any container platform
- ✅ **Scalable**: Stateless allows horizontal scaling

### Module Structure

```
ipd_vs/
├── app.py                      # Flask application
├── prophet_v2.py               # Prophet forecasting engine
├── lstm_predictor.py           # LSTM model training
├── gaff_pattern_reconiton.py   # Pattern recognition
├── requirements.txt            # Dependencies
├── Dockerfile                  # Production container
└── templates/
    └── index.html              # Web UI
```

## 🔐 Security

- Secret key management via environment variables
- No sensitive data stored on disk
- Health check endpoints for monitoring
- Thread-safe Flask configuration

## 📈 Stock Data Source

Uses **yfinance** for real-time stock data:

- Historical data: 3 years
- Frequency: Daily
- Supported markets: NSE, BSE, NYSE, NASDAQ, etc.

## ⚠️ Important Notes

1. **No Static Files**: All generated images are base64-encoded in response JSON
2. **Stateless**: Container can be restarted at any time without data loss
3. **Memory Efficient**: Charts streamed as base64 strings, not stored
4. **Port Flexibility**: Respects PORT environment variable

## 📝 Configuration Reference

See [Hugging Face Spaces Config](https://huggingface.co/docs/hub/spaces-config-reference)

## 🧪 Testing

```bash
# Test ticker analysis
curl -X POST http://localhost:7860/ \
  -d "ticker=RELIANCE.NS"

# Health check
curl http://localhost:7860/
```

## 📦 Dependencies

See [requirements.txt](requirements.txt) for full list. Key packages:

- Flask: Web framework
- Yahoo Finance: Market data
- Prophet: Time series forecasting
- TensorFlow: LSTM neural networks
- Scikit-learn: ML algorithms
- Plotly: Interactive charts (via base64)

## 🤝 Contributing

Contributions welcome! Ensure all changes maintain stateless architecture.

## 📄 License

MIT License - See LICENSE file

## 🆘 Support

For issues and questions, please create a GitHub issue.

---

**Last Updated**: 2024
**Status**: Production Ready ✅
**Hugging Face Spaces Compatible**: Yes ✅
