# Hugging Face Spaces Deployment Guide

Complete step-by-step instructions to deploy the Stock Forecasting Analytics app to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**
   - Create account at https://huggingface.co (free)
   - Verify email address

2. **Git and Command Line**
   - Git installed on your machine
   - Basic command line knowledge

3. **Git Credentials**
   - HF Hub credentials (get from https://huggingface.co/settings/tokens)
   - Create a new token with `write` permission

## Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `stock-forecasting` (or your preferred name)
   - **License**: MIT
   - **Space SDK**: Docker ✅ (important)
   - **Space hardware**: CPU Basic (sufficient)
   - **Private**: No (unless you want private access)

3. Click "Create Space"

## Step 2: Prepare Your Local Repository

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/stock-forecasting
cd stock-forecasting

# Add the source project files
# Copy all files from ipd_vs/ directory to this location
```

## Step 3: Copy Project Files

### Option A: Copy from main directory

```bash
cp -r ../ipd_vs/* .
# or on Windows:
# xcopy ..\ipd_vs\* . /E /I
```

### Option B: Manual copy

Copy these files to your Space repository:

- `app.py`
- `prophet_v2.py`
- `lstm_predictor.py`
- `gaff_pattern_reconiton.py`
- `requirements.txt`
- `Dockerfile`
- `.dockerignore`
- `.gitignore`
- `README.md`
- `templates/` folder (with index.html)

## Step 4: Verify Files

```bash
# Check that all necessary files are present
ls -la

# Should see:
# - app.py
# - prophet_v2.py
# - lstm_predictor.py
# - requirements.txt
# - Dockerfile
# - .dockerignore
# - .gitignore
# - templates/ (directory)
```

## Step 5: Configure for Hugging Face Spaces

### Update Dockerfile (if needed)

Ensure Dockerfile has:

```dockerfile
FROM python:3.10-slim
...
EXPOSE 7860
ENV PORT=7860
...
CMD ["gunicorn", "--bind", "0.0.0.0:7860", ...]
```

### Check requirements.txt

Ensure all versions are pinned:

```
flask==2.3.3
gunicorn==21.2.0
yfinance==0.2.32
... (all with ==version)
```

## Step 6: Push to Hugging Face Spaces

```bash
# Stage all files
git add .

# Commit
git commit -m "Deploy Stock Forecasting Analytics app

- Refactored for HF Spaces compliance
- Stateless architecture (no file I/O)
- In-memory base64 image encoding
- Production-ready Gunicorn setup
- Health checks and monitoring"

# Push to Hugging Face Spaces
git push origin main
```

## Step 7: Monitor Deployment

1. **Watch the build process**:
   - Go to your Space: https://huggingface.co/spaces/YOUR_USERNAME/stock-forecasting
   - Click "Logs" tab
   - Watch for build progress and any errors

2. **Expected messages**:
   - `Building image...`
   - `Running container...`
   - `Application started on port 7860`

3. **Check status**:
   - Green checkmark = Success ✅
   - Red error = Build failed, check logs

## Step 8: Access Your Space

Once deployed:

- **Public URL**: Click "View Space" button
- **Direct URL**: `https://huggingface.co/spaces/YOUR_USERNAME/stock-forecasting`

## Troubleshooting

### Build Fails with "Module not found"

- Check `requirements.txt` for typos
- Ensure all dependencies have versions pinned
- Run locally: `pip install -r requirements.txt`

### App crashes immediately

- Check Dockerfile EXPOSE port is 7860
- Verify `app.py` runs on `0.0.0.0:7860`
- Check logs for Python errors

### App runs but shows "Application Error"

- Verify Flask app imports work correctly
- Check that templates/ folder exists
- Ensure no file I/O operations

### Health check fails

- Ensure app responds within 30s to `/`
- Check logs for hung requests
- May need to increase timeout

### Out of memory

- Reduce number of workers: `--workers=1`
- Reduce threading: `--threads=1`
- Check for memory leaks in Prophet/TensorFlow

## Environment Variables

Set via Space settings:

1. Go to Space Settings → Variables
2. Add secret variables (not visible in logs):
   ```
   SECRET_KEY: your-production-secret-key
   ```

Or edit `settings.json` in the Space:

```json
{
  "variables": {
    "PORT": "7860",
    "SECRET_KEY": "production-secret"
  }
}
```

## Updates and Redeployment

To update your Space:

```bash
# Make local changes
# Update code files

# Commit and push
git add .
git commit -m "Update: describe your changes"
git push origin main

# Hugging Face automatically rebuilds and deploys
```

## Resource Limits

### CPU Basic (Free)

- ✅ 2 vCPU
- ✅ 16GB RAM
- ✅ 50GB storage (ephemeral)
- ⚠️ ~30min persistence guarantee
- ✅ Suitable for this app

### Upgrade to GPU if needed

- Settings → Hardware → Select T4 or A10G
- Charged per hour of active Space

## Cost Estimation

### Free Tier (CPU Basic)

- **Cost**: Free
- **Limitations**: Space pauses after 48 hours of inactivity
- **Best for**: Demo and development

### Paid Tier (Persistent)

- **Cost**: $7-20/month depending on hardware
- **Benefit**: Always running, guaranteed persistent
- **Best for**: Production deployments

## Security Checklist

- [x] No hardcoded secrets in code
- [x] Use environment variables for API keys
- [x] `.env` file in `.gitignore`
- [x] Never commit `.env` to git
- [x] Use long, random SECRET_KEY in production
- [x] HTTPS enabled automatically by HF
- [x] No file system writes (data leaked possible)

## Monitoring

### Log Health

```
# Via HF Spaces UI
Settings → Logs tab → view real-time logs
```

### Check Resource Usage

```
# Via HF Spaces UI
Settings → Monitoring → View CPU, RAM, Disk
```

## Advanced: CI/CD Integration

To auto-deploy on git push (optional):

```bash
# Add GitHub Actions workflow (in .github/workflows/deploy.yml)
name: Deploy to HF Spaces
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Push to HF Spaces
        run: |
          git config user.email "action@github.com"
          git config user.name "GitHub Action"
          git remote add huggingface https://huggingface.co/spaces/${{ secrets.HF_USERNAME }}/stock-forecasting
          git push huggingface main --force
```

## Quick Reference

| Task              | Command                                             |
| ----------------- | --------------------------------------------------- |
| Create new Space  | Visit huggingface.co/new-space                      |
| Clone Space repo  | `git clone https://huggingface.co/spaces/USER/NAME` |
| Deploy            | `git push origin main`                              |
| View logs         | Click "Logs" in Space settings                      |
| View app          | Click "View Space" button                           |
| Update SECRET_KEY | Space Settings → Variables                          |
| Force rebuild     | Edit Dockerfile, then push                          |
| Delete Space      | Space Settings → Delete                             |

## Support

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Issues**: Create issue on GitHub or HF Spaces
- **Discord**: Join HF Discord for community support

---

**Last Updated**: 2024
**Difficulty**: Beginner ⭐⭐
**Estimated Time**: 10-15 minutes
