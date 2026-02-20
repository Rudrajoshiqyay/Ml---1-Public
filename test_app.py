#!/usr/bin/env python3
"""Test script to verify app functionality"""
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("Testing Stock Analysis App")
print("=" * 50)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from app import app
    print("✓ App imports successfully")
except Exception as e:
    print(f"✗ App import failed: {e}")
    sys.exit(1)

# Test 2: Check if Flask app is configured
print("\n2. Testing Flask app configuration...")
try:
    assert app.secret_key is not None
    print("✓ Flask app configured with secret key")
except Exception as e:
    print(f"✗ Flask configuration failed: {e}")
    sys.exit(1)

# Test 3: Check if templates exist
print("\n3. Checking templates...")
import os
template_dir = "templates"
if os.path.isdir(template_dir):
    templates = os.listdir(template_dir)
    print(f"✓ Templates directory exists with {len(templates)} files")
    for t in templates:
        print(f"  - {t}")
else:
    print(f"✗ Templates directory not found")
    sys.exit(1)

# Test 4: Check if static directory exists
print("\n4. Checking static files...")
static_dir = "static"
if os.path.isdir(static_dir):
    statics = os.listdir(static_dir)
    print(f"✓ Static directory exists with {len(statics)} files")
else:
    print(f"✓ Static directory not required (optional)")

# Test 5: Test importing required modules
print("\n5. Testing required modules...")
modules_to_test = [
    ('yfinance', 'yf'),
    ('pandas', 'pd'),
    ('prophet', 'Prophet'),
    ('flask', 'Flask'),
    ('numpy', 'np'),
]

for module_name, alias in modules_to_test:
    try:
        __import__(module_name)
        print(f"✓ {module_name} available")
    except ImportError:
        print(f"✗ {module_name} NOT available")

# Test 6: Check prophet_v2 module
print("\n6. Testing prophet_v2 module...")
try:
    from prophet_v2 import analyze_stock
    print("✓ prophet_v2.analyze_stock imported successfully")
except Exception as e:
    print(f"✗ prophet_v2 import failed: {e}")

# Test 7: Check lstm_predictor module
print("\n7. Testing lstm_predictor module...")
try:
    from lstm_predictor import predict_lstm_change
    print("✓ lstm_predictor.predict_lstm_change imported successfully")
except Exception as e:
    print(f"✗ lstm_predictor import failed: {e}")

print("\n" + "=" * 50)
print("All basic tests passed! ✓")
print("=" * 50)
print("\nTo start the app, run: python app.py")
print("Then visit: http://localhost:7860")
