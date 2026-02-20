#!/usr/bin/env python3
"""Test stock analysis functionality"""
import warnings
warnings.filterwarnings('ignore')

print("Testing stock analysis with sample ticker...")

try:
    from prophet_v2 import analyze_stock
    
    # Test with a simple ticker
    result = analyze_stock("RELIANCE.NS")
    
    print(f"Analysis success: {result.get('success')}")
    if result.get('success'):
        print(f"✓ Analysis completed for {result.get('ticker')}")
        print(f"  Current Price: {result.get('current_price')}")
        print(f"  Summary: {result.get('summary')[:100]}...")
    else:
        print(f"✗ Analysis failed: {result.get('error')}")
        
except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()
