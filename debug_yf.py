import yfinance as yf
import pandas as pd
import numpy as np

data = yf.download('PGEL.NS', period='3y', auto_adjust=True)
print('columns:', data.columns.tolist())
print('index type:', type(data.index))
print('head:')
print(data.head())
print('Date in reset index?')
df = data.reset_index()
print(df.dtypes)
print(df.head())
print('Date column sample element type:', type(df['Date'].iloc[0]))
print('Close dtype:', df['Close'].dtype)
print('Close values shape:', getattr(df['Close'].values, 'shape', None))
