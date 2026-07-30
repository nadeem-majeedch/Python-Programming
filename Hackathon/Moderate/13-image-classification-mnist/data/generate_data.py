"""Generate MNIST-like digit dataset for image classification."""

import pandas as pd
import os
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

indices = __import__('numpy').random.RandomState(42).choice(len(X), 500, replace=False)
df = pd.DataFrame(X[indices])
df['label'] = y[indices]
df.columns = [f'pixel{i}' for i in range(64)] + ['label']

out_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'digits.csv')
df.to_csv(out_path, index=False)
print(f"Generated {out_path} ({len(df)} rows)")
