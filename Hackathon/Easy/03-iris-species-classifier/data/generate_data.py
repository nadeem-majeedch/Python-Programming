#!/usr/bin/env python3
import pandas as pd
import os
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

out = os.path.join(os.getcwd(), "iris.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
