import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
import pandas as pd

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

f_values, p_values = f_classif(X, y)

df = pd.DataFrame({
    'Feature': feature_names,
    'F-Value': f_values,
    'P-Value': p_values
})

print(df)
