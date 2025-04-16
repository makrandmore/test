import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

data = load_iris()
X = data.data
y = data.target

model = LogisticRegression(max_iter=200)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold)

print("Cross Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cv_scores)+1), cv_scores, marker='o', linestyle='-', label="Fold Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
