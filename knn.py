import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, marker='o', label="Actual", edgecolor="k")
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_pred, marker='x', label="Predicted")
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])
plt.title("KNN Classification on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
