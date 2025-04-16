import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2]

q1 = np.percentile(y, 25)
q3 = np.percentile(y, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

mask = (y >= lower_bound) & (y <= upper_bound)
X_filtered = X[mask]
y_filtered = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="green", label="Actual (Filtered)")
plt.plot(X_test, y_pred, color="orange", linewidth=2, label="Predicted")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease Progression")
plt.title("Linear Regression After Outlier Removal")
plt.legend()
plt.grid(True)
plt.show()
