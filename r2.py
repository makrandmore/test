import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("R² Score:", r2)
