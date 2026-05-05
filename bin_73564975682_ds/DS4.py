import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
raw_df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\housing.csv")

# 🔥 Ensure even rows (VERY IMPORTANT)
if len(raw_df) % 2 != 0:
    raw_df = raw_df.iloc[:-1]

# Create X and y
X = np.hstack([
    raw_df.values[::2, :],
    raw_df.values[1::2, :2]
])

y = raw_df.values[1::2, 2]

# 🔥 CHECK SHAPE
print("Shape of X:", X.shape)

# Create dynamic column names (AUTO FIX)
columns = [f"Feature_{i+1}" for i in range(X.shape[1])]

X = pd.DataFrame(X, columns=columns)

print("\nFirst Five Rows")
print(X.head())

print("\nShape of Dataset")
print(X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("\nCoefficients")
print(model.coef_)

print("\nIntercept")
print(model.intercept_)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.grid(True)
plt.show()