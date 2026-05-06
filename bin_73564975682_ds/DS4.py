# =========================================================
# Data Analytics I
# Linear Regression on Boston Housing Dataset
# =========================================================

# ---------------------------------------------------------
# 1. Import Required Libraries
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------

# Read dataset
df = pd.read_csv(r"C:\Users\Sujal\Desktop\ultimate_sem6\bin_73564975682_ds\housing.csv")

print("First 5 Rows:\n")
print(df.head())

# ---------------------------------------------------------
# 3. Check Dataset Information
# ---------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:\n")
print(df.isnull().sum())

# ---------------------------------------------------------
# 4. Handle Missing Values
# ---------------------------------------------------------

# Fill missing values with mean

df = df.fillna(df.mean())

print("\nMissing Values After Handling:\n")
print(df.isnull().sum())

# ---------------------------------------------------------
# 5. Define Features and Target Variable
# ---------------------------------------------------------

# Features (Independent Variables)
X = df.drop("medv", axis=1)

# Target Variable (Dependent Variable)
y = df["medv"]

print("\nFeatures:\n")
print(X.head())

print("\nTarget:\n")
print(y.head())

# ---------------------------------------------------------
# 6. Split Dataset into Training and Testing Data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ---------------------------------------------------------
# 7. Create Linear Regression Model
# ---------------------------------------------------------

model = LinearRegression()

# ---------------------------------------------------------
# 8. Train Model
# ---------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ---------------------------------------------------------
# 9. Predict House Prices
# ---------------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredicted Prices:\n")
print(y_pred[:10])

# ---------------------------------------------------------
# 10. Model Evaluation
# ---------------------------------------------------------

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)

# ---------------------------------------------------------
# 11. Compare Actual vs Predicted Values
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:\n")
print(comparison.head(10))