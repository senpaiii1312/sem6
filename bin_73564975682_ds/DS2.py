import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

np.random.seed(10)

data = {
    "Roll_No": range(1, 21),
    "Name": [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"
    ],
    "Maths": [78, 85, 90, np.nan, 66, 72, 88, 95, 43, 77, 82, 91, 58, 64, 70, 300, 76, 84, 69, 73],
    "Science": [80, 79, 92, 88, 67, np.nan, 85, 94, 40, 76, 81, 89, 60, 62, 71, 96, 75, 83, 68, 72],
    "English": [75, 82, 88, 84, 65, 70, 86, 91, 45, 74, 80, 87, 59, 61, 69, 93, 74, 81, 66, 71],
    "Attendance": [92, 95, 98, 90, 85, 87, 93, 97, 70, 88, 91, 96, 82, 84, 86, 99, 89, 94, 83, 85]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

print("\nMissing Values")
print(df.isnull().sum())

# ✅ FIX 1: removed inplace (better practice)
df["Maths"] = df["Maths"].fillna(df["Maths"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())

# Handling inconsistencies (same logic)
df["Maths"] = df["Maths"].clip(lower=0, upper=100)
df["Science"] = df["Science"].clip(lower=0, upper=100)
df["English"] = df["English"].clip(lower=0, upper=100)
df["Attendance"] = df["Attendance"].clip(lower=0, upper=100)

print("\nAfter Handling Missing Values and Inconsistencies")
print(df)

numeric_cols = ["Maths", "Science", "English", "Attendance"]

# Outlier detection (Z-score)
z_scores = np.abs(zscore(df[numeric_cols]))
outliers = (z_scores > 3)

print("\nOutlier Positions")
print(outliers)

# Outlier treatment (IQR)
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # ✅ FIX 2: used np.clip instead of two np.where (cleaner, same logic)
    df[col] = np.clip(df[col], lower, upper)

print("\nAfter Outlier Treatment")
print(df)

# Data transformation
df["Log_Maths"] = np.log(df["Maths"] + 1)

print("\nAfter Data Transformation")
print(df[["Maths", "Log_Maths"]])

# Visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df["Maths"], bins=10)
plt.title("Original Maths Marks")

plt.subplot(1, 2, 2)
plt.hist(df["Log_Maths"], bins=10)
plt.title("Log Transformed Maths Marks")

plt.tight_layout()
plt.show()