# =========================================================
# Data Wrangling - I
# Dataset Used: Titanic Dataset
# Using get_dummies() for Categorical Encoding
# =========================================================

# ---------------------------------------------------------
# 1. Import Required Libraries
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------

# Reading CSV file
df = pd.read_csv(r"C:\Users\Sujal\Desktop\ultimate_sem6\bin_73564975682_ds\titanic.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:\n")
print(df.head())

# ---------------------------------------------------------
# 3. Check Dataset Information
# ---------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

# ---------------------------------------------------------
# 4. Data Preprocessing
# ---------------------------------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Data Types
print("\nData Types:")
print(df.dtypes)

# ---------------------------------------------------------
# Handling Missing Values
# ---------------------------------------------------------

# Fill missing Age values with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 5. Data Formatting and Normalization
# ---------------------------------------------------------

# Convert Survived column into integer type
df['Survived'] = df['Survived'].astype(int)

print("\nUpdated Data Types:")
print(df.dtypes)

# Normalize Age and Fare columns
scaler = MinMaxScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\nNormalized Age and Fare Columns:")
print(df[['Age', 'Fare']].head())

# ---------------------------------------------------------
# 6. Convert Categorical Variables into Numeric
#    Using get_dummies()
# ---------------------------------------------------------

# Convert Sex column into dummy variables
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Convert Embarked column into dummy variables
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nDataset After Encoding:")
print(df.head())

# ---------------------------------------------------------
# Final Dataset Information
# ---------------------------------------------------------

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Data Types:")
print(df.dtypes)