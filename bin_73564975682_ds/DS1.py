import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv(r"C:\Users\Sujal\Desktop\ultimate_sem6\bin_73564975682_ds\titanic.csv")

# -----------------------------
# Normalize column names (IMPORTANT FIX)
# -----------------------------
df.columns = df.columns.str.strip().str.lower()

# Basic Info
print("First 5 Rows\n", df.head())
print("\nDataset Shape\n", df.shape)
print("\nColumn Names\n", df.columns)
print("\nData Types\n", df.dtypes)
print("\nMissing Values\n", df.isnull().sum())
print("\nStatistical Summary\n", df.describe(include="all"))

# -----------------------------
# Handling Missing Values (SAFE for both datasets)
# -----------------------------
if "age" in df.columns:
    df["age"].fillna(df["age"].mean(), inplace=True)

if "embarked" in df.columns:
    df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

if "cabin" in df.columns:
    df["cabin"].fillna("unknown", inplace=True)

if "deck" in df.columns:
    df["deck"].fillna("unknown", inplace=True)

if "embark_town" in df.columns:
    df["embark_town"].fillna(df["embark_town"].mode()[0], inplace=True)

print("\nMissing Values After Filling\n", df.isnull().sum())

# -----------------------------
# Data Type Conversion (only if present)
# -----------------------------
type_map = {
    "survived": "int",
    "pclass": "int",
    "fare": "float"
}

for col, dtype in type_map.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)

# -----------------------------
# Label Encoding (only existing columns)
# -----------------------------
label = LabelEncoder()

categorical_cols = [
    "sex", "embarked", "class", "who",
    "deck", "embark_town", "alive",
    "alone", "adult_male", "cabin"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = label.fit_transform(df[col])

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = MinMaxScaler()

scale_cols = ["age", "fare"]

for col in scale_cols:
    if col in df.columns:
        df[[col]] = scaler.fit_transform(df[[col]])

# -----------------------------
# Final Output
# -----------------------------
print("\nFormatted Data Types\n", df.dtypes)
print("\nFinal Dataset Sample\n", df.head())