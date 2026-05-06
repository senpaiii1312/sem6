import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Users\Sujal\Desktop\ultimate_sem6\bin_73564975682_ds\titanic.csv")

print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df.describe())   

print(df.dtypes)
    
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())

print("\nAge after filling in missing values\n")
print(df["Age"].head())

scaler = MinMaxScaler()

df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

print("\nNormalized Age and Fare Columns:")
print(df[['Age', 'Fare']].head())

df["Survived"] = df["Survived"].astype(int)
print("\nData type changed\n")
print(df.dtypes)
