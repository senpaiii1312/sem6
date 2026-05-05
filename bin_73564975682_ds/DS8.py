import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

print("First Five Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

plt.figure(figsize=(8, 5))
sns.countplot(x="survived", data=df)
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival Based on Gender")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="pclass", hue="survived", data=df)
plt.title("Passenger Class vs Survival")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["fare"].dropna(), bins=30, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()