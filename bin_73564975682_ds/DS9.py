import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

print("First Five Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))

sns.boxplot(
    x="sex",
    y="age",
    hue="survived",
    data=df
)

plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

print("\nObservations:")
print("1. Female passengers had higher survival compared to males.")
print("2. Many children were among survivors.")
print("3. Male non-survivors are higher in count.")
print("4. Median age differs slightly between groups.")
print("5. Several outliers are visible in higher age ranges.")