import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

print("First Five Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nFeature Types")
print(df.dtypes)

print("\nInference:")
print("sepal_length   -> Numeric")
print("sepal_width    -> Numeric")
print("petal_length   -> Numeric")
print("petal_width    -> Numeric")
print("species        -> Nominal")

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

for col in features:
    plt.figure(figsize=(7, 4))
    plt.hist(df[col], bins=15, edgecolor="black")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

for col in features:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[col])
    
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

print("\nComparison and Outlier Detection:")
print("1. Petal length and petal width show clear grouped distributions.")
print("2. Sepal width contains some outliers.")
print("3. Sepal length has moderate spread.")
print("4. Petal features separate species better than sepal features.")
print("5. Boxplots clearly show extreme values in sepal_width.")