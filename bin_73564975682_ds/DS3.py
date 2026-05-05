import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\iris.csv")

print("First Five Records")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nSummary Statistics Grouped by Species")
grouped = df.groupby("species")["sepal_length"].agg(
    ["mean", "median", "min", "max", "std"]
)

print(grouped)

species_numeric = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

df["species_value"] = df["species"].map(species_numeric)

print("\nNumeric Value for Each Category")
print(df[["species", "species_value"]].drop_duplicates())

species_list = df["species_value"].tolist()

print("\nList of Numeric Values")
print(species_list)

print("\nStatistical Details of Each Species")

for flower in df["species"].unique():
    print("\nSpecies:", flower)

    subset = df[df["species"] == flower]

    print(subset.describe())

    print("Percentiles")
    print(subset.quantile([0.25, 0.50, 0.75], numeric_only=True))