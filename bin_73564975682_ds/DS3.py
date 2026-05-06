# =========================================================
# Descriptive Statistics
# Measures of Central Tendency and Variability
# =========================================================

# ---------------------------------------------------------
# 1. Import Required Libraries
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns

# ---------------------------------------------------------
# 2. Load Iris Dataset
# ---------------------------------------------------------

df = sns.load_dataset("iris")

print("First 5 Rows:\n")
print(df.head())

# ---------------------------------------------------------
# 3. Dataset Information
# ---------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ---------------------------------------------------------
# 4. Summary Statistics Grouped by Species
# ---------------------------------------------------------

grouped = df.groupby('species')

# Mean
print("\nMean Values Grouped by Species:\n")
print(grouped.mean())

# Median
print("\nMedian Values Grouped by Species:\n")
print(grouped.median())

# Minimum
print("\nMinimum Values Grouped by Species:\n")
print(grouped.min())

# Maximum
print("\nMaximum Values Grouped by Species:\n")
print(grouped.max())

# Standard Deviation
print("\nStandard Deviation Grouped by Species:\n")
print(grouped.std())

# ---------------------------------------------------------
# 5. Create List of Sepal Length for Each Species
# ---------------------------------------------------------

species_list = {}

for species in df['species'].unique():
    
    species_list[species] = list(
        df[df['species'] == species]['sepal_length']
    )

print("\nSepal Length List for Each Species:\n")

for key, value in species_list.items():
    print(key, ":", value)

# ---------------------------------------------------------
# 6. Basic Statistical Details
# ---------------------------------------------------------

print("\nComplete Statistical Details:\n")

print(df.describe())

# ---------------------------------------------------------
# 7. Percentiles
# ---------------------------------------------------------

print("\n25th Percentile:\n")
print(df.quantile(0.25, numeric_only=True))

print("\n50th Percentile (Median):\n")
print(df.quantile(0.50, numeric_only=True))

print("\n75th Percentile:\n")
print(df.quantile(0.75, numeric_only=True))

# ---------------------------------------------------------
# 8. Mean
# ---------------------------------------------------------

print("\nMean of Numeric Columns:\n")
print(df.mean(numeric_only=True))

# ---------------------------------------------------------
# 9. Standard Deviation
# ---------------------------------------------------------

print("\nStandard Deviation of Numeric Columns:\n")
print(df.std(numeric_only=True))