# =========================================================
# Weather Data Analysis
# Find Average Temperature, Dew Point, Wind Speed
# =========================================================

# ---------------------------------------------------------
# 1. Import Required Library
# ---------------------------------------------------------

import pandas as pd

# ---------------------------------------------------------
# 2. Read Weather Dataset
# ---------------------------------------------------------

# Read text file
df = pd.read_csv(r"C:\Users\Sujal\Desktop\ultimate_sem6\bin_73564975682_ds\sample_weather.txt", sep=" ")

# Display first 5 rows
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
# 4. Calculate Average Values
# ---------------------------------------------------------

# Average Temperature
avg_temp = df['Temperature'].mean()

# Average Dew Point
avg_dew = df['DewPoint'].mean()

# Average Wind Speed
avg_wind = df['WindSpeed'].mean()

# ---------------------------------------------------------
# 5. Display Results
# ---------------------------------------------------------

print("\nAverage Temperature:")
print(avg_temp)

print("\nAverage Dew Point:")
print(avg_dew)

print("\nAverage Wind Speed:")
print(avg_wind)