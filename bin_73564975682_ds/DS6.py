# =========================================================
# Data Analytics III
# Naive Bayes Classification on Iris Dataset
# =========================================================

# ---------------------------------------------------------
# 1. Import Required Libraries
# ---------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(r"C:\Users\Sujal\Desktop\ultimate_sem6\bin_73564975682_ds\iris.csv")

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
# 4. Select Features and Target Variable
# ---------------------------------------------------------

# Input Features
X = df.drop("species", axis=1)

# Target Variable
y = df["species"]

print("\nFeatures:\n")
print(X.head())

print("\nTarget Variable:\n")
print(y.head())

# ---------------------------------------------------------
# 5. Split Dataset into Training and Testing
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ---------------------------------------------------------
# 6. Create Naive Bayes Model
# ---------------------------------------------------------

model = GaussianNB()

# ---------------------------------------------------------
# 7. Train Model
# ---------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ---------------------------------------------------------
# 8. Predict Test Results
# ---------------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredicted Values:\n")
print(y_pred)

# ---------------------------------------------------------
# 9. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# ---------------------------------------------------------
# 10. Extract TP, FP, TN, FN
# ---------------------------------------------------------

# Using first class values

TP = cm[0][0]

FP = cm[1][0]

FN = cm[0][1]

TN = cm[1][1]

print("\nTrue Positive (TP):", TP)

print("False Positive (FP):", FP)

print("False Negative (FN):", FN)

print("True Negative (TN):", TN)

# ---------------------------------------------------------
# 11. Calculate Accuracy
# ---------------------------------------------------------

accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\nAccuracy:", accuracy)

# ---------------------------------------------------------
# 12. Calculate Error Rate
# ---------------------------------------------------------

error_rate = (FP + FN) / (TP + TN + FP + FN)

print("Error Rate:", error_rate)

# ---------------------------------------------------------
# 13. Calculate Precision
# ---------------------------------------------------------

precision = TP / (TP + FP)

print("Precision:", precision)

# ---------------------------------------------------------
# 14. Calculate Recall
# ---------------------------------------------------------

recall = TP / (TP + FN)

print("Recall:", recall)