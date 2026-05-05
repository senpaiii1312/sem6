import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\iris.csv")

print(df.head())

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("\nAccuracy =", accuracy)
print("Error Rate =", error_rate)

TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (TP + FP + FN)

for i in range(len(model.classes_)):
    precision = TP[i] / (TP[i] + FP[i])
    recall = TP[i] / (TP[i] + FN[i])

    print("\nClass =", model.classes_[i])
    print("TP =", TP[i])
    print("FP =", FP[i])
    print("TN =", TN[i])
    print("FN =", FN[i])
    print("Precision =", precision)
    print("Recall =", recall)