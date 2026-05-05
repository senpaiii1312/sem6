import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(16, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Neural Network Accuracy:", accuracy)

logistic_model = Sequential()

logistic_model.add(Dense(1, activation="sigmoid", input_shape=(X_train.shape[1],)))

logistic_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

logistic_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

pred = logistic_model.predict(X_test)
pred = (pred > 0.5).astype(int)

print("Logistic Regression Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))