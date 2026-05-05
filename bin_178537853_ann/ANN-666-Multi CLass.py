import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

X_val = X_train[:5000]
y_val = y_train[:5000]

X_train_new = X_train[5000:]
y_train_new = y_train[5000:]

def build_basic_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_tuned_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same",
                     activation="relu", input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

basic_model = build_basic_model()

history_basic = basic_model.fit(
    X_train_new, y_train_new,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train_new)

tuned_model = build_tuned_model()

history_tuned = tuned_model.fit(
    datagen.flow(X_train_new, y_train_new, batch_size=64),
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)

loss1, acc1 = basic_model.evaluate(X_test, y_test, verbose=0)
loss2, acc2 = tuned_model.evaluate(X_test, y_test, verbose=0)

print("Basic Model Test Accuracy:", acc1)
print("Tuned Model Test Accuracy:", acc2)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_basic.history["accuracy"], label="Basic Train")
plt.plot(history_basic.history["val_accuracy"], label="Basic Val")
plt.plot(history_tuned.history["accuracy"], label="Tuned Train")
plt.plot(history_tuned.history["val_accuracy"], label="Tuned Val")
plt.title("Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_basic.history["loss"], label="Basic Train")
plt.plot(history_basic.history["val_loss"], label="Basic Val")
plt.plot(history_tuned.history["loss"], label="Tuned Train")
plt.plot(history_tuned.history["val_loss"], label="Tuned Val")
plt.title("Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()