from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(X_train,y_train),(X_test,y_test)=mnist.load_data()

X_train=X_train/255.0
X_test=X_test/255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

# Neural Network Architecture
model=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(100,activation='relu'),   # Hidden layer with 100 neurons
    Dense(10,activation='softmax')  # Multi-class output layer
])

# Optimization
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train,y_train,epochs=5)

# Test
loss,acc=model.evaluate(X_test,y_test)
print("Accuracy:",acc)