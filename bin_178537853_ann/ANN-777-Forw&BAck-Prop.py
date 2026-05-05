import numpy as np

np.random.seed(0)

num_samples = 300
num_features = 2
num_classes = 3

X = np.random.randn(num_samples, num_features)
y = np.zeros((num_samples, num_classes))

for i in range(num_samples):
    cls = i % num_classes
    y[i, cls] = 1
    X[i] += cls * 2

X = (X - X.mean(axis=0)) / X.std(axis=0)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

input_size = num_features
hidden_size = 100
output_size = num_classes

learning_rate = 0.001
epochs = 1000

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

for epoch in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    output = softmax(z2)

    loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))

    error_output = output - y
    dW2 = np.dot(a1.T, error_output)
    db2 = np.sum(error_output, axis=0, keepdims=True)

    error_hidden = np.dot(error_output, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X.T, error_hidden)
    db1 = np.sum(error_hidden, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

predicted_class = np.argmax(output, axis=1)
actual_class = np.argmax(y, axis=1)

accuracy = np.mean(predicted_class == actual_class)

print("\nFinal Classification Accuracy:", accuracy)