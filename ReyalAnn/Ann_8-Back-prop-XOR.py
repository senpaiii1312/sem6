import numpy as np

np.random.seed(0)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

w1 = np.random.randn(2, 2)
b1 = np.random.randn(1, 2)
w2 = np.random.randn(2, 1)
b2 = np.random.randn(1, 1)

learning_rate = 0.1
epochs = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for _ in range(epochs):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    output = sigmoid(z2)

    error = y - output

    delta_output = error * sigmoid_derivative(output)
    delta_hidden = np.dot(delta_output, w2.T) * sigmoid_derivative(a1)

    w2 += np.dot(a1.T, delta_output) * learning_rate
    b2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

    w1 += np.dot(X.T, delta_hidden) * learning_rate
    b1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

print("Final XOR Output after Training:")
print(np.round(output))