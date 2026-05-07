import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
    [-1, -1],
    [-2, -1],
    [-2, -2]
])

y = np.array([1, 1, 1, 0, 0, 0])

weights = np.zeros(2)
bias = 0
learning_rate = 0.1
epochs = 100

for _ in range(epochs):
    for xi, target in zip(X, y):
        net = np.dot(xi, weights) + bias
        output = 1 if net >= 0 else 0
        error = target - output
        weights += learning_rate * error * xi
        bias += learning_rate * error

xx, yy = np.meshgrid(
    np.linspace(-3, 3, 300),
    np.linspace(-3, 3, 300)
)

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z >= 0, alpha=0.3)

plt.scatter(X[:3, 0], X[:3, 1], color="blue", label="Class 1")
plt.scatter(X[3:, 0], X[3:, 1], color="red", label="Class 0")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron Decision Regions")
plt.legend()
plt.show()
