import numpy as np

X = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
])

y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

weights = np.zeros(4)
bias = 0
learning_rate = 0.1

def step(x):
    return 1 if x >= 0 else 0

epochs = 50

for _ in range(epochs):
    for xi, target in zip(X, y):
        net = np.dot(xi, weights) + bias
        output = step(net)
        error = target - output
        weights += learning_rate * error * xi
        bias += learning_rate * error

print("Digit | ASCII | Prediction")
print("---------------------------")

for i in range(10):
    binary = X[i]
    net = np.dot(binary, weights) + bias
    prediction = step(net)
    result = "Odd" if prediction == 1 else "Even"
    print(f"  {i}   |  {ord(str(i))}  |  {result}")



    