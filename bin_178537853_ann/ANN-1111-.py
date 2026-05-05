import numpy as np

digits = {
    0: np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]),

    1: np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1]
    ]),

    2: np.array([
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ]),

    3: np.array([
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]),

    9: np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])
}

X = np.array([digits[d].flatten() for d in [0, 1, 2, 3, 9]])
y = np.array([0, 1, 2, 3, 9])

num_classes = 10
targets = np.zeros((len(y), num_classes))

for i in range(len(y)):
    targets[i, y[i]] = 1

weights = np.random.rand(15, num_classes) - 0.5
bias = np.random.rand(num_classes) - 0.5

learning_rate = 0.1
epochs = 500

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

for _ in range(epochs):
    for xi, target in zip(X, targets):
        net = np.dot(xi, weights) + bias
        output = softmax(net)

        error = target - output

        weights += learning_rate * np.outer(xi, error)
        bias += learning_rate * error

print("Training Completed\n")

for digit in [0, 1, 2, 3, 9]:
    test = digits[digit].flatten()
    net = np.dot(test, weights) + bias
    output = softmax(net)
    prediction = np.argmax(output)

    print("Input Digit:")
    print(digits[digit])
    print("Recognized as:", prediction)
    print() 