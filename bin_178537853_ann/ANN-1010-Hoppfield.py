import numpy as np

patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    W += np.outer(p, p)

np.fill_diagonal(W, 0)

print("Weight Matrix:")
print(W)

test = np.array([1, -1, 1, 1])

print("\nInput Vector:")
print(test)

state = test.copy()

for _ in range(5):
    for i in range(n):
        net = np.dot(W[i], state)
        state[i] = 1 if net >= 0 else -1

print("\nRecovered Vector:")
print(state)