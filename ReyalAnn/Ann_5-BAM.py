import numpy as np

# Bipolar representation (-1, +1)
X = np.array([
    [1, -1, 1],
    [-1, 1, -1]
])

Y = np.array([
    [1, 1],
    [-1, -1]
])

# Step 1: Compute Weight Matrix
W = np.zeros((X.shape[1], Y.shape[1]))

for i in range(len(X)):
    W += np.outer(X[i], Y[i])

print("Weight Matrix:")
print(W)

# Activation function
def sign(x):
    return np.where(x >= 0, 1, -1)

# Recall function
def bam_recall(x_input, W):
    y = sign(np.dot(x_input, W))
    x_reconstructed = sign(np.dot(y, W.T))
    return y, x_reconstructed

print("\nTesting BAM:")
for i in range(len(X)):
    y_out, x_out = bam_recall(X[i], W)
    print(f"Input X: {X[i]}")
    print(f"Recalled Y: {y_out}")
    print(f"Reconstructed X: {x_out}")
    print()
    