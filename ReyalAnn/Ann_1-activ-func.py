import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10, 10, 400)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def tanh(x):
    return np.tanh(x)

plt.figure()
plt.plot(x, sigmoid(x), label="Sigmoid", color="blue")
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x, relu(x), label="ReLU", color="red")
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x, linear(x), label="Linear", color="green")
plt.title("Linear Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x, tanh(x), label="Tanh", color="purple")
plt.title("Tanh Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()