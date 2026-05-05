import random

p = 23   # prime number
g = 5    # generator

# Alice private key
a = random.randint(1, 10)
A = (g ** a) % p

# Bob private key
b = random.randint(1, 10)
B = (g ** b) % p

# Shared secret
key_A = (B ** a) % p
key_B = (A ** b) % p

print("Alice Key:", key_A)
print("Bob Key:", key_B)