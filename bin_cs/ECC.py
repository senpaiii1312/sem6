import random

# simple curve y^2 = x^3 + ax + b
a = 2
b = 3
p = 97  # prime

# base point
G = (3, 6)

# private key
d = random.randint(1, 10)

# public key (simple multiplication)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
Q = (d * G[0] % p, d * G[1] % p)

print("Private Key:", d)
print("Public Key:", Q)