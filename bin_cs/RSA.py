def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mod_inverse(e, phi):
    for d in range(1, phi):
        if (d * e) % phi == 1:
            return d

p = 3
q = 11

n = p * q
phi = (p - 1) * (q - 1)

e = 3
while gcd(e, phi) != 1:
    e += 1

d = mod_inverse(e, phi)

print("Public Key:", (e, n))
print("Private Key:", (d, n))

msg = 7

# Encryption
c = (msg ** e) % n
print("Encrypted:", c)

# Decryption
m = (c ** d) % n
print("Decrypted:", m)