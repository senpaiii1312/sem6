def encrypt(plain, key):
    # simple XOR encryption
    cipher = ""
    for i in range(len(plain)):
        cipher += chr(ord(plain[i]) ^ ord(key[i % len(key)]))
    return cipher

def decrypt(cipher, key):
    # same XOR used for decryption
    plain = ""
    for i in range(len(cipher)):
        plain += chr(ord(cipher[i]) ^ ord(key[i % len(key)]))
    return plain

text = "HELLO"
key = "KEY"

enc = encrypt(text, key)
print("Encrypted:", enc)

dec = decrypt(enc, key)
print("Decrypted:", dec)