from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

key = get_random_bytes(16)
cipher = AES.new(key, AES.MODE_EAX)

data = b"Hello World"
ciphertext, tag = cipher.encrypt_and_digest(data)

print("Encrypted:", ciphertext)

# Decryption
cipher = AES.new(key, AES.MODE_EAX, nonce=cipher.nonce)
plaintext = cipher.decrypt(ciphertext)

print("Decrypted:", plaintext.decode())