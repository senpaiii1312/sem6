def mc_pitts_and_not(A, B):
    w1 = 1      
    w2 = -1    
    threshold = 1


    net = w1 * A + w2 * B

    
    if net >= threshold:
        return 1
    else:
        return 0


print("A B | AND-NOT")
print("-------------")

for A in [0, 1]:
    for B in [0, 1]:
        output = mc_pitts_and_not(A, B)
        print(A, B, "|", output)