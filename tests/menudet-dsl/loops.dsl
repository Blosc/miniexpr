# me:compiler=tcc
def loops(x):
    total = 0.0
    for i in range(3):
        total += x
    j = 0
    while j < 2:
        total += 1.0
        j += 1
    return total
