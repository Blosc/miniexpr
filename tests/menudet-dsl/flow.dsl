# me:compiler=tcc
def flow(x):
    total = 0.0
    for i in range(5):
        if i == 1:
            continue
        if i == 4:
            break
        total += x
    return total
