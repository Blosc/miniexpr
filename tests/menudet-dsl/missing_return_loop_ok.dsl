# me:compiler=tcc
def return_in_loop(x):
    for i in range(3):
        if x > 0:
            return x + 1.0
        if x == 0:
            break
