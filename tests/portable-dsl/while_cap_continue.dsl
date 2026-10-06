# me:compiler=tcc
def while_cap_continue(x):
    n = 0
    while n < x:
        n = n + 1
        continue
    return n
