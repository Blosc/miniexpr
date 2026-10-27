# me:compiler=tcc
def masked_local_chain(x):
    n = 0
    if x == 0:
        n = 1
    return 0 <= n < x
