# me:compiler=tcc
def masked_bool_chain(x):
    flag = bool(0)
    if x == 0:
        flag = bool(1)
    return bool(0) <= flag < bool(x)
