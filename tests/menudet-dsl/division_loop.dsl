# me:compiler=tcc
def divide_loop(x):
    result = x
    for i in range(1, 4):
        result = result + i / 2
    return result
