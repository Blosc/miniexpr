# me:compiler=tcc
def return_after_math(x):
    y = sin(x)
    z = cos(x)
    if x > 0:
        return y + z
