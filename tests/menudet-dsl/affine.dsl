# me:compiler=tcc
def affine(x, scale, offset):
    y = x * scale + offset
    if y < 0:
        return 0.0
    return y
