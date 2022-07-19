import itertools

def Cartesian_product(a,b):
    cartesian_set = []
    for result in itertools.product(a,b):
        cartesian_set.append(list(eval('%s'%repr(result).replace('[', '').replace(']', ''))))
    return cartesian_set

def iter_all_order(alter_block_depth):
    a = alter_block_depth
    elem_a = []
    for num_a in a:
        elem_a.append([p for p in range(num_a)])

    set_1 = elem_a[0]
    for i_a in range(len(a)-1):
        set_2 = elem_a[i_a+1]
        set_1 = Cartesian_product(set_1, set_2)

    return set_1

