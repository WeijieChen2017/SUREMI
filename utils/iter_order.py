import itertools
import time

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

def iter_some_order(alter_block, order_need):
    # alter_block = [4,2,2,2,2,1,1,1,1,1]
    # alter_block = [2,2,2,2,2,2]
    order_max = 1
    for i in alter_block:
        order_max *= i
    # print(order_max)
    # order_need = 64
    assert order_need <= order_max
    order_curr = 0
    order_list = []
    order_dict = []

    while order_curr < order_need:
        order_single = [random.randint(0, alter_block[i]-1) for i in range(len(alter_block))]
        order_str = ""
        for i in order_single:
            order_str = order_str+str(i)
    #     print(order_str)
        if not order_str in order_dict:
    #         print("New!")
            order_dict.append(order_str)
            order_list.append(order_single)
            order_curr += 1
    #     else:
    #         print("Replica!")
    time_frame = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # print(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    return sorted(order_list), 
    

