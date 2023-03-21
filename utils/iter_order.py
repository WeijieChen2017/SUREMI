import itertools
import random
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

    time_frame = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # print(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    return set_1, time_frame

def iter_all_order_but(alter_block_depth, remove_blocks=[]):
    a = alter_block_depth
    elem_a = []
    for num_a in a:
        elem_a.append([p for p in range(num_a)])

    set_1 = elem_a[0]
    for i_a in range(len(a)-1):
        set_2 = elem_a[i_a+1]
        set_1 = Cartesian_product(set_1, set_2)

    qualified_set = []
    for each_order in set_1:
        is_qualified = True
        for idx, each_block in enumerate(each_order):
            if each_block in remove_blocks[idx]:
                is_qualified = False
        if is_qualified:
            qualified_set.append(each_order)

    time_frame = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # print(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    return qualified_set, time_frame

def iter_some_order_prob(alter_block, order_need=128):

    order_list = iter_all_order(alter_block)[0]
    random.shuffle(order_list)
    order_list = order_list[:order_need]
    time_frame = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    return order_list, time_frame

def iter_some_order(alter_block, order_need=128, remove_blocks=[]):
    # alter_block = [4,2,2,2,2,1,1,1,1,1]
    # alter_block = [2,2,2,2,2,2]
    order_max = 1
    for i in alter_block:
        order_max *= i
        
    if len(remove_blocks) == 0:
        remove_blocks = [[] for i in range(len(alter_block))]

    assert order_need <= order_max
    assert len(alter_block) == len(remove_blocks)

    order_available_list = []
    for idx in range(len(alter_block)):
        curr_available = []
        for idx_available in range(alter_block[idx]):
            if not idx_available in remove_blocks[idx]:
                curr_available.append(idx_available)
        order_available_list.append(curr_available)


    order_curr = 0
    order_list = []
    order_dict = []

    while order_curr < order_need:
            
        order_single = []
        for idx in range(len(alter_block)):
            curr_random_int = random.randint(0, len(order_available_list[idx])-1)
            order_single.append(order_available_list[idx][curr_random_int])

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
    return sorted(order_list), time_frame
    

