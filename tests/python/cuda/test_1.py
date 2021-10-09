import copyreg, copy, pickle
from multiprocessing.reduction import ForkingPickler
class C:
    def __init__(self, a):
        self.a = a

def pickle_c(c):
    print("pickling a C instance...")
    return C, (30,)


ForkingPickler.register(C, pickle_c)

import torch.multiprocessing as mp

def child(rank, c_obj):
    print(type(c_obj))
    print(c_obj.a)


c_obj = C(10)

if __name__ == '__main__':


    mp.spawn(child,
            args=(c_obj, ),
            nprocs=1,
            join=True)
