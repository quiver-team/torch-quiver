from multiprocessing.reduction import ForkingPickler
import quiver
import torch.multiprocessing as mp
import torch
import torch_quiver as torch_qv
import random
import numpy as np
import time
from typing import List
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import sys

def rebuild_feature(ipc_handle):
    print("rebuild feature")
    feature = quiver.Feature.lazy_from_ipc_handle(ipc_handle)
    return feature

def reduce_feature(feature):
    print("reduce feature")
    ipc_handle = feature.share_ipc()
    return (rebuild_feature, (ipc_handle, ))

def reduce_sampler():
    pass

def rebuild_sampler():
    pass    



def init_reductions():
    ForkingPickler.register(quiver.Feature, reduce_feature)



def child_proc(feature):
    NUM_ELEMENT = 10000
    SAMPLE_SIZE = 800
    rank = 3
    print(sys.argv)
    torch.cuda.set_device(rank)

    #########################
    # Init With Numpy
    ########################

    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    device_indices = indices.to(rank)

    print(feature[device_indices].shape)



def test_feature_reduction():
    rank = 2
        
    NUM_ELEMENT = 10000
    SAMPLE_SIZE = 800
    FEATURE_DIM = 600
    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)


    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))

    print("host data size", host_tensor.size * 4 // 1024  // 1024, "MB")
    tensor = torch.from_numpy(host_tensor).type(torch.float32)

    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)

    device_indices = indices.to(rank)

    ############################
    # define a quiver.Feature
    ###########################
    feature = quiver.Feature(rank=rank, device_list=[2, 3], device_cache_size="10M", cache_policy="numa_replicate")
    feature.from_cpu_tensor(tensor)

    process = Process(target=child_proc, args=(feature, ))
    process.start()
    process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    init_reductions()
    test_feature_reduction()
    



