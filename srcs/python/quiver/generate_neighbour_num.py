import numpy as np
import os
import torch
from tqdm import tqdm
from .utils import CSRTopo
from .pyg import GraphSageSampler
import torch.multiprocessing as mp
import multiprocessing as cmp

def generate_neighbour_num(node_num, edge_index, sizes, resultl_path, device_list, parallel=False, mode='CPU', num_proc=1, reverse=False, sample=False):
    
    if not parallel:
        neighbour_num = np.ascontiguousarray(np.zeros(node_num, dtype=np.int32))
        csr_topo = CSRTopo(edge_index)
        sampler = GraphSageSampler(csr_topo, sizes, device=0, mode=mode)
        
        pbar = tqdm(total=node_num)
        pbar.set_description('Generate neighbour num')
        for node in range(node_num):
            pbar.update(1)
            n_id, _, __ = sampler.sample(torch.tensor([node])) # GPU
            neighbour_num[node] = n_id.shape[0]
        
        np.save(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{reverse}.npy', neighbour_num)
    else:
        if  mode == 'GPU':
            parallel_generate_neighbour_num_GPU(node_num, edge_index, sizes, resultl_path, device_list, num_proc, reverse, sample)
        else:
            # divice == 'CPU'
            parallel_generate_neighbour_num_CPU(node_num, edge_index, sizes, resultl_path, device_list, num_proc, reverse, sample)
    
def single_generate_neighbour_num(rank, node_num, csr_topo, num_proc, sizes, mode, resultl_path, device_list, sample):
    start = rank * (node_num // num_proc)
    end = (rank+1) * (node_num // num_proc)
    if rank == num_proc - 1:
        end = node_num+1
    if mode == 'GPU':
        device = rank%len(device_list)
    else:
        device = 'CPU'
    sampler = GraphSageSampler(csr_topo, sizes, device=device, mode=mode)
    print(f'Process {rank} has started')
    
    if sample:
        num = 100000
        random_list = np.random.randint(start, end, num)
    else:
        num = end - start
        random_list = np.array(range(start, end))

    neighbour_num = np.ascontiguousarray(np.zeros(end-start, dtype=np.int32))
    for idx, node in enumerate(random_list):
        n_id, _, __ = sampler.sample(torch.tensor([node])) # GPU
        neighbour_num[node-start] = n_id.shape[0]
        if node % 10000 == 0:
            print(f'Process {rank} has finished {100*(idx)/(end-start)}%;')
    np.place(neighbour_num, neighbour_num==0, (neighbour_num.sum() // num))
    np.save(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{rank}.npy', neighbour_num)
    print(f'Process {rank} has finished')
    
def parallel_generate_neighbour_num_GPU(node_num, edge_index, sizes, resultl_path, device_list, num_proc, reverse=False, sample=False):
    neighbour_num = []
    csr_topo = CSRTopo(edge_index)
    csr_topo.share_memory_()
    mp.spawn(
        single_generate_neighbour_num,
        args=(node_num, csr_topo, num_proc, sizes, 'GPU', resultl_path, device_list, sample),
        nprocs=num_proc,
        join=True
    )
    # torch.cuda.synchronize()
    for i in range(num_proc):
        neighbour_num.append(np.load(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{i}.npy'))
        os.remove(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{i}.npy')
    neighbour_num = np.concatenate(neighbour_num, axis=0)
    np.save(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{reverse}.npy', neighbour_num)
    
def parallel_generate_neighbour_num_CPU(node_num, edge_index, sizes, resultl_path, device_list, num_proc, reverse=False, sample=False):
    neighbour_num = []
    csr_topo = CSRTopo(edge_index)
    csr_topo.share_memory_()
    pros = []
    for i in range(num_proc):
        p = cmp.Process(target=single_generate_neighbour_num, args=(i, node_num, csr_topo, num_proc, sizes, 'CPU', resultl_path, device_list, sample))
        p.start()
        pros.append(p)
        
    for p in pros:
        p.join()
        
    for i in range(num_proc):
        neighbour_num.append(np.load(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{i}.npy'))
        os.remove(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{i}.npy')
    neighbour_num = np.concatenate(neighbour_num, axis=0)
    np.save(f'{resultl_path}/{sizes[0]}_{sizes[1]}_neighbour_num_{reverse}.npy', neighbour_num)