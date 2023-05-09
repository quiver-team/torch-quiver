# fuser -v -k /dev/nvidia*
# ps -ef | grep 'stream\|spawn' | grep -v grep | awk '{print $2}' | xargs kill
import os
import os.path as osp
import sys
import torch.multiprocessing as mp

os.chdir(sys.path[0])
sys.path.append(osp.abspath(osp.join(os.getcwd(),'..'))) # For import model
sys.path.append(osp.abspath(osp.join(os.getcwd(),'../..'))) # For import src

import time
import numpy as np
from quiver import AutoBatch, ServingSampler
from quiver import ServerInference_Debug as ServerInference
import torch
import quiver
from quiver_feature import LocalTensorPGAS
from model import SAGE
        
def server_gate_local(rank, stream_queue, sizes, warmup_num):
    os.sched_setaffinity(0, [2*rank])
    np.random.seed(rank)
    time.sleep(10)

    workload = np.load(f'./{sizes[0]}_{sizes[1]}_workload.npy')
    i=0
    length = len(workload)
    while i < warmup_num:
        idx = np.random.randint(0, length)
        batch = workload[idx]
        stream_queue.put(torch.tensor(batch))
        i += 1
    for i in range(length):
        batch = workload[i]
        stream_queue.put(torch.tensor(batch))
    
def server_gate_local_preparation(rank, stream_queue, test_num, warmup_num):
    os.sched_setaffinity(0, [2*rank])
    np.random.seed(rank)
    time.sleep(10)      
    batch_size_list = [2, 4, 6, 8, 10, 12, 24, 36, 48, 60, 128, 256]
    for batch_size in batch_size_list:
        for i in range(warmup_num):
            tmp = np.random.randint(0, node_num, size=batch_size)
            stream_queue.put(torch.tensor(tmp))
            
        for i in range(test_num):
            tmp = np.random.randint(0, node_num, size=batch_size)
            stream_queue.put(torch.tensor(tmp))

if __name__ == "__main__":
    mp.set_sharing_strategy('file_system')
    np.random.seed(2022)
    
    dataset_nm = 'reddit'
    device_list = [0, 1]
    device_proc_per_device = 2
    input_proc_per_device = 4
    CPU_sampler_per_device = 4
    sizes = [25, 10]
    
    test_num = 1000
    warmup_num = 100
    
    threshold = 1670
    
    # Make preparations
    # exp_id = 'preparation'
    # request_mode = 'Preparation'
    # sample_mode = 'Auto'
    
    # Fixed Batch Size
    # exp_id = 'fixed_depatch'
    # request_mode = 'Fixed'
    # sample_mode = 'CPU' # 'CPU', 'GPU', 'Auto'
    # uva_gpu = 'GPU' # 'GPU', 'UVA'
    
    # Auto
    exp_id = 'auto_depatch'
    request_mode = 'Random'
    sample_mode = 'Auto' # 'CPU', 'GPU', 'Auto'
    uva_gpu = 'GPU' # 'GPU', 'UVA'
    
    device_num = len(device_list)
    
    ignord_length_per_proc = warmup_num * input_proc_per_device // device_proc_per_device
    
    # if (request_mode == 'Preparation') or (exp_id == 'different_throughput'):
    #     ignord_length_per_proc = 0
    # else:
    #     ignord_length_per_proc = warmup_num * input_proc_per_device // device_proc_per_device
        
    if dataset_nm == 'reddit':
        data, _ = torch.load('./data/processed/data.pt')
        model_path = './reddit_quiver_model.pth'
        neighbour_path = f'./intermediate/{sizes[0]}_{sizes[1]}_neighbour_num_False.npy'
    else:
        print('Dataset not supported')
        exit()
        
    exp_id = f'{exp_id}_{dataset_nm}_{sample_mode}'
        
        
    edge_index = data.edge_index
    node_num = data.x.shape[0]
    
    quiver.init_p2p(device_list)
    csr_topo = quiver.CSRTopo(edge_index)
    quiver_feature = quiver.Feature(rank=device_list[0], device_list=device_list, device_cache_size="4G", cache_policy="device_replicate", csr_topo=csr_topo)
    # quiver_feature = LocalTensorPGAS(device_list=device_list, device_cache_size="4G", cache_policy="device_replicate")
    quiver_feature.from_cpu_tensor(data.x)
    print("Data load finished")
    
    cpu_offset = device_num*input_proc_per_device 
    
    stream_queue_list = [mp.Manager().Queue() for i in range(device_num*input_proc_per_device)]
    auto_batch = AutoBatch(device_num=device_num, stream_queue_list=stream_queue_list, 
                           input_proc_per_device=input_proc_per_device, 
                           sample_mode=sample_mode, request_mode=request_mode, threshold=threshold, 
                           neighbour_path=neighbour_path, cpu_offset=cpu_offset)
    cpu_batched_queue_list, gpu_batched_queue_list = auto_batch.get_batched_queue()
    cpu_offset = cpu_offset + device_num*input_proc_per_device
    
    cpu_sampled_queue_list = None
    # if True:
    if sample_mode != 'GPU':
        sampler = ServingSampler(sizes=sizes, csr_topo=csr_topo, device_num=device_num, 
                                 worker_num_per_device=CPU_sampler_per_device,
                                 cpu_batched_queue_list=cpu_batched_queue_list,
                                 cpu_offset=cpu_offset)
        cpu_offset = cpu_offset + device_num*CPU_sampler_per_device
        sampler.lazy_init()
        cpu_sampled_queue_list = sampler.get_sampled_queue()
        
    result_path = osp.join(sys.path[0], 'run_result')
    inference = ServerInference(cpu_sampled_queue_list=cpu_sampled_queue_list, 
                                model_path=model_path, device_list=device_list, 
                                x_feature=quiver_feature, gpu_task_queue_list=gpu_batched_queue_list, 
                                sample_mode=sample_mode, csr_topo=csr_topo, sizes=sizes, uva_gpu=uva_gpu,
                                result_path=result_path, exp_id=exp_id, ignord_length=ignord_length_per_proc,
                                device_proc_per_device=device_proc_per_device, cpu_offset=cpu_offset)
    
    if request_mode == 'Preparation':
        for idx in range(len(stream_queue_list)):
            proc = mp.Process(target=server_gate_local_preparation, args=(idx, stream_queue_list[idx], test_num, warmup_num,))
            proc.daemon = True
            proc.start()
    else:
        for idx in range(len(stream_queue_list)):
            proc = mp.Process(target=server_gate_local, args=(idx, stream_queue_list[idx], sizes, warmup_num))
            proc.daemon = True
            proc.start()
        
    inference.start()