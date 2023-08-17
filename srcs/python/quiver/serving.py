import torch.multiprocessing as mp
import torch
import numpy as np 
import os
from .pyg import GraphSageSampler
import time

__all__ = ["RequestBatcher", "HybridSampler", "InferenceServer", "InferenceServer_Debug"]

class RequestBatcher(object):
    def __init__(self, device_num, stream_queue_list,
                 input_proc_per_device, sample_mode='GPU', request_mode='CPU',
                 threshold=800, batch_time_limit=10, fixed_batch_size=512,

                 neighbour_path=None, cpu_range=[]) -> None:

        
        self.cpu_batched_queue_list = [mp.Manager().Queue() for i in range(device_num)]
        self.gpu_batched_queue_list = [mp.Manager().Queue() for i in range(device_num)]
        
        self.stream_queue_list = stream_queue_list
        self.sample_mode = sample_mode
        self.device_num = device_num
        self.batch_time_limit = batch_time_limit / 1000
        self.threshold = threshold
        self.fixed_batch_size = fixed_batch_size
        self.neighbour_path = neighbour_path
        self.request_mode = request_mode
        self.cpu_range = cpu_range

                
        if sample_mode == 'Auto':
            for i in range(input_proc_per_device * device_num):
                child_process = mp.Process(target=self.auto_despatch, args=(i, ))
                child_process.start()
        else:
            for i in range(input_proc_per_device * device_num):
                child_process = mp.Process(target=self.fixed_despatch, args=(i, ))
                child_process.start()
            
    def fixed_despatch(self, idx):

        if len(self.cpu_range) > 0:
            os.sched_setaffinity(0, [self.cpu_range[idx%len(self.cpu_range)]])

        stream_queue = self.stream_queue_list[idx]
        if self.sample_mode == 'CPU':
            batched_queue = self.cpu_batched_queue_list[idx % self.device_num]
        else:
            batched_queue = self.gpu_batched_queue_list[idx % self.device_num]
        while 1:
            tmp = stream_queue.get()
            batched_queue.put(tmp)
            
    # def fixed_sampler(self, idx):
    #     os.sched_setaffinity(0, [2*(self.cpu_offset+idx)])
    #     stream_queue = self.stream_queue_list[idx]
    #     neighbour_num = np.load(self.neighbour_path)
    #     if self.sample_mode == 'CPU':
    #         batched_queue = self.cpu_batched_queue_list[idx % self.device_num]
    #     else:
    #         batched_queue = self.gpu_batched_queue_list[idx % self.device_num]
    #     while 1:
    #         item = stream_queue.get()
    #         tmp_sum = np.take(neighbour_num, item).sum()
            
    #         if tmp_sum > self.threshold:
    #             batched_queue.put(item)
    #         else:
    #             batched_queue.put(item) 
        
    def auto_despatch(self, idx):

        if len(self.cpu_range) > 0:
            os.sched_setaffinity(0, [self.cpu_range[idx%len(self.cpu_range)]])
            

        stream_queue = self.stream_queue_list[idx]
        gpu_batched_queue = self.gpu_batched_queue_list[idx % self.device_num]
        cpu_batched_queue = self.cpu_batched_queue_list[idx % self.device_num]
        neighbour_num = np.load(self.neighbour_path)
        if self.request_mode == 'Preparation':
            while 1:
                item = stream_queue.get()
                gpu_batched_queue.put(item)
                cpu_batched_queue.put(item)
        else:
            while 1:
                item = stream_queue.get()
                tmp_sum = np.take(neighbour_num, item).sum()
                
                if tmp_sum > self.threshold:
                    gpu_batched_queue.put(item)
                else:
                    cpu_batched_queue.put(item)
        
    def batched_request_queue_list(self):
        return [self.cpu_batched_queue_list, self.gpu_batched_queue_list]
    

class HybridSampler(object):
    def __init__(self,
                 csr_topo,
                 sizes,
                 device_num,
                 worker_num_per_device,
                 batched_queue_list,

                 cpu_range=[]):
        
        self.csr_topo = csr_topo
        self.csr_topo.share_memory_()
        self.cpu_range = cpu_range

        self.device_num = device_num
        self.cpu_num_workers = device_num * worker_num_per_device
        self.sizes = sizes
        self.cpu_batched_queue_list = batched_queue_list[0]
        self.gpu_batched_queue_list = batched_queue_list[1]
        
        self.cpu_sampled_queue_list = [mp.Manager().Queue() for i in range(device_num)]
        
    def start(self):
        
        for i in range(self.cpu_num_workers):
            child_process = mp.Process(target=self.cpu_sampler_worker_loop, 

                                       args=(i, self.cpu_batched_queue_list, self.cpu_sampled_queue_list, self.device_num, self.sizes, self.csr_topo,))
            child_process.daemon = True
            child_process.start()
            
    def cpu_sampler_worker_loop(self, rank, sample_task_queue_list, result_queue_list, device_num, sizes, csr_topo):
        if len(self.cpu_range) > 0:
            os.sched_setaffinity(0, [self.cpu_range[rank%len(self.cpu_range)]])

        cpu_sampler = GraphSageSampler(csr_topo, sizes, device='cpu', mode='CPU')
        print(f"CPU Sampler {rank} Start")   
        task_queue = sample_task_queue_list[rank % device_num]
        result_queue = result_queue_list[rank % device_num]
        while 1:
            start = time.perf_counter()
            tmp = task_queue.get()
            res = cpu_sampler.sample(tmp)
            result_queue.put((res, time.perf_counter()-start))
        
    def sampled_request_queue_list(self):
        return [self.cpu_sampled_queue_list, self.gpu_batched_queue_list]
    
        
class InferenceServer(object):
    def __init__(self, model_path, 
                 device_list, x_feature, task_queue_list, 
                 sample_mode, csr_topo, sizes, ignord_length=100,
                 proc_num_per_device=0, uva_gpu='GPU') -> None:
        self.cpu_sampled_queue_list = task_queue_list[0]
        self.model_path = model_path
        self.device_list = device_list
        self.x_feature = x_feature
        self.gpu_task_queue_list = task_queue_list[1]
        self.sample_mode = sample_mode
        self.csr_topo = csr_topo
        self.sizes = sizes
        self.ignord_length = ignord_length
        self.proc_num_per_device = proc_num_per_device
        self.uva_gpu = uva_gpu
        self.num_proc = len(self.device_list) * self.proc_num_per_device
        
        self.output_queue_list = [mp.Manager().Queue() for i in range(self.num_proc)]
        
    def start(self):
        
        mp.spawn(
            self.run,
            args=(self.device_list, self.cpu_sampled_queue_list, 
                  self.model_path, self.x_feature, 
                  self.gpu_task_queue_list, self.sample_mode, self.csr_topo, 
                  self.sizes, self.num_proc, self.uva_gpu, self.output_queue_list),
            nprocs=self.num_proc,
            join=True
        )
        
    def run(self, rank, device_list, cpu_sampled_queue_list, 
        model_path, feature, gpu_sample_task_queue_list, 
        sample_mode, csr_topo, sizes, num_proc, uva_gpu, output_queue_list):
        output_queue = output_queue_list[rank]
        if sample_mode == 'Auto':
            if rank < num_proc//2:
                self.gpu_sampler_inference_loop(rank, device_list, feature, gpu_sample_task_queue_list, model_path, csr_topo, sizes, uva_gpu, output_queue)        
            else:
                _rank = rank-num_proc//2
                self.cpu_sampler_inference_loop(_rank, device_list, feature, cpu_sampled_queue_list, model_path, output_queue)
        elif sample_mode == 'GPU':
            self.gpu_sampler_inference_loop(rank, device_list, feature, gpu_sample_task_queue_list, model_path, csr_topo, sizes, uva_gpu, output_queue)
        else:
            # mode == 'CPU'
            self.cpu_sampler_inference_loop(rank, device_list, feature, cpu_sampled_queue_list, model_path, output_queue)
            
    def gpu_sampler_inference_loop(self, rank, device_list, feature, gpu_sample_task_queue_list, model_path, csr_topo, sizes, sample_mode, output_queue):
        rank_id = rank%len(device_list)
        device = f"cuda:{device_list[rank_id]}"
        gpu_sample_task_queue = gpu_sample_task_queue_list[rank_id]
        gpu_sampler = GraphSageSampler(csr_topo, sizes, device=device_list[rank_id], mode=sample_mode)
        print(f"GPU Sampler {rank} Start")
        model = torch.load(model_path).to(device)
        model.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            while True:
                item = gpu_sample_task_queue.get()
                sample_task = item.to(device)
                n_id, batch_size, adjs = gpu_sampler.sample(sample_task)
                x_input = feature[n_id].to(device)
                out = model(x_input, adjs)
                output_queue.put(out.cpu())
                    
    def cpu_sampler_inference_loop(self, rank, device_list, feature, cpu_sampled_queue_list, model_path, output_queue):
        rank_id = rank%len(device_list)
        device = f"cuda:{device_list[rank_id]}"
        cpu_sampled_queue = cpu_sampled_queue_list[rank_id]
        model = torch.load(model_path).to(device)
        model.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            while True:
                item = cpu_sampled_queue.get()
                n_id, batch_size, adjs = item[0]
                adjs = [adj.to(device) for adj in adjs]
                x_input = feature[n_id].to(device)
                out = model(x_input, adjs)
                output_queue.put(out.cpu())
        
    def result_queue_list(self):
        return self.output_queue_list
    

class InferenceServer_Debug(object):
    def __init__(self, model_path, 
                 device_list, x_feature, task_queue_list, 
                 sample_mode, csr_topo, sizes, ignord_length=100,
                 result_path=None, exp_id=0, proc_num_per_device=0, 
                 uva_gpu='GPU') -> None:
        self.cpu_sampled_queue_list = task_queue_list[0]
        self.model_path = model_path
        self.device_list = device_list
        self.x_feature = x_feature
        self.gpu_task_queue_list = task_queue_list[1]
        self.sample_mode = sample_mode
        self.csr_topo = csr_topo
        self.sizes = sizes
        self.result_path = result_path
        self.exp_id = exp_id
        self.ignord_length = ignord_length
        self.proc_num_per_device = proc_num_per_device
        self.uva_gpu = uva_gpu
        
    def start(self):
        num_proc = len(self.device_list) * self.proc_num_per_device

        mp.spawn(
            self.run,
            args=(self.device_list, self.cpu_sampled_queue_list, 
                  self.model_path, self.x_feature, 
                  self.gpu_task_queue_list, self.sample_mode, self.csr_topo, 
                  self.sizes, self.result_path, self.exp_id,
                  num_proc, self.uva_gpu, self.ignord_length),
            nprocs=num_proc,
            join=True
        )
        
    def run(self, rank, device_list, cpu_sampled_queue_list, 
        model_path, feature, gpu_sample_task_queue_list, 
        sample_mode, csr_topo, sizes, res_path, exp_id,
        num_proc, uva_gpu, ignord_length):
        if sample_mode == 'Auto':
            if rank < num_proc//2:
                self.gpu_sampler_inference_loop(rank, device_list, feature, gpu_sample_task_queue_list, model_path, csr_topo, sizes, res_path, exp_id, uva_gpu, ignord_length)        
            else:
                _rank = rank-num_proc//2
                self.cpu_sampler_inference_loop(_rank, device_list, feature, cpu_sampled_queue_list, model_path, res_path, exp_id, ignord_length)
        elif sample_mode == 'GPU':
            self.gpu_sampler_inference_loop(rank, device_list, feature, gpu_sample_task_queue_list, model_path, csr_topo, sizes, res_path, exp_id, uva_gpu, ignord_length)
        else:
            # mode == 'CPU'
            self.cpu_sampler_inference_loop(rank, device_list, feature, cpu_sampled_queue_list, model_path, res_path, exp_id, ignord_length)
            
    def gpu_sampler_inference_loop(self, rank, device_list, feature, gpu_sample_task_queue_list, model_path, csr_topo, sizes, res_path, exp_id, sample_mode, ignord_length):
        rank_id = rank%len(device_list)
        device = f"cuda:{device_list[rank_id]}"
        gpu_sample_task_queue = gpu_sample_task_queue_list[rank_id]
        gpu_sampler = GraphSageSampler(csr_topo, sizes, device=device_list[rank_id], mode=sample_mode)
        print(f"GPU Sampler {rank} Start")
        model = torch.load(model_path).to(device)
        model.eval()
        result = []
        with torch.no_grad():
            torch.cuda.synchronize()
            while True:
                try:
                    start_time = time.perf_counter() # Debug
                    item = gpu_sample_task_queue.get(timeout=10)
                    sample_task = item.to(device)
                    n_id, batch_size, adjs = gpu_sampler.sample(sample_task)
                    sample_time = time.perf_counter() # Debug
                    x_input = feature[n_id].to(device)
                    out = model(x_input, adjs)
                    end_time = time.perf_counter()
                    result.append([start_time, sample_time, end_time, batch_size, n_id.shape[0]])
                except:
                    if len(result) != 0:
                        torch.cuda.synchronize()
                        print("Test task done!")
                        result = np.array(result[ignord_length:])
                        avg_latency = np.average(result[:, 2] - result[:, 0], axis=0, weights=result[:, 3])
                        tp99_latency = np.percentile(result[:, 2] - result[:, 0], 99, axis=0) * 1000
                        avg_sampled_latency = np.average(result[:, 1] - result[:, 0], axis=0, weights=result[:, 3])
                        avg_F_I_latency = np.average(result[:, 2] - result[:, 1], axis=0, weights=result[:, 3])
                        throughput = np.sum(result[:, 3]) / (np.max(result[:, 2]) - np.min(result[:, 2]))
                        total = np.sum(result[:, 3])
                        msg = f"GPU Rank {rank}: Avg Latency: {avg_latency}, TP99 Latency: {tp99_latency}, Avg Sampled Latency: {avg_sampled_latency}, Avg F_I Latency: {avg_F_I_latency}, Throughput: {throughput}, Total: {total}\n"
                        print(msg, flush=True)
                        if res_path is not None:
                            np.save(res_path+f"/{exp_id}_GPU_{rank}", result)
                        result = []
                    
    def cpu_sampler_inference_loop(self, rank, device_list, feature, cpu_sampled_queue_list, model_path, res_path, exp_id, ignord_length):
        rank_id = rank%len(device_list)
        device = f"cuda:{device_list[rank_id]}"
        cpu_sampled_queue = cpu_sampled_queue_list[rank_id]
        model = torch.load(model_path).to(device)
        model.eval()
        result = []
        with torch.no_grad():
            torch.cuda.synchronize()
            while 1:
                try:
                    start_time = time.perf_counter()
                    item = cpu_sampled_queue.get(timeout=10)
                    n_id, batch_size, adjs = item[0]
                    sample_time = item[1]
                    adjs = [adj.to(device) for adj in adjs]
                    x_input = feature[n_id].to(device)
                    out = model(x_input, adjs)
                    end_time = time.perf_counter()
                    result.append([start_time, sample_time, end_time, batch_size, n_id.shape[0]])
                except:
                    if len(result) != 0:
                        torch.cuda.synchronize()
                        print("Test task done!")
                        result = np.array(result[ignord_length:])
                        avg_latency = np.average(result[:, 2] - result[:, 0] + result[:, 1], axis=0, weights=result[:, 3])
                        tp99_latency = np.percentile(result[:, 2] - result[:, 0] + result[:, 1], 99, axis=0) * 1000
                        avg_sampled_latency = np.average(result[:, 1], axis=0, weights=result[:, 3])
                        avg_F_I_latency = np.average(result[:, 2] - result[:, 0], axis=0, weights=result[:, 3])
                        throughput = np.sum(result[:, 3]) / (np.max(result[:, 2]) - np.min(result[:, 2]))
                        total = np.sum(result[:, 3])
                        msg = f"CPU Rank {rank}: Avg Latency: {avg_latency}, TP99 Latency: {tp99_latency}, Avg Sampled Latency: {avg_sampled_latency}, Avg F_I Latency: {avg_F_I_latency}, Throughput: {throughput}, Total: {total}\n"
                        print(msg, flush=True)
                        if res_path is not None:
                            np.save(res_path+f"/{exp_id}_CPU_{rank}", result)
                        result = []