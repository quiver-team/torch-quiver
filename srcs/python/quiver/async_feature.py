import torch
import torch_quiver as qv

import time


class FeatureRequest:
    def __init__(self, src, dst, nodes):
        self.src = src
        self.dst = dst
        self.nodes = nodes


class FeatureResponse:
    def __init__(self, src, dst, features):
        self.src = src
        self.dst = dst
        self.features = features


class FeatureConfig:
    def __init__(self, rank, ws, cpu, gpu, request_queues, response_queues,
                 cpu_split, to_local):
        self.rank = rank
        self.ws = ws
        self.cpu = cpu
        self.gpu = gpu
        self.request_queues = request_queues
        self.response_queues = response_queues
        self.cpu_split = cpu_split
        self.to_local = to_local


class AsyncFeature:
    def __init__(self, config, device, local_feature, cpu_feature):
        self.device = device
        self.config = config
        self.local_feature = local_feature.to(device)
        # if cpu, put shard tensor stores whole cpu tensor only
        if config.cpu:
            self.cpu_feature = qv.ShardTensor(device)
            self.cpu_feature.append(cpu_feature, -1)
        else:
            self.cpu_feature = None

    def hybrid_dispatch(self, nodes):
        ws = self.config.ws
        beg = 0
        res = []
        input_orders = torch.arange(nodes.size(0),
                                    dtype=torch.long,
                                    device=nodes.device)
        cpu_mask = torch.lt(nodes, self.config.cpu_split)
        gpu_mask = torch.ge(nodes, self.config.cpu_split)
        cpu_nodes = torch.masked_select(nodes, cpu_mask)
        cpu_orders = torch.masked_select(input_orders, cpu_mask)
        beg += cpu_nodes.size(0)
        res.append(cpu_nodes)
        # TODO: General split
        ranks = torch.fmod(nodes, ws)
        reorder = torch.empty_like(input_orders)
        for i in range(ws):
            rank_mask = torch.eq(ranks, i)
            rank = torch.bitwise_and(rank_mask, gpu_mask)
            part_nodes = torch.masked_select(nodes, mask)
            part_orders = torch.masked_select(input_orders, mask)
            reorder[beg:beg + part_nodes.size(0)] = part_orders
            beg += part_nodes.size(0)
            res.append(part_nodes)
        return reorder, res

    def gpu_dispatch(self, nodes):
        ws = self.config.ws
        beg = 0
        res = []
        input_orders = torch.arange(nodes.size(0),
                                    dtype=torch.long,
                                    device=nodes.device)
        # TODO: General split
        ranks = torch.fmod(nodes, ws)
        reorder = torch.empty_like(input_orders)
        for i in range(ws):
            mask = torch.eq(ranks, i)
            part_nodes = torch.masked_select(nodes, mask)
            part_orders = torch.masked_select(input_orders, mask)
            reorder[beg:beg + part_nodes.size(0)] = part_orders
            beg += part_nodes.size(0)
            res.append(part_nodes)
        return reorder, res

    def collect(self, nodes):
        nodes = nodes.to(self.device)
        # local, hybrid or gpu request
        if not self.config.cpu and not self.config.gpu:
            return self.feature[nodes]
        elif self.config.cpu:
            feature_reorder, feature_args = self.hybrid_dispatch(
                nodes, self.config.ws)
            gpu_args = feature_args[1:]
            cpu_result = self.cpu_feature[nodes]
        else:
            feature_reorder, gpu_args = self.gpu_dispatch(
                nodes, self.config.ws)
        feature_results = []
        for rank, part_nodes in enumerate(gpu_args):
            if rank == self.config.rank:
                result = self.feature[part_nodes]
            else:
                result = None
                req = FeatureRequest(self.config.rank, rank, part_nodes)
                self.config.request_queues[rank].put(req)
            feature_results.append(result)
        for i in range(self.config.ws - 1):
            req = self.config.request_queues[self.config.rank].get()
            src = req.src
            dst = req.dst
            part_nodes = req.nodes.to(self.device)
            part_nodes = t
            feature = self.feature[part_nodes]
            resp = FeatureResponse(src, dst, feature)
            self.config.response_queues[src].put(resp)
        for i in range(self.config.ws - 1):
            resp = self.config.response_queues[self.config.rank].get()
            src = resp.src
            dst = resp.dst
            feature = resp.features
            feature_results[dst] = feature.to(self.device)
        total_features = []
        if self.config.cpu:
            feature_results.insert(0, cpu_result)
        for feature in feature_results:
            total_features.append(feature)
        total_features = torch.cat(total_features)
        total_features = total_features[feature_reorder]
        return total_features


class TorchShardTensor:
    def __init__(self, rank, ws, cpu_tensor, gpu_tensors, range_list):
        self.rank = rank
        self.ws = ws
        self.local_tensor = qv.ShardTensor(rank)
        self.local_tensor.append(gpu_tensors[rank], rank)
        self.local_tensor.append(cpu_tensor, -1)
        self.local_tensor.finish_init()
        self.gpu_tensors = gpu_tensors
        self.range_list = range_list
        self.stream_list = []
        for i in range(ws):
            s = torch.cuda.Stream(i)
            self.stream_list.append(s)

    def collect(self, nodes):
        t0 = time.time()
        torch.cuda.set_device(self.rank)
        nodes = nodes.to(self.rank)
        input_orders = torch.arange(nodes.size(0),
                                    dtype=torch.long,
                                    device=torch.device(self.rank))
        beg_r = self.range_list[self.rank]
        end_r = self.range_list[self.rank + 1]
        beg_mask = torch.ge(nodes, beg_r)
        end_mask = torch.lt(nodes, end_r)
        local_mask = torch.bitwise_and(beg_mask, end_mask)
        local_gpu_nodes = torch.masked_select(nodes, local_mask) - beg_r
        local_gpu_order = torch.masked_select(input_orders, local_mask)
        beg_r = self.range_list[self.ws]
        end_r = self.range_list[self.ws + 1]
        beg_mask = torch.ge(nodes, beg_r)
        end_mask = torch.lt(nodes, end_r)
        cpu_mask = torch.bitwise_and(beg_mask, end_mask)
        cpu_nodes = torch.masked_select(
            nodes, cpu_mask) - beg_r + self.range_list[self.rank]
        cpu_order = torch.masked_select(input_orders, cpu_mask)
        local_nodes = torch.cat([local_gpu_nodes, cpu_nodes])
        local_order = torch.cat([local_gpu_order, cpu_order])
        with torch.cuda.stream(self.stream_list[self.rank]):
            local_result = self.local_tensor[local_nodes]
        t1 = time.time()
        remote_orders = []
        remote_results = []
        for rank in range(self.ws):
            if rank == self.rank:
                continue
            # print(f'remote {rank}')
            t_beg = time.time()
            beg_r = self.range_list[rank]
            end_r = self.range_list[rank + 1]
            beg_mask = torch.ge(nodes, beg_r)
            end_mask = torch.lt(nodes, end_r)
            mask = torch.bitwise_and(beg_mask, end_mask)
            part_nodes = torch.masked_select(nodes, mask) - beg_r
            part_orders = torch.masked_select(input_orders, mask)
            remote_orders.append(part_orders)
            torch.cuda.set_device(rank)
            t_mid = time.time()
            with torch.cuda.stream(self.stream_list[rank]):
                result = self.gpu_tensors[rank][part_nodes]
                t_local = time.time()
                result = result.to(self.rank, non_blocking=True)
                remote_results.append(result)
                t_move = time.time()
            # print(f'pre {t_mid - t_beg}')
            # print(f'collect {t_local - t_mid}')
            # print(f'move {t_move - t_local}')
        t2 = time.time()
        for rank in range(self.ws):
            if rank == self.rank:
                continue
            self.stream_list[rank].synchronize()
        torch.cuda.set_device(self.rank)
        self.stream_list[self.rank].synchronize()
        remote_results.insert(0, local_result)
        remote_orders.insert(0, local_order)
        total_results = torch.cat(remote_results)
        total_orders = torch.cat(remote_orders)
        total_results[total_orders] = total_results
        torch.cuda.synchronize(self.rank)
        # print(f'local {t1 - t0}')
        # print(f'remote {t2 - t1}')
        # print(f'sync {time.time() - t2}')
        return total_results

    def __getitem__(self, nodes):
        return self.collect(nodes)
