import torch
import torch_quiver as qv

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
    def __init__(self, rank, ws, cpu, gpu, request_queues, response_queues, cpu_split, to_local):
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
        input_orders = torch.arange(nodes.size(
            0), dtype=torch.long, device=nodes.device)
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
        input_orders = torch.arange(nodes.size(
            0), dtype=torch.long, device=nodes.device)
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
