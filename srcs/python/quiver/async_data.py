import torch

from typing import List, NamedTuple, Optional, Tuple


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class SampleBuffer:
    def __init__(self, batch_size, size, total_layer, sample_device, reindex_device, train_device):
        self.batch_size = batch_size
        self.total_layer = total_layer
        self.temp_layer = 0
        self.sample_device = sample_device
        self.reindex_device = reindex_device
        self.train_device = train_device
        self.reindex_results = [None] * total_layer
        self.feature_reorder = None
        self.inputs = None
        self.n_ids = None
        self.feature_results = [None] * size
        self.state = "sample"


class DataManager:
    def __init__(self, device, feature, feature_devices, feature_to_local, feature_rank):
        self.device = device
        self.feature = feature
        self.feature_to_local = feature_to_local
        self.feature_rank = feature_rank
        self.feature_devices = feature_devices
        self.buffers = dict()

    def dispatch(self, nodes, ws):
        ranks = self.feature_rank(nodes)
        input_orders = torch.arange(nodes.size(
            0), dtype=torch.long, device=nodes.device)
        reorder = torch.empty_like(input_orders)
        beg = 0
        res = []
        for i in range(ws):
            mask = torch.eq(ranks, i)
            part_nodes = torch.masked_select(nodes, mask)
            part_orders = torch.masked_select(input_orders, mask)
            reorder[beg:beg + part_nodes.size(0)] = part_orders
            beg += part_nodes.size(0)
            res.append(part_nodes)
        return reorder, res

    def init_entry(self, batch_id, size, total_layer, sample_device, reindex_device, train_device):
        buffer = SampleBuffer(
            batch_id, size, total_layer, sample_device, reindex_device, train_device)
        self.buffers[batch_id] = buffer

    def prepare_request(self, batch_id, reorder, node_group):
        size = len(node_group)
        res = []
        self.buffers[batch_id].feature_reorder = reorder.to(
            self.buffers[batch_id].train_device)
        for rank in range(size):
            nodes = self.feaure_to_local(node_group[i], rank, size)
            nodes = nodes.to(self.feature_devices[rank])
            res.append(nodes)
        return res

    def recv_sample(self, batch_id, nodes, counts):
        buffer = self.buffers[batch_id]
        reindex_device = buffer.reindex_device
        nodes = nodes.to(reindex_device)
        counts = counts.to(reindex_device)
        inputs = buffer.inputs
        buffer.inputs = None
        buffer.state = "reindex"
        return inputs, nodes, counts

    def recv_feature(self, batch_id, rank, size, features):
        buffer = self.buffers[batch_id]
        buffer.feature_results[rank] = features.to(buffer.train_device)
        cnt = 0
        for res in buffer.feature_results:
            if res is not None:
                cnt += 1
        if cnt == size:
            all_features = []
            for feature in buffer.feature_results:
                all_features.append(feature)
            all_features = torch.cat(all_features)
            buffer.feature_results = [None] * size
            reorder = buffer.feature_reorder
            buffer.feature_reorder = None
            buffer.state = "finished"
            all_features[reorder] = all_features
            return all_features

    def recv_reindex(self, batch_id, nodes, row, col):
        buffer = self.buffers[batch_id]
        temp_layer = buffer.temp_layer
        buffer.reindex_results[temp_layer] = (
            nodes, row.to(buffer.train_device), col.to(buffer.train_device))
        buffer.temp_layer += 1
        finished = buffer.temp_layer < buffer.total_layer
        buffer.state = "feature" if finished else "sample"
        if not finished:
            return nodes.to(self.sample_device), buffer.temp_layer
        else:
            buffer.n_ids = nodes
            return nodes, -1

    def prepare_train(self, batch_id):
        buffer = self.buffers[batch_id]
        assert buffer.state == "finished"
        n_ids = buffer.n_ids.to(buffer.train_device)
        batch_size = buffer.batch_size
        adjs = []
        last_size = batch_size
        for nodes, row_idx, col_idx in buffer.reindex_results:
            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            size = torch.LongTensor([
                nodes.size(0),
                last_size,
            ])
            e_id = torch.tensor([])
            adjs.append(Adj(edge_index, e_id, size))
            last_size = nodes.size(0)
        return batch_size, n_ids, adjs[::-1]
