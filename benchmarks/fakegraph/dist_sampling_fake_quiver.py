# Reaches around 0.7870 ± 0.0036 test accuracy.
import os
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
import torch_quiver as qv

####################
# Import Quiver
####################
import quiver

class FakeDataset:
    def __init__(self, root, gpu_portion):
        data_dir = osp.join(root, 'fake')
        feat_root = osp.join(data_dir, 'feat', 'feat.pt')
        indptr_root = osp.join(data_dir, 'csr', 'indptr.pt')
        indices_root = osp.join(data_dir, 'csr', 'indices.pt')
        label_root = osp.join(data_dir, 'label', 'label.pt')
        index_root = osp.join(data_dir, 'index', 'index.pt')
        feat = torch.load(feat_root)
        print('load feature')
        node_count = feat.shape[0]
        prev_order = torch.arange(node_count, dtype=torch.long)
        total_range = torch.arange(node_count, dtype=torch.long)
        perm_range = torch.randperm(int(node_count * gpu_portion))
        new_order = torch.zeros_like(total_range)
        prev_order[: int(node_count * gpu_portion)] = prev_order[perm_range]
        new_order[prev_order] = total_range
        print('reorder feature')
        self.feature = feat.share_memory_()
        self.indptr = torch.load(indptr_root).share_memory_()
        self.indices = torch.load(indices_root).share_memory_()
        self.label = torch.load(label_root).squeeze().share_memory_()
        self.train_idx = torch.load(index_root).share_memory_()
        self.new_order = new_order
        self.prev_order = prev_order

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, world_size, quiver_sampler, quiver_feature, y, train_idx, num_features, num_classes):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    torch.manual_seed(123 + 45 * rank)

    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True, shuffle=True)

    model = SAGE(num_features, 256, num_classes, num_layers=3).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = y.to(rank)

    for epoch in range(1, 21):
        model.train()

        epoch_start = time.time()
        step = 0
        iter_times = []
        for seeds in train_loader:
            iter_start = time.time()
            s_beg = time.time()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            adjs = [adj.to(rank) for adj in adjs]
            torch.cuda.synchronize()
            f_beg = time.time()
            feat = quiver_feature[n_id]
            torch.cuda.synchronize()
            t_beg = time.time()
            optimizer.zero_grad()
            out = model(feat, adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t_end = time.time()
            iter_times.append(time.time() - iter_start)
            if rank == 0:
                print(f"sample {f_beg - s_beg}")
                print(f"feat {t_beg - f_beg}")
                print(f"train {t_end - t_beg}")

        dist.barrier()

        iter_times = sorted(iter_times)
        

        if rank == 0:
            # remove 10% minium values and 10% maximum values
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {np.mean(iter_times[int(0.1 * len(iter_times)): -int(0.1 * len(iter_times))]) * len(train_loader)}')

        # if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
        #     model.eval()
        #     with torch.no_grad():
        #         out = model.module.inference(quiver_feature, rank, subgraph_loader)
        #     res = out.argmax(dim=-1) == y.cpu()
        #     acc1 = int(res[train_idx].sum()) / train_idx.numel()
        #     acc2 = int(res[val_idx].sum()) / val_idx.numel()
        #     acc3 = int(res[test_idx].sum()) / test_idx.numel()
        #     print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    root = "."
    world_size = torch.cuda.device_count()
    world_size = 2
    dataset = FakeDataset(root, 0.2 * min(world_size, 2))
    
    ##############################
    # Create Sampler And Feature
    ##############################
    csr_topo = quiver.CSRTopo(indptr=dataset.indptr, indices=dataset.indices)
    csr_topo.feature_order = dataset.new_order
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size="8G", cache_policy="p2p_clique_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(dataset.feature)
    l = list(range(world_size))
    qv.init_p2p(l)
    del dataset.feature

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run,
        args=(world_size, quiver_sampler, quiver_feature, dataset.label, dataset.train_idx, 1024, 100),
        nprocs=world_size,
        join=True
    )