import os
import os.path as osp
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
import time
import numpy as np

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class InterProcData:
    edge_index = None
    train_idx = None
    val_idx = None
    test_idx = None
    y = None
    num_features = None
    num_classes = None


class SAGE(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
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


def run(rank, world_size, x, inter_proc_data: InterProcData):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    train_idx = inter_proc_data.train_idx

    batch_size = 1024

    train_loader = NeighborSampler(inter_proc_data.edge_index,
                                   node_idx=train_idx,
                                   sizes=[15, 10, 5],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8)

    torch.manual_seed(12345)

    model = SAGE(inter_proc_data.num_features,
                 256,
                 inter_proc_data.num_classes,
                 num_layers=3).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    y = inter_proc_data.y.squeeze().to(rank)
    time_points = []
    iter_points = []
    first = True
    for epoch in range(1, 21):
        model.train()
        start_time = time.time()
        iter_step = 0
        for batch_size, n_id, adjs in train_loader:
            feature = x[n_id].to(rank)
            time_points.append(time.time() - start_time)
            iter_step += 1
            if rank == 0 and iter_step % 20 == 0:
                print(
                    f"average data time = {np.mean(np.array(time_points[1:]))}"
                )

            adjs = [adj.to(rank) for adj in adjs]

            optimizer.zero_grad()
            out = model(feature, adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
            if rank == 0 and iter_step > 10:
                iter_points.append(time.time() - start_time)
                print(
                    f"average iter time = {np.mean(np.array(iter_points[10:]))}, throughput = {world_size * batch_size  / np.mean(np.array(iter_points[10:]))}"
                )

            start_time = time.time()

        iter_points.clear()
        time_points.clear()
        dist.barrier()
        exit()

    dist.destroy_process_group()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    data = dataset[0]
    inter_proc_data = InterProcData()
    data.edge_index.share_memory_()

    inter_proc_data.edge_index = data.edge_index
    inter_proc_data.num_classes = dataset.num_classes
    inter_proc_data.num_features = dataset.num_features
    inter_proc_data.train_idx = train_idx

    inter_proc_data.y = data.y

    data.x.share_memory_()
    world_size = 2
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run,
             args=(world_size, data.x, inter_proc_data),
             nprocs=world_size,
             join=True)
