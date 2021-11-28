import os
import time
import glob
import argparse
import os.path as osp
from torch.utils import data
from tqdm import tqdm

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator

import quiver
from quiver.feature import DeviceConfig, Feature, DistFeature
import gc

ROOT = '/data/papers'
CPU_CACHE_GB = 18
GPU_CACHE_GB = 4
LOCAL_ADDR = '192.168.0.78'
MASTER_ADDR = '192.168.0.78'
MASTER_PORT = 19216


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


class MAG240M(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 sizes: List[int],
                 host,
                 host_size,
                 local_size,
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.host = host
        self.host_size = host_size
        self.local_size = local_size
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        return 128

    @property
    def num_classes(self) -> int:
        return 172

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(row=edge_index[0],
                                 col=edge_index[1],
                                 sparse_sizes=(dataset.num_papers,
                                               dataset.num_papers),
                                 is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)

        self.train_idx = torch.load(
            '/data/papers/ogbn_papers100M/index/train_idx.pt')
        self.train_idx.share_memory_()

        if self.in_memory:
            self.x = torch.from_numpy(dataset.all_paper_feat).share_memory_()
        else:
            host_size = self.host_size
            gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (128 * 4)
            cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (128 * 4)
            host = self.host
            t0 = time.time()
            cpu_part = torch.zeros((cpu_size, 128)).share_memory_()
            gpu_parts = []
            for i in range(self.local_size):
                gpu_part = torch.zeros((gpu_size, 128))
                gpu_parts.append(gpu_part)
            feat = Feature(0, list(range(self.local_size)), 0,
                           'p2p_clique_replicate')
            device_config = DeviceConfig(gpu_parts, cpu_part)
            feat.from_mmap(None, device_config)
            self.x = feat
            print(f'feat init {time.time() - t0}')
        self.y = torch.load('/data/papers/ogbn_papers100M/label/label.pt'
                            ).squeeze().share_memory_()

        self.indptr = torch.load(
            "/data/papers/ogbn_papers100M/csr/indptr.pt").share_memory_()
        self.indices = torch.load(
            "/data/papers/ogbn_papers100M/csr/indices.pt").share_memory_()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        csr_topo = quiver.CSRTopo(indptr=self.indptr, indices=self.indices)
        quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10],
                                                     0,
                                                     mode="UVA")
        return quiver_sampler

    def val_dataloader(self):
        return NeighborSampler(self.adj_t,
                               node_idx=self.val_idx,
                               sizes=self.sizes,
                               return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size,
                               num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t,
                               node_idx=self.val_idx,
                               sizes=self.sizes,
                               return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size,
                               num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t,
                               node_idx=self.test_idx,
                               sizes=self.sizes,
                               return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size,
                               num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = self.x[n_id]
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=adjs)


class GNN(torch.nn.Module):
    def __init__(self,
                 model: str,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.model = model.lower()
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'gat':
            self.convs.append(
                GATConv(in_channels, hidden_channels // heads, heads))
            self.skips.append(Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // heads, heads))
                self.skips.append(Linear(hidden_channels, hidden_channels))

        elif self.model == 'graphsage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward_step(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs_t):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if self.model == 'gat':
                x = x + self.skips[i](x_target)
                x = F.elu(self.norms[i](x))
            elif self.model == 'graphsage':
                x = F.relu(self.norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def forward(self, batch, batch_idx: int):
        y_hat = self.forward_step(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        # self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
        #          on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc',
                 self.val_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc',
                 self.test_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


def run(rank, args, quiver_sampler, quiver_feature, label, train_idx,
        num_features, num_classes, id, local_size, host, host_size):
    torch.cuda.set_device(rank)
    print(f'{rank} beg')
    global_rank = rank + host * local_size
    global_size = host_size * local_size
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=global_rank, world_size=global_size)

    train_idx = train_idx.split(train_idx.size(0) // global_size)[global_rank]

    torch.manual_seed(123 + 45 * rank)

    gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (128 * 4)
    cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (128 * 4)

    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)

    model = GNN(args.model,
                num_features,
                num_classes,
                args.hidden_channels,
                num_layers=len(args.sizes),
                dropout=args.dropout).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    global2host = torch.load(f'/data/papers/{host_size}h/global2host.pt')
    replicate = torch.load(f'/data/papers/{host_size}h/replicate{host}.pt')
    info = quiver.feature.PartitionInfo(rank, host, host_size, global2host,
                                        replicate)
    comm = quiver.comm.NcclComm(global_rank, global_size, id, host_size,
                                local_size)
    print('comm')
    quiver_feature.lazy_init_from_ipc_handle()
    local_order = torch.load(f'/data/papers/{host_size}h/local_order{host}.pt')
    quiver_feature.set_local_order(local_order)
    dist_feature = DistFeature(quiver_feature, info, comm)
    # prev_order = torch.load(
    #     osp.join('/data/mag/mag240m_kddcup2021', 'processed', 'paper',
    #              'prev_order2.pt'))
    # disk_map = torch.zeros(prev_order.size(0), device=rank,
    #                        dtype=torch.int64) - 1
    # mem_range = torch.arange(end=cpu_size + 2 * gpu_size,
    #                          device=rank,
    #                          dtype=torch.int64)
    # disk_map[prev_order[:2 * gpu_size + cpu_size]] = mem_range
    # print(f'{rank} disk map')
    # quiver_feature.set_mmap_file(
    #     osp.join('/data/mag/mag240m_kddcup2021', 'processed', 'paper',
    #              'node_feat.npy'), disk_map)
    # print(f'{rank} mmap file')
    torch.cuda.empty_cache()
    for epoch in range(1, args.epochs + 1):
        model.train()

        sample_time = []
        feat_time = []
        train_time = []

        epoch_beg = time.time()
        for cnt, seeds in enumerate(train_loader):
            t0 = time.time()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            t1 = time.time()
            x = dist_feature[n_id]
            y = label[n_id[:batch_size]].to(torch.long)
            batch = Batch(x=x, y=y, adjs_t=adjs).to(rank)
            t2 = time.time()
            optimizer.zero_grad()
            loss = model(batch, 0)
            loss.backward()
            optimizer.step()
            t3 = time.time()
            sample_time.append(t1 - t0)
            feat_time.append(t2 - t1)
            train_time.append(t3 - t2)

            if rank == 0:
                print(f'sample {sample_time[-1]}')
                print(f'feat {feat_time[-1]}')
                print(f'train {train_time[-1]}')

        dist.barrier()

        if rank == 0:
            # remove 10% minium values and 10% maximum values
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_beg}'
            )

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model',
                        type=str,
                        default='graphsage',
                        choices=['gat', 'graphsage'])
    parser.add_argument('--sizes', type=str, default='25-10')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    host_size = 3
    local_size = 2
    host = 0
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, host, host_size,
                         local_size, args.in_memory)

    if not args.evaluate:
        store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, host_size,
                              MASTER_ADDR == LOCAL_ADDR)
        if MASTER_ADDR == LOCAL_ADDR:
            id = quiver.comm.getNcclId()
            store.set("id", id)
        else:
            id = store.get("id")

        ##############################
        # Create Sampler And Feature
        ##############################
        datamodule.setup()
        quiver_sampler = datamodule.train_dataloader()
        quiver_feature = datamodule.x
        y, train_idx, num_features, num_classes = datamodule.y, datamodule.train_idx, datamodule.num_features, datamodule.num_classes
        l = list(range(local_size))
        quiver.init_p2p(l)

        del datamodule
        gc.collect()
        os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

        print('Let\'s use', local_size, 'GPUs!')

        mp.spawn(run,
                 args=(args, quiver_sampler, quiver_feature, y, train_idx,
                       num_features, num_classes, id, local_size, host,
                       host_size),
                 nprocs=local_size,
                 join=True)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
        model = GNN.load_from_checkpoint(checkpoint_path=ckpt,
                                         hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res,
                                       f'results/{args.model}',
                                       mode='test-dev')
