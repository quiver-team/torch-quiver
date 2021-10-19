import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel
import tqdm
import os
import os.path as osp
import sklearn.linear_model as lm
import sklearn.metrics as skm
from quiver.shard_tensor import ShardTensor as PyShardTensor
from quiver.shard_tensor import ShardTensorConfig
import torch_quiver as torch_qv
from scipy.sparse import csr_matrix
from ogb.nodeproppred import PygNodePropPredDataset

# 0: CPU, 1: GPU, 2: ShardTensor
GPU_FEATURE = 2
# 0: CPU, 1: GPU, 2: ZeroCopy
GPU_SAMPLE = 2
# 0: Device, 1: NUMA
CACHE_MODE = 1


def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst)) + 1
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat


def split(ratio, name, root):
    dataset = PygNodePropPredDataset(name, root)
    data = dataset[0]
    csr = get_csr_from_coo(data.edge_index)
    indptr = csr.indptr
    prev = th.LongTensor(indptr[:-1])
    sub = th.LongTensor(indptr[1:])
    deg = sub - prev
    sorted_deg, prev_order = th.sort(deg, descending=True)
    total_num = data.x.shape[0]
    total_range = th.arange(total_num, dtype=th.long)
    if isinstance(ratio, float):
        perm_range = th.randperm(int(total_num * ratio))
        prev_order[:int(total_num * ratio)] = prev_order[perm_range]
    new_order = th.zeros_like(prev_order)
    new_order[prev_order] = total_range
    index = 0
    res = []
    if isinstance(ratio, list):
        for i in range(len(ratio) - 1):
            num = int(ratio[i] * total_num)
            gpu_tensor = data.x[prev_order[index:index + num]].share_memory_()
            res.append(gpu_tensor)
            index += num
        cpu_tensor = data.x[prev_order[index:]].clone().share_memory_()
        res.append(cpu_tensor)
    return res, prev_order, new_order


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation,
             dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test


def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes


def load_ogb(name, root=None):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    home = os.getenv('HOME')
    root = osp.join(home, 'products')
    data = DglNodePropPredDataset(name=name, root=root)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx[
        'valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(), ), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(), ), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(), ), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size,
                               args.num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])


def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(dev_id)
    batch_labels = labels[seeds].to(dev_id)
    return batch_inputs, batch_labels


# Entry point


def run(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    th.cuda.set_device(dev_id)
    print('ready')
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)

    # Unpack data
    n_classes, train_g, train_nfeat, train_labels = data
    train_g = train_g.formats(['csc'])
    val_g = train_g
    test_g = train_g

    if GPU_FEATURE == 1:
        train_nfeat = train_nfeat.to(dev_id)
        # train_labels = train_labels.to(dev_id)
    if GPU_FEATURE == 2:
        if CACHE_MODE == 1:
            ipc_handle, new_order = train_nfeat
            train_nfeat = PyShardTensor.new_from_share_ipc(ipc_handle, dev_id)
        else:
            shards, prev_order, new_order = train_nfeat
            train_nfeat = torch_qv.ShardTensor(dev_id)
            train_nfeat.append(shards[0], dev_id)
            train_nfeat.append(shards[1], -1)
        new_order = new_order.to(dev_id)
        in_feats = 100
    else:
        in_feats = train_nfeat.shape[1]

    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')],
        zero_copy=GPU_SAMPLE == 2)
    if GPU_SAMPLE == 1:
        train_g = train_g.to(dev_id)
        train_nid = train_nid.to(dev_id)
    dataloader = dgl.dataloading.NodeDataLoader(train_g,
                                                train_nid,
                                                sampler,
                                                use_ddp=False,
                                                device=dev_id,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=args.num_workers)
    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu,
                 args.dropout)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model,
                                        device_ids=[dev_id],
                                        output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        # if n_gpus > 1:
        #     dataloader.set_epoch(epoch)
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        t0 = time.time()
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            t1 = time.time()
            # Load the input features as well as output labels
            if GPU_FEATURE == 2:
                input_nodes = new_order[input_nodes]
            batch_inputs, batch_labels = load_subtensor(
                train_nfeat, train_labels, seeds, input_nodes, dev_id)
            blocks = [block.int().to(dev_id) for block in blocks]
            t2 = time.time()
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()
            if proc_id == 0:
                iter_tput.append(
                    len(seeds) * n_gpus / (time.time() - tic_step))
                tic_step = time.time()
            if step % args.log_every == 0 and proc_id == 0:
                print(f'sample took {t1 - t0}')
                print(f'feature took {t2 - t1}')
                print(f'train took {t3 - t2}')
                acc = compute_acc(batch_pred, batch_labels)
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'
                    .format(epoch, step, loss.item(), acc.item(),
                            np.mean(iter_tput[3:]),
                            th.cuda.max_memory_allocated() / 1000000))
            t0 = time.time()

        # if n_gpus > 1:
        #     th.distributed.barrier()

        # toc = time.time()
        # if proc_id == 0:
        #     print('Epoch Time(s): {:.4f}'.format(toc - tic))
        #     if epoch >= 5:
        #         avg += toc - tic
        #     if epoch % args.eval_every == 0 and epoch != 0:
        #         if n_gpus == 1:
        #             eval_acc = evaluate(
        #                 model, val_g, val_nfeat, val_labels, val_nid, devices[0])
        #             test_acc = evaluate(
        #                 model, test_g, test_nfeat, test_labels, test_nid, devices[0])
        #         else:
        #             eval_acc = evaluate(
        #                 model.module, val_g, val_nfeat, val_labels, val_nid, devices[0])
        #             test_acc = evaluate(
        #                 model.module, test_g, test_nfeat, test_labels, test_nid, devices[0])
        #         print('Eval Acc {:.4f}'.format(eval_acc))
        #         print('Test Acc: {:.4f}'.format(test_acc))

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu',
                           type=str,
                           default='0,1',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='15,10,5')
    argparser.add_argument('--batch-size', type=int, default=2048)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive',
                           action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument(
        '--data-cpu',
        action='store_false',
        help="By default the script puts all node features and labels "
        "on GPU when using it to save time for data copy. This may "
        "be undesired if they cannot fit in GPU memory at once. "
        "This flag disables that.")
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products')
    else:
        raise Exception('unknown dataset')

    # Construct graph
    g = dgl.as_heterograph(g)
    # g.create_formats_()
    train_nfeat = g.ndata.pop('features')
    train_labels = g.ndata.pop('labels')
    train_nfeat.share_memory_()
    train_labels.share_memory_()
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    g = g.shared_memory("shared_g")
    g.ndata['val_mask'] = val_mask
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    if GPU_FEATURE == 2:
        NUM_ELEMENT = train_nfeat.size(0)
        # distributed feature on GPUs and CPU
        home = os.getenv('HOME')
        root = osp.join(home, 'products')
        if CACHE_MODE == 1:
            shard_tensor_config = ShardTensorConfig({0: "0.2G", 1: "0.2G"})
            shard_tensor = PyShardTensor(0, shard_tensor_config)
            _, prev_order, new_order = split(0.42, "ogbn-products", root)
            train_nfeat = train_nfeat[prev_order]
            shard_tensor.from_cpu_tensor(train_nfeat)
            ipc_handle = shard_tensor.share_ipc()
            shard = ipc_handle, new_order
            data = n_classes, g, shard, train_labels
        else:
            ratio = [0.15, 0.85]
            shard_tensor = split(ratio, "ogbn-products", root)
            data = n_classes, g, shard_tensor, train_labels
    else:
        data = n_classes, g, train_nfeat, train_labels

    mp.set_start_method('spawn')
    procs = []
    for proc_id in range(n_gpus):
        p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
