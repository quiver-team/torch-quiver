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
import os.path as osp
import sklearn.linear_model as lm
import sklearn.metrics as skm
from quiver.shard_tensor import ShardTensor as PyShardTensor
from quiver.shard_tensor import ShardTensorConfig
from ogb.lsc import MAG240MDataset
from scipy.sparse import csc_matrix


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


def load_240m():
    train_idx = th.load(
        '/data/papers/ogbn_papers100M/index/train_idx.pt').share_memory_()
    indptr = th.load(
        '/data/papers/ogbn_papers100M/csr/indptr.pt').share_memory_()
    indices = th.load(
        '/data/papers/ogbn_papers100M/csr/indices.pt').share_memory_()
    label = th.load('/data/papers/ogbn_papers100M/label/label.pt').squeeze(
    ).share_memory_()

    return (indptr, indices), train_idx, label, 172


def load_ogb(name, root=None):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name)
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


def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id, n_gpus, host,
                   host_size, temp):
    """
    Extracts features and labels for a subset of nodes.
    """
    per_host_nodes = nfeat.size(0)
    res = []
    for src in range(host_size):
        for dst in range(host_size):
            if src == dst:
                continue
            if src == host:
                peer = dst * n_gpus + dev_id
                th.distributed.send(temp, peer)
            elif dst == host:
                peer = src * n_gpus + dev_id
                th.distributed.recv(temp, peer)
    input_nodes = input_nodes // host_size
    input_nodes = input_nodes.cpu()
    batch_inputs = nfeat[input_nodes].to(dev_id).to(th.float32)
    batch_labels = labels[seeds].to(dev_id).to(th.long)
    return batch_inputs, batch_labels


#### Entry point


def run(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    th.cuda.set_device(dev_id)
    print('ready')
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='192.168.0.78', master_port='12975')
        world_size = n_gpus * args.host_size
        th.distributed.init_process_group(backend="gloo",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id + n_gpus * args.host)
    print('comm')
    # Unpack data
    n_classes, train_g, train_idx, train_labels, train_nfeat = data
    val_g = train_g
    test_g = train_g
    indptr, indices = train_g
    nodes = indptr.size(0) - 1
    index = np.zeros(indices.size(0), dtype=np.int8)

    csc = csc_matrix((index, indices.numpy(), indptr.numpy()),
                     shape=[nodes, nodes])
    train_g = dgl.from_scipy(csc)

    in_feats = 128

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')], replace=False)
    dataloader = dgl.dataloading.NodeDataLoader(train_g,
                                                train_idx,
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
    sizes = [int(fanout) for fanout in args.fan_out.split(',')]
    comm_sizes = args.batch_size
    for size in sizes:
        comm_sizes *= size
    comm_sizes = comm_sizes // 2 * (args.host_size - 1) // args.host_size
    temp = th.zeros((comm_sizes, 128))
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
            batch_inputs, batch_labels = load_subtensor(
                train_nfeat, train_labels, seeds, input_nodes, dev_id, n_gpus,
                args.host, args.host_size, temp)
            blocks = [block.int().to(dev_id) for block in blocks]
            t2 = time.time()
            # Compute loss and prediction
            optimizer.zero_grad()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
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
                            np.mean(iter_tput[-10:]),
                            th.cuda.max_memory_allocated() / 1000000))
            t0 = time.time()

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--host', type=int, default=0)
    argparser.add_argument('--host_size', type=int, default=3)
    argparser.add_argument('--gpu',
                           type=str,
                           default='0,1',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--dataset', type=str, default='ogbn-mag240m')
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='25,10')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
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
    nodes = 111059956
    per_host_nodes = (nodes + args.host_size - 1) // args.host_size
    train_nfeat = th.zeros((per_host_nodes, 128)).share_memory_()

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products')
    else:
        g, train_idx, label, n_classes = load_240m()

    # Construct graph
    indptr, _ = g

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    data = n_classes, g, train_idx, label, train_nfeat

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        mp.set_start_method('spawn')
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run,
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
