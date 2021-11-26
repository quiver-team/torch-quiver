# Reaches around 0.7866 ± 0.0041 test accuracy.

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset

import quiver

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
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
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(
                self.layers) - 1 else self.n_classes).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes(), device=g.device),
                sampler,
                device=device,
                batch_size=args.batch_size*4,
                shuffle=False,
                drop_last=False,
                num_workers=0 if args.sample_gpu else args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

# Entry point

def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        g = g.formats(['csc'])
        g = g.to(device)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0 if args.sample_gpu else args.num_workers,
        persistent_workers=not args.sample_gpu)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes,
                 args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    best_val_acc = final_test_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        model.train()
        pbar = tqdm(total=train_nid.size(0))
        pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_correct = 0
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for input_nodes, seeds, blocks in dataloader:

            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(
                nfeat, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()
            pbar.update(args.batch_size)

        pbar.close()

        loss = total_loss / len(dataloader)
        approx_acc = total_correct / (len(dataloader) * args.batch_size)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}, Epoch Time: {time.time() - tic:.4f}')

        if epoch >= 5:
            val_acc, test_acc = evaluate(model, g, nfeat, labels, val_nid, test_nid, device)
            print(f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    return final_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--sample-gpu', action='store_true')
    argparser.add_argument('--data', type=str, choices=('cpu', 'gpu', 'quiver', 'unified'))
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load ogbn-products data
    data = DglNodePropPredDataset(name='ogbn-products', root='/data/products')
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]
    labels = labels[:, 0].to(device)

    feat = graph.ndata.pop('feat')
    if args.data == 'cpu':
        nfeat = feat
    elif args.data == 'gpu':
        nfeat = feat.to(device)
    elif args.data == 'quiver':
        csr_topo = quiver.CSRTopo(th.stack(graph.edges('uv')))
        nfeat = quiver.Feature(rank=args.gpu, device_list=[args.gpu], 
                               device_cache_size="200M", cache_policy="device_replicate", 
                               csr_topo=csr_topo)
        nfeat.from_cpu_tensor(feat)
    elif args.data == 'unified':
        from distutils.version import LooseVersion
        assert LooseVersion(dgl.__version__) >= LooseVersion('0.8.0'), \
            f'Current DGL version ({dgl.__version__}) does not support UnifiedTensor.'
        nfeat = dgl.contrib.UnifiedTensor(feat, device=device)
    else:
        raise ValueError(f'Unsupported feature storage location {args.data}.')

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph

    test_accs = []
    for i in range(1, 11):
        print(f'\nRun {i:02d}:\n')
        test_acc = run(args, device, data)
        test_accs.append(test_acc)
    test_accs = th.tensor(test_accs)
    print('============================')
    print(f'Final Test: {test_accs.mean():.4f} ± {test_accs.std():.4f}')
