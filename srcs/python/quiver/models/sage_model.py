from torch_geometric.nn import SAGEConv
from tqdm import tqdm
import torch.nn.functional as F
import torch


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

    def inference(self, x_all, subgraph_loader, device):
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

    def train_m(self, loader, w, optimizer, device, x, y, train_idx, epoch,
                mode, tot_epoch):
        # w1 = StopWatch('train loop')
        super().train()
        # w1.tick('set mode to train')

        # pbar = tqdm(total=train_idx.size(0))
        # pbar.set_description(f'Epoch {epoch:02d}')
        if epoch > 1 and mode == 'prefetch':
            loader.reset()
        total_loss = total_correct = 0
        w.turn_on('sample')
        for batch_size, n_id, adjs in loader:
            w.turn_off('sample')
            w.turn_on('train')
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            # w1.tick('prepro')
            adjs = [adj.to(device) for adj in adjs]

            optimizer.zero_grad()
            out = self(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
            # w1.tick('train')

            total_loss += float(loss)
            total_correct += int(
                out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
            # pbar.update(batch_size)
            # print('\n\n')
            w.turn_on('sample')
            w.turn_off('train')
        if epoch == tot_epoch and mode == 'prefetch':
            loader.close()
        w.turn_off('sample')

        # pbar.close()

        loss = total_loss / len(loader)
        approx_acc = total_correct / train_idx.size(0)

        # del w1
        return loss, approx_acc
