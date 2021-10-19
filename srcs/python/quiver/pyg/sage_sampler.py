import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch_quiver as qv
from typing import List, Tuple, NamedTuple

from .. import utils as quiver_utils


__all__ = ["GraphSageSampler"]


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)
                   
class GraphSageSampler:
    r"""
    Quiver's GraphSageSampler behaves just like Pyg's `NeighborSampler` but with much higher performance.
    It can work in `UVA` mode or `GPU` mode. You can set `mode=GPU` if you have enough GPU memory to place graph's topology data which will offer the best sample performance.
    When your graph is too big for GPU memory, you can set `mode=UVA` to still use GPU to perform sample but place the data in host memory. `UVA` mode suffers 30%-40% performance loss compared to `GPU` mode
    but is much faster than CPU sampling(normally 16x~20x) and it consumes much less GPU memory compared to `GPU` mode.

    Args:
        csr_topo (quiver.CSRTopo): A quiver.CSRTopo for graph topology
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to `sizes[l] = -1`, all neighbors are included
            in layer `l`.
        device (int): Device which sample kernel will be launched
        mode (str): Sample mode, choices are [`UVA`, `GPU`], default is `UVA`.
    """

    def __init__(self, csr_topo: quiver_utils.CSRTopo, sizes: List[int], device, mode="UVA"):

        assert mode in ["UVA", "GPU"], f"sampler mode should be one of [UVA, GPU]"
        self.sizes = sizes
        self.quiver = None
        self.csr_topo = csr_topo
        self.mode = mode
        if device >= 0:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.new_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices, edge_id, device, self.mode != "UVA")
    
        self.device = device

        self.ipc_handle_ = None

    
    def sample_layer(self, batch, size):
        self.lazy_init_quiver()
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch.to(torch.device(self.device))
        size = size if size != -1 else self.csr_topo.node_count
        n_id, count = self.quiver.sample_neighbor(0, n_id, size)
        return n_id, count

    def lazy_init_quiver(self):
        if self.quiver is not None:
            return 
        self.device = torch.cuda.current_device()
        edge_id = torch.zeros(1, dtype=torch.long)
        self.quiver = qv.new_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices, edge_id, self.device, self.mode != "UVA")

    def reindex(self, inputs, outputs, counts):
        return qv.reindex_single(inputs, outputs, counts)

    def sample(self, input_nodes):
        """Sample k-hop neighbors from input_nodes

        Args:
            input_nodes (torch.LongTensor): seed nodes ids to sample from

        Returns:
            Tuple: Return results are the same with Pyg's sampler
        """
        self.lazy_init_quiver()
        nodes = input_nodes.to(self.device)
        adjs = []

        batch_size = len(nodes)
        for size in self.sizes:
            out, cnt = self.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.reindex(nodes, out, cnt)
            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            adj_size = torch.LongTensor([
                frontier.size(0),
                nodes.size(0),
            ])
            e_id = torch.tensor([])
            adjs.append(Adj(edge_index, e_id, adj_size))
            nodes = frontier

        return nodes, batch_size, adjs[::-1]

    def share_ipc(self):
        """Create ipc handle for multiprocessing

        Returns:
            tuple: ipc handle tuple
        """
        return self.csr_topo, self.sizes, self.mode
    
    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        """Create from ipc handle

        Args:
            ipc_handle (tuple): ipc handle got from calling `share_ipc`

        Returns:
            quiver.pyg.GraphSageSampler: Sampler created from ipc handle
        """
        csr_topo, sizes, mode = ipc_handle
        return cls(csr_topo, sizes, -1, mode)