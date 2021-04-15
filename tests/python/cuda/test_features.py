from torch_geometric.data import NeighborSampler
from quiver.cuda_sampler import CudaNeighborSampler
import torch
from quiver.profile_utils import StopWatch


class ProductsDataset:
    def __init__(self, train_idx, edge_index, x, y, f, c):
        self.train_idx = train_idx
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.num_features = f
        self.num_classes = c
root = './products.pt'
dataset = torch.load(root)
train_idx = dataset.train_idx
edge_index = dataset.edge_index
x = dataset.x
x = x.share_memory_()
batch_size = 1000000
print(batch_size)
torch.set_num_threads(4)
max_index = dataset.num_nodes

w = StopWatch('feature')
cpu_device = torch.device('cpu')
cuda_device = torch.device('cuda:3')

for i in range(1, 51):
    n_id = torch.randint(0, max_index - 1, (batch_size,))
    temp_x = x[n_id]

w.tick('start cpu')
for i in range(1, 101):
    n_id = torch.randint(0, max_index - 1, (batch_size,))
    w.turn_on('cpu')
    temp_x = x[n_id]
    w.turn_off('cpu')
w.tick('finish cpu')

x = x.to(cuda_device)
for i in range(1, 51):
    n_id = torch.randint(0, max_index - 1, (batch_size,))
    temp_x = x[n_id].to(cpu_device)

# torch.set_num_threads(4)
w.tick('start cuda')
for i in range(1, 101):
    w.turn_on('cuda')
    temp_x = x[n_id].to(cpu_device)
    w.turn_off('cuda')

w.tick('finish')
del w