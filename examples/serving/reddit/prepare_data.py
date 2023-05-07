from torch_geometric.datasets import Reddit
import os.path as osp
import os
import torch
import torch_quiver as qv

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = Reddit(path)
data = dataset[0]

sizes = [25, 10]
reverse = False
intermediate_path = osp.join(osp.dirname(osp.realpath(__file__)), 'intermediate')

edge_index = data.edge_index
if reverse:
    edge_index = torch.stack((edge_index[1], edge_index[0]), 0)
node_num = data.x.shape[0]

if osp.exists(intermediate_path):
    print("intermediate_path exists")
else:
    os.mkdir(intermediate_path)
    print("intermediate_path created")
    
qv.generate_neighbour_num(node_num, edge_index, sizes, intermediate_path, parallel=False, mode='GPU', reverse=reverse)

result_path = osp.join(osp.dirname(osp.realpath(__file__)), 'result')
if osp.exists(result_path):
    print("result_path exists")
else:
    os.mkdir(result_path)
    print("result_path created")