# from quiver.saint_sampler import QuiverSAINTEdgeSampler
# import torch
# from torch_sparse import SparseTensor
# import os.path as osp
# from torch_geometric.datasets import Flickr
# from torch_geometric.data import GraphSAINTEdgeSampler
#
# # -----below is flicker-----------
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
# dataset = Flickr(path)
# data = dataset[0]
# row, col = data.edge_index
#
# # i = torch.tensor([[0,0,1,1,2,2,3,3,4,4], [2,3,2,4,0,1,0,4,1,3]])
# # v = torch.tensor([1,2,3,4,5,6,7,8,9,10])
# # s = SparseTensor(row=i[0], col=i[1], value=v, sparse_sizes=(5,5))
#
# # print(s.to_dense())
# loader = GraphSAINTEdgeSampler(data,
#                                  batch_size = 100,
#                                  num_steps = 1,
#                                  sample_coverage=0,
#                                  save_dir=dataset.processed_dir,
#                                  num_workers = 0)
#
# for data in loader:
#     print("x is "+ str(data.x.size(0)))
#
# loader = QuiverSAINTEdgeSampler(data,
#                                  batch_size = 100,
#                                  num_steps = 1,
#                                  sample_coverage=0,
#                                  save_dir=dataset.processed_dir,
#                                  num_workers = 0)
# for data in loader:
#     print("x is" +str(data.x.size(0)))
#
