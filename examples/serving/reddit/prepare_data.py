import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
import sys
os.chdir(sys.path[0])
sys.path.append(osp.abspath(osp.join(os.getcwd(),'..'))) # For import model

import torch
import quiver
from torch_geometric.datasets import Reddit
import torch.multiprocessing as mp

from model import SAGE

if __name__ == "__main__":
    mp.set_start_method('spawn')
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = Reddit(path)
    data = dataset[0]

    sizes = [25, 10]
    reverse = False
    intermediate_path = osp.join(osp.dirname(osp.realpath(__file__)), 'intermediate')
    
    if osp.exists(osp.join(intermediate_path, f'{sizes[0]}_{sizes[1]}_neighbour_num_{reverse}.npy')):
        print("neighbour_num file exists")
    else:
        print("neighbour_num file not exists")
        edge_index = data.edge_index
        if reverse:
            edge_index = torch.stack((edge_index[1], edge_index[0]), 0)
        node_num = data.x.shape[0]

        if osp.exists(intermediate_path):
            print("intermediate_path exists")
        else:
            os.mkdir(intermediate_path)
            print("intermediate_path created")
            
        device_list = [i for i in range(torch.cuda.device_count())]
        quiver.generate_neighbour_num(node_num, edge_index, sizes, intermediate_path, parallel=True, mode='GPU', device_list=device_list, num_proc=4, reverse=reverse)

    result_path = osp.join(osp.dirname(osp.realpath(__file__)), 'result')
    if osp.exists(result_path):
        print("result_path exists")
    else:
        os.mkdir(result_path)
        print("result_path created")
    
    if osp.exists('reddit_quiver_model.pth'):
        print("model file exists")
    else:
        print("model file not exists")
        model = SAGE(data.num_features, 256, dataset.num_classes, 2)
        torch.save(model, 'reddit_quiver_model.pth')