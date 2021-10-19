[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.org/project/torch-quiver/

<p align="center">
  <img height="150" src="docs/multi_medias/imgs/logo.png" />
</p>

--------------------------------------------------------------------------------


**Quiver** is a high-performance GNN training add-on which can fully utilize the hardware to achive the best GNN trainning performance. By integrating Quiver into your GNN training pipeline with **just serveral lines of code change**, you can enjoy **much better end-to-end performance** and **much better scalability with multi-gpus**, you can even achieve **super linear scalability** if your GPUs are connected with NVLink, Quiver will help you make full use of NVLink.

--------------------------------------------------------------------------------

## Library Highlights

If you are a GNN researcher or you are a `PyG`'s or `DGL`'s user and you are suffering from consuming too much time on graph sampling and feature collection when training your GNN models, ere are some reasons to try out Quiver for your GNN training.

* **Really Easy-to-use and unified API**:
  All it takes is 5-10 lines of code to integrate Quiver into your training pipepline, whether you are using `PyG` or `DGL` (see the next section for a [quick tour](#quick-tour-for-new-users)). 
* **Impressed performance and scalibility**: pass


## Quick Tour for New Users

In this quick tour, we highlight the ease of creating and training a GNN model with only a few lines of code.

### Train your own GNN model

In the first glimpse of Quiver, we integrate it into the training pipeline of [offical redit examples from PyG](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py) and we show how simplicity it is to use Quiver.

```python
#############################
# Original Pyg Code
#############################
# train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
#                               sizes=[15, 10, 5], batch_size=1024,
#                               shuffle=True, num_workers=12)

#############################################
# Integrate Quiver: Using Quiver's sampler
############################################
train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True)

csr_topo = quiver.CSRTopo(data.edge_index)

quiver_sampler = GraphSageSampler(csr_topo, sizes=[15, 10, 5], device=0)

```

<details>
<summary>We can now optimize the model in a training loop, similar to the <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation">standard PyTorch training procedure</a>.</summary>

```python
import torch.nn.functional as F

data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
</details>

More information about evaluating final model performance can be found in the corresponding [example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py).

### Create your own GNN layer

In addition to the easy application of existing GNNs, PyG makes it simple to implement custom Graph Neural Networks (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) for the accompanying tutorial).
For example, this is all it takes to implement the [edge convolutional layer](https://arxiv.org/abs/1801.07829) from Wang *et al.*:

<p align="center">
  <img height="40" src="https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_figures/edge_conv.svg?sanitize=true" />
</p>

```python
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")  # "Max" aggregation.
        self.mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)  # shape [num_edges, out_channels]
```

### Manage experiments with GraphGym

GraphGym allows you to manage and launch GNN experiments, using a highly modularized pipeline (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/graphgym.html) for the accompanying tutorial).

```
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric/graphgym
bash run_single.sh  # run a single GNN experiment (node/edge/graph-level)
bash run_batch.sh   # run a batch of GNN experiments, using differnt GNN designs/datasets/tasks
```

Users are highly encouraged to check out the [documentation](https://pytorch-geometric.readthedocs.io/en/latest), which contains additional tutorials on the essential functionalities of PyG, including data handling, creation of datasets and a full list of implemented methods, transforms, and datasets.
For a quick start, check out our [examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples) in `examples/`.

## Architecture Overview

PyG provides a multi-layer framework that enables users to build Graph Neural Network solutions on both low and high levels.
It comprises of the following components:

* The PyG **engine** utilizes the powerful PyTorch deep learning framework, as well as additions of efficient CUDA libraries for operating on sparse data, *e.g.*, [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse) and [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster).
* The PyG **storage** handles data processing, transformation and loading pipelines. It is capable of handling and processing large-scale graph datasets, and provides effective solutions for heterogeneous graphs. It further provides a variety of sampling solutions, which enable training of GNNs on large-scale graphs.
* The PyG **operators** bundle essential functionalities for implementing Graph Neural Networks. PyG supports important GNN building blocks that can be combined and applied to various parts of a GNN model, ensuring rich flexibility of GNN design.
* Finally, PyG provides an abundant set of GNN **models**, and examples that showcase GNN models on standard graph benchmarks. Thanks to its flexibility, users can easily build and modify custom GNN models to fit their specific needs.

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_static/img/architecture.svg?sanitize=true" />
</p>

etero_mag.py)]
</details>


### Pip Wheels

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://data.pyg.org/whl).

#### PyTorch 1.9.0/1.9.1

To install the binaries for PyTorch 1.9.0 and 1.9.1, simply run

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu111` depending on your PyTorch installation (`torch.version.cuda`).

|             | `cpu` | `cu102` | `cu111` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

For additional but optional functionality, run

```
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
```

#### PyTorch 1.8.0/1.8.1

To install the binaries for PyTorch 1.8.0 and 1.8.1, simply run

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html
pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu101`, `cu102`, or `cu111` depending on your PyTorch installation (`torch.version.cuda`).

|             | `cpu` | `cu101` | `cu102` | `cu111` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ❌      | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |

For additional but optional functionality, run

```
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html
```

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, PyTorch 1.5.0, PyTorch 1.6.0 and PyTorch 1.7.0/1.7.1 (following the same procedure).

### From master

In case you want to experiment with the latest PyG features which are not fully released yet, ensure that `torch-scatter` and `torch-sparse` are installed by [following the steps mentioned above](#pip-wheels), and install PyG from master via:

```
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

## Cite

Please cite [our paper](https://arxiv.org/abs/1903.02428) (and the respective papers of the methods used) if you use this code in your own work:

```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```

Feel free to [email us](mailto:matthias.fey@tu-dortmund.de) if you wish your work to be listed in the [external resources](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html).
If you notice anything unexpected, please open an [issue](https://github.com/pyg-team/pytorch_geometric/issues) and let us know.
If you have any questions or are missing a specific feature, feel free [to discuss them with us](https://github.com/pyg-team/pytorch_geometric/discussions).
We are motivated to constantly make PyG even better.
