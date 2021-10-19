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

* **Impressed performance and scalibility**: Graph sampling and feature collection often consume much of training time in GNN thus cause low utilization of GPU resources. What's even worse is that sample and feature collection with CPU have severe scalability problem because of limited CPU resources. Quiver tackles these two problem and achieve much better performance and scales much better.


## Quick Tour for New Users

In this quick tour, we highlight the ease of creating and training a GNN model with only a few lines of code change.

### A simple example

In the first glimpse of Quiver, we integrate it into the training pipeline of [offical redit examples from PyG](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py). With about 5 lines of code change, you can achieve about 2x training speedup. You can check [our reddit example](examples/pyg/reddit_quiver.py) for details.

```python
...
#############################################
# Original PyG's Code About Sampler
############################################
#train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
#                               sizes=[25, 10], batch_size=1024, shuffle=True,
#                               num_workers=12)

#############################################
# Integrate Quiver: Using Quiver's sampler
############################################
train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True)

csr_topo = quiver.CSRTopo(data.edge_index)
quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10])
...

######################
# Original Pyg's Code
######################
#x = data.x.to(device)

#############################################
# Integrate Quiver: Using Quiver's Feature
############################################
x = quiver.Feature(rank=0, device_list=[0], device_cache_size="1G", cache_policy="device_replicate", csr_topo=csr_topo)
x.from_cpu_tensor(data.x)
...

```


## Architecture Overview




### Pip Wheels

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations.

To install the binaries, simply run

```
pip install torch-quiver
```

cuda10.2 and cuda11.1 on Linux are fully tested

|     OS        | `cu102` | `cu111` |
|-------------|---------|---------|
| **Linux**   | ✅      | ✅      |



### Build from  source

clone this project run script

```cmd
$ git clone git@github.com:quiver-team/torch-quiver.git
$ ./U
```


If you notice anything unexpected, please open an [issue](https://github.com/quiver-team/torch-quiver/issues) and let us know.
If you have any questions or are missing a specific feature, feel free to discuss them with us.
We are motivated to constantly make Quiver even better.
