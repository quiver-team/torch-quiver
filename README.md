[pypi-image]: https://badge.fury.io/py/torch-geometric.svg
[pypi-url]: https://pypi.org/project/torch-quiver/

<p align="center">
  <img height="150" src="docs/multi_medias/imgs/logo.png" />
</p>

--------------------------------------------------------------------------------

Quiver is a distributed graph learning library for PyTorch. The goal of Quiver is to make distributed graph learning fast and easy to use.

<!-- **Quiver** is a high-performance GNN training add-on which can fully utilize the hardware to achive the best GNN trainning performance. By integrating Quiver into your GNN training pipeline with **just serveral lines of code change**, you can enjoy **much better end-to-end performance** and **much better scalability with multi-gpus**, you can even achieve **super linear scalability** if your GPUs are connected with NVLink, Quiver will help you make full use of NVLink. -->

--------------------------------------------------------------------------------

## Why Quiver?

The primary motivation for this project to make it easy to take a single-GPU `PyG` script, and efficiently scale it across many GPUs and CPUs in parallel. To achieve this, Quiver provides several features:
<!-- 
If you are a GNN researcher or you are a `PyG`'s or `DGL`'s user and you are suffering from consuming too much time on graph sampling and feature collection when training your GNN models, then here are some reasons to try out Quiver for your GNN model trainning. -->

* **High performance**: Quiver enables GPUs to be efficiently used in accelerating graph sampling, feature construction and data parallel training, which usually become bottlenecks in large-scale graph learning.

* **High scalability**: Quiver can achieve even super linear scalablity in distributed graph learning. This is contributed by novel communication-efficient data/processor management techniques and an effective utilisation of emerging networking technologies (e.g., NVLink and RDMA).

<!-- * **Greate performance and scalibility**: Using CPU to do graph sample and feature collection not only leads to poor performance, but also leads to poor scalability because of CPU contention. Quiver, however, can achieve much better scalability and can even achieve `super linear scalibility` on machines equipped with NVLink. -->

* **Easy to use**: Quiver requires only a few lines of code changes in existing `PyG` programs, and it has no external dependency. This makes Quiver easy to be adopted by both `PyG` beginners and professional users.

<!-- * **Easy-to-use and unified API**:
Integrate Quiver into your training pipeline in `PyG` or `DGL` is just a matter of several lines of code change. We've also implemented IPC mechanism which makes it also a piece of cake to use Quiver to speedup your multi-gpu GNN model training (see the next section for a [quick tour](#quick-tour-for-new-users)).  -->

Below is a chart represeting the benchmark that evaluates the performance of Quiver, PyG and DGL on X servers with X GPUs each. 

![e2e_benchmark](docs/multi_medias/imgs/benchmark_e2e_performance.png)

For system design details, see Quiver's [design overview](docs/Introduction_en.md) (Chinese version: [设计简介](docs/Introduction_cn.md)).

## Install

Assuming `PyG` has been installed locally, you can install Quiver as follow:

```
pip install torch-quiver
```

Quiver has been tested with Cuda 10.2 and 11.1 on Linux:

|     OS        | `cu102` | `cu111` |
|-------------|---------|---------|
| **Linux**   | ✅      | ✅      |


Quiver can be also built from source:

```cmd
$ git clone git@github.com:quiver-team/torch-quiver.git
$ sh ./install.sh
```

## Quick Start

Quiver comes into the play by replacing PyG's slow graph sampler and feature collector with `quiver.Sampler` and `quiver.Feature`, respectively. This replacement can be done by changing a few lines of code in existing PyG programs. In the below example, the `PyG` user wants to modify a single-GPU program to leverage a 4-GPU server:

```python
...

## Step 1: Parallel graph sampling
# train_loader = NeighborSampler(...) # Comment out PyG sampler

# Start: Enable Quiver sampler
train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True)
csr_topo = quiver.CSRTopo(data.edge_index)
quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10])
# End

...

## Step 2: Parallel feature collection
# x = data.x.to(device) # Comment out PyG feature collector

# Start: Eanble Quiver feature collector
x = quiver.Feature(rank=0, device_list=[0], device_cache_size="1G", cache_policy="device_replicate", csr_topo=csr_topo)
x.from_cpu_tensor(data.x)
# End

## Step 3: Data parallel training
# TODO


...

```

To run a Quiver program on a 4-GPU server, try:

```cmd
$ python ..........
```

A full example is available [here](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py) where Quiver can achieve 2x performance improvement with the Reddit dataset on a 4-GPU server.

If you have multiple multi-GPU servers, try:

```cmd
$ python ..........
```

<!-- You can check [our reddit example](examples/pyg/reddit_quiver.py) for details. -->

## Examples

We provide a large collection of examples to demonsrate how to use Quiver in practice:

- Quiver can be eaisly enabled in the PyG examples for [ogbn-product](examples/pyg/) and [reddit](examples/pyg/).
- Multi-GPU Quiver is also easy to be enabled in PyG's examples for [ogbn-product](examples/multi-gpu/pyg/ogb-products/) and [reddit](examples/multi-gpu/pyg/reddit/).

## Documentation

Please refer to the [API Document](docs/) to learn more about the arguments passed to the Quiver's graph sampler and feature collector.


## License

Quiver is released under the Apache 2.0 license. 

<!-- ## Architecture Overview
Key reasons behind Quiver's high performance are that it provides two key components: `quiver.Feature` and `quiver.Sampler`.

Quiver provide users with **UVA-Based**（Unified Virtual Addressing Based）graph sampling operator, supporting storing graph topology data in CPU memory and sampling the graph with GPU. In this way, we not only get performance benefits beyond CPU sampling, but can also process graphs whose size are too large to host in GPU memory. With UVA, Quiver achieves nearly **20x** sample performance compared with CPU doing graph sample. Besides `UVA mode`, Quiver also support `GPU` sampling mode which will host graph topology data all into GPU memory and will give you 40% ~ 50% performance benifit w.r.t `UVA` sample.

![uva_sample](docs/multi_medias/imgs/UVA-Sampler.png)


A training batch in GNN also consumed hundreds of MBs memory and move memory of this size across CPU memory or between CPU memory and GPU memory consumes hundreds of milliseconds.Quiver utilizes high throughput between page locked memory and GPU memory, high throughput of p2p memory access between different GPUs' memory when they are connected with NVLinks and high throughput of local GPU global memory access to achieve 4-10x higher feature collection throughput compared to conventional method(i.e. use CPU to do sparse feature collection and transfer data to GPU). It partitons data to local GPU memory, other GPUs's memory(if they connected to current GPU with NVLink) and CPU page locked memory. 

We also discovered that real graphs nodes' degree often obeys power-law distribution and nodes with high degree are more often to be accessed during training and sampling. `quiver.Feature` can also do some preprocess to ensure that hottest data are always in GPU's memory(local GPU's memory or other GPU's memory which can be p2p accessed) and this will furtherly improve feature collection performance during training.

![feature_collection](docs/multi_medias/imgs/single_device.png)

For system design details, you can read our (introduction)[docs/Introduction_en.md], we also provide chinese version: [中文版本系统介绍](docs/Introduction_cn.md) -->


<!-- ## Benchmarks

Here we show benchmark about graph sample, feature collection and end2end training. They are all tested on open dataset.

### Sample benchmark
Quiver's sampling can be configured to use UVA sampling (`mode='UVA'`) or GPU sampling(`mode='GPU'`), hosting the whole graph structure in CPU memory and GPU memory respectively.
We use **S**ampled **E**dges **P**er **S**econd (**SEPS**) as metrics to evaluate sample performance. **Without storing the graph on GPU, Quiver get 20x speedup on real datasets**.

![sample benchmark](docs/multi_medias/imgs/benchmark_img_sample.png)

### Feature collection benchmark

We constrain each GPU caching 20% of feature data. Quiver can achieve **10x throughput** on ogbn-product data compared to CPU feature collection.

![single_device](docs/multi_medias/imgs/benchmark_img_feature_single_device.png)

If your GPUs are connected with NVLink, Quiver can make full use of it and achieve **super linear throughput increase**. Our test machine has 2 GPUs connected with NVLink and we still constrain each GPU caching 20% percent of feature data(which means 40% feature data are cached on GPU with 2 GPUs), we achieve 4~5x total throughput increase with the second GPU comes in.

![p2p_access](docs/multi_medias/imgs/p2p_access.png)

![super_linear](docs/multi_medias/imgs/super_linear_feature_bench.png)

### End2End training benchmark

With high performance sampler and feature collection, Quiver not only achieve good performance with single GPU training, but also enjoys good scalability. We modify [PyGs official multi-gpu training example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py) to train `ogbn-product`([code file is here](example/multi_gpu/pyg/ogb-products)). By constraining each GPU to cache only 20% of feature data, we can achieve better scalability even compared with placing all of feature data in GPU in PyG. 

![e2e_benchmark](docs/multi_medias/imgs/benchmark_e2e_performance.png)

When training with multi-GPU and there are no NVLinks between these GPUs, Quiver will use `device_replicate` cache policy by default(you can refer to our [introduction](docs/Introductions_en.md) to learn more about this cache policy). If you have NVLinks, Quiver can make several GPUs share their GPU memory and cache more data to achieve higher feature collection throughput. Our test machine has 2 GPUs connected with NVLink and we still constrain each GPU caching 20% percent of feature data(which means 40% feature data are cached on GPU with 2 GPUs), we show our scalability results here:

![](docs/multi_medias/imgs/nvlink_e2e.png) -->



<!-- ## Note

If you notice anything unexpected, please open an [issue](https://github.com/quiver-team/torch-quiver/issues) and let us know.
If you have any questions or are missing a specific feature, feel free to discuss them with us.
We are motivated to constantly make Quiver even better. -->
