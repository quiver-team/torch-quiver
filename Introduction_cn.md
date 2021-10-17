# Gpex

## 一、整体介绍
所有研究图机器学习系统的团队都知道基于采样的图模型训练性能瓶颈在图采样和特征聚合上，但两个瓶颈背后的本质到底是什么？Gpex核心观点是：
- 图采样是一个**latency critical problem**，高性能的采样核心在于通过大量并行掩盖访问延迟。
- 特征聚合是一个**bandwidth critical problem**，高性能的特征聚合在于优化聚合带宽。


![GNN训练流程](multi_medias/imgs/overview.png)

一般情况下，我们选择使用CPU进行图采样和特征聚合，这种方案不仅会带来单卡训练的性能问题，同时由于采样与特征聚合均为CPU密集型操作，多卡训练时会**由于CPU计算资源的瓶颈导致训练扩展性不理想**。我们以`ogbn-product`数据集为例benchmark `Pyg`和`DGL`两个框架使用CPU采样和特征聚合时的多卡训练扩展性结果如下：

`说明`: 示例代码见[这里](examples/multi_gpu/pyg/ogb-products/dist_sampling_ogb_products_pyg.py)

| Framework | Device Num | Epoch Time(s) |Scalability|
| ------ | ------ | ------ |------|
| Pyg | 1 | 36.5 |1.0|
| Pyg | 2 | 31.3 |1.16|
| Pyg | 3 | 28.1|1.30|
| Pyg | 4 |  29.1|1.25|
| DGL | 1 |  ||
| DGL | 2 |  ||
| DGL | 3 |  ||
| DGL | 4 |  ||



我们本次开源的单机版本的Gpex是一个即插即用并且充分压榨硬件潜力的高性能GNN训练组件，用户可使用Gpex在单机上训练GNN时拥有**更好的性能**，并且拥有**更好的多卡扩展性**，甚至在拥有NVLink的情况下，获得**多卡超线性加速收益**。接下来我们分别介绍我们在图采样和特征聚合性能优化上的工作与经验。

## 二、图采样

### 2.1 当前已有方案
当前的开源系统中已经支持CPU采样和GPU采样，其中GPU采样需要将整个图存储到GPU显存中。CPU采样往往受困于采样性能以及训练扩展性，而GPU采样因为显存大小的限制，使得能够处理的图的尺寸往往有限。

### 2.2 Gpex方案
Gpex中向用户提供**UVA-Based**（Unified Virtual Addressing Based）图采样算子，支持用户在图拓扑数据较大时选择将图存储在CPU内存中的同时使用GPU进行采样。这样我们不仅获得了远高于CPU采样的性能收益，同时能够处理的图的大小从GPU显存大小限制扩展到了CPU内存大小(一般远远大于GPU显存)。我们在`ogbn-products`和`reddit`上的两个数据集上进行采样性能测试显示，UVA-Based的采样性能远远高于CPU采样性能(CPU采样使用Pyg的采样实现为基线)，我们衡量采样性能的指标为单位时间内的采样边数(**S**ampled **E**dges **P**er **S**econd, **SEPS**)

| Dataset | Parameter | Sampler |SEPS|Speedup Over CPU|
| ------ | ------ | ------ |------|------|
| ogbn-product | [15, 10, 5] | CPU |2.23 M|1.0|
| ogbn-product | [15, 10, 5] | UVA |48.53 M|21.76|
| reddit | [25, 10] | CPU |3.53 M|1.0|
| reddit | [25, 10] | UVA |41.2 M|11.67|



![uva-sampler](./multi_medias/imgs/UVA-Sampler.png)

```python

    dataset = PygNodePropPredDataset('ogbn-products', root)
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)

    # You can set mode='GPU' to choose place graph data in GPU memory
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], device=0, mode="UVA")
```














