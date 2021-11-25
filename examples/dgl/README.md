# Examples of using Quiver with DGL backend

Currently, the `quiver.Sampler` does not work with DGL, but we can leverage `quiver.Feature` to accelerate the feature transferring in DGL.

## Requirements

- dgl >= 0.7.1

> The `UnifiedTensor` requires DGL version >= 0.8.0 and is not available in the current release.

## Ogbn-products

The script `ogbn_products_sage_quiver.py` supports loading feature data from multiple locations:
- `cpu`: features are stored in the host memory and need to be transfered to the GPU memory for training.
- `gpu`: features are already stored in the GPU/device memory. This is the fastest way but consumes the most amount of GPU memory as well.
- `quiver`: use `quiver.Feature` to manage feature storage and transferring.
- `unified`: use DGL's (version >= 0.8.0) `UnifiedTensor`, which utilizes NVIDIA GPU's unified virtual address (UVA) and zero-copy access capabilities to accelerate data transferring. Refer to [this](https://docs.dgl.ai/en/latest/api/python/dgl.contrib.UnifiedTensor.html) for more details.

### Training Scripts

```bash
python ogbn_products_sage_quiver.py --data cpu|gpu|quiver|unified
```

You can also specify `--sample-gpu` to enable DGL's GPU sampling.

### Benchmark Results

Run the training script on a single RTX 3090 GPU card, the epoch times are listed below.

| Feature        | Sampler | Epoch Time (s) |
|----------------|---------|----------------|
| CPU            | CPU     |           24.8 |
| CPU            | GPU     |           13.9 |
| Cache (21%)    | CPU     |           7.95 |
| Cache (21%)    | GPU     |           3.46 |
| Unified Tensor | CPU     |           17.4 |
| Unified Tensor | GPU     |           4.25 |
| GPU            | CPU     |           6.77 |
| GPU            | GPU     |           2.33 |
