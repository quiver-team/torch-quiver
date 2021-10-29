# OGBn-Papers100M

This dataset is large, so we need to preprocess the dataset. We assume that you have downloaded the raw dataset from [OGB](https://snap.stanford.edu/ogb/data/nodeproppred/) and decompress it to `\data\papers` (also the files in `split/time/`). Then `preprocess.py` can help you transform the data into the appropriate format.

When this is done, you can run `python3 benchmarks/ogbn-papers100M/dist_sampling_ogb_paper100M_quiver.py` to start training with multiple GPUs.