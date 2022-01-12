# OGBn-mag240M

This dataset is large, so we need to preprocess the dataset. We assume that you have downloaded the raw dataset from [OGB](https://snap.stanford.edu/ogb/data/nodeproppred/) and decompress it to `\data\mag` (similar to ogbn-papers100M). Then `preprocess.py` can help you transform the data into the appropriate format. You can change the GPU and CPU memory size to match your hardware configurations. The `p2p_group` and `p2p_size` are NVLink-connected GPU cliques and the number of cliques seperately.

Also, Quiver uses large shared memory to hold the dataset. If your program is killed silently or has bus error, make sure your physical memory can hold the dataset. You should make sure your shared memory limit is set properly, and we recommend that it is greater than 400G.

When this is done, you can run `python3 benchmarks/ogbn-mag240m/train_quiver_multi_node.py` to start training with multiple hosts. Every host should launch the script at the same time, with correct arguments such as `host` and `host_size`.
