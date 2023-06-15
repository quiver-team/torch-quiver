# OGBn-Papers100M

This dataset is large, so we need to preprocess the dataset. We assume that you have downloaded the raw dataset from [OGB](https://snap.stanford.edu/ogb/data/nodeproppred/) and decompress it to `/data` (also the files in `split/time/`). Then `preprocess.py` can help you transform the data into the appropriate format.

Also, Quiver uses large shared memory to hold the dataset. If your program is killed silently or has bus error, make sure your physical memory can hold the dataset. You should make sure your shared memory limit is set properly, and we recommend that it is greater than 128G:

```
  echo 128000000000 > /proc/sys/kernel/shmmax
  mount -o remount,size=128G /dev/shm
```

When this is done, you can run `python3 benchmarks/ogbn-papers100M/dist_sampling_ogb_paper100M_quiver.py` to start training with multiple GPUs.
