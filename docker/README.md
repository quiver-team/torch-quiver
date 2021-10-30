# Docker

This docker file is a modification of [PyG](https://github.com/pyg-team/pytorch_geometric/tree/master/docker), and it has been tested with cuda 10.1 driver. 

Make sure you have docker installed with [NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) support.

1. Download the dockerfile to your host server.
2. `$ docker build -t "custom image name"`
3. `$ docker run --rm -it --gpus=all --shm-size=4g "custom image name" /bin/bash`
4. Inside your container run `$ pip install torch_quiver` or install from source and try our examples.

You can test the installation by "`>>> import quiver`" in your Python interpreter. If you have issues building the image, you could also get our pre-built image with `$ docker pull zenotan/torch_quiver:test`.

Since Quiver uses shared memory to store data, `--shm-size` should be set to be large enough to hold your dataset.