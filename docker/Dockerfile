#!/usr/bin/env -S sh -c 'docker build --rm -t torch_quiver:snapshot . -f $0 && docker run --rm -it torch_quiver:snapshot'

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

# Install PyG.
RUN CPATH=/usr/local/cuda/include:$CPATH && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && \
    DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install scipy==1.5.0

RUN pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu102.html && \
    pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu102.html && \
    pip install torch-geometric

WORKDIR /quiver
ADD . .
RUN pip install -v .

# Set the default command to python3.
CMD ["python3"]
