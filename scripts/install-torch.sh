#!/bin/sh
set -e

cd $(dirname $0)/..

pip3 install -r requirements.torch.txt -f https://download.pytorch.org/whl/torch_stable.html
