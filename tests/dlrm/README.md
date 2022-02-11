### Test Quiver Embedding on Deep Learning Recommendation Model

This example shows how to replace torch.nn.Embedding with Quiver Embedding. 
In this example, we adopt [DLRM](https://github.com/facebookresearch/dlrm) [[1]](#1) to test the correctness and training/inference performance of Quiver Embedding.

Note that we made essential modifications on the original DLRM code 
mainly because there are critical unsolved issues such as https://github.com/facebookresearch/dlrm/issues/216 
and https://github.com/facebookresearch/dlrm/issues/219. 
We only keep PyTorch training scripts and necessary util files.
We also remove the mlperf testing code to make the training script relatively easy to follow.

## Reference
<a id="1">[1]</a> 
Naumov, Maxim, et al. 
"Deep learning recommendation model for personalization and recommendation systems." 
arXiv preprint arXiv:1906.00091 (2019).
