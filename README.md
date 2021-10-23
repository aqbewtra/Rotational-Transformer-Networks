# Spatial-Transformer-Networks
### Reverse engineering Spatial Transformer Networks to approximate rotational orientation.

### Problem: Estimate the rotational orientation of an image.
The central idea of this solution is to learn the optimal rotation for a given image in a rotational transformer. I've essentially clamped affine transform learned by the novel spatial transformer network to a simple rotation. Read the original paper [here](https://arxiv.org/abs/1506.02025).  
  

By reconstructing the spatial transformer into a, as I'll call it, rotational transformer, we can extract our predicted angle `theta`.