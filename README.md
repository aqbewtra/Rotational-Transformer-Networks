# Rotational Transformer Networks
Reverse engineering Spatial Transformer Networks to approximate rotational orientation.

### Problem: Estimate the rotational orientation of an image.
The central idea of a rotational transformer network is to learn the optimal ***rotation*** for a given image. 

The novel spatial transformer network learns an affine transformation; the modified rotational transformer clamps the transform into a simple rotation. The network learns an angle of rotation `theta` - a single value - instead of an 3x2 affine transformation matrix. 

Read the original Spatial Transformer Network paper [here](https://arxiv.org/abs/1506.02025). 
  

By reconstructing the spatial transformer into a, as I'll call it, rotational transformer, we can extract our predicted angle `theta`.