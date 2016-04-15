<div align="center">
  <img width='600px' src="http://i.imgur.com/XjCXXap.png"><br><br>
</div>
-----------------

**awesome-very-deep-learning** is a curated list for papers and code about implementing and training very deep neural networks.

## Deep Residual Learning 

**Deep Residual Networks** are a family of extremely deep architectures (up to 1000 layers) showing compelling accuracy and nice convergence behaviors. Instead of learning a new representation at each layer, deep residual networks use identity mappings to learn residuals. 

### Papers

- [Deep Networks with Stochastic Depth (2016)](http://arxiv.org/abs/1603.09382) [[original code](https://github.com/yueatsprograms/Stochastic_Depth)], dropout with residual layers as regularizer
- [Identity Mappings in Deep Residual Networks (2016)](http://arxiv.org/abs/1603.05027) [[original code](https://github.com/KaimingHe/resnet-1k-layers)], improving the original proposed residual units by reordering batchnorm and activation layers
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (2016)](http://arxiv.org/abs/1602.07261), inception network with residual connections
- [Deep Residual Learning for Image Recognition (2015)](http://arxiv.org/abs/1512.03385) [[original code](https://github.com/KaimingHe/deep-residual-networks)], original paper introducing residual neural networks

### Implementations

0. Torch by Facebook AI Research (FAIR), with **training code in Torch and pre-trained ResNet-18/34/50/101 models for ImageNet**: [blog](http://torch.ch/blog/2016/02/04/resnets.html), [code](https://github.com/facebook/fb.resnet.torch)
0. Torch, CIFAR-10, with ResNet-20 to ResNet-110, training code, and curves: [code](https://github.com/gcr/torch-residual-networks)
0. Lasagne, CIFAR-10, with ResNet-32 and ResNet-56 and training code: [code](https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning)
0. Neon, CIFAR-10, with pre-trained ResNet-32 to ResNet-110 models, training code, and curves: [code](https://github.com/apark263/cfmz)
0. Torch, MNIST, 100 layers: [blog](https://deepmlblog.wordpress.com/2016/01/05/residual-networks-in-torch-mnist/), [code](https://github.com/arunpatala/residual.mnist)
0. A winning entry in Kaggle's right whale recognition challenge: [blog](http://blog.kaggle.com/2016/02/04/noaa-right-whale-recognition-winners-interview-2nd-place-felix-lau/), [code](https://github.com/felixlaumon/kaggle-right-whale)
0. Neon, Place2 (mini), 40 layers: [blog](http://www.nervanasys.com/using-neon-for-scene-recognition-mini-places2/), [code](https://github.com/hunterlang/mpmz/)
0. Tensorflow with tflearn, with CIFAR-10 and MNIST: [code](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py)
0. Tensorflow with skflow, with MNIST: [code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/skflow/resnet.py)
0. Stochastic dropout in Keras: [code](https://github.com/dblN/stochastic_depth_keras)

In addition, this [code] (https://github.com/ry/tensorflow-resnet) by Ryan Dahl helps to convert the pre-trained models to TensorFlow.


## Highway Networks

**Highway Networks** take inspiration from Long Short Term Memory (LSTM) and allow training of deep, efficient networks (with hundreds of layers) with conventional gradient-based methods

### Papers

- [Training Very Deep Networks (2015)](http://arxiv.org/abs/1507.06228), introducing highway neural networks

### Implementations

0. Lasagne: [code](https://github.com/Lasagne/Lasagne/blob/highway_example/examples/Highway%20Networks.ipynb)
0. Caffe: [code](https://github.com/flukeskywalker/highway-networks)
0. Torch: [code](https://github.com/yoonkim/lstm-char-cnn/blob/master/model/HighwayMLP.lua)
