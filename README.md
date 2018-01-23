<div align="center">
  <img width='600px' src="http://i.imgur.com/XjCXXap.png"><br><br>
</div>

-----------------

**awesome-very-deep-learning** is a curated list for papers and code about implementing and training very deep neural networks.

## Value Iteration Networks

**Value Iteration Networks** are very deep networks that have tied weights and perform approximate value iteration. They are used as an internal (model-based) planning module.

### Papers

- [Value Iteration Networks (2016)](https://arxiv.org/abs/1602.02867) [[original code](https://github.com/avivt/VIN)], introduces VINs (Value Iteration Networks). The author shows that one can perform value iteration using iterative usage of convolutions and channel-wise pooling. It is able to generalize better in environments where a network needs to plan. NIPS 2016 best paper. 

## Densely Connected Convolutional Networks

**Densely Connected Convolutional Networks** are very deep neural networks consisting of dense blocks. Within dense blocks, each layer receives the the feature maps of all preceding layers. This leverages feature reuse and thus substantially reduces the model size (parameters).

### Papers

- [Densely Connected Convolutional Networks (2016)](https://arxiv.org/abs/1608.06993) [[original code](https://github.com/liuzhuang13/DenseNet)], introduces DenseNets and shows that it outperforms ResNets in CIFAR10 and 100 by a large margin (especially when not using data augmentation), while only requiring half the parameters. CVPR 2017 best paper.

### Implementations

0. Authors' [Caffe Implementation](https://github.com/liuzhuang13/DenseNetCaffe)
0. Authors' more memory-efficient [Torch Implementation](https://github.com/gaohuang/DenseNet_lite).
0. [Tensorflow Implementation](https://github.com/YixuanLi/densenet-tensorflow) by Yixuan Li.
0. [Tensorflow Implementation](https://github.com/LaurentMazare/deep-models/tree/master/densenet) by Laurent Mazare.
0. [Lasagne Implementation](https://github.com/Lasagne/Recipes/tree/master/papers/densenet) by Jan Schlüter.
0. [Keras Implementation](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet) by tdeboissiere. 
0. [Keras Implementation](https://github.com/robertomest/convnet-study) by Roberto de Moura Estevão Filho.
0. [Chainer Implementation](https://github.com/t-hanya/chainer-DenseNet) by Toshinori Hanya.
0. [Chainer Implementation](https://github.com/yasunorikudo/chainer-DenseNet) by Yasunori Kudo.
0. [PyTorch Implementation (including BC structures)](https://github.com/andreasveit/densenet-pytorch) by Andreas Veit
0. [PyTorch Implementation](https://github.com/bamos/densenet.pytorch)


## Deep Residual Learning 

**Deep Residual Networks** are a family of extremely deep architectures (up to 1000 layers) showing compelling accuracy and nice convergence behaviors. Instead of learning a new representation at each layer, deep residual networks use identity mappings to learn residuals. 

### Papers

- [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/abs/1707.04585v1) [[code](https://github.com/renmengye/revnet-public)] constructs reversible residual layers (no need to store activations) and surprisingly finds out that reversible layers don't impact final performance. 
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) [[original code](https://github.com/hujie-frank/SENet)], introduces Squeeze-and-Excitation (SE) block, that adaptively recalibrates channel-wise feature responses. It achieved the 1st place on ILSVRC17.
- [Aggregated Residual Transformation for Deep Neural Networks (2016)](https://arxiv.org/abs/1611.05431), introduces ResNeXt, which aggregates a set of transformations within a a res-block. It achieved the 2nd place on ILSVRC16.
- [Residual Networks of Residual Networks: Multilevel Residual Networks (2016)](https://arxiv.org/abs/1608.02908), adds multi-level hierarchical residual mappings and shows that this improves the accuracy of deep networks
- [Wide Residual Networks (2016)](http://arxiv.org/abs/1605.07146) [[orginal code](https://github.com/szagoruyko/wide-residual-networks)], studies wide residual neural networks and shows that making residual blocks wider outperforms deeper and thinner network architectures
- [Swapout: Learning an ensemble of deep architectures (2016)](https://arxiv.org/pdf/1605.06465v1.pdf), improving accuracy by randomly applying dropout, skipforward and residual units per layer
- [Deep Networks with Stochastic Depth (2016)](http://arxiv.org/abs/1603.09382) [[original code](https://github.com/yueatsprograms/Stochastic_Depth)], dropout with residual layers as regularizer
- [Identity Mappings in Deep Residual Networks (2016)](http://arxiv.org/abs/1603.05027) [[original code](https://github.com/KaimingHe/resnet-1k-layers)], improving the original proposed residual units by reordering batchnorm and activation layers
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (2016)](http://arxiv.org/abs/1602.07261), inception network with residual connections
- [Deep Residual Learning for Image Recognition (2015)](http://arxiv.org/abs/1512.03385) [[original code](https://github.com/KaimingHe/deep-residual-networks)], original paper introducing residual neural networks

### Implementations

0. Torch by Facebook AI Research (FAIR), with **training code in Torch and pre-trained ResNet-18/34/50/101 models for ImageNet**: [blog](http://torch.ch/blog/2016/02/04/resnets.html), [code](https://github.com/facebook/fb.resnet.torch)
0. Torch, CIFAR-10, with ResNet-20 to ResNet-110, training code, and curves: [code](https://github.com/gcr/torch-residual-networks)
0. Lasagne, CIFAR-10, with ResNet-32 and ResNet-56 and training code: [code](https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning)
0. Neon, CIFAR-10, with pre-trained ResNet-32 to ResNet-110 models, training code, and curves: [code](https://github.com/NervanaSystems/ModelZoo/tree/master/ImageClassification/CIFAR10/DeepResNet)
0. Neon, Preactivation layer implementation: [code](https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_msra.py)
0. Torch, MNIST, 100 layers: [blog](https://deepmlblog.wordpress.com/2016/01/05/residual-networks-in-torch-mnist/), [code](https://github.com/arunpatala/residual.mnist)
0. A winning entry in Kaggle's right whale recognition challenge: [blog](http://blog.kaggle.com/2016/02/04/noaa-right-whale-recognition-winners-interview-2nd-place-felix-lau/), [code](https://github.com/felixlaumon/kaggle-right-whale)
0. Neon, Place2 (mini), 40 layers: [blog](http://www.nervanasys.com/using-neon-for-scene-recognition-mini-places2/), [code](https://github.com/hunterlang/mpmz/)
0. Tensorflow with tflearn, with CIFAR-10 and MNIST: [code](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py)
0. Tensorflow with skflow, with MNIST: [code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/skflow/resnet.py)
0. Stochastic dropout in Keras: [code](https://github.com/dblN/stochastic_depth_keras)
0. ResNet in Chainer: [code](https://github.com/yasunorikudo/chainer-ResNet)
0. Stochastic dropout in Chainer: [code](https://github.com/yasunorikudo/chainer-ResDrop)
0. Wide Residual Networks in Keras: [code](https://github.com/asmith26/wide_resnets_keras)
0. ResNet in TensorFlow 0.9+ with pretrained caffe weights: [code](https://github.com/ry/tensorflow-resnet)
0. ResNet in PyTorch: [code](https://github.com/ruotianluo/pytorch-resnet)

In addition, this [code] (https://github.com/ry/tensorflow-resnet) by Ryan Dahl helps to convert the pre-trained models to TensorFlow.


## Highway Networks

**Highway Networks** take inspiration from Long Short Term Memory (LSTM) and allow training of deep, efficient networks (with hundreds of layers) with conventional gradient-based methods

### Papers

- [Recurrent Highway Networks (2016)](https://arxiv.org/abs/1607.03474) [[original code](https://github.com/julian121266/RecurrentHighwayNetworks)], introducing recurrent highway networks, which increases space depth in recurrent networks 
- [Training Very Deep Networks (2015)](http://arxiv.org/abs/1507.06228), introducing highway neural networks

### Implementations

0. Lasagne: [code](https://github.com/Lasagne/Lasagne/blob/highway_example/examples/Highway%20Networks.ipynb)
0. Caffe: [code](https://github.com/flukeskywalker/highway-networks)
0. Torch: [code](https://github.com/yoonkim/lstm-char-cnn/blob/master/model/HighwayMLP.lua)
0. Tensorflow: [blog](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.r2msk226f), [code](https://github.com/fomorians/highway-cnn)
0. PyTorch: [code](https://github.com/c0nn3r/pytorch_highway_networks/blob/master/layers/highway.py)

## Very Deep Learning Theory

**Theories** in very deep learning concentrate on the ideas that very deep networks with skip connections are able to efficiently approximate recurrent computations (similar to the recurrent connections in the visual cortex) or are actually exponential ensembles of shallow networks

### Papers

- [Identity Matters in Deep Learning](https://arxiv.org/abs/1611.04231) considers identity parameterizations from a theoretical perspective and proofs that arbitrarily deep linear residual networks have no spurious local optima 
- [The Shattered Gradients Problem: If resnets are the answer, then what is the question?](https://arxiv.org/abs/1702.08591) argues that gradients of very deep networks resemble white noise (thus are harder to optimize). Resnets are more resistant to shattering (decaying sublinearly)
- [Skip Connections as Effective Symmetry-Breaking](https://arxiv.org/pdf/1701.09175) hypothesizes that ResNets improve performance by breaking symmetries
- [Highway and Residual Networks learn Unrolled Iterative Estimation](https://arxiv.org/abs/1612.07771), argues that instead of learning a new representation at each layer, the layers within a stage rather work as an iterative refinement of the same features.
- [Demystifying ResNet](https://arxiv.org/abs/1611.01186), shows mathematically that 2-shortcuts in ResNets achieves the best results because they have non-degenerate depth-invariant initial condition numbers (in comparison to 1 or 3-shortcuts), making it easy for the optimisation algorithm to escape from the initial point.
- [Wider or Deeper? Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080v1), extends results from Veit et al. and shows that it is actually a linear ensemble of subnetworks. Wide ResNet work well, because current very deep networks are actually over-deepened (hence not trained end-to-end), due to the much shorter effective path length. 
- [Residual Networks are Exponential Ensembles of Relatively Shallow Networks](http://arxiv.org/abs/1605.06431), shows that ResNets behaves just like ensembles of shallow networks in test time. This suggests that in addition to describing neural networks in terms of width and depth, there is a third dimension: multiplicity, the size of the implicit ensemble
- [Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex](http://arxiv.org/abs/1604.03640), shows that ResNets with shared weights work well too although having fewer parameters 
- [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](https://arxiv.org/abs/1504.00941), pre-ResNet Hinton paper that suggested, that the identity matrix could be useful for the initialization of deep networks

