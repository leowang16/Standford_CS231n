from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a 
    Batch Normalization, then by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a1, fc_cache = affine_forward(x, w, b)
    a2, bn_cache = batchnorm_forward(a1, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a2)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_relu_dropout_forward(x, w, b, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a 
    ReLU, then by a dropout

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a1, fc_cache = affine_forward(x, w, b)
    a2, relu_cache = relu_forward(a1)
    out, dropout_cache = dropout_forward(a2, dropout_param)
    cache = (fc_cache, relu_cache, dropout_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu-dropout convenience layer
    """
    fc_cache, relu_cache, dropout_cache = cache
    da1 = dropout_backward(dout, dropout_cache)
    da2 = relu_backward(da1, relu_cache)
    dx, dw, db = affine_backward(da2, fc_cache)    
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
    """
    Convenience layer that performs a convolution, a spatial batch 
    normalization, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a1, conv_cache = conv_forward_fast(x, w, b, conv_param)
    a2, norm_cache = spatial_batchnorm_forward(a1, gamma, beta, bn_param)
    a3, relu_cache = relu_forward(a2)
    out, pool_cache = max_pool_forward_fast(a3, pool_param)
    cache = (conv_cache, norm_cache, relu_cache, pool_cache)
    return out, cache


def conv_norm_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-norm-relu-pool convenience layer
    """
    conv_cache, norm_cache, relu_cache, pool_cache = cache
    da1 = max_pool_backward_fast(dout, pool_cache)
    da2 = relu_backward(da1, relu_cache)
    da3, dgamma, dbeta = spatial_batchnorm_backward(da2, norm_cache)
    dx, dw, db = conv_backward_fast(da3, conv_cache)
    return dx, dw, db, dgamma, dbeta
