import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        #######################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #######################################################################
        self.params['W1'] = np.random.normal(
            0, weight_scale, [num_filters, input_dim[0], filter_size, filter_size])
        self.params['b1'] = np.zeros(num_filters)

        W2_row_size = num_filters * input_dim[1] / 2 * input_dim[2] / 2
        self.params['W2'] = np.random.normal(
            0, weight_scale, [W2_row_size, hidden_dim])
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(
            0, weight_scale, [hidden_dim, num_classes])
        self.params['b3'] = np.zeros(num_classes)
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################
        z1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        z2, cache2 = affine_relu_forward(z1, W2, b2)
        scores, cache3 = affine_forward(z2, W3, b3)
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #######################################################################
        loss, da3 = softmax_loss(scores, y)
        loss += 0.5 * self.reg * \
            (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        dz2, dW3, db3 = affine_backward(da3, cache3)
        dz1, dW2, db2 = affine_relu_backward(dz2, cache2)
        dz0, dW1, db1 = conv_relu_pool_backward(dz1, cache1)

        grads["W1"] = dW1 + self.reg * W1
        grads["b1"] = db1
        grads["W2"] = dW2 + self.reg * W2
        grads["b2"] = db2
        grads["W3"] = dW3 + self.reg * W3
        grads["b3"] = db3
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads


class cnn1(object):
    """
    A Convolutional neural network with an arbitrary number of Convolutional
    layers and affine (fully-connected) layers, then with a softmax layer.
    The architecture will be:

    {conv - relu - 2x2 max pool} x N - {affine - Relu} x M - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated several times.

    Learnable parameters are stored in the self.params dictionary and will be 
    learned using the Solver class.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=7,
                 hidden_dim=[100], num_classes=10, weight_scale=1e-3, reg=0.0,
                 dropout=0, use_batchnorm=False, dtype=np.float32, seed=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.bn_params = {}
        self.params = {}
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.reg = reg
        self.dtype = dtype

        # pass conv_param to the forward pass for the convolutional layer
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        #######################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #######################################################################
        N_conv = len(num_filters)
        N_fc = len(hidden_dim)

        F1 = filter_size
        # pass conv_param to the forward pass for the convolutional layer
        S1 = self.conv_param['stride']
        P = self.conv_param['pad']

        # pass pool_param to the forward pass for the max-pooling layer
        FH = self.pool_param['pool_height']
        FW = self.pool_param['pool_width']
        S3 = self.pool_param['stride']

        # Conv Layers
        D1, W1, H1 = input_dim
        for i in xrange(N_conv):
            self.params['Wconv_%d' % (i + 1)] = np.random.normal(
                0, weight_scale, [num_filters[i], D1, F1, F1])
            self.params['bconv_%d' % (i + 1)] = np.zeros(num_filters[i])
            if self.use_batchnorm:
                self.params["gamma_conv%d" %
                    (i + 1)] = np.ones([1, num_filters[i]])
                self.params["beta_conv%d" %
                    (i + 1)] = np.zeros([1, num_filters[i]])
                self.bn_params[i + 1] = {'mode': 'train',
                'running_mean': np.zeros(num_filters[i]),
                'running_var': np.ones(num_filters[i])}
            W2 = (W1 - F1 + 2 * P) / S1 + 1
            H2 = (H1 - F1 + 2 * P) / S1 + 1
            D2 = num_filters[i]
            W3 = (W2 - FW) / S3 + 1
            H3 = (H2 - FH) / S3 + 1
            D3 = D2
            # Update input dimension of the next Conv layer
            D1, W1, H1 = D3, W3, H3

        # Full-connected layers
        # Use Xavier initialization
        hidden_dim0 = D1 * W1 * H1
        for i in xrange(N_fc):
            self.params['Wfc_%d' % (i + 1)] = np.random.normal(
                0, weight_scale, [hidden_dim0, hidden_dim[i]])
            self.params['Wfc_%d' % (i + 1)] /= np.sqrt(hidden_dim0)
            self.params['bfc_%d' % (i + 1)] = np.zeros(hidden_dim[i])
            if self.use_batchnorm:
                self.params["gamma_fc%d" %
                    (i + 1)] = np.ones(hidden_dim[i])
                self.params["beta_fc%d" %
                    (i + 1)] = np.zeros(hidden_dim[i])
                self.bn_params[i + N_conv + 1] = {'mode': 'train',
                    'running_mean': np.zeros(hidden_dim[i]),
                    'running_var': np.ones(hidden_dim[i])}
            hidden_dim0 = hidden_dim[i]

        # Last layer
        # Use Xavier initialization
        self.params['WL'] = np.random.normal(
            0, weight_scale, [hidden_dim[-1], num_classes])
        self.params['WL'] /= np.sqrt(hidden_dim[-1])
        self.params['bL'] = np.zeros(num_classes)
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convolutional network.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode

        N_conv = len(self.num_filters)
        N_fc = len(self.hidden_dim)

        scores = None

        cache_conv = {}
        z_conv = X
        for i in xrange(N_conv):
            if self.use_batchnorm:
                z_conv, cache_conv[i + 1] = conv_norm_relu_pool_forward(z_conv,
                    self.params['Wconv_%d' % (i + 1)], self.params['bconv_%d' % (i + 1)], 
                    self.conv_param, self.pool_param, self.params['gamma_conv%d' % (i + 1)],
                    self.params['beta_conv%d' % (i + 1)], self.bn_params[i + 1])
            else:
                z_conv, cache_conv[i + 1] = conv_relu_pool_forward(z_conv,
                    self.params['Wconv_%d' % (i + 1)], self.params['bconv_%d' % (i + 1)],
                    self.conv_param, self.pool_param)

        cache_fc = {}
        z_fc = z_conv
        for i in xrange(N_fc):
            if self.use_batchnorm:
                z_fc, cache_fc[i + 1] = affine_batchnorm_relu_forward(z_fc,
                    self.params['Wfc_%d' % (i + 1)], self.params['bfc_%d' % (i + 1)], 
                    self.params['gamma_fc%d' % (i + 1)], self.params['beta_fc%d' % (i + 1)],
                    self.bn_params[i + N_conv + 1])
            elif self.use_dropout:
                z_fc, cache_fc[i + 1] = affine_relu_dropout_forward(z_fc, 
                    self.params['Wfc_%d' % (i + 1)], self.params['bfc_%d' % (i + 1)], 
                    self.dropout_param)
            else:
                z_fc, cache_fc[i + 1] = affine_relu_forward(z_fc,
                    self.params['Wfc_%d' % (i + 1)], self.params['bfc_%d' % (i + 1)])

        scores, cache_L = affine_forward(z_fc, self.params['WL'], self.params['bL'])

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, da_L = softmax_loss(scores, y)

        W_sum = 0
        for i in xrange(N_conv):
            W_sum += np.sum(self.params['Wconv_%d' % (i + 1)] ** 2)
        for i in xrange(N_fc):
            W_sum += np.sum(self.params['Wfc_%d' % (i + 1)] ** 2)
        loss += 0.5 * self.reg * W_sum

        # Backprop
        dz_fc, dWL, dbL = affine_backward(da_L, cache_L)
        grads['WL'] = dWL + self.reg * self.params['WL']
        grads['bL'] = dbL

        for i in xrange(N_fc, 0, -1):
            if self.use_batchnorm:
                dz_fc, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dz_fc, 
                    cache_fc[i])
                grads["gamma_fc%d" % i] = dgamma
                grads["beta_fc%d" % i] = dbeta
            elif self.use_dropout:
                dz_fc, dW, db = affine_relu_dropout_backward(dz_fc, cache_fc[i])
            else:
                dz_fc, dW, db = affine_relu_backward(dz_fc, cache_fc[i])
            grads["Wfc_%d" % i] = dW + self.reg * self.params['Wfc_%d' % i]
            grads["bfc_%d" % i] = db

        dz_conv = dz_fc
        for i in xrange(N_conv, 0, -1):
            if self.use_batchnorm:
                dz_conv, dW, db, dgamma, dbeta = conv_norm_relu_pool_backward(dz_conv,
                    cache_conv[i])
                grads["gamma_conv%d" % i] = dgamma
                grads["beta_conv%d" % i] = dbeta
            else:
                dz_conv, dW, db = conv_relu_pool_backward(dz_conv, cache_conv[i])
            grads["Wconv_%d" % i] = dW + self.reg * self.params['Wconv_%d' % i]
            grads["bconv_%d" % i] = db
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################
        return loss, grads
