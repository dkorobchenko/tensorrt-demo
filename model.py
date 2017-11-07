'''
Model definition
'''

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

def model(inputs, is_training, **kwargs):
    '''
    VGG-A + BatchNorm
    '''
    net = tf.subtract(inputs, np.reshape((104., 117., 123.), [1, 1, 1, 3]))

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):

        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):

            net = slim.conv2d(net, 64, [3, 3], scope='conv1_1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6', is_training=is_training)
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, 0.5, scope='dropout7', is_training=is_training)
            net = slim.fully_connected(net, 2, activation_fn=None, scope='fc8')

    return net


