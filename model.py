import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_v1
import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
from time import time
from tabulate import tabulate
import logging


class resnet(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes


class regression_resnet(object):
    def __init__(self):
        self.regularize_scale = 1e-5  # L2 regularizer scale

    def _loss_(self, predicts, labels, loss_type, use_regular=True):
        loss_type = loss_type.lower()
        assert loss_type in ['rmse', 'cross_entropy', 'ordinal']
        if loss_type is 'rmse':  # Root-Means-Squared-Error, 均方根误差，标准差
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predicts, labels))))
        elif loss_type is 'cross_entropy':
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predicts)
        elif loss_type is 'ordinal':
            pass
        if use_regular:
            reg = layers.l2_regularizer(self.regularize_scale)
            loss += layers.apply_regularization(reg, tf.trainable_variables())
        return loss

    def _get_resnet(self, resnet_depth=50, version=1):
        assert version in [1, 2]
        assert resnet_depth in [50, 101, 152, 200]
        if version == 1:
            resnet = resnet_v1
        else:
            resnet = resnet_v2
        if resnet_depth == 50:
            return resnet.resnet_v1_50
        elif resnet_depth == 101:
            return resnet.resnet_v1_101
        elif resnet_depth == 152:
            return resnet.resnet_v1_152
        elif resnet_depth == 200:
            return resnet.resnet_v1_200
        else:
            raise NotImplementedError
