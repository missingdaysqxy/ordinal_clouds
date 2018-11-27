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
from .config import Config


class resnet(object):
    def __init__(self, mode, config, checkpoints_root_dir):
        self.num_classes = config.NUM_CLASSES
        assert mode in ['train', 'evaluate', 'predict'], \
            "mode must choose one from 'train','evaluate' or 'predict'"
        assert isinstance(config, Config)
        assert os.path.isdir(checkpoints_root_dir)
        self.mode = mode
        self.config = config
        self.ckpt_root_dir = checkpoints_root_dir
        self.sess = tf.InteractiveSession(config=config.SessionConfig)
        self.resnet = self._get_resnet(config.RESNET_DEPTH, config.RESNET_VERSION)
        self._batch_xs = tf.placeholder(tf.float32)
        self.features, self.end_points = resnet(self._batch_xs, num_classes=self.num_classes)

    def _get_resnet(self, resnet_depth, version):
        assert version in ['v1', 'v2']
        assert resnet_depth in [50, 101, 152, 200]
        if version == 'v1':
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

    def _get_ckpt_dir(self, ckpt_root_dir, target='last', create=False):
        """
        get or create the checkpoints files path with current configuration
        :param ckpt_root_dir: where to save or load the checkpoints files
        :param target: choose from 'last' to load the lasted trained model,
         or 'new' to create a new folder to save new training model.
         :param create: create new folder if the ckpt path does not exist.
        :return: a path to the checkpoints files
        """
        assert target in ['last', 'new']
        ckpt_name = 'resnet{}_{}_{}_{}_'.format(
            self.config.RESNET_DEPTH, self.config.RESNET_VERSION,
            self.config.LOSS_TYPE, self.config.OPTIMIZER)
        try:
            list = os.listdir(ckpt_root_dir)
            list = [s for s in list if s.startswith(ckpt_name)]
            list = [s[len(ckpt_name):] for s in list]
            num = int(max(list))
        except:
            num = 0
        if target is 'last':
            ckpt_name += str(num)
        else:
            ckpt_name += str(num + 1)
        path = os.path.join(ckpt_root_dir, ckpt_name)
        if create and not os.path.exists(path):
            os.makedirs(path)
        return path

    def _init_loss(self, predicts, labels):
        loss_type = self.config.LOSS_TYPE.lower()
        if loss_type is 'cross_entropy':
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predicts)
        if self.config.USE_REGULARIZER:
            reg = layers.l2_regularizer(self.config.REGULARIZE_SCALE)
            loss += layers.apply_regularization(reg, tf.trainable_variables())
        return loss

    def load_weights(self, model_path):
        if os.path.isdir(model_path):  # saver write_version=tf.train.SaverDef.V2
            pass
        elif os.path.isfile(model_path):  # saver write_version=tf.train.SaverDef.V1
            pass
        else:
            raise FileNotFoundError

    def find_last(self):
        """Find the lasted trained model in current configuration"""
        ckpt_dir = self._get_ckpt_dir(self.ckpt_root_dir, target='last')
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return ckpt.model_checkpoint_path
        else:
            print('Failed to find the last checkpoints files in %s' % ckpt_dir)
            return None

    def initialize_weights(self):
        # ToDO: create new folder in ckpt dir
        pass

    def save(self, save_path, version=2):
        pass


class regression_resnet(resnet):
    def __init__(self, mode, config, checkpoints_root_dir):
        assert mode in ['train', 'evaluate', 'predict'], \
            "mode must choose one from 'train','evaluate' or 'predict'"
        assert isinstance(config, Config)
        assert os.path.isdir(checkpoints_root_dir)
        self.mode = mode
        self.config = config
        self.ckpt_root_dir = checkpoints_root_dir
        self.sess = tf.InteractiveSession(config=config.SessionConfig)
        self.resnet = self._get_resnet(config.RESNET_DEPTH, config.RESNET_VERSION)
        self._batch_xs = tf.placeholder(tf.float32)
        # set the num_class to None for further regression
        self.features, self.end_points = resnet(self._batch_xs, num_classes=None)
        # ToDO : convolution/dense the features into one value presented for regression value

    def _init_loss(self, predicts, labels):
        loss_type = self.config.LOSS_TYPE.lower()
        if loss_type is 'rmse':  # Root-Means-Squared-Error, 均方根误差，标准差
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predicts, labels))))
        elif loss_type is 'cross_entropy':
            loss = super(predicts,labels)
        elif loss_type is 'ordinal':
            # ToDO
            pass
        if self.config.USE_REGULARIZER:
            reg = layers.l2_regularizer(self.config.REGULARIZE_SCALE)
            loss += layers.apply_regularization(reg, tf.trainable_variables())
        return loss
    def train(self, batch_xs, batch_ys, epochs):
        assert self.mode is 'train'
        self.ckpt_dir = self._get_ckpt_dir(self.ckpt_root_dir, target='new', create=True)
        suffix = str(int(time()))
        writer = tf.summary.FileWriter(self.ckpt_dir, self.sess.graph, filename_suffix=suffix)
        logging.basicConfig(filename=os.path.join(self.ckpt_dir, 'train.output-{}.txt'.format(suffix)),
                            level=logging.DEBUG)

        pass

    def evaluate(self, dataset, epochs=1):
        pass

    def predict(self, data):
        pass
