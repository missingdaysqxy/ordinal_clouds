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
        self.ckpt_base_name == 'resnet{}_{}_{}_{}_'.format(
            self.config.RESNET_DEPTH, self.config.RESNET_VERSION,
            self.config.LOSS_TYPE, self.config.OPTIMIZER)
        self.sess = tf.InteractiveSession(config=config.SessionConfig)
        self.resnet = self._get_resnet(config.RESNET_DEPTH, config.RESNET_VERSION)
        self._batch_xs = tf.placeholder(tf.float32)
        self.features, self.end_points = resnet(self._batch_xs, num_classes=self.num_classes)
        self.initialized = False

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

    def _get_last_ckpt_folder(self, create_if_none=False):
        """
        Find the folder which stored the last checkpoint under current configuration
        :param create_if_none: Create a new folder if there's no existed checkpoint folder
        :return: a path to the checkpoint folder
        """
        _ckpt_name = self.ckpt_name
        try:
            list = os.listdir(self.ckpt_root_dir)
            list = [s[len(_ckpt_name):] for s in list if s.startswith(_ckpt_name)]
            num = int(max(list))
        except:
            return None
        _ckpt_name += str(num)
        path = os.path.join(self.ckpt_root_dir, _ckpt_name)
        if create_if_none and not os.path.exists(path):
            os.makedirs(path)
        return path

    def _create_new_ckpt_folder(self, makedir=True):
        '''
        Create a new folder to store checkpoints under current configuration
        :param makedir: If 'False', this function only return the path, but not create the folder whether it existed or not
        :return: a path to the checkpoint folder
        '''
        _ckpt_name = self.ckpt_name
        try:
            list = os.listdir(self.ckpt_root_dir)
            list = [s[len(_ckpt_name):] for s in list if s.startswith(_ckpt_name)]
            num = int(max(list)) + 1
        except:
            num = 0
        _ckpt_name += str(num)
        path = os.path.join(self.ckpt_root_dir, _ckpt_name)
        if makedir and not os.path.exists(path):
            os.makedirs(path)
        return path

    def find_last(self):
        """Find the lasted trained model under current configuration"""
        ckpt_dir = self._get_last_ckpt_folder(create_if_none=False)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return ckpt.model_checkpoint_path
        else:
            print('[!]Failed to find the last checkpoints files in %s' % ckpt_dir)
            return None

    def load_weights(self, model_path):
        def get_global_step_init_value(model_dir):
            import re
            # print('-[*] Find last global step value in {}'.format(model_dir))
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                global_step_init_value = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                # print("-[*] The last global step value of {} is {}".format(ckpt_name, global_step_init_value))
            else:
                # Load the model pretrained by TensorFlow official
                global_step_init_value = 0
            return global_step_init_value

        self.global_step_init_value = get_global_step_init_value(model_path)
        if os.path.isdir(model_path):  # saver write_version=tf.train.SaverDef.V2
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(self.sess, os.path.join(model_path, ckpt_name))
                print("[*]Loaded weights from {}".format(model_path))
            else:
                print("[!]Failed to load weights from {}".format(model_path))
                raise RuntimeError
            self.ckpt_dir=model_path
        elif os.path.isfile(model_path):  #  Update saver write_version from tf.train.SaverDef.V1 into V2
            self.ckpt_dir = self._create_new_ckpt_folder(makedir=True)
            pass
        else:
            print('[!]No checkpoint files were found in '+model_path)
            raise FileNotFoundError
        self.initialized = True

    def initialize_weights(self):
        self.global_step_init_value = 0
        # create new folder in ckpt dir
        self.ckpt_dir=self._create_new_ckpt_folder(makedir=True)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        print('[*]Initialized all variables')
        self.initialized = True

    def _init_loss(self, predicts, labels):
        loss_type = self.config.LOSS_TYPE.lower()
        if loss_type is 'cross_entropy':
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predicts)
        if self.config.USE_REGULARIZER:
            reg = layers.l2_regularizer(self.config.REGULARIZE_SCALE)
            loss += layers.apply_regularization(reg, tf.trainable_variables())
        return loss

    def train(self, batch_xs, batch_ys, epochs=1):
        assert self.mode is 'train', 'current mode is %s, not training mode' % self.mode
        assert self.initialized, 'initialize_weights() or load_weights() must be called before call train()'
        logits = tf.squeeze(self.features, name='probability')
        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32, name='prediction')
        # Summaries log confugrations
        loss_t = self._init_loss(logits, batch_ys)
        mAP_t = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), dtype=tf.float32))
        mAP_log = tf.summary.scalar('mAP', mAP_t)
        loss_log = tf.summary.scalar('loss', loss_t)
        summaries = tf.summary.merge([mAP_log, loss_log])
        # global_steps
        global_step = tf.get_variable('global_step', dtype=tf.int32,
                                      initializer=tf.constant_initializer(self.global_step_init_value), trainable=False)
        # variable averages operation
        variable_averages = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        # Exponential decay learning rate and optimizer configurations
        learning_rate = tf.train.exponential_decay(self.config.LEARNING_RATE, global_step, decay_steps=100,
                                                   decay_rate=0.98, staircase=True, name='learning_rate')
        if self.config.OPTIMIZER == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learning_rate)
        elif self.config.OPTIMIZER == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate)
        else:
            raise NotImplementedError
        train_step = optim.minimize(loss_t, global_step=global_step, name=self.config.OPTIMIZER)
        train_op = tf.group(train_step, variable_averages_op)
        # Init and Train
        init_op = tf.group(global_step.initializer, learning_rate.initializer)
        self.sess.run(init_op)

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
            loss = super(predicts, labels)
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
