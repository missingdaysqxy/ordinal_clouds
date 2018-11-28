# -*- coding: utf-8 -*-
# @Time    : 2018/11/2/002 21:07 下午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : train.py.py
# @Software: PyCharm

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
from .model import resnet, regression_resnet
from .model import Config

TRAIN_DATA_DIR = r'./datasets/separate_relabel2/train'
VAL_DATA_DIR = r'./datasets/separate_relabel2/validation'
CKPT_DIR = r'./checkpoints/'
PRETRAINED_MODEL_PATH = r'./pretrained/resnet101.ckpt'
MODEL_SAVE_DIR = r'./saved_models/'
MODEL_SAVE_NAME = 'model.ckpt'
TRAIN_EPOCHS = 50
PRE_MODEL = 'none'


class Dataset(object):
    def __init__(self, data_dir, config):
        self.config = config
        self.resize = config.IMG_RESIZE is not None and type(config.IMG_RESIZE) in [list, np.ndarray] and len(
            config.IMG_RESIZE) == 2

        def _parse_function(file_path, label):
            '''Decode image
            :param file_path: shape[]
            :param label: shape[]
            :return a string of image file path, a tensor of image, a label string of image
            '''
            x_img_str = tf.read_file(file_path)  # shape[]
            org_name = os.path.basename(file_path).split('_')[0]
            x_img = tf.image.decode_jpeg(x_img_str, channels=config.IMG_CHANNEL)  # shape[?,?,channels]
            if self.resize:
                x_img = tf.image.resize_images(x_img, size=config.IMG_RESIZE,
                                               method=tf.image.ResizeMethod.BILINEAR)  # shape[img_resize,channels]
            # if adjust:  # 随机亮度、对比度与翻转
            #     x_img = tf.image.random_brightness(x_img, max_delta=0.25)
            #     x_img = tf.image.random_contrast(x_img, lower=0.75, upper=1.5)
            #     # x_img = tf.image.random_hue(x_img, max_delta=0.5)
            #     x_img = tf.image.random_flip_up_down(x_img)
            #     x_img = tf.image.random_flip_left_right(x_img)
            return x_img, label, org_name

        files = []
        labels = []
        cloudcover = [0.0, 0.1, 0.25, 0.75, 1.0]
        for i in range(config.NUM_CLASSES):
            dir = os.path.join(data_dir, config.CLASS_LIST[i])
            if not os.path.exists(dir):
                print('path %s not exist' % dir)
                continue
            fs = os.listdir(dir)
            fs = [os.path.join(dir, item) for item in fs]
            files.extend(fs)
            if config.LOSS_TYPE == 'rmse':
                labels.extend([cloudcover[i]] * len(fs))
            else:
                labels.extend([i] * len(fs))
        count = len(files)
        assert count > config.BATCH_SIZE
        if config.SHUFFLE:
            import random
            idx = list(range(count))
            random.shuffle(idx)
            sfl_files = []
            sfl_labels = []
            for i in idx:
                sfl_files.append(files[i])
                sfl_labels.append(labels[i])
            files = sfl_files
            labels = sfl_labels
        if count % config.BATCH_SIZE > 0:
            count = count - count % config.BATCH_SIZE
            # files = files[:batch_count]
            # labels = labels[:batch_count]
        # Initialize as a tensorflow tensor object
        data = tf.data.Dataset.from_tensor_slices((tf.constant(files, dtype=tf.string, name='file_path'),
                                                   tf.constant(labels, name='label')))
        data = data.map(_parse_function)
        self.data = data.batch(config.BATCH_SIZE).repeat()
        self.batch_count = count // config.BATCH_SIZE

    def get_batch_pipeline(self):
        batch_xs, batch_ys, batch_org = self.data.make_one_shot_iterator().get_next()
        return batch_xs,batch_ys,batch_org


def main():
    config = Config()
    train_data=Dataset(TRAIN_DATA_DIR, config, repeat=TRAIN_EPOCHS)
    train_xs,train_ys,train_org = train_data.get_batch_pipeline()
    net = resnet(mode='train', config=config, checkpoints_root_dir=CKPT_DIR)
    # Load Existed Models
    if PRE_MODEL == 'resnet':
        net.load_weights(PRETRAINED_MODEL_PATH)
    elif PRE_MODEL == 'last':
        net.load_weights(net.find_last())
    else:
        net.initialize_weights()
    # Train Model
    net.train(train_xs, train_ys, train_data.batch_count)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    savepath = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
    net.save(savepath)
    # Test Model
    val_data=Dataset(VAL_DATA_DIR)
    val_xs,val_ys,val_org = val_data.get_batch_pipeline()
    net.test(val_xs,val_ys,val_org)
