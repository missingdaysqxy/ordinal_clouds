# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 11:24
# @Author  : qxliu
# @Email   : qixuan.lqx@qq.com
# @File    : dataset.py
# @Software: PyCharm

import os
import tensorflow as tf
import numpy as np


class Dataset(object):
    def __init__(self, data_dir, config):
        self.config = config

        def _parse_function(file_path, label, org_name):
            '''Decode image
            :param file_path: shape[]
            :param label: shape[]
            :return a string of image file path, a tensor of image, a label string of image
            '''
            x_img_str = tf.read_file(file_path)  # shape[]

            x_img = tf.image.decode_jpeg(x_img_str, channels=config.IMG_CHANNEL)  # shape[?,?,channels]
            # if self.resize:
            x_img = tf.image.resize_images(x_img, size=config.IMG_SIZE,
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
        org_names = []
        cloudcover = [0.0, 0.1, 0.25, 0.75, 1.0]
        for i in range(config.NUM_CLASSES):
            dir = os.path.join(data_dir, config.CLASS_LIST[i])
            if not os.path.exists(dir):
                print('path %s not exist' % dir)
                continue
            fs = os.listdir(dir)
            org = [os.path.basename(f).split('_')[0] for f in fs]
            fs = [os.path.join(dir, item) for item in fs]
            files.extend(fs)
            org_names.extend(org)
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
            sfl_orgs = []
            sfl_labels = []
            for i in idx:
                sfl_files.append(files[i])
                sfl_orgs.append(org_names[i])
                sfl_labels.append(labels[i])
            files = sfl_files
            org_names = sfl_orgs
            labels = sfl_labels
        if count % config.BATCH_SIZE > 0:
            count = count - count % config.BATCH_SIZE
            # files = files[:batch_count]
            # labels = labels[:batch_count]
        # Initialize as a tensorflow tensor object
        data = tf.data.Dataset.from_tensor_slices((tf.constant(files, dtype=tf.string, name='file_path'),
                                                   tf.constant(labels, name='label'),
                                                   tf.constant(org_names, name='org_name')))
        data = data.map(_parse_function)
        self.data = data.batch(config.BATCH_SIZE).repeat()
        self.batch_count = count // config.BATCH_SIZE

    def get_batch_pipeline(self):
        batch_xs, batch_ys, batch_org = self.data.make_one_shot_iterator().get_next()
        return batch_xs, batch_ys, batch_org
