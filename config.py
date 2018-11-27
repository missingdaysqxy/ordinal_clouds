# -*- coding: utf-8 -*-
# @Time    : 2018/11/2/002 21:09 下午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : config.py
# @Software: PyCharm

import  tensorflow as tf

class Config(object):

    CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'nodata']

    NUM_CLASSES = len(CLASS_LIST)

    # 1 for grayscale or 3 for rgb
    IMG_CHANNEL = 3

    # 'None' or a list as [height, width]
    IMG_RESIZE = None

    BATCH_SIZE = 256

    SHUFFLE = True

    # loss function, choose from 'rmse', 'cross_entropy', 'ordinal'
    LOSS_TYPE='rmse'

    # choose from 'SGD','ADAM'
    OPTIMIZER='SGD'

    # initial value of learning rate
    LEARNING_RATE=8e-3

    # whether use the L2 regularizer or not
    USE_REGULARIZER=True

    # L2 regularizer scale
    REGULARIZE_SCALE=1e-5

    # choose from resnet 50, 101, 152, 200
    RESNET_DEPTH=50

    # 'v1' was originally proposed in: Deep Residual Learning for Image Recognition. arXiv:1512.03385
    # 'v2' was introduced by: Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
    RESNET_VERSION='v1'

    # allow automatically changing from GPU to CPU when necessary
    TF_ALLOW_CPU=True
    # allow auto growth of the GPU memory usage
    TF_GPU_GROWTH=True
    # when TF_GPU_GROWTH = FALSE, this option set the GPU memory fraction
    TF_GPU_FRACTION=0.8

    def __init__(self):
        """Set values of computed attributes."""
        assert self.LOSS_TYPE in ['rmse', 'cross_entropy', 'ordinal']
        assert self.RESNET_VERSION in ['v1','v2']
        assert self.RESNET_DEPTH in [50, 101, 152, 200]
        self.OPTIMIZER=self.OPTIMIZER.upper()
        assert self.OPTIMIZER in ['SGD','ADAM']
        self.SessionConfig = tf.ConfigProto()
        self.SessionConfig.allow_soft_placement = self.TF_ALLOW_CPU
        self.SessionConfig.gpu_options.allow_growth =self. TF_GPU_GROWTH
        if not self.TF_GPU_GROWTH:
            self.SessionConfig.gpu_options.per_process_gpu_memory_fraction  =self. TF_GPU_FRACTION

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")