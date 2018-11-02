# -*- coding: utf-8 -*-
# @Time    : 2018/11/2/002 21:07 下午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : train.py.py
# @Software: PyCharm

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from .model import regression_resnet
from .model import Config

TRAIN_DATA_DIR=r'./datasets/separate_relabel2/train'
VAL_DATA_DIR=r'./datasets/separate_relabel2/validation'
CKPT_DIR=r'./checkpoints/'
PRETRAINED_MODEL_PATH=r'./pretrained/resnet101.ckpt'
MODEL_SAVE_DIR=r'./saved_models/'
MODEL_SAVE_NAME='model.ckpt'
TRAIN_EPOCHS=50
PRE_MODEL='none'
def InitDataset(data_dir):
    # ToDO
    return 0

def main():
    config=Config()
    dataset_train=InitDataset(TRAIN_DATA_DIR)
    net=regression_resnet(mode='train', config=config, checkpoints_root_dir=CKPT_DIR)
    # Load Existed Models
    if PRE_MODEL=='resnet':
        net.load_weights(PRETRAINED_MODEL_PATH)
    elif PRE_MODEL=='last':
        net.load_weights(net.find_last())
    else:
        net.initialize_weights()
    # Train Model
    net.train(dataset_train,epochs=TRAIN_EPOCHS)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    savepath=os.path.join(MODEL_SAVE_DIR,MODEL_SAVE_NAME)
    net.save(savepath)
    # Test Model
    dataset_val = InitDataset(VAL_DATA_DIR)
    net.test(dataset_val)