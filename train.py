# -*- coding: utf-8 -*-
# @Time    : 2018/11/2/002 21:07 下午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : train.py.py
# @Software: PyCharm

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse

from model import resnet, regression_resnet
from model import Config
from dataset import Dataset

TRAIN_DATA_DIR = r'./datasets/separate_relabel2/train'
VAL_DATA_DIR = r'./datasets/separate_relabel2/validation'
CKPT_DIR = r'./checkpoints/'
PRETRAINED_MODEL_PATH = r'./pretrained/resnet101.ckpt'
MODEL_SAVE_DIR = r'./saved_models/'
MODEL_SAVE_NAME = 'model.ckpt'
PRE_MODEL = 'none'



def main(args):
    config = Config()
    train_data = Dataset(TRAIN_DATA_DIR, config)
    batch_count = train_data.batch_count
    train_xs, train_ys, train_org = train_data.get_batch_pipeline()
    net = resnet(mode='train', config=config, checkpoints_root_dir=CKPT_DIR)
    # Load Existed Models
    if PRE_MODEL == 'resnet':
        net.load_weights(PRETRAINED_MODEL_PATH)
    elif PRE_MODEL == 'last':
        net.load_weights(net.find_last())
    else:
        net.initialize_weights()
    # Train Model
    net.train(train_xs, train_ys, batch_count)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    savepath = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
    net.save(savepath)
    return
    # Test Model
    val_data = Dataset(VAL_DATA_DIR)
    val_xs, val_ys, val_org = val_data.get_batch_pipeline()
    net.test(val_xs, val_ys, val_org)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument('--dataset')
    args = parse.parse_args()
    main(args)
