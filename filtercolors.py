# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 19:21
# @Author  : qxliu
# @Email   : qixuan.lqx@qq.com
# @File    : filtercolors.py
# @Software: PyCharm

import tensorflow as tf
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import logging
import os
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 可以使用的GPU
INPUT_DIR = './datasets/separate_relabel/'
OUTPUT_DIR = './datasets/'
OUTPUT_CHANNEL = 3
IMG_RESIZE = [512, 512]
CUT_NUM = [4, 2]

def countwhite(imgpath):
    img=Image.open(imgpath)
    img_arr=img.load()
    width=img.size[0]
    height=img.size[1]
    for x in range(width):
        for y in range(height):
            r,g,b=img_arr[x,y]




def main():
    for root,dir,name in os.walk(INPUT_DIR):
        if os.path.split(name) in ['jpg','jpeg','png']:
            cw=countwhite(os.path.join(root,name))
            if cw == 0:
                pass
            elif cw <=0.1:
                pass
            elif cw <=0.25:
                pass
            elif cw<=0.75:
                pass
            else:
                pass


if __name__ == '__main__':
    main()
