# -*- coding: utf-8 -*-
# @Time    : 2018/10/14 1:10
# @Author  : qxliu
# @Email   : qixuan.lqx@qq.com
# @File    : test_tfimageadjust.py
# @Software: PyCharm

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES']='2'

input_dir=r'D:\qxliu\ordinal_clouds\datasets\separate_relabel\train\E'
file_list=['41_0','39_5','11_6','489_5']

with tf.Session() as sess:
    pltidx=1
    for i in range(len(file_list)):
        path=os.path.join(input_dir,file_list[i]+'.jpg')

        x_img_str = tf.read_file(path)
        x_img = tf.image.decode_jpeg(x_img_str, channels=3)
        x_img_b = tf.image.adjust_brightness(x_img, -0.25)
        x_img_c = tf.image.adjust_contrast(x_img, 0.5)
        x_img_h = tf.image.adjust_hue(x_img, -0.25)
        img,b,c,h = sess.run([x_img,x_img_b,x_img_c,x_img_h])

        plt.subplot(len(file_list),4, pltidx)
        plt.imshow(img)
        plt.title(file_list[i])
        pltidx += 1
        plt.subplot(len(file_list), 4, pltidx)
        plt.imshow(b)
        plt.title('brightness-0.25')
        pltidx+=1
        plt.subplot(len(file_list), 4, pltidx)
        plt.imshow(c)
        plt.title('contrast0.5')
        pltidx += 1
        plt.subplot(len(file_list), 4, pltidx)
        plt.imshow(h)
        plt.title('hue-0.25')
        pltidx += 1
    plt.show()