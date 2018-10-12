# -*- coding: utf-8 -*-
# @Time    : 2018/10/9 22:31
# @Author  : qxliu
# @Email   : qixuan.lqx@qq.com
# @File    : makedataset.py
# @Software: PyCharm

import tensorflow as tf
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 可以使用的GPU
INPUT_DIR = './datasets/mode_2004/'
OUTPUT_DIR = './datasets/separate_relabel/'
OUTPUT_CHANNEL = 3
IMG_RESIZE = [512, 512]
CUT_NUM = [4, 2]

CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'nodata','may_nodata','may_abcde']


def mk_childfolders(parent_dir, child_list=[]):
    for dir in child_list:
        path = os.path.join(parent_dir, dir)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


def _cut_images(image_to_cut, img_resize=[64, 64], cut_num=[2, 2], cut_order='horizontal'):
    '''
    :param img: image string from tf.image.decode
    :param img_resize: size to resize the orgin image, [height, width]
    :param cut_num: numbers to cut in horizontal and vertical, [vertical, horizontal]
    :param cut_order: the output squeence for cutted images, 'horizontal' first or 'vertical' first
    :param channel:  output images channel, 1 for grayscale or 3 for rgb
    :return: s list of small images in type of uint8 tensor
    '''
    assert type(img_resize) in [list, np.ndarray] and len(img_resize) == 2
    assert type(cut_num) in [list, np.ndarray] and len(cut_num) == 2
    assert img_resize[0] % cut_num[0] == 0 and img_resize[1] % cut_num[1] == 0
    assert cut_order in ['horizontal', 'vertical']
    x_img_resized = tf.image.resize_images(image_to_cut, size=img_resize,
                                           method=tf.image.ResizeMethod.BILINEAR)  # shape[512,512,?]
    h = img_resize[0] // cut_num[0]  # height of small img
    w = img_resize[1] // cut_num[1]  # width of small img
    if cut_order == 'horizontal':
        off_hs = list((np.arange(cut_num[0]) * h).repeat(cut_num[1]))
        off_ws = list(np.arange(cut_num[1]) * w) * cut_num[0]
    else:
        off_ws = list((np.arange(cut_num[1]) * h).repeat(cut_num[0]))
        off_hs = list(np.arange(cut_num[0]) * w) * cut_num[1]
    x_img_cuts = [tf.image.crop_to_bounding_box(x_img_resized, hs, ws, h, w) \
                  for hs, ws in zip(off_hs, off_ws)]  # [shape[128,256,?] * batch_size]

    x_img_cuts = [tf.cast(x, tf.uint8) for x in x_img_cuts]
    return x_img_cuts


def init_reader(input_dir, is_train, start_index=None, max_count=None, channels=3):
    assert channels in [1, 3]

    def _parse_function(files, labels):
        '''Decode images and devide into 8 small images
        :param files: shape[]
        :param labels: shape[8]
        '''
        batch_size = CUT_NUM[0] * CUT_NUM[1]
        x_img_str = tf.read_file(files)  # shape[]
        x_img_decoded = tf.image.decode_jpeg(x_img_str, channels=channels)  # shape[?,?,channels]
        batch_xs = _cut_images(x_img_decoded, IMG_RESIZE, CUT_NUM, 'horizontal')
        batch_ys = tf.reshape(tf.split(labels, batch_size, axis=0), [-1, 1], name='batch_ys')  # shape[batch_size,1]
        return files, batch_xs, batch_ys

    # Processing the image filenames
    fs = os.listdir(input_dir)
    csv_name = os.path.join(input_dir, [it for it in fs if '.csv' in it][0])

    frame = pd.read_csv(csv_name)

    # Add one more column named "Train" to split the training set and validation set
    if is_train:
        frame = frame.loc[frame['Train'] == 'T']
        if isinstance(start_index, int) and start_index > 0:
            frame = frame[start_index:]
        if isinstance(max_count, int) and max_count > 0:
            frame = frame[:max_count]
        print(' [*] {} images initialized as training data'.format(frame['num_id'].count()))
    else:
        frame = frame.loc[frame['Train'] == 'F']
        if isinstance(start_index, int) and start_index > 0:
            frame = frame[start_index:]
        if isinstance(max_count, int) and max_count > 0:
            frame = frame[:max_count]
        print(' [*] {} images initialized as validation data'.format(frame['num_id'].count()))
    count = frame['num_id'].count()

    num_idx = frame['num_id'].values.astype(str).tolist()
    t_names = [item + '.jpg' for item in num_idx]
    file_names = [os.path.join(input_dir, item) for item in t_names]
    labels = frame['Cloud_Cover'].values.tolist()
    t_labels = [list('F'.join(item.split('*'))) for item in labels]
    for it in range(len(t_labels)):
        t_labels[it] = list(map(lambda x: ord(x) - ord('A'), t_labels[it]))
    # Initialize as a tensorflow tensor object
    data = tf.data.Dataset.from_tensor_slices((tf.constant(file_names, name='file_names'),
                                               tf.constant(t_labels)))
    data = data.map(_parse_function)
    return data, count


def init_dataset(tag='train', start_index=None, max_count=None, datatype='jpg'):
    assert tag in ['train', 'validation']
    assert datatype in ['jpg', 'jpeg', 'png', 'tfrecord', 'json', 'h5']
    # ToDO: arrange more datatype
    _output_dir = os.path.join(OUTPUT_DIR, tag)
    mk_childfolders(_output_dir, child_list=CLASS_LIST)
    reader, count = init_reader(INPUT_DIR, tag == 'train', start_index, max_count, channels=OUTPUT_CHANNEL)
    batch_path, batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
    # param batch_path: shape []
    # param batch_xs: shape [batch_size, 128, 256, 3] type tf.uint8
    # param batch_ys: shape [batch_size, 1] type tf.int32
    xs = [tf.squeeze(x, axis=0) for x in
          tf.split(batch_xs, batch_xs.shape[0], axis=0)]  # a list of single images, [shape[1] * batch_size]
    ys = [tf.squeeze(y, axis=0) for y in
          tf.split(batch_ys, batch_ys.shape[0], axis=0)]  # a list of single label, [shape[1] * batch_size]
    logging.basicConfig(filename=os.path.join(OUTPUT_DIR, 'log.txt'), level=logging.DEBUG)
    extname='.'+datatype
    with tf.Session() as sess:
        perc = count / 100
        perc = 1 if perc < 1 else int(perc)
        step = 0
        while True:
            try:
                org_path, imgs, labels = sess.run([batch_path, xs, ys])
                org_name = os.path.basename(org_path.decode()).split('.')[0]
                for i in range(len(imgs)):
                    if datatype in ['jpg', 'jpeg', 'png']:
                        new_name = CLASS_LIST[labels[i][0]] + '/' + org_name + '_' + str(i) + extname
                        if imgs[i].sum()==0: # 全是0
                            if labels[i][0] != 5: # 原label不是nodata
                                logging.error('{} is nodata, not a-e'.format(new_name))
                            new_name = CLASS_LIST[5] + '/' + org_name + '_' + str(i) + extname
                        else: # 不全是0
                            if 0 in [x.sum() for x in imgs[i]] and labels[i][0]!=5: # 有一行是0，可能是nodata
                                new_name = 'may_nodata' + '/' + org_name + '_' + str(i) + extname
                            elif labels[i][0]==5: # 没有一行0且原label是nodata
                                new_name = 'may_abcde' + '/' + org_name + '_' + str(i) + extname
                        save_path = os.path.join(_output_dir, new_name)
                        im = Image.fromarray(imgs[i])
                        im.save(save_path)
                    elif datatype == 'tfrecord':
                        pass
                    elif datatype == 'json':
                        pass
                    elif datatype == 'h5':
                        pass
                if int(org_name) % perc == 0:
                    print('progress: {}/{}'.format(step, count))
                step += 1
            except tf.errors.OutOfRangeError:
                print('Finish!')
                break
            except Exception as e:
                print('an error accrue when open file %s' % org_path.decode())
                print(e)
                pass


def main():
    begintime = datetime.now()
    print('Begin to initialize training dataset...')
    init_dataset('train', 0, -1)
    print('Begin to initialize validation dataset...')
    init_dataset('validation', 0, -1)
    endtime = datetime.now()
    print('All dataset initialized!  Span Time:%s' % (endtime - begintime))


if __name__ == '__main__':
    main()
