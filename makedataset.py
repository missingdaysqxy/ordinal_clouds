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
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 可以使用的GPU
INPUT_DIR = './datasets/separate_relabel/'
OUTPUT_DIR = './datasets/'
OUTPUT_CHANNEL = 3
IMG_RESIZE = [512, 512]
CUT_NUM = [4, 2]

CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'nodata']
CLASS_LIST_MAY = ['may_nodata', 'may_abcde', 'may_A', 'may_B', 'may_C', 'may_D', 'may_E']


def processBar(num, total, msg='', length=30):
    rate = num / total
    rate_num = int(rate * 100)
    clth = int(rate * length)
    if len(msg) > 0:
        msg += ':'
    if rate_num == 100:
        r = '\r%s[%s%d%%]\n' % (msg, '*' * length, rate_num,)
    else:
        r = '\r%s[%s%s%d%%]' % (msg, '*' * clth, '-' * (length - clth), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush


def mk_childfolders(parent_dir, child_list=[]):
    os.makedirs(parent_dir, exist_ok=True)
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


def init_csv_reader(input_dir, is_train, start_index=None, max_count=None, channels=3):
    '''read records from .csv file and separate original images into 8 small images'''
    assert channels in [1, 3]

    def _parse_function(file_paths, labels):
        '''Decode images and devide into 8 small images
        :param file_paths: shape[]
        :param labels: shape[8]
        '''
        batch_size = CUT_NUM[0] * CUT_NUM[1]
        x_img_str = tf.read_file(file_paths)  # shape[]
        x_img_decoded = tf.image.decode_jpeg(x_img_str, channels=channels)  # shape[?,?,channels]
        batch_xs = _cut_images(x_img_decoded, IMG_RESIZE, CUT_NUM, 'horizontal')
        batch_ys = tf.reshape(tf.split(labels, batch_size, axis=0), [-1, 1], name='batch_ys')  # shape[batch_size,1]
        return file_paths, batch_xs, batch_ys

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


def init_separate_dataset(tag='train', start_index=None, max_count=None, datatype='jpg'):
    assert tag in ['train', 'validation']
    assert datatype in ['jpg', 'jpeg', 'png']

    _output_dir = os.path.join(OUTPUT_DIR, tag)
    mk_childfolders(_output_dir, child_list=CLASS_LIST + CLASS_LIST_MAY)
    reader, count = init_csv_reader(INPUT_DIR, tag == 'train', start_index, max_count, channels=OUTPUT_CHANNEL)
    batch_path, batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
    # param batch_path: shape []
    # param batch_xs: shape [batch_size, 128, 256, channels] type tf.uint8
    # param batch_ys: shape [batch_size, 1] type tf.int32
    xs = [tf.squeeze(x, axis=0) for x in
          tf.split(batch_xs, batch_xs.shape[0], axis=0)]  # a list of single images, [shape[1] * batch_size]
    ys = [tf.squeeze(y, axis=0) for y in
          tf.split(batch_ys, batch_ys.shape[0], axis=0)]  # a list of single label, [shape[1] * batch_size]
    logging.basicConfig(filename=os.path.join(OUTPUT_DIR, 'log.txt'), level=logging.DEBUG)
    extname = '.' + datatype
    with tf.Session() as sess:
        perc = count / 100
        perc = 1 if perc < 1 else int(perc)
        step = 0
        while True:
            try:
                org_path, imgs, labels = sess.run([batch_path, xs, ys])
                org_name = os.path.basename(org_path.decode()).split('.')[0]
                for i in range(len(imgs)):
                    new_name = CLASS_LIST[labels[i][0]] + '/' + org_name + '_' + str(i) + extname
                    if imgs[i].sum() == 0:  # 全是0
                        if labels[i][0] != 5:  # 原label不是nodata
                            logging.error('{} is nodata, not a-e'.format(new_name))
                        new_name = CLASS_LIST[5] + '/' + org_name + '_' + str(i) + extname
                    else:  # 不全是0
                        if 0 in [x.sum() for x in imgs[i]] and labels[i][0] != 5:  # 有一行是0，可能是nodata
                            new_name = 'may_nodata' + '/' + org_name + '_' + str(i) + extname
                        elif labels[i][0] == 5:  # 没有一行0且原label是nodata
                            new_name = 'may_abcde' + '/' + org_name + '_' + str(i) + extname
                    save_path = os.path.join(_output_dir, new_name)
                    im = Image.fromarray(imgs[i])
                    im.save(save_path)
                if int(org_name) % perc == 0:
                    # print('progress: {}/{}'.format(step, count))
                    processBar(step, count, msg="Initizing " + _output_dir)
                step += 1
            except tf.errors.OutOfRangeError:
                processBar(step, count, msg="Initizing " + _output_dir)
                print('Finish!')
                break
            except Exception as e:
                print('an error accrue when open file %s' % org_path.decode())
                print(e)
                pass


def init_img_reader(input_dir, class_list, img_resize=None, channels=3, shuffle=False):
    assert channels in [1, 3]
    resize = img_resize is not None and type(img_resize) in [list, np.ndarray] and len(img_resize) == 2

    def _parse_function(file_path, label):
        '''Decode image
        :param file_path: shape[]
        :param label: shape[]
        :return a string of image file path, a tensor of image, a label string of image
        '''
        x_img_str = tf.read_file(file_path)  # shape[]
        x_img = tf.image.decode_jpeg(x_img_str, channels=channels)  # shape[?,?,channels]
        if resize:
            x_img = tf.image.resize_images(x_img, size=img_resize,
                                           method=tf.image.ResizeMethod.BILINEAR)  # shape[img_resize,channels]
        if shuffle:  # 随机亮度对比度色相翻转
            # ToDO: all images do with these
            x_img = tf.image.random_brightness(x_img, max_delta=0.25)
            x_img = tf.image.random_contrast(x_img, lower=0.75, upper=1.5)
            # x_img = tf.image.random_hue(x_img, max_delta=0.5)
            x_img = tf.image.random_flip_up_down(x_img)
            x_img = tf.image.random_flip_left_right(x_img)
        return file_path, x_img, label

    files = []
    labels = []
    for cls in class_list:
        dir = os.path.join(input_dir, cls)
        if not os.path.exists(dir):
            print('path %s not exist' % dir)
            continue
        fs = os.listdir(dir)
        fs = [os.path.join(dir, item) for item in fs]
        files.extend(fs)
        labels.extend([cls] * len(fs))
    count = len(files)
    if shuffle:
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
    # Initialize as a tensorflow tensor object
    data = tf.data.Dataset.from_tensor_slices((tf.constant(files, dtype=tf.string, name='file_path'),
                                               tf.constant(labels, name='label')))
    data = data.map(_parse_function)
    # if shuffle:
    #     data = data.shuffle(count)
    return data, count


def tfrecord_reader(filepaths, batch_size=24, num_epochs=1):
    # ToDO: try this
    assert batch_size >= 0 and type(batch_size) is int
    reader = tf.TFRecordReader()
    if type(filepaths) is not list:
        filepaths = [filepaths]
    fqueue = tf.train.string_input_producer(filepaths, num_epochs=num_epochs)  # 此处不指定epochs就会一直持续下去
    _, serialized_example = reader.read(fqueue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'name': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
        })
    name = tf.decode_raw(features['name'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)
    image = tf.decode_raw(features['image'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channel = tf.cast(features['channel'], tf.int32)
    image_shape = tf.stack([height, width, channel])
    image = tf.reshape(image, image_shape)
    # set_shape是为了固定维度，方便后面使用tf.train.batch
    label.set_shape([1])
    name.set_shape([1])
    image.set_shape([32, 32, 3])
    # num_thread可以选择用几个线程同时读取example queue，
    # min_after_dequeue表示读取一次之后队列至少需要剩下的样例数目，capacity表示队列的容量
    names, images, labels = tf.train.batch([name, image, label], batch_size=batch_size,
                                                  capacity=512 + 4 * batch_size)
    return names, images, labels


def check_tfrecord():
    path = r'D:\qxliu\ordinal_clouds\datasets\clouds.shuffle.train.tfrecord'
    name_op, label_op, image_op = tfrecord_reader(path)
    import cv2 as cv
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while True:
                name, label, img = sess.run([name_op, label_op, image_op])
                img = np.squeeze(img, axis=0)
                label = np.squeeze(label, axis=0)
                print(name.tostring().decode('utf8'))
                print(label.tostring().decode('utf8'))
                # plt.imshow(img)
                cv.imshow('img', img)
                cv.waitKey(50)
        except tf.errors.OutOfRangeError:
            print('End')
        finally:
            coord.request_stop()
        coord.join(threads)  # 等待各线程关闭


def init_binary_dataset(save_name, tag, datatype, shuffle):
    assert tag in ['train', 'validation']
    assert datatype in ['tfrecord', 'json', 'h5']

    def _bytes_feature(value):  # 生成字符串型的属性
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):  # 生成整数型的属性
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # ToDO: arrange more datatype
    input_dir = os.path.join(INPUT_DIR, tag)
    output_dir = OUTPUT_DIR
    mk_childfolders(output_dir)
    reader, count = init_img_reader(input_dir, class_list=CLASS_LIST,
                                    img_resize=[32, 32], channels=OUTPUT_CHANNEL, shuffle=shuffle)
    fpath, xs, ys = reader.make_one_shot_iterator().get_next()
    # param batch_xs: shape [32, 32, channels] type tf.uint8
    # param batch_ys: shape [1] type tf.int32
    logging.basicConfig(filename=os.path.join(OUTPUT_DIR, 'log.txt'), level=logging.DEBUG)
    if shuffle:
        save_name += '.shuffle'
    else:
        save_name += '.noshuffle'
    output_path = os.path.join(output_dir, '{}.{}.{}'.format(save_name, tag, datatype))

    with tf.Session() as sess:
        if datatype == 'tfrecord':
            with tf.python_io.TFRecordWriter(output_path) as writer:
                perc = count / 100
                perc = 1 if perc < 1 else int(perc)
                step = 0
                while True:
                    try:
                        org_path, img, label = sess.run([fpath, xs, ys])
                        org_name = os.path.basename(org_path.decode()).split('.')[0]
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'name': _bytes_feature(org_name.encode('utf8')),
                            'label': _bytes_feature(label),
                            'height': _int64_feature(32),
                            'width': _int64_feature(32),
                            'channel': _int64_feature(3),
                            'image': _bytes_feature(img.tostring())
                        }))
                        writer.write(example.SerializeToString())
                        if int(org_name) % perc == 0:
                            processBar(step, count, msg="Initizing " + output_path)
                        step += 1
                    except tf.errors.OutOfRangeError:
                        processBar(step, count, msg="Initizing " + output_path)
                        print('Finish!')
                        break
        elif datatype == 'json':
            pass
        elif datatype == 'h5':
            pass


def main():
    def _make_separate_dataset():
        print('Begin to initialize training dataset...')
        init_separate_dataset('train', 0, -1)
        print('Begin to initialize validation dataset...')
        init_separate_dataset('validation', 0, -1)

    def _make_tfrecord_dataset():
        print('Begin to initialize training dataset...')
        init_binary_dataset('clouds', tag='train', datatype='tfrecord', shuffle=True)
        print('Begin to initialize validation dataset...')
        init_binary_dataset(save_name='clouds', tag='validation', datatype='tfrecord', shuffle=True)

    begintime = datetime.now()
    # _make_separate_dataset()
    # _make_tfrecord_dataset()
    check_tfrecord()
    endtime = datetime.now()
    print('All dataset initialized!  Span Time:%s' % (endtime - begintime))


if __name__ == '__main__':
    main()
