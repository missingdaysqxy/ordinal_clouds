# -*- coding: utf-8 -*-
# @Time    : 2018/10/9 22:31
# @Author  : qxliu
# @Email   : qixuan.lqx@qq.com
# @File    : makedataset.py
# @Software: PyCharm

import tensorflow as tf
from datetime import datetime
import pandas as pd
import os

input_dir = './datasets/mode_2004/'
output_dir = './datasets/separate/'
batch_size = 8

_class_list = ['A', 'B', 'C', 'D', 'E', 'nodata']


def mk_childfolders(parent_dir, child_list=[]):
    for dir in child_list:
        path = os.path.join(parent_dir, dir)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


def init_reader(input_dir, is_train, start_index=None, max_count=None):
    def _parse_function(files, labels):
        '''Decode images and devide into 8 small images
        :param files: shape[]
        :param labels: shape[8]'''
        x_img_str = tf.read_file(files)  # shape[]
        x_img_decoded = tf.image.decode_jpeg(x_img_str)  # shape[?,?,?]
        x_img_resized = tf.image.resize_images(x_img_decoded, size=[512, 512],
                                               method=tf.image.ResizeMethod.BILINEAR)  # shape[512,512,?]

        off_ws = [0, 0, 0, 0, 256, 256, 256, 256]
        off_hs = [0, 128, 256, 384, 0, 128, 256, 384]
        x_img_cuts = [tf.image.crop_to_bounding_box(x_img_resized, hs, ws, 128, 256) \
                      for hs, ws in zip(off_hs, off_ws)]  # [shape[128,256,?] * batch_size]
        x_img_cuts = [tf.cast(x, tf.uint8) for x in x_img_cuts]
        tf_imgs = [tf.image.encode_jpeg(x, 'rgb', name='rgb_img') for x in
                   x_img_cuts]  # a list of small image, type: [tf.string * batch_size]
        batch_xs = tf.reshape(tf_imgs, [batch_size, 1], name='batch_xs')  # shape[batch_size,1]
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
        else:
            start_index = 0
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


def init_dataset(tag='train', start_index=None, max_count=None):
    if tag not in ['train', 'validation']:
        raise NotImplementedError
    _output_dir = os.path.join(output_dir, tag)
    mk_childfolders(_output_dir, child_list=_class_list)
    reader, count = init_reader(input_dir, tag == 'train', start_index, max_count)
    batch_path, batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
    # param batch_path: shape []
    # param batch_xs: shape [batch_size, 128, 256, 3] type tf.uint8
    # param batch_ys: shape [batch_size, 1] type tf.int32
    xs = [tf.squeeze(x, axis=0) for x in
          tf.split(batch_xs, batch_size, axis=0)]  # a list of single images, [shape[1] * batch_size]
    ys = [tf.squeeze(y, axis=0) for y in
          tf.split(batch_ys, batch_size, axis=0)]  # a list of single label, [shape[1] * batch_size]
    with tf.Session() as sess:
        perc = count / 100
        perc = 1 if perc < 1 else int(perc)
        step = 0
        while True:
            try:
                path, imgs, labels = sess.run([batch_path, xs, ys])
                img_idx = os.path.basename(path.decode()).split('.')[0]
                for i in range(batch_size):
                    save_path = os.path.join(_output_dir,
                                             _class_list[labels[i][0]] + '/' + img_idx + '_' + str(i) + '.jpg')
                    with tf.gfile.GFile(save_path, "wb") as f:
                        f.write(imgs[i][0])
                if int(img_idx) % perc == 0:
                    print('progress: {}/{}'.format(step, count))
                step += 1
            except tf.errors.OutOfRangeError:
                print('Finish!')
                break
            except Exception:
                print('lack of file %s' % path.decode())
                pass

def main():
    begintime = datetime.now()
    print('Begin to initialize training dataset...')
    init_dataset('train', 0, 500)
    print('Begin to initialize validation dataset...')
    init_dataset('validation', 0, 200)
    endtime = datetime.now()
    print('All dataset initialized!  Span Time:%s' % (endtime - begintime))


if __name__ == '__main__':
    main()
