import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_v1
import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
from time import time
from tabulate import tabulate
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'  # 可以使用的GPU
gpu_net = '/gpu:1'
gpu_opt = '/gpu:0'
gpu_evl = '/gpu:1'
# TODO: output the results to file "results.csv" and compute the confusion matrix

flags = tf.app.flags
ERROR_FLAG = 0
CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'nodata']
CLASS_COUNT = len(CLASS_LIST)

flags.DEFINE_integer('resnet_model', 101, 'The layers count of resnet: 50, 101 or 152')
flags.DEFINE_string('pretrain_path', './pretrained/resnet_v1_{}.ckpt', '')
flags.DEFINE_string('data_dir', './datasets/separate_relabel/', '')
flags.DEFINE_string('save_dir', './checkpoints/models.separate_{}_{}_{}-',
                    'Save user models and logs, ends with 1.unpre/pre* 2.optimizer 3.loss type')
flags.DEFINE_string('model_name', 'resnet.model', 'model name')
flags.DEFINE_string('optimizer', 'SGD', 'Either Adam or SGD')
flags.DEFINE_string('losstype', 'cross_entropy', 'Either ordinal or cross_entropy')
flags.DEFINE_integer('batch_size', 256, 'How many big images in a batch, so the small images count is 8 * batch_size')
flags.DEFINE_integer('epoch', 5, 'Count of epoch')
flags.DEFINE_boolean('fullytrain', True, 'Train all images in dataset')
flags.DEFINE_integer('loops', 10000, 'Number of iteration in training, only works when fullytrain is False')
flags.DEFINE_float('learning_rate', 8e-3, 'Initial learning rate')
flags.DEFINE_float('regularize_scale', 1e-5, 'L2 regularizer scale')
flags.DEFINE_boolean('is_training', False, 'Train or evaluate?')
flags.DEFINE_boolean('pretrained', False,
                     'Whether using the pretrained model given by TensorFlow or not')
flags.DEFINE_boolean('use_last_model', True,
                     'Whether using the last model trained by ourselves or not. '
                     'Only works when \'pretrained\' is False')
FLAGS = flags.FLAGS


def _get_session_config():
    config = tf.ConfigProto()
    # config.gpu_options.report_tensor_allocations_upon_oom = True
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction  = 0.5
    return config


def _get_save_dir(flags):
    if flags.pretrained:
        mtype = 'pre' + str(flags.resnet_model)
    else:
        mtype = 'unpre' + str(flags.resnet_model)
    dir = flags.save_dir.format(mtype, flags.optimizer, flags.losstype)
    dir = os.path.abspath(dir)
    basedir, name = os.path.split(dir)
    try:
        list = os.listdir(basedir)
        list = [s for s in list if s.startswith(name)]
        list = [s[len(name):] for s in list]
        num = int(max(list))
    except:
        num = 0
    if flags.is_training and not flags.use_last_model:
        dir += str(num + 1)
    else:
        dir += str(num)
    return dir


def save(sess, model_path, counter):
    saver = tf.train.Saver(max_to_keep=3)
    save_dir, model_name = os.path.split(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = saver.save(sess, model_path, global_step=counter)
    print('MODEL RESTORED IN: ' + save_path)


def load_user_model(sess, model_dir):
    import re
    print(' [*] Load last model in {}...'.format(model_dir))
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver(max_to_keep=1)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return counter
    else:
        print(" [*] No checkpoint files were found")
        return ERROR_FLAG


def init_csv_reader(path, batch_size, epoch, is_training):
    '''
    get a tensorflow dataset
    :param path:
    :param batch_size:
    :param epoch:
    :param is_training:
    :return: dataset, count
    '''

    def _parse_function(xs, ys):
        x_img_str = tf.read_file(xs)
        x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(x_img_str), tf.float32)
        x_img_resized = tf.image.resize_images(x_img_decoded, size=[128, 64],
                                               method=tf.image.ResizeMethod.BILINEAR)
        x_img = tf.reshape(x_img_resized, [128, 64, 3])
        return x_img, ys

    # Processing the image filenames
    fs = os.listdir(path)
    csv_name = os.path.join(path, [it for it in fs if '.csv' in it][0])

    # Add one more column named "Train" to split the training set and validation set
    frame = pd.read_csv(csv_name)

    frame = frame.loc[frame['Train'] == ('T' if is_training else 'F')]
    count = frame['num_id'].count()
    print(' [*] {} images initialized as {} data'.format(count, ('training' if is_training else 'validation')))

    num_idx = frame['num_id'].values.astype(str).tolist()
    t_names = [item + '.jpg' for item in num_idx]
    file_names = [os.path.join(path, item) for item in t_names]
    labels = frame['Cloud_Cover'].values.tolist()
    t_labels = [list('F'.join(item.split('*'))) for item in labels]
    for it in range(len(t_labels)):
        t_labels[it] = list(map(lambda x: ord(x) - ord('A'), t_labels[it]))
    # Initialize as a tensorflow tensor object
    data = tf.data.Dataset.from_tensor_slices((tf.constant(file_names),
                                               tf.constant(t_labels)))
    # larger buffer_size increase the confusion level, but cost more time and memory
    # however, there's evidence that larger buffer_size results in less caculating time
    data = data.shuffle(buffer_size=batch_size * 1024)
    data = data.map(_parse_function)
    if is_training:
        return data.batch(batch_size).repeat(epoch), count
    else:
        return data.batch(batch_size), count


def init_img_reader(input_dir, batch_size, epoch, class_list, img_resize=None, channels=3, shuffle=False):
    '''

    :param input_dir:
    :param batch_size:
    :param epoch:
    :param class_list:
    :param img_resize:
    :param channels:
    :param shuffle:
    :return: dataset:
            coutn: dataset可以get_next的次数（在当前epoch与batch_size下）
    '''
    assert channels in [1, 3]
    assert batch_size > 0
    assert epoch > 0
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
        return x_img, label

    files = []
    labels = []
    for i in range(len(class_list)):
        dir = os.path.join(input_dir, class_list[i])
        if not os.path.exists(dir):
            print('path %s not exist' % dir)
            continue
        fs = os.listdir(dir)
        fs = [os.path.join(dir, item) for item in fs]
        files.extend(fs)
        labels.extend([i] * len(fs))
    count = len(files)
    assert count > batch_size
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
    if count % batch_size > 0:
        count = count - count % batch_size
        files = files[:count]
        labels = labels[:count]
    # Initialize as a tensorflow tensor object
    data = tf.data.Dataset.from_tensor_slices((tf.constant(files, dtype=tf.string, name='file_path'),
                                               tf.constant(labels, name='label')))
    data = data.map(_parse_function)
    # if shuffle:
    #     data = data.shuffle(count)
    return data.batch(batch_size).repeat(epoch), count // batch_size * epoch


class Confusion(object):
    '''
     To compute the confusion matrix
     The __str__ function is overrided for better visualization effect
     Use the following command to install necessary packages
     $ pip install tabulate
    '''

    def __init__(self, headers=['A', 'B', 'C', 'D', 'E', '*'], predictions=None, labels=None):
        self.headers = headers
        self.class_count = len(headers)
        self.matrix = np.zeros((self.class_count, self.class_count), dtype=np.int64)
        try:
            if predictions.shape == labels.shape and predictions.shape[0] > 0:
                self.add_data(predictions, labels)
        except:
            pass

    def add_data(self, predictions, labels):
        if predictions.shape == labels.shape and predictions.shape[0] > 0:
            for it in range(predictions.shape[0]):
                self.matrix[predictions[it]][labels[it]] += 1

    def __str__(self):
        matrix = self.matrix.tolist()
        return tabulate(matrix, headers=self.headers, tablefmt='grid')


def train(sess, optim, loss, summaries, loop, global_step=tf.Variable(0, False),
          predct=None, labels=None, accuracy=None, logiter=20, ):
    def _get_loss_log_summary(sess, loss, summaries, predct, labels, writer, step_val):
        sum_str, lossval = sess.run([summaries, loss])
        if predct != None and labels != None and accuracy != None:
            predval, ys, accval = sess.run([predct, labels, accuracy])
            cf = Confusion(predictions=predval, labels=ys)
            msg = '[*]step:{}  accuracy:{}  Confusion Matrix:\n{}'.format(step_val, accval, cf)
            print(msg)
            logging.info(msg)
        writer.add_summary(sum_str, step_val)
        return lossval

    time_begin = datetime.now()
    savedir = _get_save_dir(FLAGS)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    suffix = str(int(time()))
    writer = tf.summary.FileWriter(savedir, sess.graph, filename_suffix=suffix)
    logging.basicConfig(filename=os.path.join(savedir, 'train.output-{}.txt'.format(suffix)), level=logging.DEBUG)
    for it in range(loop):
        try:
            _, step_val = sess.run([optim, global_step])
            if it % logiter == 0:
                lossval = _get_loss_log_summary(sess, loss, summaries, predct, labels, writer, step_val)
                time_elapse = datetime.now() - time_begin
                time_remain = time_elapse / (it + 1) * (loop - it - 1)
                msg = 'elapsed time:{} remaining time:{} iteration:{}/{} loss:{}'. \
                    format(time_elapse, time_remain, it, loop, lossval)
                print(msg)
                logging.info(msg)
                if lossval < 1.5:
                    save(sess, os.path.join(savedir, 'tmp_loss{:.3f}'.format(lossval) + FLAGS.model_name), step_val)
        except tf.errors.InvalidArgumentError as e:
            print('An error of type tf.errors.InvalidArgumentError has been ignored...')
            print(e.message)
            logging.error('tf.errors.InvalidArgumentError:\r\n' + e.message)
            continue
        except tf.errors.OutOfRangeError:
            lossval = _get_loss_log_summary(sess, loss, summaries, predct, labels, writer, step_val)
            msg = '[*]Epoch reach the end, final loss value is {}'.format(lossval)
            print(msg)
            logging.info(msg)
            break
    time_elapse = datetime.now() - time_begin
    print('Training finish, elapsed time %s...Trying to save the model...' % time_elapse)
    save(sess, os.path.join(savedir, FLAGS.model_name), step_val)


def evaluate(sess, probabilities, labels, loop=1000, logiter=50):
    cnt = 0;
    accsum = 0.0
    suffix = int(time())
    logpath = os.path.join(_get_save_dir(FLAGS), 'evaluate.output-{}.txt'.format(suffix))
    logging.basicConfig(filename=logpath, level=logging.DEBUG)
    cf = Confusion(headers=['A', 'B', 'C', 'D', 'E', '*'])
    with tf.device(gpu_evl):
        prediction = tf.argmax(probabilities, axis=-1)
        accuracy, acc_update = tf.metrics.accuracy(labels, prediction)
        sess.run(tf.local_variables_initializer())
        while cnt < loop:
            cnt += 1
            try:
                accval, acc_up, probsval, predval, ys = sess.run(
                    [accuracy, acc_update, probabilities, prediction, labels])
                accsum = accsum + accval
                cf.add_data(predval, ys)
                if cnt % logiter == 0:
                    msg = 'iteration: {}/{}  accuracy: {}\r\nconfusion matrix:\r\n{}'.format(cnt, loop, accval, cf)
                    print(msg)
                    #print('pre:{}\nlable:{}'.format(predval, ys))
                    logging.info(msg)
            except tf.errors.InvalidArgumentError:
                print('An error of type tf.errors.InvalidArgumentError has been ignored...')
                logging.error('tf.errors.InvalidArgumentError')
                continue
            except tf.errors.OutOfRangeError:
                print('Dataset reach the end.')
                break
            except tf.errors.NotFoundError:
                break
    return accsum / cnt


def init_loss(logits, labels, end_points=None, losstype='ordinal'):
    with tf.device(gpu_opt):
        if losstype == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.regularize_scale),
                                                         tf.trainable_variables())
            loss += reg
        elif losstype == 'ordinal':
            # ToDO: these codes below can`t get a valid loss value
            import math
            ks = [np.arange(1, 7).astype(np.float32)[None, :] \
                  for _ in range(FLAGS.batch_size)]
            ks = np.concatenate(ks, axis=0)
            kfac = [[math.factorial(it) for it in range(1, 7)] for _ in range(FLAGS.batch_size)]
            kfac = np.array(kfac, dtype=np.float32)
            k_vector = tf.constant(ks, name='k_vector')
            k_factor = tf.constant(kfac, name='k_factor')
            softmaxed = tf.nn.softmax(logits, axis=-1, name='softmax')
            log_exp = tf.log(softmaxed)
            poisson = k_vector * log_exp - logits - tf.log(k_factor)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=poisson)
            loss = tf.reduce_mean(loss)
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.regularize_scale),
                                                         tf.trainable_variables())
            loss += reg
        else:
            raise NotImplementedError
        return loss


def main(_):
    reader, data_count = init_img_reader(os.path.join(FLAGS.data_dir, 'train' if FLAGS.is_training else 'validation')
                                         , FLAGS.batch_size, FLAGS.epoch, CLASS_LIST, img_resize=[32, 32], shuffle=True)
    batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
    # param batch_xs: shape [batch_size, 32, 32, 3] type tf.float32
    # param batch_ys: shape [batch_size] type tf.int32

    # off_hs = [0, 0, 32, 32, 64, 64, 96, 96]
    # off_ws = [0, 32, 0, 32, 0, 32, 0, 32]
    # x_img_cuts = [tf.image.crop_to_bounding_box(batch_xs, hs, ws, 32, 32) \
    #               for hs, ws in zip(off_hs, off_ws)]
    # batch_xs = tf.reshape(tf.concat(x_img_cuts, axis=0), [-1, 32, 32, 3])  # shape [batch_size * 8, 32, 32, 3]
    # batch_ys = tf.reshape(batch_ys, [-1])  # shape [batch_size * 8]

    config = _get_session_config()
    sess = tf.InteractiveSession(config=config)

    def _get_resnet():
        if FLAGS.resnet_model == 50:
            return resnet_v1.resnet_v1_50
        elif FLAGS.resnet_model == 101:
            return resnet_v1.resnet_v1_101
        elif FLAGS.resnet_model == 152:
            return resnet_v1.resnet_v1_152
        elif FLAGS.resnet_model == 200:
            return resnet_v1.resnet_v1_200
        else:
            raise NotImplementedError

    resnet = _get_resnet()

    if FLAGS.is_training:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            with tf.device(gpu_net):
                # param logits: shape [batch_size, CLASS_COUNT]
                logits, end_points = resnet(batch_xs, num_classes=CLASS_COUNT, is_training=True)
                logits = tf.reshape(logits, [-1, CLASS_COUNT], name='logits_2d')
                prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
            with tf.device(gpu_evl):
                mAP = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), dtype=tf.float32))
                loss = init_loss(logits, batch_ys, end_points=end_points, losstype=FLAGS.losstype)
                mAP_sum = tf.summary.scalar('mAP', mAP)
                loss_sum = tf.summary.scalar('loss', loss)
                summaries = tf.summary.merge([mAP_sum, loss_sum])

        step_num = 0
        if FLAGS.pretrained:
            # Load the pretrained model given by TensorFlow official
            exclusions = ['resnet_v1_{}/logits'.format(FLAGS.resnet_model), 'predictions']
            resnet_except_logits = slim.get_variables_to_restore(exclude=exclusions)
            path = FLAGS.pretrain_path.format(FLAGS.resnet_model)
            init_fn = slim.assign_from_checkpoint_fn(path, resnet_except_logits,
                                                     ignore_missing_vars=True)
            init_fn(sess)
            print('Pretrained model %s successfully loaded' % path)
        elif FLAGS.use_last_model:
            # Load the last model trained by ourselves
            step_num = load_user_model(sess, _get_save_dir(FLAGS))

        with tf.device(gpu_opt):
            global_step = tf.Variable(step_num, trainable=False)
            # Exponential decay learning rate and optimizer configurations
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 100, 0.98, staircase=True)
            if 'SGD' in FLAGS.optimizer:
                optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            elif 'Adam' in FLAGS.optimizer:
                optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            else:
                raise NotImplementedError

        # Ready to train
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        loops = FLAGS.loops
        if FLAGS.fullytrain:
            loops = data_count
        else:
            loops = min(loops, data_count)
        train(sess, optim, loss, summaries, loops, global_step, prediction, batch_ys, mAP)
        print('Training finished')
    else:  # Evaluate
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            probs, end_points = resnet(batch_xs, num_classes=CLASS_COUNT, is_training=False)
            probs = tf.reshape(probs, [-1, CLASS_COUNT], name='probability')
            # _, binary = init_loss(probs, batch_ys, end_points, losstype=FLAGS.losstype)
        if FLAGS.pretrained:
            # Load the pretrained model given by TensorFlow official
            exclusions = ['resnet_v1_{}/logits'.format(FLAGS.resnet_model), 'predictions']
            resnet_except_logits = slim.get_variables_to_restore(exclude=exclusions)
            path = FLAGS.pretrain_path.format(FLAGS.resnet_model)
            init_fn = slim.assign_from_checkpoint_fn(path, resnet_except_logits,
                                                     ignore_missing_vars=True)
            init_fn(sess)
            print('Pretrained model %s successfully loaded' % path)
        elif FLAGS.use_last_model:
            # Load the last model trained by ourselves
            load_user_model(sess, _get_save_dir(FLAGS))
        else:
            raise NotImplementedError
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # TODO: Consider the binary classification...
        # which is even more tricky to be implemented...
        accsum_val = evaluate(sess, probabilities=probs, labels=batch_ys)
        print('The model accuracy is {}'.format(accsum_val))


if __name__ == '__main__':
    tf.app.run()
