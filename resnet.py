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
gpu_train = '/gpu:1'
gpu_test = '/gpu:0'
# TODO: output the results to file "results.csv" and compute the confusion matrix

flags = tf.app.flags
ERROR_FLAG = 0
CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'nodata']
CLASS_COUNT = len(CLASS_LIST)

flags.DEFINE_integer('resnet_model', 101, 'The layers count of resnet: 50, 101 or 152')
flags.DEFINE_string('official_model_path', './officialmodels/resnet_v1_{}.ckpt', '')
flags.DEFINE_string('data_dir', './datasets/separate_relabel/', '')
flags.DEFINE_string('save_dir', './checkpoints/models.separate_{}_{}_{}-',
                    'Save user models and logs, ends with 1.unpre/pre* 2.optimizer 3.loss type')
flags.DEFINE_string('model_name', 'resnet.model', 'model name')
flags.DEFINE_string('optimizer', 'SGD', 'Either Adam or SGD')
flags.DEFINE_string('loss_type', 'cross_entropy', 'Either ordinal or cross_entropy')
flags.DEFINE_integer('batch_size', 256, 'How many big images in a batch, so the small images count is 8 * batch_size')
flags.DEFINE_integer('epoch', 5, 'Count of epoch')
flags.DEFINE_boolean('fullytrain', True, 'Train all images in dataset')
flags.DEFINE_integer('loops', 10000, 'Number of iteration in training, only works when fullytrain is False')
flags.DEFINE_float('learning_rate', 8e-3, 'Initial learning rate')
flags.DEFINE_float('regularize_scale', 1e-5, 'L2 regularizer scale')
flags.DEFINE_boolean('is_training', False, 'Train or evaluate?')
flags.DEFINE_boolean('test_after_train',True,'Test the model on validation dataset after train')
flags.DEFINE_string('model_to_load', 'last',
                    "Which pretrained model to use, choose from 'offical','last','none'")

FLAGS = flags.FLAGS


def processBar(num, total, msg='', length=50):
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
    return r.replace('\r', ':')


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
    dir = flags.save_dir.format(mtype, flags.optimizer, flags.loss_type)
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


def init_loss(logits, labels, loss_type='ordinal'):
    with tf.device(gpu_train):
        if loss_type == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.regularize_scale),
                                                         tf.trainable_variables())
            loss += reg
        elif loss_type == 'ordinal':
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


def train(sess, train_op, global_step_t, loss_t, summary_t, savedir, loops, logiter):
    '''
    Main function to train the resnet
    :param sess: tensorflow session
    :param train_op: tensor operation
    :param loss_t: tensor operation, loss function that trains every iteration
    :param summary_t: tensor operation
    :param global_step_t: tensor operation
    :param loops: int
    :param savedir: string
    :param logiter: int, log the summaries after each 'logiter' iterations
    :return:
    '''
    time_begin = datetime.now()
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    suffix = str(int(time()))
    writer = tf.summary.FileWriter(savedir, sess.graph, filename_suffix=suffix)
    logging.basicConfig(filename=os.path.join(savedir, 'train.output-{}.txt'.format(suffix)), level=logging.DEBUG)
    for it in range(loops):
        try:
            sess.run(train_op)  # main train step
            if it % logiter == 0:  # log summaries
                loss_val, sum_str, step_val, = sess.run([loss_t, summary_t, global_step_t, ])
                writer.add_summary(sum_str, step_val)
                time_elapse = datetime.now() - time_begin
                time_remain = time_elapse / (it + 1) * (loops - it - 1)
                msg = '[*]elapsed time:{} remaining time:{} step:{} loss:{}'. \
                    format(time_elapse, time_remain, step_val, loss_val)
                logmsg = processBar(it, loops, msg, 50)
                logging.info(logmsg)
                if loss_val < 1.5:
                    save(sess, os.path.join(savedir, 'tmp_loss{:.3f}'.format(loss_val) + FLAGS.model_name), step_val)
        except tf.errors.InvalidArgumentError as e:
            print('An error of type tf.errors.InvalidArgumentError has been ignored...')
            print(e.message)
            logging.error('tf.errors.InvalidArgumentError:\r\n' + e.message)
            continue
        except tf.errors.OutOfRangeError:
            loss_val, sum_str, step_val, = sess.run([loss_t, summary_t, global_step_t, ])
            writer.add_summary(sum_str, step_val)
            msg = 'Epoch reach the end, final loss value is {}'.format(loss_val)
            logmsg = processBar(it, loops, msg, 50)
            logging.info(logmsg)
            break
    time_elapse = datetime.now() - time_begin
    print('Training finish, elapsed time %s' % time_elapse)
    return step_val


def evaluate(sess, acc_t, probs_t, labels_t, summary_t, loops, logiter, savedir):
    accuracies = []
    suffix = int(time())
    logpath = os.path.join(savedir, 'evaluate.output-{}.txt'.format(suffix))
    logging.basicConfig(filename=logpath, level=logging.DEBUG)
    writer = tf.summary.FileWriter(savedir, sess.graph, filename_suffix=suffix)
    cf = Confusion(headers=['A', 'B', 'C', 'D', 'E', '*'])
    predict_t = tf.argmax(probs_t, axis=-1)
    sess.run(tf.local_variables_initializer())
    for it in range(loops):
        try:
            accuracy, predictions, labels, sum_str = sess.run(
                [acc_t, predict_t, labels_t, summary_t])
            accuracies.append(accuracy)
            cf.add_data(predictions, labels)
            if it % logiter == 0:
                msg = 'iteration: {}/{}  accuracy: {}\r\nconfusion matrix:\r\n{}'.format(it, loops, accuracy, cf)
                print(msg)
                logging.info(msg)
                writer.add_summary(sum_str, it)
        except tf.errors.InvalidArgumentError as e:
            print('An error of type tf.errors.InvalidArgumentError accrue:')
            print(e.message)
            logging.error('tf.errors.InvalidArgumentError:\n%s' % e.message)
            continue
        except tf.errors.OutOfRangeError:
            print('Dataset reach the end.')
            break
        except tf.errors.NotFoundError:
            break
    return accuracies


def main(_):
    reader, data_count = init_img_reader(os.path.join(FLAGS.data_dir, 'train' if FLAGS.is_training else 'validation')
                                         , FLAGS.batch_size, FLAGS.epoch, CLASS_LIST, img_resize=[32, 32], shuffle=True)
    batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
    # param batch_xs: shape [batch_size, 32, 32, 3] type tf.float32
    # param batch_ys: shape [batch_size] type tf.int32

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
    savedir = _get_save_dir(FLAGS)
    if FLAGS.is_training:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()), tf.device(gpu_train):
            # param logits: shape [batch_size, CLASS_COUNT]
            logits, end_points = resnet(batch_xs, num_classes=CLASS_COUNT, is_training=True)
            prediction = tf.argmax(logits, axis=-1, output_type=tf.int32, name='prediction')
            loss_t = init_loss(logits, batch_ys, loss_type=FLAGS.loss_type)
            mAP_t = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), dtype=tf.float32))
            # Load pretrained model
            if FLAGS.model_to_load.lower() == 'last':
                # Load the last model trained by ourselves
                global_step_init_value = load_user_model(sess, savedir)
            elif FLAGS.model_to_load.lower() == 'official':
                # Load the pretrained model given by TensorFlow official
                global_step_init_value = 0
                exclusions = ['resnet_v1_{}/logits'.format(FLAGS.resnet_model), 'predictions']
                resnet_except_logits = slim.get_variables_to_restore(exclude=exclusions)
                path = FLAGS.official_model_path.format(FLAGS.resnet_model)
                init_fn = slim.assign_from_checkpoint_fn(path, resnet_except_logits,
                                                         ignore_missing_vars=True)
                init_fn(sess)
                print('Pretrained model %s successfully loaded' % path)
            elif FLAGS.model_to_load.lower() == 'none':
                global_step_init_value = 0
            global_step = tf.Variable(global_step_init_value, trainable=False, name='global_step')
            # variable averages operation
            variable_averages = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
            variable_averages_op = variable_averages.apply(tf.trainable_variables())
            # Exponential decay learning rate and optimizer configurations
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=data_count,
                                                       decay_rate=0.98, staircase=True, name='learning_rate')
            if FLAGS.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate)
            elif FLAGS.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate)
            else:
                raise NotImplementedError
            train_step = optim.minimize(loss_t, global_step=global_step, name=FLAGS.optimizer)
            train_op = tf.group(train_step, variable_averages_op)
            # Summaries log confugrations
            mAP_log = tf.summary.scalar('mAP', mAP_t)
            loss_log = tf.summary.scalar('loss', loss_t)
            summaries = tf.summary.merge([mAP_log, loss_log])
            # Init all variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            loops = data_count if FLAGS.fullytrain else min(FLAGS.loops, data_count)
            # Ready to Train
            final_step_val = train(sess, train_op, global_step, loss_t, summaries, savedir, loops, logiter=10)
            print('Training finished, try to save the model...')
            save(sess, os.path.join(savedir, FLAGS.model_name), final_step_val)
            if FLAGS.test_after_train:
                print('Prepare to test the model...')
                # ToDO: finish this
    if not FLAGS.is_training:  # Evaluate
        with slim.arg_scope(resnet_v1.resnet_arg_scope()), tf.device(gpu_test):
            probs, end_points = resnet(batch_xs, num_classes=CLASS_COUNT, is_training=False)
            # probs = tf.reshape(probs, [-1, CLASS_COUNT], name='probability')
            prediction = tf.argmax(probs, axis=-1, output_type=tf.int32, name='prediction')
            acc_t = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), dtype=tf.float32))
            summary_t = tf.summary.scalar('accuracy', acc_t)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            # Load pretrained model
            if FLAGS.model_to_load.lower() == 'last':
                # Load the last model trained by ourselves
                load_user_model(sess, savedir)
            elif FLAGS.model_to_load.lower() == 'official':
                # Load the pretrained model given by TensorFlow official
                exclusions = ['resnet_v1_{}/logits'.format(FLAGS.resnet_model), 'predictions']
                resnet_except_logits = slim.get_variables_to_restore(exclude=exclusions)
                path = FLAGS.official_model_path.format(FLAGS.resnet_model)
                init_fn = slim.assign_from_checkpoint_fn(path, resnet_except_logits,
                                                         ignore_missing_vars=True)
                init_fn(sess)
                print('Pretrained model %s successfully loaded' % path)
            elif FLAGS.model_to_load.lower() == 'none':
                pass
            savedir = os.path.join(savedir, 'test')
            accuracies = evaluate(sess, acc_t, probs, batch_ys, summary_t, savedir, loops=data_count, logiter=10)
            print('The model accuracy is {}'.format(sum(accuracies) / len(accuracies)))


if __name__ == '__main__':
    tf.app.run()
