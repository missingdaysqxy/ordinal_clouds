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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 可以使用的GPU
gpu_train = '/gpu:0'
gpu_test = '/gpu:0'
# TODO: output the results to file "results.csv" and compute the confusion matrix

flags = tf.app.flags
ERROR_FLAG = 0
CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'nodata']
CLASS_COUNT = len(CLASS_LIST)

flags.DEFINE_integer('resnet_model', 101, 'The layers batch_count of resnet: 50, 101 or 152')
flags.DEFINE_string('official_model_path', './pretrained/resnet_v1_{}.ckpt',
                    'If \'model_to_load\' is \'pretrained\', then this file will be loaded as a pretrained model')
flags.DEFINE_string('data_dir', './datasets/separate_relabel/', 'Where is the input data in')
flags.DEFINE_string('save_dir', './checkpoints/models.separate_{}_{}_{}-',
                    'Save user models and logs, ends with 1.unpre/pre* 2.optimizer 3.loss type')
flags.DEFINE_string('model_name', 'ordinal_clouds.ckpt', 'model name')
flags.DEFINE_string('optimizer', 'SGD', 'Either Adam or SGD')
flags.DEFINE_string('loss_type', 'cross_entropy', 'Either ordinal or cross_entropy')
flags.DEFINE_integer('batch_size', 256,
                     'How many big images in a batch, so the small images batch_count is 8 * batch_size')
flags.DEFINE_integer('epoch', 1, 'Count of epoch, if zero, the dataset will be empty')
flags.DEFINE_integer('loops', 500, 'Number of iterations, only works when loop_all is False. '
                                   'Note: it will be modified when the data batch_count is less')
flags.DEFINE_float('learning_rate', 8e-3, 'Initial learning rate')
flags.DEFINE_float('regularize_scale', 1e-5, 'L2 regularizer scale')
flags.DEFINE_boolean('random_adjust', False, 'Randomly adjust the brightness, contrast and flip with dataset')
flags.DEFINE_boolean('loop_all', True, 'Train all images in dataset')
flags.DEFINE_boolean('is_training', False, 'Train or evaluate?')
flags.DEFINE_boolean('auto_save', False, 'Auto save the model when loss value is small enough')
flags.DEFINE_boolean('test_after_train', False, 'Test the model on validation dataset after train')
flags.DEFINE_string('model_to_load', 'last',
                    "Which pretrained model to use, choose from 'pretrained','last','new'")

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


def _checkflags(flags):
    assert flags.resnet_model in [50, 101, 152, 200]
    assert os.path.exists(flags.data_dir)
    assert flags.optimizer in ['SGD', 'Adam']
    assert flags.loss_type in ['ordinal', 'cross_entropy']
    assert flags.batch_size % 8 == 0 and flags.batch_size > 0
    assert flags.epoch >= 0
    assert flags.loops >= 0
    assert flags.learning_rate > 0
    assert flags.regularize_scale > 0
    assert flags.model_to_load in ['pretrained', 'last', 'new']
    print(flags)


def _get_session_config():
    config = tf.ConfigProto()
    # config.gpu_options.report_tensor_allocations_upon_oom = True
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction  = 0.5
    return config


def _get_save_dir(flags):
    if flags.model_to_load == 'pretrained':
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
    if flags.is_training and flags.model_to_load != 'last':
        dir += str(num + 1)
    else:
        dir += str(num)
    return dir


def save(sess, model_path, counter):
    saver = tf.train.Saver()
    save_dir, model_name = os.path.split(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = saver.save(sess, model_path, global_step=counter)
    return save_path


def init_csv_reader(path, batch_size, epoch, is_training):
    '''
    get a tensorflow dataset
    :param path:
    :param batch_size:
    :param epoch:
    :param is_training:
    :return: dataset, batch_count
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


def init_img_reader(input_dir, batch_size, epoch, class_list, img_resize=None, channels=3, shuffle=True, adjust=False):
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

    def _parse_function(file_path, label, org_name):
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
        if adjust:  # 随机亮度、对比度与翻转
            x_img = tf.image.random_brightness(x_img, max_delta=0.25)
            x_img = tf.image.random_contrast(x_img, lower=0.75, upper=1.5)
            # x_img = tf.image.random_hue(x_img, max_delta=0.5)
            x_img = tf.image.random_flip_up_down(x_img)
            x_img = tf.image.random_flip_left_right(x_img)
        return x_img, label, org_name

    files = []
    labels = []
    org_names = []
    cloudcover = [0.0, 0.1, 0.25, 0.75, 1.0]
    for i in range(len(class_list)):
        dir = os.path.join(input_dir, class_list[i])
        if not os.path.exists(dir):
            print('path %s not exist' % dir)
            continue
        fs = os.listdir(dir)
        org = [os.path.basename(f).split('_')[0] for f in fs]
        fs = [os.path.join(dir, item) for item in fs]
        files.extend(fs)
        org_names.extend(org)
        if FLAGS.loss_type == 'ordinal':
            labels.extend([cloudcover[i]] * len(fs))
        else:
            labels.extend([i] * len(fs))
    count = len(files)
    assert count > batch_size
    if shuffle:
        import random
        idx = list(range(count))
        random.shuffle(idx)
        sfl_files = []
        sfl_orgs = []
        sfl_labels = []
        for i in idx:
            sfl_files.append(files[i])
            sfl_orgs.append(org_names[i])
            sfl_labels.append(labels[i])
        files = sfl_files
        org_names = sfl_orgs
        labels = sfl_labels
    if count % batch_size > 0:
        count = count - count % batch_size
        # files = files[:batch_count]
        # labels = labels[:batch_count]
    # Initialize as a tensorflow tensor object
    data = tf.data.Dataset.from_tensor_slices((tf.constant(files, dtype=tf.string, name='file_path'),
                                               tf.constant(labels, name='label'),
                                               tf.constant(org_names, name='org_name')))
    data = data.repeat(epoch)
    data = data.map(_parse_function)
    # if shuffle:
    #     data = data.shuffle(batch_count)
    return data.batch(batch_size), count // batch_size * epoch


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
        self.super_class = {}
        try:
            if predictions.shape == labels.shape and predictions.shape[0] > 0:
                self.add_data(predictions, labels)
        except:
            pass

    def add_data(self, predictions, labels, super_classes=None):
        if predictions.shape == labels.shape and predictions.shape[0] > 0:
            for it in range(predictions.shape[0]):
                self.matrix[predictions[it]][labels[it]] += 1
                if super_classes != None:
                    if self.super_class[super_classes[it]] is None:
                        self.super_class[super_classes[it]] = []
                self.super_class[super_classes[it]].append([predictions[it], labels[it], predictions[it] == labels[it]])

    def __str__(self):
        matrix = self.matrix.tolist()
        msg = tabulate(matrix, headers=self.headers, tablefmt='grid')
        if self.super_class != None:
            super_acc = {}
            for items in self.super_class:
                right_count = 0
                for item in self.super_class:
                    if item[2]:
                        right_count += 1
                if super_acc.__contains__(right_count):
                    super_acc[right_count] += 1
                else:
                    super_acc[right_count] = 1
            accs = super_acc[max(super_acc.keys())] / sum(super_acc.values())
            msg += "\nSuper accuracy is {}, distribution: {}".format(accs, super_acc)
        return msg


def init_loss(logits, labels, end_points, loss_type='ordinal'):
    with tf.device(gpu_train):
        if loss_type == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.regularize_scale),
                                                         tf.trainable_variables())
            loss += reg
        elif loss_type == 'ordinal':
            # ToDO: these codes below can`t get a valid loss value
            with tf.variable_scope('ordinal_loss'):
                # 将原logits卷积成一个值
                conv_1 = slim.conv2d(end_points['resnet_v1_{}/block4'.format(FLAGS.resnet_model)], 6, [3, 3],
                                     scope='conv_1')
                conv_2 = slim.conv2d(conv_1, 1, [3, 3], scope='conv_2')
                reshaped = tf.reshape(conv_2, [FLAGS.batch_size, -1], name='reshaped')
                # sigmoid到[0:1)区间
                sigmoid = tf.nn.sigmoid(reshaped, name='sigmoid')
                # 将groud truth归一化到[0:1)区间
                # r_labels = tf.cast(tf.divide(labels ,CLASS_COUNT,name='groudtruth'),tf.float32)
                # 计算sigmoid与groud truth的距离作为loss
                loss = tf.square(sigmoid - labels)
                loss = tf.reduce_mean(loss)
                reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.regularize_scale),
                                                             tf.trainable_variables())
                loss += reg
            # import math
            #
            # ks = [np.arange(1, 7).astype(np.float32)[None, :] \
            #       for _ in range(FLAGS.batch_size)]
            # ks = np.concatenate(ks, axis=0)
            # kfac = [[math.factorial(it) for it in range(1, 7)] for _ in range(FLAGS.batch_size)]
            # kfac = np.array(kfac, dtype=np.float32)
            # k_vector = tf.constant(ks, name='k_vector')
            # k_factor = tf.constant(kfac, name='k_factor')
            # softmaxed = tf.nn.softmax(logits, axis=-1, name='softmax')
            # log_exp = tf.log(softmaxed)
            # poisson = k_vector * log_exp - logits - tf.log(k_factor)
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=poisson)
            # loss = tf.reduce_mean(loss)
            # reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.regularize_scale),
            #                                              tf.trainable_variables())
            # loss += reg
        else:
            raise NotImplementedError
        return loss


def train(sess, train_op, global_step_t, loss_t, summary_t, savedir, loops, logiter, save_limit=1.2):
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
    :param save_limit: float, when the loss value is less than save_limit, the model will save a temp copy.
            this value will change to the minimize loss value during training
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
            _, loss_val, sum_str, step_val, = sess.run(
                [train_op, loss_t, summary_t, global_step_t, ])  # main train step
            if it % logiter == 0:  # log summaries
                step_val -= 1  # As 'global_step_t' already increased after 'sess.run(train_op)', here we decrease 'step_val' by one
                writer.add_summary(sum_str, step_val)
                time_elapse = datetime.now() - time_begin
                time_remain = time_elapse / (it + 1) * (loops - it - 1)
                msg = 'elapsed time:{} remaining time:{} step:{} loss:{}'. \
                    format(time_elapse, time_remain, step_val, loss_val)
                logmsg = processBar(it, loops, msg, 50)
                logging.info(logmsg)
                if loss_val < save_limit and FLAGS.auto_save:
                    save_limit = loss_val
                    save(sess, os.path.join(savedir, 'tmp_loss{:.3f}'.format(loss_val) + FLAGS.model_name), step_val)
        except tf.errors.InvalidArgumentError as e:
            print('An error of type tf.errors.InvalidArgumentError has been ignored...')
            print(e.message)
            logging.error('tf.errors.InvalidArgumentError:\r\n' + e.message)
            continue
        except tf.errors.OutOfRangeError:
            writer.add_summary(sum_str, step_val)
            msg = 'Epoch reach the end, final loss value is {}'.format(loss_val)
            logmsg = processBar(it, loops, msg, 50)
            logging.info(logmsg)
            break
    time_elapse = datetime.now() - time_begin
    print('Training finish, elapsed time %s' % time_elapse)
    return step_val


def evaluate(sess, acc_t, probs_t, labels_t, org_names_t, summary_t, savedir, loops, logiter):
    accuracies = []
    suffix = str(int(time()))
    logpath = os.path.join(savedir, 'evaluate.output-{}.txt'.format(suffix))
    logging.basicConfig(filename=logpath, level=logging.DEBUG)
    writer = tf.summary.FileWriter(savedir, sess.graph, filename_suffix=suffix)
    cf = Confusion(headers=['A', 'B', 'C', 'D', 'E', '*'])
    predict_t = tf.argmax(probs_t, axis=-1)
    sess.run(tf.local_variables_initializer())
    for it in range(loops):
        try:
            accuracy, predictions, labels, org_names, sum_str = sess.run(
                [acc_t, predict_t, labels_t, org_names_t, summary_t])
            accuracies.append(accuracy)
            cf.add_data(predictions, labels, org_names)
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
    _checkflags(FLAGS)
    reader, data_count = init_img_reader(os.path.join(FLAGS.data_dir, 'train' if FLAGS.is_training else 'train')
                                         , FLAGS.batch_size, FLAGS.epoch, CLASS_LIST, img_resize=[32, 32], shuffle=True,
                                         adjust=FLAGS.random_adjust)
    batch_xs, batch_ys, batch_names = reader.make_one_shot_iterator().get_next()
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
    os.makedirs(savedir, exist_ok=True)
    confile = open(os.path.join(savedir, 'flags.configs.txt'), mode='a', encoding='utf8')
    confile.write(str(FLAGS))
    confile.close()

    def load_user_model(sess, model_dir):
        print('-[*] Load last model in {}'.format(model_dir))
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(model_dir, ckpt_name))
            print("-[*] Success to read {}".format(ckpt_name))
        else:
            print("-[*] No checkpoint files were found")
            return ERROR_FLAG

    if FLAGS.is_training:
        print('[*]Training logs and models will be saved into %s' % os.path.abspath(savedir))
        with slim.arg_scope(resnet_v1.resnet_arg_scope()), tf.device(gpu_train):
            # param logits: shape [batch_size, CLASS_COUNT]
            logits, end_points = resnet(batch_xs, num_classes=CLASS_COUNT, is_training=True)
            logits = tf.squeeze(logits, name='probability')
            prediction = tf.argmax(logits, axis=-1, output_type=tf.int32, name='prediction')
            # prediction = tf.cast(prediction, tf.float32)
            loss_t = init_loss(logits, batch_ys, end_points, loss_type=FLAGS.loss_type)
            mAP_t = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), dtype=tf.float32))

            def get_global_step_init_value(model_dir):
                if FLAGS.model_to_load.lower() == 'last':
                    import re
                    print('-[*] Find last global step value in {}'.format(model_dir))
                    ckpt = tf.train.get_checkpoint_state(model_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                        global_step_init_value = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                        print("-[*] The last global step value of {} is {}".format(ckpt_name, global_step_init_value))
                else:
                    # Load the model pretrained by TensorFlow official
                    global_step_init_value = 0
                return global_step_init_value

            global_step_init_value = get_global_step_init_value(savedir)
            print('[*]Init global_step with value of %d' % global_step_init_value)
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

            # Load pretrained model
            if FLAGS.model_to_load.lower() == 'last':
                # Load the last model trained by ourselves
                load_user_model(sess, savedir)
            elif FLAGS.model_to_load.lower() == 'pretrained':
                # Load the model pretrained by TensorFlow official
                exclusions = ['resnet_v1_{}/logits'.format(FLAGS.resnet_model), 'predictions']
                resnet_except_logits = slim.get_variables_to_restore(exclude=exclusions)
                path = FLAGS.official_model_path.format(FLAGS.resnet_model)
                init_fn = slim.assign_from_checkpoint_fn(path, resnet_except_logits,
                                                         ignore_missing_vars=True)
                init_fn(sess)
                print('[*]Pretrained model %s successfully loaded' % path)

            loops = data_count if FLAGS.loop_all else min(FLAGS.loops, data_count)
            # Ready to Train
            final_step_val = train(sess, train_op, global_step, loss_t, summaries, savedir, loops, logiter=10)
            print('[*]Training finished, try to save the model...')
            save_path = save(sess, os.path.join(savedir, FLAGS.model_name), final_step_val)
            print('model saved into %s' % os.path.abspath(save_path))
            if FLAGS.test_after_train:
                print('[*]Prepare to test the model...')
                t_savedir = os.path.join(savedir, 'test')
                print('[*]Test logs and summaries will be saved into %s' % os.path.abspath(t_savedir))
                os.makedirs(t_savedir, exist_ok=True)
                # ToDO: finish this
    if not FLAGS.is_training:  # Evaluate
        with slim.arg_scope(resnet_v1.resnet_arg_scope()), tf.device(gpu_test):
            probs, end_points = resnet(batch_xs, num_classes=CLASS_COUNT, )
            probs = tf.squeeze(probs, name='probability')
            prediction = tf.argmax(probs, axis=-1, output_type=tf.int32, name='prediction')
            acc_t = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), dtype=tf.float32))
            summary_t = tf.summary.scalar('accuracy', acc_t)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            # Load pretrained model
            if FLAGS.model_to_load.lower() == 'last':
                # Load the last model trained by ourselves
                load_user_model(sess, savedir)
            elif FLAGS.model_to_load.lower() == 'pretrained':
                # Load the pretrained model given by TensorFlow official
                exclusions = ['resnet_v1_{}/logits'.format(FLAGS.resnet_model), 'predictions']
                resnet_except_logits = slim.get_variables_to_restore(exclude=exclusions)
                path = FLAGS.official_model_path.format(FLAGS.resnet_model)
                init_fn = slim.assign_from_checkpoint_fn(path, resnet_except_logits,
                                                         ignore_missing_vars=True)
                init_fn(sess)
                print('[*]Pretrained model %s successfully loaded' % path)
            elif FLAGS.model_to_load.lower() == 'new':
                pass
            e_savedir = os.path.join(savedir, 'evaluate')
            print('[*]Evaluate logs and summaries will be saved into %s' % os.path.abspath(e_savedir))
            os.makedirs(e_savedir, exist_ok=True)
            accuracies = evaluate(sess, acc_t, probs, batch_ys, batch_names, summary_t, e_savedir, loops=data_count,
                                  logiter=10)
            print('[*]The model accuracy is {}'.format(sum(accuracies) / len(accuracies)))


if __name__ == '__main__':
    tf.app.run()
