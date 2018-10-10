import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_v1
import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
from tabulate import tabulate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 可以使用的GPU
gpu_net = '/gpu:0'
gpu_loss = '/gpu:0'
# TODO: output the results to file "results.csv" and compute the confusion matrix

flags = tf.app.flags
ERROR_FLAG = 0

flags.DEFINE_integer('resnet_model', 50, 'The layers count of resnet: 50, 101 or 152')
flags.DEFINE_string('pretrain_path', './pretrained/resnet_v1_{}.ckpt', '')
flags.DEFINE_string('data_dir', './datasets/separate/', '')
flags.DEFINE_string('model_dir', './checkpoints/models_{}_{}_{}-',
                    'Folder path ends with 1.unpre/pre* 2.optimizer 3.loss type')
flags.DEFINE_string('model_name', 'resnet.model', 'model name')
flags.DEFINE_string('logdir', './logs/logs_{}_{}_{}-', 'Folder path ends with 1.unpre/pre* 2.optimizer 3.loss type')
flags.DEFINE_string('optimizer', 'SGD', 'Either Adam or SGD')
flags.DEFINE_string('losstype', 'cross_entropy', 'Either ordinal or cross_entropy')
flags.DEFINE_integer('batch_size', 16 * 8, '')
flags.DEFINE_integer('epoch', 4, 'Count of epoch')
flags.DEFINE_integer('loops', 1000, 'Number of iteration in training')
flags.DEFINE_float('learning_rate', 8e-3, 'Initial learning rate')
flags.DEFINE_float('regularize_scale', 1e-5, 'L2 regularizer scale')
flags.DEFINE_boolean('is_training', True, 'Train or evaluate?')
flags.DEFINE_boolean('pretrained', False,
                     'Whether using the pretrained model given by TensorFlow or not')
flags.DEFINE_boolean('use_last_model', False,
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


def __parse_dir(flags, dir_to_parse):
    if flags.pretrained:
        mtype = 'pre' + str(flags.resnet_model)
    else:
        mtype = 'unpre' + str(flags.resnet_model)
    dir = dir_to_parse.format(mtype, flags.optimizer, flags.losstype)
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


def _get_model_dir(flags=FLAGS):
    return __parse_dir(flags, flags.model_dir)


def _get_log_dir(flags=FLAGS):
    return __parse_dir(flags, flags.logdir)


def save(sess, model_path, counter):
    saver = tf.train.Saver(max_to_keep=3)
    model_dir, model_name = os.path.split(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
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


def init_reader(path=FLAGS.data_dir, batch_size=64, epoch=1, is_training=True):
    def _parse_function(xs, ys):
        x_img_str = tf.read_file(xs)
        x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(x_img_str), tf.float32)
        x_img_resized = tf.image.resize_images(x_img_decoded, size=[32, 32],
                                               method=tf.image.ResizeMethod.BILINEAR)
        x_img = tf.reshape(x_img_resized, [32, 32, 3])
        return x_img, ys

    # Processing the image filenames
    fullpath = os.path.join(path, ('train' if is_training else 'validation'))
    count = 0
    dicts = {}
    for dir in os.listdir(fullpath):
        files = os.listdir(os.path.join(fullpath, dir))
        files = [(os.path.join(fullpath, dir + '/' + f)) for f in files]
        count_sub = len(files)
        count += count_sub
        label = os.path.basename(dir)
        if label == 'nodata':
            label = 'F'
        t_label = ord(label) - ord('A')
        t_labels = [t_label] * count_sub
        dicts.update(dict(zip(files, t_labels)))
    print(' [*] {} images initialized as validation data'.format(count))
    # Initialize as a tensorflow tensor object
    xs = tf.constant(list(dicts.keys()))
    ys = tf.constant(list(dicts.values()))
    data = tf.data.Dataset.from_tensor_slices((xs, ys))
    data = data.map(_parse_function)
    if is_training:
        return data.shuffle(buffer_size=512).batch(batch_size).repeat(epoch)
    else:
        return data.batch(batch_size)


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


def train(sess, optim, loss, summaries, loop=50000, counter=tf.Variable(0, False), logiter=100, predct=None,
          labels=None):
    time_begin = datetime.now()
    logdir = _get_log_dir(FLAGS)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = tf.summary.FileWriter(logdir, sess.graph)
    for it in range(loop):
        try:
            optimval, global_counter = sess.run([optim, counter])
            if it % logiter == 0:
                sum_str, lossval = sess.run([summaries, loss])
                if predct != None and labels != None:
                    correct_prediction = tf.equal(predct, labels)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    predval, ys, accval = sess.run([predct, labels, accuracy])
                    cf = Confusion(predictions=predval, labels=ys)
                    # print('[*]accuracy:{}\n[*]predict:{}\n[*]labels:{}'.format(accval, predval, ys))
                    print('[*]accuracy:{}  Confusion Matrix:\n{}'.format(accval, cf))
                writer.add_summary(sum_str, global_counter)
                time_elapse = datetime.now() - time_begin
                time_remain = time_elapse / (it + 1) * (loop - it - 1)
                words = 'elapsed time:{} remaining time:{} iteration:{} loss:{}'.format(time_elapse, time_remain, it,
                                                                                        lossval)
                print(words)
                if lossval < 2.0:
                    save(sess,
                         os.path.join(_get_model_dir(FLAGS), 'tmp_loss{:.3f}'.format(lossval, ) + FLAGS.model_name),
                         global_counter)
        except tf.errors.InvalidArgumentError as e:
            print('An error of type tf.errors.InvalidArgumentError has been ignored...')
            print(e.message)
            continue
        except tf.errors.OutOfRangeError:
            print('Epoch reach the end...')
            break
        # global_counter += 1
    time_elapse = datetime.now() - time_begin
    print('Training finish, elapsed time %s...Trying to save the model...' % time_elapse)
    save(sess, os.path.join(_get_model_dir(FLAGS), FLAGS.model_name), global_counter)


def evaluate(sess, probabilities, labels, loop=1000, logiter=50):
    cnt = 0;
    accsum = 0.0
    cf = Confusion(headers=['A', 'B', 'C', 'D', 'E', '*'])
    prediction = tf.argmax(probabilities, axis=-1)
    accuracy, acc_update = tf.metrics.accuracy(labels, prediction)
    sess.run(tf.local_variables_initializer())
    while cnt < loop:
        cnt += 1
        try:
            accval, acc_up, probsval, predval, ys = sess.run([accuracy, acc_update, probabilities, prediction, labels])
            accsum = accsum + accval
            cf.add_data(predval, ys)
            if cnt % logiter == 0:
                print('accuracy of %d batches is %f' % (cnt, accsum / cnt))
                print('confusion matrix: {}'.format(cf))
        except tf.errors.InvalidArgumentError:
            print('An error of type tf.errors.InvalidArgumentError has been ignored...')
            continue
        except tf.errors.OutOfRangeError:
            break
        except tf.errors.NotFoundError:
            break
    return accsum / cnt


def init_loss(logits, labels, end_points=None, losstype='ordinal'):
    with tf.device(gpu_loss):
        if losstype == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.regularize_scale),
                                                         tf.trainable_variables())
            loss += reg
        elif losstype == 'ordinal':
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


def init_loss_old(logits, labels, end_points=None, losstype='ordinal'):
    if end_points is not None:
        # Definition of binary network for better classification of "*"
        # The network has only 3 layers, with the front-end being resnet_v1_*/block4
        # See the graph in tensorboard for more detailed information
        with tf.variable_scope('binary_classification_for_nodata'):
            conv_1 = slim.conv2d(end_points['resnet_v1_{}/block4'.format(FLAGS.resnet_model)], 64, [3, 3],
                                 scope='conv_1')
            conv_2 = slim.conv2d(conv_1, 1, [3, 3], scope='conv_2')
            reshaped = tf.reshape(conv_2, [FLAGS.batch_size, -1], name='reshaped')
            binary = slim.fully_connected(reshaped, 1, activation_fn=None, scope='fc_3')
            binary_labels = tf.reshape(tf.cast(tf.equal(labels, 5), tf.float32), [-1, 1])
            binary_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_labels,
                                                                  logits=binary)
            binary_loss = tf.reduce_mean(binary_loss, name='binary_loss')

    # Here we start our cross entropy loss definition
    # Note that the "nodata" class "*" is assigned with 0 loss
    if losstype == 'cross_entropy':
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        mask = 1.0 - tf.cast(tf.equal(labels, 5), tf.float32)
        return tf.reduce_mean(mask * loss, name='loss') + binary_loss, binary
    elif losstype == 'ordinal':
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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=poisson)
        mask = 1.0 - tf.cast(tf.equal(labels, 5), tf.float32)
        return tf.reduce_mean(loss * mask, name='loss') + binary_loss, binary
    else:
        raise NotImplementedError


def main(_):
    reader = init_reader(FLAGS.data_dir, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch, is_training=FLAGS.is_training)
    batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
    # param batch_xs: shape [batch_size, 32, 32, 3] type tf.float32
    # param batch_ys: shape [batch_size] type tf.int32

    num_classes = 6

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
                # param logits: shape [64, 6]
                logits, end_points = resnet(batch_xs, num_classes=num_classes, is_training=True)
                logits = tf.reshape(logits, [-1, num_classes], name='logits_2d')
                prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
            mAP = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), dtype=tf.float32))
            loss = init_loss(logits, batch_ys, end_points=end_points, losstype=FLAGS.losstype)
            mAP_sum = tf.summary.scalar('mAP', mAP)
            loss_sum = tf.summary.scalar('loss', loss)
            summaries = tf.summary.merge([mAP_sum, loss_sum])

        counter = 0
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
            counter = load_user_model(sess, _get_model_dir(FLAGS))

        global_step = tf.Variable(counter, trainable=False)
        # Exponential decay learning rate and optimizer configurations
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 100, 0.98, staircase=True)
        if 'SGD' in FLAGS.optimizer:
            optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        elif 'Adam' in FLAGS.optimizer:
            optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        else:
            optim = None
            raise NotImplementedError

        # Ready to train
        sess.run(tf.global_variables_initializer(), options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
        train(sess, optim, loss, summaries, FLAGS.loops, counter=global_step, predct=prediction, labels=batch_ys)
        print('Training finished')
    else:  # Evaluate
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            probs, end_points = resnet(batch_xs, num_classes=num_classes, is_training=False)
            probs = tf.reshape(probs, [-1, num_classes], name='probability')
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
            load_user_model(sess, _get_model_dir(FLAGS))

        sess.run(tf.global_variables_initializer())
        # TODO: Consider the binary classification...
        # which is even more tricky to be implemented...
        print('The model accuracy is {}'.format(evaluate(sess, probabilities=probs, labels=batch_ys)))


if __name__ == '__main__':
    tf.app.run()
