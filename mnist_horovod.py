#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import horovod.tensorflow as hvd

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

def main(_):
    mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
    test_x = mnist.test.images[:2000]
    test_y = mnist.test.labels[:2000]

    # plot one example
    print(mnist.train.images.shape)     # (55000, 28 * 28)
    print(mnist.train.labels.shape)   # (55000, 10)

    # Init horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
    image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

    # get global step
    global_step = tf.train.get_or_create_global_step()

    # CNN
    conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
        inputs=image,
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )           # -> (28, 28, 16)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2,
    )           # -> (14, 14, 16)
    conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
    flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
    output = tf.layers.dense(flat, 10)              # output layer

    accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
        labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
    optimizer = tf.train.AdamOptimizer(LR * hvd.size()) # Increase learning rate
    optimizer = hvd.DistributedOptimizer(optimizer) # Add Horovod Distributed Optimizer
    train_op = optimizer.minimize(loss, global_step=global_step)

    # define hooks
    hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.StopAtStepHook(last_step=600 // hvd.size()),
    ]
    if hvd.rank() == 0:
        hooks.append(tf.train.LoggingTensorHook(
            tensors={'step': global_step, 'loss': loss},every_n_iter=10))

    # Use MonitoredTrainingSession
    with tf.train.MonitoredTrainingSession(config=config,
                                           hooks=hooks) as mon_sess:
        start = BATCH_SIZE * hvd.rank()
        end = BATCH_SIZE * (hvd.rank() + 1)
        print(start, end, BATCH_SIZE * hvd.size())
        while not mon_sess.should_stop():
            b_x, b_y = mnist.train.next_batch(BATCH_SIZE * hvd.size())
            b_x, b_y = b_x[start:end], b_y[start:end]
            mon_sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})

if __name__ == '__main__':
    tf.app.run()