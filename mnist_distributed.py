#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('server', help='Parameter servers, seperated by commas.')
parser.add_argument('worker', help='Workers, seperated by commas.')
parser.add_argument('jobname', help='Name of this tensorflow job.', type=str)
parser.add_argument('index', help='Index of job.', type=int)
args = parser.parse_args()

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

# 1. Define ClusterSpec and Server
cluster = tf.train.ClusterSpec({
  "ps": args.server.split(','),
  "worker": args.worker.split(',')
})
server = tf.train.Server(
  cluster, 
  job_name=args.jobname,
  task_index=args.index
)

if args.jobname == 'ps':
  server.join()
  exit()

# 2. Assign model varaibles to ps and ops to workers
with tf.device(tf.train.replica_device_setter(
  cluster=cluster, 
  worker_device="/job:worker/task:%d" % args.index
)):
  # create global step
  global_step = tf.train.get_or_create_global_step()

  mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
  test_x = mnist.test.images[:2000]
  test_y = mnist.test.labels[:2000]

  tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
  image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
  tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

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

  loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
  accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
      labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
  init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op

  optimizer = tf.train.AdamOptimizer(LR) # remove minimize
  optimizer = tf.train.SyncReplicasOptimizer( # Use synchronous training here
    optimizer, 
    replicas_to_aggregate=len(args.worker.split(',')),
    total_num_replicas=len(args.worker.split(','))
  )

  # 3. Configure and launch a tf.train.MonitoredTrainingSession
  hooks = [
    tf.train.StopAtStepHook(last_step=600),
    optimizer.make_session_run_hook((args.index == 0))
  ]
  train_op = optimizer.minimize(
      loss, 
      global_step=global_step
  )
  with tf.train.MonitoredTrainingSession(
    master=server.target,
    is_chief=(args.index == 0),
    hooks=hooks
  ) as sess:
    sess.run(init_op)     # initialize var in graph

    # for step in range(6000):
    while True:
      b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
      _, loss_, step = sess.run([train_op, loss, global_step], {tf_x: b_x, tf_y: b_y})
      # print("Job %s, global step: %d" % (args.jobname, step))
      if sess.should_stop():
        break
      if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
      else:
        print('Step:', step, '| train loss: %.4f' % loss_)
