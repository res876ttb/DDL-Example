#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

def model_fn(features, labels, mode):
  # CNN
  conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=tf.reshape(features['tf_x'], [-1, 28, 28, 1]),
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

  # Compute prediction
  predictedClass = tf.argmax(output, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'class': predictedClass
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute loss
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output)           # compute cost

  # Create training op
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_global_step()
    train_op = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  # Compute evaluation metrics.
  eval_metric_ops = {
    'accuracy': tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
                labels=tf.argmax(labels, axis=1), predictions=tf.argmax(output, axis=1),)
  }
  return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(_):
  # Create estimator
  classifier = tf.estimator.Estimator(model_fn=model_fn)

  # Do training
  train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'tf_x': mnist.train.images}, 
    y=mnist.train.labels, 
    num_epochs=None, shuffle=True, 
    batch_size=BATCH_SIZE
  )
  classifier.train(input_fn=train_input_fn, steps=600)

  # predict
  test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'tf_x': test_x}, y=test_y, num_epochs=1, shuffle=False
  )
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy: %f' % scores['accuracy'])

if __name__ == '__main__':
  tf.app.run()