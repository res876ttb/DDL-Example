#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--numgpus', help='Number of GPUs to train.', default=2, type=int)
args = parser.parse_args()

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model = tf.keras.utils.multi_gpu_model(model, 
  gpus=args.numgpus, cpu_merge=True, cpu_relocation=False)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)
model.evaluate(x_test, y_test)
