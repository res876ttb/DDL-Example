#!/usr/bin/env python3

import argparse, os, json
parser = argparse.ArgumentParser()
parser.add_argument('--worker', help='Set worker in ClusterSpec.', default='10.18.18.1:2223,10.18.18.2:2223', type=str)
parser.add_argument('--ps', help='Set ps in ClusterSpec.', default='10.18.18.1:2222', type=str)
parser.add_argument('--numgpus', help='Number of GPUs.', default=2, type=int)
parser.add_argument('jobname', help='Set job type: ps or worker.')
parser.add_argument('index', help='Set job index.', type=int)
args = parser.parse_args()

import tensorflow as tf
mnist = tf.keras.datasets.mnist

# Set environment
environ = {
  'cluster': {
    'worker': args.worker.split(','),
    'ps': args.ps.split(',')
  },
  'task': {
    'type': args.jobname,
    'index': args.index,
  }
}
print(environ)
os.environ['TF_CONFIG'] = json.dumps(environ)

# Load data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Configure distribute strategy
ps_strategy = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=args.numgpus)

# Build model
with ps_strategy.scope():
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

# Training
model.fit(x_train, y_train, epochs=5, batch_size=32)
