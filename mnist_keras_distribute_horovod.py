#!/usr/bin/env python3

import math
import keras
import tensorflow as tf
import horovod.keras as hvd
from keras import backend as K

# Initialize horovod
hvd.init()

# Pin one process on one GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config)) # Adjust keras backend config

# Adjust epoch
epochs = int(math.ceil(16 / hvd.size()))

mnist = keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(512, activation=tf.nn.relu),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Adjust learning rate based on number of GPUs
opt = keras.optimizers.Adadelta(1.0 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Broadcast initial variable states from rank 0 to ensure consistent initialization
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks)
model.evaluate(x_test, y_test)
