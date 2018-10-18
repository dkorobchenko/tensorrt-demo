'''
Build and train a ConvNet using Tensorflow
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import data_provider
import model

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

### Settings

TRAIN_DATA_ROOT = '/imagenet/train/' # ADJUST
TRAIN_LIST_FILE = '/imagenet/train.txt' # ADJUST
BATCH_SIZE = 64  # ADJUST
CROP_SIZE = 224 # ADJUST
CLASSES = [281, 239] # [Tabby cat; Bernese mountain dog]
LR_START = 0.01 # ADJUST
LR_END = LR_START / 1e4 # ADJUST
MOMENTUM = 0.9 # ADJUST
NUM_EPOCHS = 1000 # ADJUST
OUTPUT_ROOT = 'data/checkpoints/vggA_BN' # ADJUST
LOG_EVERY_N = 10 # ADJUST

### Prepare training data queue

train_image, train_label, num_samples = data_provider.imagenet_data(
    TRAIN_DATA_ROOT,
    TRAIN_LIST_FILE,
    batch_size=BATCH_SIZE,
    crop_size=(CROP_SIZE, CROP_SIZE),
    classes=CLASSES,
)
iters = NUM_EPOCHS * num_samples // BATCH_SIZE
print('Number of train samples: {}'.format(num_samples))
print('Number of train iterations: {}'.format(iters))

### Build network graph

with tf.variable_scope('net'):
    logits = model.model(train_image, is_training=True)

### Add loss function

with tf.name_scope('loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=train_label, logits=logits)
tf.summary.scalar('loss/loss', loss)

### Setup training parameters and structures

global_step = tf.train.get_or_create_global_step()
lr = tf.train.polynomial_decay(LR_START, global_step, iters, LR_END)
tf.summary.scalar('learning_rate', lr)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=MOMENTUM)
train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

### Run training loop

with tf.train.MonitoredTrainingSession(checkpoint_dir=OUTPUT_ROOT) as sess:
    start_iter = sess.run(global_step)
    for it in range(start_iter, iters):
        loss_value = sess.run(train_op)
        if it % LOG_EVERY_N == 0:
            print('[{} / {}] loss_net = {}'.format(it, iters, loss_value))
