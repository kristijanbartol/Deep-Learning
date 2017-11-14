import tensorflow as tf
import tensorflow.contrib.layers as layers

from skimage import io
import numpy as np
import math
import os

# conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)

DATA_DIR = '/home/kristijan/FER/DU/cnn/datasets/CIFAR10/'
SAVE_DIR = '/home/kristijan/FER/DU/cnn/source/out/'

config = dict()
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-2


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    # special case when drawing MNIST
    num_channels = w.shape[2] if w.shape[2] in [3, 4] else 1
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, 3 if num_channels == 1 else num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    io.imsave(os.path.join(save_dir, filename), img)


def build_model(inputs, num_classes, config):
    weight_decay = config['weight_decay']
    conv1sz = 16
    conv2sz = 32
    fc1sz = 512

    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        pass


def evaluate():
    pass
