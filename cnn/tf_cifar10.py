import tensorflow as tf
import tensorflow.contrib.layers as layers

import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
import math
import os
import pickle
import time
import numpy as np

DATA_DIR = '/home/kristijan/FER/DU/cnn/datasets/CIFAR10/'
SAVE_DIR = '/home/kristijan/FER/DU/cnn/source/out/CIFAR10/'

config = dict()
config['max_epochs'] = 30
config['batch_size'] = 50
config['weight_decay'] = 1e-2


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    # special case when drawing MNIST
    num_channels_ = w.shape[2] if w.shape[2] in [3, 4] else 1
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels_, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, 3 if num_channels_ == 1 else num_channels_])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def build_model(inputs, num_classes_, weight_decay):
    conv1sz = 16
    conv2sz = 32
    fc1sz = 256
    fc2sz = 128

    with tf.name_scope('reshape'):
        inputs = tf.reshape(inputs, [-1, 32, 32, 3])

    # conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() ->
    # -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)
    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.convolution2d(inputs, conv1sz,
                                   weights_initializer=layers.variance_scaling_initializer(), scope='conv1')
        with tf.contrib.framework.arg_scope([layers.max_pool2d],
                                            kernel_size=5, stride=1, padding='SAME'):
            net = layers.max_pool2d(net, scope='max_pool1')
            net = layers.convolution2d(net, conv2sz, scope='conv2')
            net = layers.max_pool2d(net, scope='max_pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.flatten(net, scope='flatten')
        net = layers.fully_connected(net, fc1sz, scope='fc1')
        net = layers.fully_connected(net, fc2sz, scope='fc2')
        logits_ = layers.fully_connected(net, num_classes_, activation_fn=None, scope='logits')

    return logits_


def prepare_cifar(img_height_, img_width_, num_channels_, data_dir):
    train_x_ = np.ndarray((0, img_height_ * img_width_ * num_channels_), dtype=np.float32)
    train_y_ = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(data_dir, 'data_batch_%d' % i))
        train_x_ = np.vstack((train_x_, subset['data']))
        train_y_ += subset['labels']
    train_x_ = train_x_.reshape((-1, num_channels_, img_height_, img_width_)).transpose(0, 2, 3, 1)
    train_y_ = np.array(train_y_, dtype=np.int32)

    subset = unpickle(os.path.join(data_dir, 'test_batch'))
    test_x_ = subset['data'].reshape((-1, num_channels_, img_height_, img_width_)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y_ = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x_, train_y_ = shuffle_data(train_x_, train_y_)
    valid_x_ = train_x_[:valid_size, ...]
    valid_y_ = train_y_[:valid_size, ...]
    train_x_ = train_x_[valid_size:, ...]
    train_y_ = train_y_[valid_size:, ...]
    data_mean = train_x_.mean((0, 1, 2))
    data_std = train_x_.std((0, 1, 2))

    train_x_ = (train_x_ - data_mean) / data_std
    valid_x_ = (valid_x_ - data_mean) / data_std
    test_x_ = (test_x_ - data_mean) / data_std

    return train_x_, valid_x_, test_x_, train_y_, valid_y_, test_y_


def evaluate(run_ops_, x, y):
    eval_x = np.array(x).reshape(-1, img_width * img_height * num_channels)
    # convert to one-hot
    temp_eval_y = np.zeros((y.size, 10))
    temp_eval_y[np.arange(y.size), y] = 1
    eval_y = temp_eval_y.copy().reshape(-1, num_classes)

    # if CPU:
    if True:
        eval_x = eval_x[:500, :]
        eval_y = eval_y[:500, :]
    # if GPU: (feed full datasets)
    else:
        pass
    return sess.run(run_ops_, feed_dict={node_x: eval_x, node_y: eval_y})


img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x, valid_x, test_x, train_y, valid_y, test_y = prepare_cifar(img_height, img_width, num_channels, DATA_DIR)

num_epochs = config['max_epochs']
batch_size = config['batch_size']

plot_data = dict()
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

node_x = tf.placeholder(tf.float32, [None, img_width * img_height * num_channels])
node_y = tf.placeholder(tf.float32, [None, num_classes])

logits = build_model(node_x, num_classes, config['weight_decay'])

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=node_y, logits=logits))

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss)

# with tf.name_scope('confusion_matrix'):
#    tf.contrib.metrics.confusion_matrix(labels=node_y, predictions=tf.argmax(logits, 1))

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(node_y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_num in range(1, num_epochs + 1):
        train_x, train_y = shuffle_data(train_x, train_y)
        num_batches = train_x.shape[0] // batch_size
        for step in range(1, num_batches + 1):
            offset = step * batch_size
            batch_x = np.array(train_x[offset:(offset + batch_size), ...])\
                .reshape(-1, img_width * img_height * num_channels)
            batch_y = train_y[offset:(offset + batch_size)]
            # convert to one-hot
            temp_batch_y = np.zeros((batch_size, num_classes))
            temp_batch_y[np.arange(batch_size), batch_y] = 1
            batch_y = temp_batch_y.copy().reshape(-1, num_classes)
            feed_dict = {node_x: batch_x, node_y: batch_y}
            start_time = time.time()
            run_ops = [train_op, loss, logits]
            ret_val = sess.run(run_ops, feed_dict=feed_dict)
            _, loss_val, logits_val = ret_val
            duration = time.time() - start_time
            if step % 50 == 0:
                sec_per_batch = float(duration)
                format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
                print(format_str % (epoch_num, step, num_batches, loss_val, sec_per_batch))
            if step % 100 == 0:
                conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
                conv1_weights = conv1_var.eval(session=sess)
                draw_conv_filters(epoch_num, step, conv1_weights, SAVE_DIR)

        run_ops = [loss, accuracy]
        train_loss, train_acc = evaluate(run_ops, train_x, train_y)
        print('Train error %.2f, Train accuracy %.2f' % (train_loss, train_acc))
        valid_loss, valid_acc = evaluate(run_ops, valid_x, valid_y)
        print('Validation error %.2f, Validation accuracy %.2f' % (valid_loss, valid_acc))
        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [valid_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [valid_acc]
        plot_data['lr'] += [optimizer._lr]
        plot_training_progress(SAVE_DIR, plot_data)
        print('\n')
