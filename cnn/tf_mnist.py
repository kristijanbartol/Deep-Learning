import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

from tf_cifar10 import draw_conv_filters

DATA_DIR = '/home/kristijan/FER/DU/cnn/datasets/MNIST/'
SAVE_DIR = '/home/kristijan/FER/DU/cnn/source/out/MNIST/'

config = dict()
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-2
# TODO: add lr_policy for gradient descent optimizer
config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}


def build_model(inputs, num_classes, config):
    weight_decay = config['weight_decay']
    conv1sz = 16
    conv2sz = 32
    fc1sz = 512

    with tf.name_scope('reshape'):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])

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
        logits_ = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')

    return logits_


def train(sess, train_set, accuracy_f, train_f, loss_f, config):
    batch_size = config['batch_size']
    max_epoch = config['max_epochs']

    num_examples = train_set.images.shape[0]
    num_batches = num_examples // batch_size

    for epoch in range(1, max_epoch + 1):
        # TODO: how to permute the dataset after each epoch?
        for i in range(num_batches):
            batch = train_set.next_batch(batch_size)
            sess.run(train_f, feed_dict={x: batch[0], y: batch[1]})
            if i % 5 == 0:
                loss = sess.run(loss_f, feed_dict={x: batch[0], y: batch[1]})
                print('epoch %d/%d, step %d/%d, batch loss = %.2f'
                      % (epoch, max_epoch, i * batch_size, num_examples, loss))
            if i % 100 == 0:
                conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
                conv1_weights = conv1_var.eval(session=sess)
                draw_conv_filters(epoch, i, conv1_weights, SAVE_DIR)
            if i > 0 and i % 50 == 0:
                accuracy = sess.run(accuracy_f, feed_dict={x: batch[0], y: batch[1]})
                print('Train accuracy = %.2f' % accuracy)


mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

logits = build_model(x, 10, config)

with tf.name_scope('loss'):
    cross_entropy_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits))

with tf.name_scope('optimizer'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    train(session, mnist.train, accuracy, train_op, cross_entropy_loss, config)

    print('test accuracy {}'.format(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})))
