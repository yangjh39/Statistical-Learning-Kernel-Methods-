# -*- coding: utf-8 -*-
# @Time    : 2018/2/26 0:37
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import input_data
import time

# Import Minist Dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Placeholder for input images and output predicted labels
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

sess = tf.InteractiveSession()


# Initial weight function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Initial bias function
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First convolution layer
W_conv1 = weight_variable([5, 5, 1, 100])
b_conv1 = bias_variable([100])

# Input Layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First pooling layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolution layer
W_conv2 = weight_variable([5, 5, 100, 100])
b_conv2 = bias_variable([100])

# Second pooling layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Full connection layer
W_fc1 = weight_variable([7 * 7 * 100, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*100])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropping layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer(Softmax layer)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and evaluate the CNN model
# loss function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# Train algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Initialization
sess.run(tf.initialize_all_variables())

train_accuracy = 0.
i = 0
start = time.time()

while train_accuracy < 0.975:
    batch = mnist.test.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        print("Time consuming: %d sec" % (time.time() - start))
        start = time.time()
    i += 1
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

f0_cnn = mnist.test.images

f1_cnn = h_pool1.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
f1_cnn = f1_cnn.reshape(10000, -1)

f2_cnn = h_pool2.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
f2_cnn = f2_cnn.reshape(10000, -1)

f3_cnn = y_conv.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

label_cnn = mnist.test.labels

pd.DataFrame(f0_cnn).to_csv('f0_cnn.csv')
pd.DataFrame(f1_cnn).to_csv('f1_cnn.csv')
pd.DataFrame(f2_cnn).to_csv('f2_cnn.csv')
pd.DataFrame(f3_cnn).to_csv('f3_cnn.csv')
pd.DataFrame(label_cnn).to_csv('label_cnn.csv')
