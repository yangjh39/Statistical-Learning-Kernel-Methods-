# -*- coding: utf-8 -*-
# @Time    : 2018/3/4 16:15
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

# Input Layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First full connection hidden layer
W_fc1 = weight_variable([784, 1600])
b_fc1 = bias_variable([1600])

input_flat = tf.reshape(x_image, [-1, 28*28])
h_fc1 = tf.nn.relu(tf.matmul(input_flat, W_fc1) + b_fc1)

# Second full connection hidden layer
W_fc2 = weight_variable([1600, 1600])
b_fc2 = bias_variable([1600])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Dropping layer
keep_prob = tf.placeholder("float")
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Output layer(Softmax layer)
W_fc3 = weight_variable([1600, 10])
b_fc3 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

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

f0_mlp = mnist.test.images

f1_mlp = h_fc1.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
f1_mlp = f1_mlp.reshape(10000, -1)

f2_mlp = h_fc2.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
f2_mlp = f2_mlp.reshape(10000, -1)

f3_mlp = y_conv.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

label_mlp = mnist.test.labels

pd.DataFrame(f0_mlp).to_csv('f0_mlp.csv')
pd.DataFrame(f1_mlp).to_csv('f1_mlp.csv')
pd.DataFrame(f2_mlp).to_csv('f2_mlp.csv')
pd.DataFrame(f3_mlp).to_csv('f3_mlp.csv')
pd.DataFrame(label_mlp).to_csv('label_mlp.csv')
