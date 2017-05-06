import tensorflow as tf
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
import numpy as np
import sys

import matplotlib.pyplot as plt
import scipy.misc

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

n_nodes_hl1 = 1000
n_nodes_hl2 = 1500
n_nodes_hl3 = 2000


n_input = 3072
n_classes = 10
batch_size = 64

print("&&&&&&&&&&")


def to_categorical(y_train, nb_classes):
    """ to_categorical.
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.
    """
    y_train = np.asarray(y_train, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y_train)+1
    y_test = np.zeros((len(y_train), nb_classes))
    for i in range(len(y_train)):
        y_test[i, y_train[i]] = 1.
    return y_test


print("#######")

def onehot(l):
    m = max(l)
    return [[1 if i == r else 0 for r in range(m + 1)] for i in l]


X_train = X_train.reshape(-1,3072)
y_train = np.array(onehot(list(y_train.reshape(-1))))

X_test = X_test.reshape(-1,3072)
y_test = np.array(onehot(list(y_test.reshape(-1))))




print("$#$#$#$#")


x = tf.placeholder('float', shape=[None, n_input])
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([3072, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

    
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(n_classes)):
                epoch_x = np.array(X_train[:])
                epoch_y = np.array(y_train[:])
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 1})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x: X_train, y: y_train}))
    
train_neural_network(x)
