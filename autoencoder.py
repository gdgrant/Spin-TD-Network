import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from parameters import *
import stimulus
import os, time

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Autoencoder:

    def __init__(self, input_data):
        self.y_hat = input_data
        self.test()
        self.optimize()

    def test(self):

        #tf.layers.conv2d(INPUTS, FILTERS, KERNEL_SIZE, STRIDES, PADDING)
        conv = tf.layers.conv2d
        deco = tf.layers.conv2d_transpose
        data1 = self.y_hat


        conv1 = conv(data1, 32, [9, 9], strides=1, activation=tf.nn.relu)
        conv2 = conv(conv1, 32, [9, 9], strides=1, activation=tf.nn.relu)
        conv3 = conv(conv2, 16, [5, 5], strides=1, activation=tf.nn.relu)
        conv4 = conv(conv3, 16, [5, 5], strides=1, activation=tf.nn.relu)
        conv5 = conv(conv4, 8,  [3, 3], strides=1, activation=tf.nn.relu)
        conv6 = conv(conv5, 8,  [3, 3], strides=1, activation=tf.nn.relu)

        x = tf.reshape(conv6, [par['batch_size'],-1])

        deco1 = deco(conv6, 8,  [3, 3], strides=1, activation=tf.nn.relu)
        deco2 = deco(deco1, 8,  [3, 3], strides=1, activation=tf.nn.relu)
        deco3 = deco(deco2, 16, [5, 5], strides=1, activation=tf.nn.relu)
        deco4 = deco(deco3, 16, [5, 5], strides=1, activation=tf.nn.relu)
        deco5 = deco(deco4, 32, [9, 9], strides=1, activation=tf.nn.relu)
        deco6 = deco(deco5, 3,  [9, 9], strides=1, activation=tf.nn.relu)
        self.y = deco6

        print('Encoding -', data1.shape)
        print('Decoding -', x.shape , 'from', conv6.shape)
        print('Result   -', deco6.shape)

    def optimize(self):
        self.loss = tf.reduce_mean(tf.square(self.y_hat - self.y))

        optimizer = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
        gradients = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(gradients)


def main():

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    x  = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'stim')
    print('Batch size:', par['batch_size'])
    print('Input size:', 32*32*3, '\n')

    with tf.Session() as sess:
        model   = Autoencoder(x)
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        s = stimulus.Stimulus()
        print(' Iter. | Task | Loss')
        print('----------------------------')
        for i in range(1000):
            task_id = np.random.randint(11)
            stim, _, _ = s.make_batch(task_id, test=False)
            _, loss = sess.run([model.train_op, model.loss], {x:stim})
            print('', str(i).ljust(5), '|', str(task_id).ljust(4), '|', loss)

main()
