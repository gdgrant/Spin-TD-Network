import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from parameters import *
import stimulus
import os, time
import matplotlib.pyplot as plt

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

        k9  = [9, 9]
        k7  = [7, 7]

        conv1 = conv(data1, 32, k9, strides=1, activation=tf.nn.relu)
        conv2 = conv(conv1, 16, k9, strides=1, activation=tf.nn.relu)
        conv3 = conv(conv2, 16, k7, strides=1, activation=tf.nn.relu)
        conv4 = conv(conv3, 8,  k7, strides=1, activation=tf.nn.relu)

        x = tf.reshape(conv4, [par['batch_size'],-1])

        deco1 = deco(conv4, 8,  k7, strides=1, activation=tf.nn.relu)
        deco2 = deco(deco1, 16, k7, strides=1, activation=tf.nn.relu)
        deco3 = deco(deco2, 16, k9, strides=1, activation=tf.nn.relu)
        deco4 = deco(deco3, 3,  k9, strides=1, activation=tf.nn.relu)
        self.y = deco4

        print('Encoding -', data1.shape)
        print('Decoding -', x.shape , 'from', conv4.shape)
        print('Result   -', deco4.shape)

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
        for i in range(1000):
            task_id = np.random.randint(11)
            y_hat, _, _ = s.make_batch(task_id, test=False)
            _, loss, y = sess.run([model.train_op, model.loss, model.y], {x:y_hat})

            if i%100 == 0:
                print('\n Iter. | Task | Loss')
                print('----------------------------')
            print('', str(i).ljust(5), '|', str(task_id).ljust(4), '|', loss)

            if i%20 == 0:
                f, axarr = plt.subplots(2, 2)
                axarr[0,0].imshow(y_hat[0])
                axarr[0,0].set_title('Original 0')
                axarr[0,1].imshow(y[0])
                axarr[0,1].set_title('Reconstructed 0')
                axarr[1,0].imshow(y_hat[1])
                axarr[1,0].set_title('Original 1')
                axarr[1,1].imshow(y[1])
                axarr[1,1].set_title('Reconstructed 1')
                plt.savefig('./encoder_testing/'+str(i))


main()
