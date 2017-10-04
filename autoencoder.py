import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from parameters import *
import stimulus
import pickle
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
        # 24 24 64

        k3  = [3, 3]

        drop_pct = 1.0


        ### Encoder
        conv1 = tf.layers.conv2d(inputs=data1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.dropout(conv2, drop_pct)
        # Now 64x64x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        # Now 16x16x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.dropout(conv4, drop_pct)
        # Now 16x16x32
        maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')
        # Now 8x8x32
        #conv5 = tf.layers.conv2d(inputs=maxpool4, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        #conv6 = tf.layers.conv2d(inputs=conv5, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 8x8x16
        #maxpool6 = tf.layers.max_pooling2d(conv6, pool_size=(2,2), strides=(2,2), padding='same')
        # Now 4x4x16

        self.spike_loss = tf.reduce_mean(tf.square(maxpool4))

        ### Decoder
        upsample1 = tf.image.resize_images(maxpool4, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 8x8x16
        deconv1 = tf.layers.conv2d(inputs=upsample1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        deconv2 = tf.layers.conv2d(inputs=deconv1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 8x8x16
        upsample2 = tf.image.resize_images(deconv2, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 16x16x16
        deconv3 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        deconv4 = tf.layers.conv2d(inputs=deconv3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 32x32x64
        #upsample4 = tf.image.resize_images(deconv4, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 32x32x32
        #deconv5 = tf.layers.conv2d(inputs=upsample4, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        #deconv6 = tf.layers.conv2d(inputs=deconv5, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # Now 32x32x64

        self.y = tf.layers.conv2d(inputs=deconv4, filters=3, kernel_size=(3,3), padding='same', activation=None)


        """
        conv1 = conv(data1, 32, k3, strides=1, activation=tf.nn.relu)
        conv1 = tf.layers.dropout(conv1, drop_pct)
        conv2 = conv(conv1, 32, k3, strides=2, activation=tf.nn.relu)
        conv2 = tf.layers.dropout(conv2, drop_pct)
        conv3 = conv(conv2, 64, k3, strides=1, activation=tf.nn.relu)
        conv3 = tf.layers.dropout(conv3, drop_pct)
        conv4 = conv(conv3, 64,  k3, strides=2, activation=tf.nn.relu)
        conv4 = tf.layers.dropout(conv4, drop_pct)
        print('CONV4', conv4)

        deco1 = deco(conv4, 64,  k3, strides=2, activation=tf.nn.relu)
        deco2 = deco(deco1, 32, k3, strides=1, activation=tf.nn.relu)
        deco3 = deco(deco2, 16, k3, strides=1, activation=tf.nn.relu)
        deco4 = deco(deco3, 3,  k3, strides=2, activation=tf.nn.relu)
        self.y = deco4
        """
        print('Y', self.y.shape)
        #print('Encoding -', data1.shape)
        #print('Decoding -', x1.shape , 'from', conv4.shape)
        #print('Result   -', deco4.shape)

    def optimize(self):

        self.loss = tf.reduce_mean(tf.square(self.y_hat - self.y)) + 0.1*self.spike_loss

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
        with tf.device('/gpu:0'):
            model   = Autoencoder(x)
            sess.run(tf.global_variables_initializer())
        t_start = time.time()

        s = stimulus.Stimulus()
        for i in range(2000):
            task_id = np.random.randint(10)
            y_hat, _, _ = s.make_batch(task_id, test=False)
            _, loss, spike_loss, y = sess.run([model.train_op, model.loss, model.spike_loss, model.y], {x:y_hat})

            if i%100 == 0:
                print('\n Iter. | Task | Loss')
                print('----------------------------')
            print('', str(i).ljust(5), '|', str(task_id).ljust(4), '|', loss, '|', spike_loss)

            if i%100 == 0:
                print(y_hat.shape, y.shape)
                f, axarr = plt.subplots(2, 2)
                axarr[0,0].imshow(y_hat[0], aspect='auto', interpolation='none')
                axarr[0,0].set_title('Original 0')
                axarr[0,1].imshow(y[0], aspect='auto', interpolation='none')
                axarr[0,1].set_title('Reconstructed 0')
                axarr[1,0].imshow(y_hat[1], aspect='auto', interpolation='none')
                axarr[1,0].set_title('Original 1')
                axarr[1,1].imshow(y[1], aspect='auto', interpolation='none')
                axarr[1,1].set_title('Reconstructed 1')
                plt.savefig('./encoder_testing/'+str(i))

                W = {}
                for var in tf.trainable_variables():
                    W[var.op.name] = var.eval()
                pickle.dump(W, open('./encoder_testing/conv_weights.pkl','wb'))




main()
