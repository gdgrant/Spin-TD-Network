import tensorflow as tf
import numpy as np

from parameters import *

import os, time

###################
### Model setup ###
###################

class Model:

    def __init__(self, input_data, td_data, target_data, learning_rate, stim_train):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data     = tf.unstack(input_data, axis=1)
        self.td_data        = tf.unstack(td_data, axis=1)
        self.target_data    = tf.unstack(target_data, axis=1)
        self.learning_rate  = learning_rate
        self.stim_train     = stim_train
        self.td_train       = not stim_train

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    def run_model(self):
        with tf.variable_scope('parameters'):
            W_stim = tf.get_variable('W_stim',   initializer=np.float32(par['w_stim0']), trainable=self.stim_train)
            W_td   = tf.get_variable('W_td',     initializer=np.float32(par['w_td0']),   trainable=self.td_train)
            W_out  = tf.get_variable('W_out',    initializer=np.float32(par['w_out0']),  trainable=self.stim_train)

            b_hid  = tf.get_variable('b_hid',    initializer=np.float32(par['b_hid0']),  trainable=self.stim_train)
            b_out  = tf.get_variable('b_out',    initializer=np.float32(par['b_out0']),  trainable=self.stim_train)

        ### Tensordot specs:
        # W_in   = [n_hidden_neurons x dendrites_per_neuron x n_input_neurons]
        # input  = [n_input_neurons x batch_train_size]
        # output = [n_hidden_neurons x dendrites_per_neuron x batch_train_size]

        d_stim  = tf.nn.relu(tf.tensordot(W_stim, stim_in, ([2],[0])))
        d_td    = tf.nn.relu(tf.tensordot(W_td, td_in, ([2],[0])))
        d_state = d_stim * d_td

        h_state = tf.reduce_sum(d_state, 1)

        self.y = tf.matmul(tf.nn.relu(W_out),h_state)+b_out

    def optimize(self):
        pass


def main():
    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, td_data, target_data, learning_rate, stim_train
    x   = tf.placeholder(tf.float32, [par['n_stim'], par['batch_train_size']], 'stim')
    td  = tf.placeholder(tf.float32, [par['n_td'], par['batch_train_size']], 'TD')
    lr  = tf.placeholder(tf.float32, [], 'learning_rate')
    st  = tf.placeholder(tf.bool, [], 'training_selection')

    with tf.Session as sess:
        model   = Model(x, td, learnin_rate, stim_train)
        init    = tf.global_variables_initializer()
        t_start = time.time()
        sess.run(init)

        stim_in  = np.ones([par['n_stim'], par['batch_size']])
        td_in    = np.ones([par['n_td'], par['batch_size']])
        rate     = 0.001
        train    = True

        sess.run(model.y, feed_dict={x:stim_in, td:td_in, lr:rate, st:train})
