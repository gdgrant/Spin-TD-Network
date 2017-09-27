import tensorflow as tf
import numpy as np

from parameters import *
import os, time

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################

class Model:

    def __init__(self, input_data, td_data, target_data, learning_rate, train_set):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data     = input_data
        self.td_data        = td_data
        self.target_data    = target_data
        self.learning_rate  = learning_rate

        # Choose whether to train TD or stim
        self.select_trainable(train_set)

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def select_trainable(self, train_set):
        """
        If train_set is true, train the network in general.
        If train_set is false, train just W_td.
        """
        def tr():
            self.stim_train = True
            self.td_train   = False
            return 0
        def fa():
            self.stim_train = False
            self.td_train   = True
            return 0

        tf.cond(train_set, tr, fa)


    def run_model(self):
        with tf.variable_scope('parameters'):
            W_stim = tf.get_variable('W_stim',   initializer=par['w_stim0'], trainable=self.stim_train)
            W_td   = tf.get_variable('W_td',     initializer=par['w_td0'],   trainable=self.td_train)
            W_out  = tf.get_variable('W_out',    initializer=par['w_out0'],  trainable=self.stim_train)

            b_hid  = tf.get_variable('b_hid',    initializer=par['b_hid0'],  trainable=self.stim_train)
            b_out  = tf.get_variable('b_out',    initializer=par['b_out0'],  trainable=self.stim_train)

        ### Tensordot specs:
        # W_in   = [n_hidden_neurons x dendrites_per_neuron x n_input_neurons]
        # input  = [n_input_neurons x batch_size]
        # output = [n_hidden_neurons x dendrites_per_neuron x batch_size]

        self.d_stim  = tf.nn.relu(tf.tensordot(W_stim, self.input_data, ([2],[0])))
        self.d_td    = tf.nn.relu(tf.tensordot(W_td, self.td_data, ([2],[0])))
        self.d_state = self.d_stim * self.d_td

        self.h_state = tf.nn.relu(tf.reduce_sum(self.d_state, 1) + b_hid)

        self.y = tf.nn.relu(tf.matmul(tf.nn.relu(W_out),self.h_state)+b_out)

    def optimize(self):
        if self.td_train:
            with tf.variable_scope('parameters', reuse=True):
                W_td = tf.get_variable('W_td')

            td_cases = []
            for td in par['td_cases']:
                td_cases.append(tf.nn.relu(tf.tensordot(W_td, tf.constant(td, dtype=tf.float32), ([2],[0]))))

            self.td_cases = tf.stack(td_cases)

        else:
            print('Training loss not yet implemented')


def main():
    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, td_data, target_data, learning_rate, stim_train
    x   = tf.placeholder(tf.float32, [par['n_stim'], par['batch_size']], 'stim')
    td  = tf.placeholder(tf.float32, [par['n_td'], par['batch_size']], 'TD')
    y   = tf.placeholder(tf.float32, [par['n_out'], par['batch_size']], 'out')
    lr  = tf.placeholder(tf.float32, [], 'learning_rate')
    st  = tf.placeholder(tf.bool, [], 'training_selection')

    with tf.Session() as sess:
        model   = Model(x, td, y, lr, st)
        init    = tf.global_variables_initializer()
        t_start = time.time()
        sess.run(init)

        stim_in  = np.ones([par['n_stim'], par['batch_size']])
        td_in    = np.ones([par['n_td'], par['batch_size']])
        y_hat    = np.ones([par['n_out'], par['batch_size']])
        rate     = par['learning_rate']
        train    = True

        proj_td = sess.run(model.td_cases, feed_dict={x:stim_in, td:td_in, y:y_hat, lr:rate, st:train})

        print(proj_td)

    print('\nModel execution complete.')

try:
    main()
except KeyboardInterrupt:
    quit('\nQuit by KeyboardInterrupt.')
