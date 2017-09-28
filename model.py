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


        self.x = self.input_data
        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):
                W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['n_dendrites'],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable=self.stim_train)
                b = tf.get_variable('b',initializer = tf.zeros([1,par['n_dendrites'],par['layer_dims'][n+1]]), trainable=self.stim_train)
                W_td = tf.get_variable('W_td',initializer = tf.random_uniform([par['n_td'],par['n_dendrites'],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable=self.td_train)

                print('SHAPES')
                print('td_data',self.td_data)
                print('W_td',W_td)
                td  = tf.nn.relu(tf.tensordot(self.td_data, W_td, ([1],[0])))
                print('td',td)
                print('W',W)
                print('x_data', self.x)

                dend_activity = tf.nn.relu(tf.tensordot(self.x, W, ([1],[0]))  + b)
                print('dend_activity', dend_activity)

                if n < par['n_layers']-2:
                    self.x = tf.reduce_sum(dend_activity*td, axis=1)
                else:
                    self.y = tf.reduce_sum(dend_activity*td, axis=1)
                #W_effective = tf.reduce_sum(W*td, axis=1)
                #b_effective = tf.reduce_sum(b*td, axis=1)
                #W_effective = tf.reduce_sum(W*td, axis=1)
                #b_effective = tf.reduce_sum(b*td, axis=1)

                #self.x = tf.nn.relu(tf.matmul(self.x,W_effective) + b_effective)



        """
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
        """

    def optimize(self):

        for var in tf.trainable_variables():
            print(var)


        if self.td_train:
            td_cases = []
            self.task_loss = 0
            count = 0
            self.td_set = []
            for n in range(par['n_layers']-1):
                scope_name = 'layer' + str(n)
                with tf.variable_scope(scope_name, reuse = True):
                    W_td = tf.get_variable('W_td')
                    for td in par['td_cases']:
                        print(tf.constant(td, dtype=tf.float32))
                        self.td_set.append(tf.nn.softmax(tf.nn.relu(tf.tensordot(tf.constant(td, dtype=tf.float32), W_td, ([0],[0]))), dim = 0))
                        #print('TD SET', self.td_set[-1])
                        z = tf.constant(par['td_targets'][count], dtype=tf.float32)
                        self.task_loss += tf.reduce_sum(tf.square(self.td_set[-1]-z))
                        count += 1


            """
            if self.td_train:
                with tf.variable_scope('parameters', reuse=True):
                    W_td = tf.get_variable('W_td')

            td_cases = []
            for td in par['td_cases']:
                td_cases.append(tf.nn.relu(tf.tensordot(W_td, tf.constant(td, dtype=tf.float32), ([2],[0]))))
            self.td_set = tf.stack(td_cases)

            if par['td_loss_type'] == 'naive_dot':
                td_cases = tf.unstack(self.td_set, axis=1)
                self.task_loss = 0.
                for neuron in td_cases:
                    t = tf.reduce_prod(neuron, axis=0)  # Across tasks
                    d = tf.reduce_prod(neuron, axis=1)  # Across dendrites
                    self.task_loss += tf.square(tf.reduce_sum(t)+tf.reduce_sum(d))

            elif par['td_loss_type'] == 'z_dot':
                td_set = tf.nn.softmax(self.td_set, dim=2)
                y_prime = tf.tensordot(td_set, td_set, [[2],[2]])
                z = tf.constant(par['TD_Z'], dtype=tf.float32)
                self.task_loss = tf.reduce_sum(tf.square(y_prime-z))

            elif par['td_loss_type'] == 'pairwise_random':
                td_set = self.td_set
                z = tf.constant(par['TD_Z'], dtype=tf.float32)
                self.task_loss = tf.reduce_sum(tf.square(td_set-z))
            """

            opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.grads_and_vars = opt.compute_gradients(self.task_loss)
            self.train_op = opt.apply_gradients(self.grads_and_vars)

        else:
            print('Training loss not yet implemented')


def main():
    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, td_data, target_data, learning_rate, stim_train
    x   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][0]], 'stim')
    td  = tf.placeholder(tf.float32, [par['batch_size'], par['n_td']], 'TD')
    y   = tf.placeholder(tf.float32, [ par['batch_size'], par['layer_dims'][-1]], 'out')
    lr  = tf.placeholder(tf.float32, [], 'learning_rate')
    st  = tf.placeholder(tf.bool, [], 'training_selection')

    with tf.Session() as sess:
        model   = Model(x, td, y, lr, st)
        init    = tf.global_variables_initializer()
        t_start = time.time()
        sess.run(init)

        # TD Training
        stim_in  = np.ones([par['batch_size'], par['layer_dims'][0]])
        td_in    = np.ones([par['batch_size'], par['n_td']])
        y_hat    = np.ones([par['batch_size'], par['layer_dims'][-1]])
        rate     = par['learning_rate']
        train    = True

        for i in range(20000):

            _, td_set, loss, gvs = sess.run([model.train_op, model.td_set, model.task_loss, model.grads_and_vars], \
                                            feed_dict={x:stim_in, td:td_in, y:y_hat, lr:rate, st:train})
            if i//1000 == i/1000:
                print(i, loss)
        print(td_set)
        # Task training
        rate     = par['learning_rate']
        train   = False

    print('\nModel execution complete.')

try:
    main()
except KeyboardInterrupt:
    quit('\nQuit by KeyboardInterrupt.')
