import numpy as np
import tensorflow as tf
import pickle
import time
import stimulus
from parameters import *

class TrainTopDown:

    def __init__(self, save_fn):

        if par['train_top_down']:
            self.train(save_fn)
        else:
            print('Loading top-down weights')
            top_down_results = pickle.load(open(save_fn,'rb'))
            par['W_td0'] = top_down_results['W_td']
            par['td_cases']  = top_down_results['td_cases']

    def model(self):

        if par['clamp'] is None:
            self.td_set = np.ones((par['n_tasks']), dtype = np.float32)
            return -1

        td_cases = []
        self.task_loss = 0
        self.td_set = []

        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):
                W_td = tf.get_variable('W_td', initializer = tf.random_uniform([par['n_td'],par['n_dendrites'],par['layer_dims'][n+1]], -0.5, 0.5), trainable = True)
                for task in range(par['n_tasks']):

                    if par['clamp'] == 'dendrites':
                        self.td_set.append(tf.nn.softmax(tf.tensordot(tf.constant(par['td_cases'][task, :], dtype=tf.float32), W_td, ([0],[0])), dim = 0))
                    elif par['clamp'] == 'neurons':
                        self.td_set.append(tf.tensordot(tf.constant(par['td_cases'][task, :], dtype=tf.float32), W_td, ([0],[0])))
                    #z = tf.constant(par['td_targets'][task][n], dtype=tf.float32)
                    z = tf.constant(par['td_targets'][n][task,:,:], dtype=tf.float32)
                    #print(np.mean(par['td_targets'][n]), np.mean(par['td_cases'][task, :]))
                    self.task_loss += tf.reduce_sum(tf.square(self.td_set[-1]-z))

        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.grads_and_vars = opt.compute_gradients(self.task_loss)
        self.train_op = opt.apply_gradients(self.grads_and_vars)

    def train(self, save_fn):

        print('Training top-down weights')
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = self.model()
            sess.run(tf.global_variables_initializer())

            for i in range(par['n_batches_top_down']):
                _, loss = sess.run([self.train_op, self.task_loss])
                if i//1000 == i/1000:
                    print('Iteration = ',i,' Loss = ', loss)
            # extract top-down weights
            W_td = []
            for n in range(par['n_layers']-1):
                scope_name = 'layer' + str(n)
                with tf.variable_scope(scope_name, reuse = True):
                    W = tf.get_variable('W_td')
                    W_td.append(W.eval())

        top_down_results = {'W_td': W_td, 'td_cases': par['td_cases']}
        par['W_td0'] = W_td
        print('Finished training top-down weights... saving data')

        pickle.dump(top_down_results, open(save_fn,'wb'))
