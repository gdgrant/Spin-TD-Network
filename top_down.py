import numpy as np
import tensorflow as tf
from parameters import *

class TrainTopDown:

    def __init__(self):

        td_cases = []
        self.task_loss = 0
        self.td_set = []

        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):
                W_td = tf.get_variable('W_td', initializer = tf.random_uniform([par['n_td'],par['n_dendrites'],par['layer_dims'][n+1]], -0.5, 0.5), trainable = True)
                for task in range(len(par['td_cases'])):

                    self.td_set.append(tf.nn.softmax(tf.tensordot(tf.constant(par['td_cases'][task, :], dtype=tf.float32), W_td, ([0],[0])), dim = 0))
                    #z = tf.constant(par['td_targets'][task][n], dtype=tf.float32)
                    z = tf.constant(par['td_targets'][n][task,:,:], dtype=tf.float32)
                    self.task_loss += tf.reduce_sum(tf.square(self.td_set[-1]-z))

        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.grads_and_vars = opt.compute_gradients(self.task_loss)
        self.train_op = opt.apply_gradients(self.grads_and_vars)
