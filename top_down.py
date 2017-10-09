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



class ConvolutionalLayers:

    def __init__(self):

        # Reset TensorFlow graph
        tf.reset_default_graph()
        # Train on CIFAR-10 task
        task_id = 0
        self.droput_keep_pct = 1.0

        # Create placeholders for the model
        input_data  = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'stim')
        target_data  = tf.placeholder(tf.float32, [par['batch_size'], 100], 'target')
        top_down  = tf.placeholder(tf.float32, [par['batch_size'], par['n_td']], 'TD')

        print('Batch size:', par['batch_size'])
        print('Input size:', 32*32*3, '\n')

        with tf.Session() as sess:
            cifar_model   = self.model(input_data, target_data, top_down)
            sess.run(tf.global_variables_initializer())
            t_start = time.time()

            s = stimulus.Stimulus(include_cifar10 = True)
            print('\n Iter. | Task | Loss')
            print('----------------------------')

            for i in range(2500):

                x, y, td, m = s.make_batch(task_id, test=False)
                _, loss, spike_loss  = sess.run([self.train_op, self.loss, self.spike_loss], feed_dict = {input_data:x, target_data: y, top_down: td})
                print(i, end='\r')

                if i%100 == 0:
                    print('', str(i).ljust(5), '|', str(task_id).ljust(4), '|', loss, '|', spike_loss)


            W = {}
            for var in tf.trainable_variables():
                W[var.op.name] = var.eval()
            pickle.dump(W, open('./encoder_testing/conv_weights.pkl','wb'))



    def model(self, input_data, target_data, top_down):

        drop_pct = 0.0
        self.spike_loss = 0

        ### Encoder
        conv1 = tf.layers.conv2d(inputs=input_data, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        #conv2 = tf.layers.dropout(conv2, drop_pct)
        # Now 64x64x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        # Now 16x16x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        #conv4 = tf.layers.dropout(conv4, drop_pct)
        # Now 16x16x64
        maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')


        self.x = tf.reshape(maxpool4,[par['batch_size'], -1])
        self.spike_loss += tf.reduce_sum(self.x)

        #self.td_gating = []

        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):
                if n < par['n_layers']-2 or par['dendrites_final_layer']:
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['n_dendrites'],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['n_dendrites'],par['layer_dims'][n+1]]), trainable = True)
                    W_td = tf.get_variable('W_td', initializer = par['W_td0'][n], trainable = False)
                    """
                    if par['clamp'] == 'dendrites':
                        self.td_gating.append(tf.nn.softmax(tf.tensordot(top_down, W_td, ([1],[0])), dim = 1))
                    elif par['clamp'] == 'neurons':
                        self.td_gating.append(tf.tensordot(top_down, W_td, ([1],[0])))
                    """
                else:
                    # final layer -> no dendrites
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], -1/np.sqrt(par['layer_dims'][n]), 1/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['layer_dims'][n+1]]), trainable = True)


                if n < par['n_layers']-2:
                    dend_activity = tf.nn.relu(tf.tensordot(self.x, W, ([1],[0]))  + b)

                    #self.x = tf.nn.dropout(tf.reduce_sum(dend_activity*self.td_gating[n], axis=1), self.droput_keep_pct)
                    self.x = tf.nn.dropout(tf.reduce_sum(dend_activity, axis=1), self.droput_keep_pct)

                    #self.spike_loss += tf.reduce_sum(self.x)

                else:
                    if par['dendrites_final_layer']:
                        dend_activity = tf.tensordot(self.x, W, ([1],[0])) + b
                        #self.y = tf.nn.softmax(tf.reduce_sum(dend_activity*self.td_gating[n], axis=1), dim = 1)
                        self.y = tf.nn.softmax(tf.reduce_sum(dend_activity, axis=1), dim = 1)
                        print('Y',self.y)
                    else:
                        self.y = tf.nn.softmax(tf.matmul(self.x,W) + b, dim = 1)
                        print('Y',self.y)


        epsilon = 1e-4
        self.loss = -tf.reduce_sum(target_data*tf.log(self.y+epsilon) + (1.-target_data)*tf.log(1.-self.y+epsilon) )


        optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        #gradients = optimizer.compute_gradients(self.loss)
        #print('HERE')
        #self.train_op = optimizer.apply_gradients(gradients)
        self.train_op = optimizer.minimize(self.loss + 0.0005*self.spike_loss)
