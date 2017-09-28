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
                W_td = tf.get_variable('W_td',initializer = tf.random_uniform([par['n_td'],par['n_dendrites'],par['layer_dims'][n+1]], -0.5, 0.5), trainable=self.td_train)

                td  = tf.nn.relu(tf.tensordot(self.td_data, W_td, ([1],[0])))
                dend_activity = tf.nn.relu(tf.tensordot(self.x, W, ([1],[0]))  + b)

                if n < par['n_layers']-2:
                    self.x = tf.reduce_sum(dend_activity*td, axis=1)
                else:
                    self.y = tf.reduce_sum(dend_activity*td, axis=1)


    def optimize(self):

        for var in tf.trainable_variables():
            print(var)


        if self.td_train:
            td_cases = []
            self.task_loss = 0
            self.td_set = []
            for n in range(par['n_layers']-1):
                scope_name = 'layer' + str(n)
                with tf.variable_scope(scope_name, reuse = True):
                    W_td = tf.get_variable('W_td')
                    for task in range(len(par['td_cases'])):
                        #print(tf.constant(td, dtype=tf.float32))
                        self.td_set.append(tf.nn.softmax(tf.tensordot(tf.constant(par['td_cases'][task, :], dtype=tf.float32), W_td, ([0],[0])), dim = 0))
                        #self.td_set.append(tf.nn.relu(tf.tensordot(tf.constant(td, dtype=tf.float32), W_td, ([0],[0]))))
                        #print('TD SET', self.td_set[-1])
                        z = tf.constant(par['td_targets'][task][n], dtype=tf.float32)
                        #print('z', z)
                        #quit()
                        self.task_loss += tf.reduce_sum(tf.square(self.td_set[-1]-z))

        elif self.stim_train:

            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Implementation of the intelligent synapses model
            variables = [var for var in tf.trainable_variables()]

            small_omega_var = {}
            previous_weights_mu_minus_1 = {}
            big_omega_var = {}
            aux_loss = 0.0

            reset_small_omega_ops = []
            update_small_omega_ops = []
            update_big_omega_ops = []
            #for var, task_num in zip(variables, range(n_tasks)):
            if param_c > 0:

            	for var in variables:
                    print(var.op.name)
                    print(var.get_shape())
                    quit()
            		small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            		reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )

            		#small_omega_var[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            		previous_weights_mu_minus_1[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            		big_omega_var[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

            		aux_loss += tf.reduce_sum(tf.multiply( big_omega_var[var.op.name, task_num], tf.square(previous_weights_mu_minus_1[var.op.name, task_num] - var) ))

            		reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name, task_num], var ) )
            		#reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name, task_num], small_omega_var[var.op.name, task_num]*0.0 ) )

            		#update_big_omega_ops.append( tf.assign_add( big_omega_var[var.op.name, task_num],  task_vector[task_num]*tf.div(small_omega_var[var.op.name, task_num], \
            			#(param_xi + tf.square(var-previous_weights_mu_minus_1[var.op.name, task_num])))))

            		update_big_omega_ops.append( tf.assign_add( big_omega_var[var.op.name, task_num],  task_vector[task_num]*tf.div(small_omega_var[var.op.name], \
            			(param_xi + tf.square(var-previous_weights_mu_minus_1[var.op.name, task_num])))))

            	# After each task is complete, call update_big_omega and reset_small_omega
            	update_big_omega = tf.group(*update_big_omega_ops)
            	#new_big_omega_var = big_omega_var

            	# Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
            	reset_small_omega = tf.group(*reset_small_omega_ops)


            # Gradient of the loss function for the current task
            gradients = optimizer.compute_gradients(cross_entropy, var_list=variables)

            # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
            if param_c > 0:
            	gradients_with_aux = optimizer.compute_gradients(cross_entropy + param_c*aux_loss, var_list=variables)
            else:
            	gradients_with_aux = gradients




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

        for i in range(25000):

            _, td_set, loss, gvs = sess.run([model.train_op, model.td_set, model.task_loss, model.grads_and_vars], \
                                            feed_dict={x:stim_in, td:td_in, y:y_hat, lr:rate, st:train})
            if i//1000 == i/1000:
                print(i, loss)
        print(td_set[0].shape, len(td_set))
        print(td_set[0])
        # Task training
        rate     = par['learning_rate']
        train   = False

    print('\nModel execution complete.')

try:
    main()
except KeyboardInterrupt:
    quit('\nQuit by KeyboardInterrupt.')
