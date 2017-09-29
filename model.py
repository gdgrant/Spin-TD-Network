import tensorflow as tf
import numpy as np
import stimulus
from parameters import *
import os, time

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################

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
                    z = tf.constant(par['td_targets'][task][n], dtype=tf.float32)
                    self.task_loss += tf.reduce_sum(tf.square(self.td_set[-1]-z))

        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.grads_and_vars = opt.compute_gradients(self.task_loss)
        self.train_op = opt.apply_gradients(self.grads_and_vars)

class Model:

    def __init__(self, input_data, td_data, target_data, droput_keep_pct):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data     = input_data
        self.td_data        = td_data
        self.target_data    = target_data
        self.droput_keep_pct = droput_keep_pct

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    """
    def select_trainable(self, train_set):

        #If train_set is true, train the network in general.
        #If train_set is false, train just W_td.

        def tr():
            self.stim_train = True
            self.td_train   = False
            return 0
        def fa():
            self.stim_train = False
            self.td_train   = True
            return 0

        tf.cond(train_set, tr, fa)
    """


    def run_model(self):


        self.x = self.input_data
        self.td_gating = []
        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):
                W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['n_dendrites'],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                b = tf.get_variable('b', initializer = tf.zeros([1,par['n_dendrites'],par['layer_dims'][n+1]]), trainable = True)
                W_td = tf.get_variable('W_td', initializer = par['W_td0'][n], trainable = False)

                self.td_gating.append(tf.nn.softmax(tf.tensordot(self.td_data, W_td, ([1],[0])), dim = 1))
                print('gating ', self.td_gating[-1].shape)

                if n < par['n_layers']-2:
                    dend_activity = tf.nn.relu(tf.tensordot(self.x, W, ([1],[0]))  + b)
                    self.x = tf.nn.dropout(tf.reduce_sum(dend_activity*self.td_gating[-1], axis=1), self.droput_keep_pct)
                else:
                    dend_activity = tf.tensordot(self.x, W, ([1],[0]))  + b
                    self.y = tf.nn.softmax(tf.reduce_sum(dend_activity*self.td_gating[-1], axis=1), dim = 1)


    def optimize(self):

        epsilon = 1e-4
        optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

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
        if par['omega_c'] > 0:
            for var in variables:

                layer_num = int([s for s in var.op.name if s.isdigit()][0])
                var_dim = var.get_shape()[0].value
                td_gating = tf.tile(tf.reduce_mean(self.td_gating[layer_num], axis=0, keep_dims = True),[var_dim,1,1])

                print(var.op.name, var.get_shape(), layer_num, td_gating.shape)

                small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )

                #small_omega_var[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

                aux_loss += tf.reduce_sum(tf.multiply( big_omega_var[var.op.name], tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))

                reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name], var ) )
                #reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name, task_num], small_omega_var[var.op.name, task_num]*0.0 ) )

                #update_big_omega_ops.append( tf.assign_add( big_omega_var[var.op.name, task_num],  task_vector[task_num]*tf.div(small_omega_var[var.op.name, task_num], \
                	#(param_xi + tf.square(var-previous_weights_mu_minus_1[var.op.name, task_num])))))

                update_big_omega_ops.append( tf.assign_add( big_omega_var[var.op.name],  td_gating*tf.div(small_omega_var[var.op.name], \
                	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))

            # After each task is complete, call update_big_omega and reset_small_omega
            self.update_big_omega = tf.group(*update_big_omega_ops)
            #new_big_omega_var = big_omega_var

            # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
            self.reset_small_omega = tf.group(*reset_small_omega_ops)

        self.task_loss = -tf.reduce_sum( self.target_data*tf.log(self.y+epsilon) + (1.-self.target_data)*tf.log(1.-self.y+epsilon) )
        # Gradient of the loss function for the current task
        gradients = optimizer.compute_gradients(self.task_loss, var_list=variables)

            # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        if par['omega_c'] > 0:
        	gradients_with_aux = optimizer.compute_gradients(self.task_loss + par['omega_c']*aux_loss, var_list=variables)
        else:
        	gradients_with_aux = gradients

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in gradients_with_aux:
        	capped_gvs.append((tf.clip_by_norm(grad, 1), var))

        # This is called every batch
        #print(small_omega_var.keys())
        if par['omega_c'] > 0:
        	for i, (grad,var) in enumerate(gradients_with_aux):
        		update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], par['learning_rate']*capped_gvs[i][0]*gradients[i][0] ) )
        		#for j in range(n_tasks):
        			#update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name, j], task_vector[j]*learning_rate*capped_gvs[i][0]*gradients[i][0] ) ) # small_omega -= delta_weight(t)*gradient(t)
        	self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

        self.train_op = optimizer.apply_gradients(capped_gvs)

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.target_data,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def main():


    print('Training top-down weights')

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = TrainTopDown()
        sess.run(tf.global_variables_initializer())

        for i in range(20000):
            _, loss = sess.run([model.train_op, model.task_loss])
            if i//1000 == i/1000:
                print('Iteration = ',i,' Loss = ', loss)
        # extract top-down weights
        W_td = []
        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name, reuse = True):
                W = tf.get_variable('W_td')
                W_td.append(W.eval())

    par['W_td0'] = W_td
    print('Finished training top-down weights')


    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, td_data, target_data, learning_rate, stim_train
    x   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][0]], 'stim')
    td  = tf.placeholder(tf.float32, [par['batch_size'], par['n_td']], 'TD')
    y   = tf.placeholder(tf.float32, [ par['batch_size'], par['layer_dims'][-1]], 'out')
    droput_keep_pct = tf.placeholder(tf.float32, [] , 'dropout')

    stim = stimulus.Stimulus()


    with tf.Session() as sess:
        model   = Model(x, td, y, droput_keep_pct)
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        for task in range(par['n_tasks']):
            for i in range(par['n_train_batches']):

                stim_in, td_in, y_hat = stim.make_batch(task, test = False)
                if par['omega_c'] > 0:
                    sess.run([model.train_op,model.update_small_omega], feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct:0.75})
                else:
                    sess.run(model.train_op, feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct:0.75})

            if par['omega_c'] > 0:
                sess.run(model.update_big_omega,feed_dict={td:td_in})
                sess.run(model.reset_small_omega)

            accuracy = np.zeros((task+1))
            for test_task in range(task+1):
                stim_in, td_in, y_hat = stim.make_batch(test_task, test = True)
                accuracy[test_task] = sess.run(model.accuracy, feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct:1.0})
            print(task, accuracy)



    print('\nModel execution complete.')

try:
    main()
except KeyboardInterrupt:
    quit('\nQuit by KeyboardInterrupt.')
