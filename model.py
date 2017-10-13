import tensorflow as tf
import numpy as np
import stimulus
import AdamOpt
from parameters import *
import os, time
import pickle
import top_down


# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################
class Model:

    def __init__(self, input_data, td_data, target_data, mask, droput_keep_pct):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data         = input_data
        self.td_data            = td_data
        self.target_data        = target_data
        self.droput_keep_pct    = droput_keep_pct
        self.mask               = mask

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        if par['task'] == 'cifar':
            self.x = self.apply_convulational_layers()

        elif par['task'] == 'mnist':
            self.x = self.input_data

        self.apply_dense_layers()


    def apply_dense_layers(self):

        self.spike_loss = 0
        for n, scope_name in enumerate(['layer'+str(n) for n in range(par['n_layers']-1)]):
            with tf.variable_scope(scope_name):
                if n < par['n_layers']-2 or par['dendrites_final_layer']:
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['n_dendrites'],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['n_dendrites'],par['layer_dims'][n+1]]), trainable = True)
                    W_td = tf.get_variable('W_td', initializer = par['W_td0'][n], trainable = False)

                    proj_W_td = tf.tensordot(self.td_data, W_td, ([1],[0]))
                    W_effective = W*tf.tile(proj_W_td,[par['layer_dims'][n],1,1])
                    b_effective = b*proj_W_td

                else:
                    # final layer -> no dendrites
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], -1/np.sqrt(par['layer_dims'][n]), 1/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['layer_dims'][n+1]]), trainable = True)


                if n < par['n_layers']-2:
                    dend_activity = tf.nn.relu(tf.tensordot(self.x, W_effective, ([1],[0]))  + b_effective)
                    self.x = tf.nn.dropout(tf.reduce_sum(dend_activity, axis=1), self.droput_keep_pct)
                    self.spike_loss += tf.reduce_mean(self.x)

                else:
                    if par['dendrites_final_layer']:
                        dend_activity = tf.tensordot(self.x, W_effective, ([1],[0])) + b_effective
                        self.y = tf.nn.softmax(tf.reduce_sum(dend_activity, axis=1), dim = 1)

                    else:
                        self.y = tf.nn.softmax(tf.matmul(self.x,W) + b, dim = 1)

    def apply_convulational_layers(self):

        conv_weights = pickle.load(open('./encoder_testing/conv_weights.pkl','rb'))

        conv1 = tf.layers.conv2d(inputs=self.input_data,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv2 = tf.layers.conv2d(inputs=conv1,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_1/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_1/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')

        conv3 = tf.layers.conv2d(inputs=conv2,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_2/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_2/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv4 = tf.layers.conv2d(inputs=conv3,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_3/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_3/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='SAME')

        return tf.reshape(conv4,[par['batch_size'], -1])

    def optimize(self):

        epsilon = 1e-4

        # Use all trainable variables, except those in the convolutional layers
        #variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]

        adam_optimizer = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])
        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  par['learning_rate'])
        #adam_optimizer_task = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])


        small_omega_var = {}

        previous_weights_mu_minus_1 = {}
        self.big_omega_var = {}

        self.aux_loss = 0.0

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in variables:

            # Create variables
            #gates[var.op.name] = tf.Variable(tf.ones(var.get_shape()), trainable=False)
            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

            # Create assing operations
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name], var ) )


            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))

            self.aux_loss += par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))

        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)
        #new_big_omega_var = big_omega_var

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        self.task_loss = -tf.reduce_sum(self.mask*self.target_data*tf.log(self.y+epsilon) + \
            self.mask*(1.-self.target_data)*tf.log(1.-self.y+epsilon) )

        #self.task_loss = -tf.reduce_sum(self.target_data*tf.log(self.y+epsilon) + (1.-self.target_data)*tf.log(1.-self.y+epsilon) )

        self.total_loss = self.task_loss + 0.0000001*self.spike_loss

        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        self.train_op = adam_optimizer.compute_gradients(self.total_loss + self.aux_loss)
        #self.task_op = adam_optimizer_task.compute_gradients(self.task_loss, gates)
        self.delta_grads = adam_optimizer.return_delta_grads()
        self.gradients = optimizer_task.compute_gradients(self.task_loss)


        #self.gradients = adam_optimizer.return_grads_and_vars()
        reset_op = []
        reset_op.append(adam_optimizer.reset_params())
        self.reset_adam_op = adam_optimizer.reset_params()

        # This is called every batch
        for grad,var in self.gradients:
            update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )

        self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

        correct_prediction = tf.equal(tf.argmax(self.y - (1-self.mask)*9999,1), tf.argmax(self.target_data - (1-self.mask)*9999,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def main(save_fn):

    #determine_top_down_weights()

    if par['task'] == 'cifar' and par['train_convolutional_layers']:
        top_down.ConvolutionalLayers()

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, td_data, target_data, learning_rate, stim_train
    if par['task'] == 'mnist':
        x  = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][0]], 'stim')
    elif par['task'] == 'cifar':
        x  = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'stim')
    #td  = tf.placeholder(tf.float32, [par['batch_size'], par['n_td']], 'TD')
    td  = tf.placeholder(tf.float32, [1, par['n_td']], 'TD')
    y   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'out')
    mask   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'mask')
    droput_keep_pct = tf.placeholder(tf.float32, [] , 'dropout')


    stim = stimulus.Stimulus()
    accuracy_full = []

    with tf.Session() as sess:

        model = Model(x, td, y, mask, droput_keep_pct)
        sess.run(tf.global_variables_initializer())
        t_start = time.time()
        sess.run(model.reset_small_omega)

        for task in range(0,par['n_tasks']):

            for i in range(par['n_train_batches']):

                stim_in, y_hat, td_in, mk = stim.make_batch(task, test = False)

                _,loss,spike_loss,_,AL = sess.run([model.train_op, model.task_loss,model.spike_loss, model.update_small_omega, \
                    model.aux_loss], feed_dict={x:stim_in, td:td_in, y:y_hat, mask:mk, droput_keep_pct:par['keep_pct']})

                if (i+1)//400 == (i+1)/400:
                    print('Iter: ', i, 'Loss: ', loss, 'Aux Loss: ',  AL)

            # Update big omegaes, and reset other values before starting new task
            sess.run(model.update_big_omega,feed_dict={td:td_in})
            big_omegas = sess.run(model.big_omega_var)
            sess.run(model.reset_adam_op)
            sess.run(model.reset_small_omega)

            accuracy = np.zeros((task+1))
            for test_task in range(task+1):
                stim_in, y_hat, td_in, mk = stim.make_batch(test_task, test = True)
                accuracy[test_task] = sess.run(model.accuracy, feed_dict={x:stim_in, td:td_in, y:y_hat, mask:mk,droput_keep_pct:1.0})


            print('Task ',task, ' Mean ', np.mean(accuracy), ' First ', accuracy[0], ' Last ', accuracy[-1])
            accuracy_full.append(np.mean(accuracy))

        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nModel execution complete.')
