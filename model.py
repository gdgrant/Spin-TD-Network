import tensorflow as tf
import numpy as np
import stimulus
import AdamOpt
from parameters import *
import os, time
import pickle
import top_down
import matplotlib.pyplot as plt


# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################
class Model:

    def __init__(self, input_data, td_data, target_data):

        # Load the input activity, the target data, and the time mask
        # for this batch of trials
        self.input_data = tf.unstack(input_data)
        self.td_data    = tf.unstack(td_data)
        self.y_hat      = target_data
        self.time_mask  = tf.constant(par['output_time_mask'], dtype=tf.float32)
        self.rnn_mask   = tf.constant(par['rnn_mask'], dtype=tf.float32)

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):


        W_in  = tf.get_variable('W_in',  initializer=np.float32(par['W_in0']))
        W_td  = tf.get_variable('W_td',  initializer=np.float32(par['W_td0']))
        W_rnn = tf.get_variable('W_rnn', initializer=np.float32(par['W_rnn0']))
        W_out = tf.get_variable('W_out', initializer=np.float32(par['W_out0']))

        b_hid = tf.get_variable('b_hid', initializer=tf.zeros([1,par['n_hidden']]))
        b_out = tf.get_variable('b_out', initializer=tf.zeros([1,par['n_output']]))

        d = []  # Dendrite activity logging across time
        h = []  # Hidden layer activity logging across time
        y = []  # Output activity logging across time

        h_act = tf.zeros(shape=[par['n_hidden'], par['batch_size']])
        for x, td in zip(self.input_data, self.td_data):

            d_in  = tf.tensordot(tf.nn.relu(W_in), x, ([2],[0]))
            d_td  = tf.constant(1.) # tf.tensordot(tf.nn.relu(W_td), td, ([2],[0]))
            d_rnn = tf.tensordot(tf.nn.relu(W_rnn), h_act, ([2],[0]))

            d_act = (d_in + d_rnn) * d_td
            d.append(d_act)

            h_act = tf.reduce_sum(d_act, 1)
            h.append(h_act)

            y_act = tf.matmul(tf.nn.relu(W_out), h_act)
            y.append(y_act)

        print('Stimulus:     ', x.shape)
        print('Top-Down:     ', td.shape)
        print('Dendrites:    ', d_act.shape)
        print('Hidden layer: ', h_act.shape)
        print('Output layer: ', y_act.shape)

        self.d = tf.stack(d)
        self.h = tf.stack(h)
        self.y = tf.stack(y)


    def optimize(self):

        if par['loss_function'] == 'MSE':
            self.train_loss = tf.reduce_mean(self.time_mask*tf.square(self.y-self.y_hat))
        elif par['loss_function'] == 'cross_entropy':
            print('Cross entropy is currently broken.')
            epsilon = 1e-4
            logistic = self.y*tf.log(self.y_hat+epsilon) + (1-self.y)*tf.log(1-self.y_hat+epsilon)
            self.train_loss = tf.reduce_mean(self.time_mask * logistic)

        optimizer = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
        print('\nTrainable Variables:')
        [print('-->', v) for v in tf.trainable_variables()]

        self.grads_and_vars = optimizer.compute_gradients(self.train_loss)
        self.train_op       = optimizer.apply_gradients(self.grads_and_vars)





        """
        quit()

        epsilon = 1e-4


        if adam_opt_method:
            adam_optimizer = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])
            adam_optimizer_task = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])
        else:
            adam_optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = par['learning_rate'])

        small_omega_var = {}

        previous_weights_mu_minus_1 = {}
        self.big_omega_var = {}
        gates = {}

        self.aux_loss = 0.0

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        #for var, task_num in zip(variables, range(n_tasks)):
        if par['omega_c'] > 0:
            for var in variables:

                # Create variables
                gates[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

                # Create assing operations
                reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
                reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name], var ) )


                update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(small_omega_var[var.op.name], \
                	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))

                self.aux_loss += tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))

            # After each task is complete, call update_big_omega and reset_small_omega
            self.update_big_omega = tf.group(*update_big_omega_ops)
            #new_big_omega_var = big_omega_var

            # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
            self.reset_small_omega = tf.group(*reset_small_omega_ops)

        self.task_loss = -tf.reduce_sum(self.mask*self.target_data*tf.log(self.y+epsilon) + self.mask*(1.-self.target_data)*tf.log(1.-self.y+epsilon) )

        self.total_loss = self.task_loss + 0.0005*self.spike_loss

        # Gradient of the loss function for the current task
        #grads_and_vars = optimizer.compute_gradients(self.task_loss, var_list = variables)

        """
        #Apply any applicable weights masks to the gradient and clip
        """
        update_gate_ops = []
        for var in variables:

            # fully connected layer
            layer_num = int([s for s in var.op.name if s.isdigit()][0])
            var_dim = var.get_shape()[0].value
            if par['clamp'] == 'dendrites' and (layer_num < par['n_layers']-2 or par['dendrites_final_layer']):
                td_gating = tf.tile(tf.reduce_mean(self.td_gating[layer_num], axis=0, keep_dims = True), [var_dim, 1, 1])
                update_gate_ops.append(tf.assign(gates[var.op.name], td_gating))

        self.update_gate = tf.group(*update_gate_ops)

        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        if par['omega_c'] > 0:
            if adam_opt_method:
                self.train_op = adam_optimizer.compute_gradients(self.total_loss + par['omega_c']*self.aux_loss, gates)
                self.task_op = adam_optimizer_task.compute_gradients(self.task_loss, gates)
                self.delta_grads = adam_optimizer_task.return_delta_grads()
                self.gradients = adam_optimizer_task.return_grads_and_vars()
            else:
                adam_grads_and_vars = adam_optimizer.compute_gradients(self.total_loss + par['omega_c']*self.aux_loss)
                self.delta_grads = []
                for g,v in adam_grads_and_vars:
                    self.delta_grads.append(g)
                self.train_op = adam_optimizer.apply_gradients(adam_grads_and_vars)
        else:
            self.train_op = adam_optimizer.compute_gradients(self.total_loss, gates)

        #self.gradients = adam_optimizer.return_grads_and_vars()
        reset_op = []
        reset_op.append(adam_optimizer.reset_params())
        self.reset_adam_op = adam_optimizer.reset_params()


        # This is called every batch
        #print(small_omega_var.keys())
        if par['omega_c'] > 0:
            if adam_opt_method:
                for grad,var in self.gradients:
                #for var in variables:
                    print(var.op.name)
                    #update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
                    update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], gates[var.op.name]*par['learning_rate']*grad*grad ) )
                    #update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], gates[var.op.name]*self.delta_grads[var.op.name]*self.delta_grads[var.op.name]/par['learning_rate'] ) )
            else:
                for (grad,var), g_adam in zip(grads_and_vars, self.delta_grads):
                    update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], gates[var.op.name]*self.learning_rate*g_adam*grad ) )

            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

        #self.train_op = adam_optimizer.apply_gradients()

        correct_prediction = tf.equal(tf.argmax(self.y - (1-self.mask)*9999,1), tf.argmax(self.target_data - (1-self.mask)*9999,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        """

def main():

    #determine_top_down_weights()

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, td_data, target_data, learning_rate, stim_train
    x   = tf.placeholder(tf.float32, par['stim_shape'], 'stim')
    td  = tf.placeholder(tf.float32, par['td_shape'], 'TD')
    y   = tf.placeholder(tf.float32, par['output_shape'], 'out')

    stim = stimulus.Stimulus()


    with tf.Session() as sess:

        model = Model(x, td, y)
        sess.run(tf.global_variables_initializer())
        t_start = time.time()
        #sess.run(model.reset_small_omega)

        print('')
        print('Iter. |', 'Loss'.ljust(10), '|', 'Accuracy')
        print('-'*30)
        for i in range(par['n_train_batches']):

            x_all, y_hat = stim.make_batch(0)
            x_st   = x_all[:,:par['n_input'],:]
            x_td   = x_all[:,par['n_input']:,:]

            _, loss, d, h, y_out = sess.run([model.train_op, model.train_loss, model.d, model.h, model.y],
                         feed_dict={x:x_st, td:x_td, y: y_hat})

            if i%50==0:
                print(' ' + str(i).ljust(4), '|', str(loss).ljust(10), '|', str(determine_accuracy(y_out, y_hat)))








            #sess.run(model.update_gate, feed_dict={td:td_in})


            #stim_in += np.random.normal(0,0.02,size=stim_in.shape)
            #print(stim_in.shape, y_hat.shape, td_in.shape, np.mean(td_in,axis=0))

            #if par['omega_c'] > 0:
            #
            #    _,_,loss,spike_loss,_,AL,_ = sess.run([model.update_gate, model.train_op,model.task_loss,model.spike_loss, \
            #    model.update_small_omega, model.aux_loss, model.task_op], feed_dict={x:stim_in, td:td_in, y:y_hat, mask:mk, droput_keep_pct:keep_pct})
            #else:
            #    sess.run(model.train_op, feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct: par['keep_pct'], \
            #        learning_rate: lr, gate_conv: gate})
            #
            #if i//100 == i/100:
            #    if task == 0:
            #        print(i, loss, spike_loss, AL)
            #    else:
            #        big_om = [np.mean(bo) for bo in big_omegas.values()]
            #        print(i, loss, AL, big_om)

        # if training on the cifar task, don't update omegas on the 0th task
        #if par['omega_c'] > 0:
        #    sess.run(model.update_big_omega,feed_dict={td:td_in})
        #    big_omegas = sess.run(model.big_omega_var)
        #    sess.run(model.reset_adam_op)
        #    sess.run(model.reset_small_omega)
        #
        #if par['task']=='mnist' or (par['task']=='cifar' and task > -1):
        #    accuracy = np.zeros((task+1))
        #    for test_task in range(task+1):
        #        stim_in, y_hat, td_in = stim.make_batch(test_task, test = True)
        #        accuracy[test_task] = sess.run(model.accuracy, feed_dict={x:stim_in, td:td_in, y:y_hat, mask:mk,droput_keep_pct:1.0})
        #else:
        #    accuracy = [-1]
        #
        #print('Task ',task, ' Mean ', np.mean(accuracy), ' First ', accuracy[0], ' Last ', accuracy[-1])
        #
        #if par['save_analysis'] and (par['task']=='mnist' or (par['task']=='cifar' and task > 0)):
        #    save_results = {'task': task, 'accuracy': accuracy, 'big_omegas': big_omegas, 'par': par}
        #    pickle.dump(save_results, open(par['save_dir'] + 'analysis.pkl', 'wb'))

    print('\nModel execution complete.')


def determine_top_down_weights():

    # file to store/load top-down weights
    td_weight_fn = par['save_dir'] + 'top_down_weights_' + par['clamp'] + '_' + par['task'] + '_dfl_.pkl'
    top_down.TrainTopDown(td_weight_fn)


def determine_accuracy(y, y_hat):
    y     = y[-par['steps_per_input']:,:,:]
    y_hat = y_hat[-par['steps_per_input']:,:,:]

    y     = np.mean(np.argmax(y, 1), axis=0)
    y_hat = np.mean(np.argmax(y_hat, 1), axis=0)

    return np.mean(np.float32(y==y_hat))



try:
    main()
except KeyboardInterrupt:
    quit('\nQuit by KeyboardInterrupt.')
