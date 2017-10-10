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

        self.spike_loss = 0
        self.td_gating = []

        for scope_name in ['layer'+str(n) for n in range(par['n_layers']-1)]:
            with tf.variable_scope(scope_name):
                if n < par['n_layers']-2 or par['dendrites_final_layer']:
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['n_dendrites'],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['n_dendrites'],par['layer_dims'][n+1]]), trainable = True)
                    W_td = tf.get_variable('W_td', initializer = par['W_td0'][n], trainable = False)

                    if par['clamp'] == 'dendrites':
                        self.td_gating.append(tf.nn.softmax(tf.tensordot(self.td_data, W_td, ([1],[0])), dim = 1))
                    elif par['clamp'] == 'neurons':
                        self.td_gating.append(tf.tensordot(self.td_data, W_td, ([1],[0])))
                else:
                    # final layer -> no dendrites
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], -1/np.sqrt(par['layer_dims'][n]), 1/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['layer_dims'][n+1]]), trainable = True)


                if n < par['n_layers']-2:
                    dend_activity = tf.nn.relu(tf.tensordot(self.x, W, ([1],[0]))  + b)
                    self.x = tf.nn.dropout(tf.reduce_sum(dend_activity*self.td_gating[n], axis=1), self.droput_keep_pct)

                    self.spike_loss += tf.reduce_sum(self.x)

                else:
                    if par['dendrites_final_layer']:
                        dend_activity = tf.tensordot(self.x, W, ([1],[0])) + b
                        self.y = tf.nn.softmax(tf.reduce_sum(dend_activity*self.td_gating[n], axis=1), dim = 1)
                        print('Y',self.y)
                    else:
                        self.y = tf.nn.softmax(tf.matmul(self.x,W) + b, dim = 1)
                        print('Y',self.y)


    def optimize(self):

        adam_opt_method = True

        epsilon = 1e-4
        #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)

        # Use all trainable variables, except those in the convolutional layers
        #variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        print('Trainable Variables:')
        [print(v) for v in variables]

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
        Apply any applicable weights masks to the gradient and clip
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



def main():

    determine_top_down_weights()

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
    td  = tf.placeholder(tf.float32, [par['batch_size'], par['n_td']], 'TD')
    y   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'out')
    mask   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'mask')
    droput_keep_pct = tf.placeholder(tf.float32, [] , 'dropout')


    stim = stimulus.Stimulus()


    with tf.Session() as sess:

        model = Model(x, td, y, mask, droput_keep_pct)
        sess.run(tf.global_variables_initializer())
        t_start = time.time()
        sess.run(model.reset_small_omega)

        for task in range(0,par['n_tasks']):

            #gate = 1 if (par['task'] == 'mnist' or task == 0) else 0
            gate = 0
            keep_pct = par['keep_pct'] if (par['task'] == 'mnist' or task > 0) else 1.0
            #gate = 1

            for i in range(par['n_train_batches']):

                stim_in, y_hat, td_in, mk = stim.make_batch(task, test = False)
                #sess.run(model.update_gate, feed_dict={td:td_in})


                #stim_in += np.random.normal(0,0.02,size=stim_in.shape)
                #print(stim_in.shape, y_hat.shape, td_in.shape, np.mean(td_in,axis=0))

                if par['omega_c'] > 0:

                    _,_,loss,spike_loss,_,AL,_ = sess.run([model.update_gate, model.train_op,model.task_loss,model.spike_loss, \
                    model.update_small_omega, model.aux_loss, model.task_op], feed_dict={x:stim_in, td:td_in, y:y_hat, mask:mk, droput_keep_pct:keep_pct})
                else:
                    sess.run(model.train_op, feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct: par['keep_pct'], \
                        learning_rate: lr, gate_conv: gate})

                if i//100 == i/100:
                    if task == 0:
                        print(i, loss, spike_loss, AL)
                    else:
                        big_om = [np.mean(bo) for bo in big_omegas.values()]
                        print(i, loss, AL, big_om)

            # if training on the cifar task, don't update omegas on the 0th task
            if par['omega_c'] > 0:
                sess.run(model.update_big_omega,feed_dict={td:td_in})
                big_omegas = sess.run(model.big_omega_var)
                sess.run(model.reset_adam_op)
                sess.run(model.reset_small_omega)

            if par['task']=='mnist' or (par['task']=='cifar' and task > -1):
                accuracy = np.zeros((task+1))
                for test_task in range(task+1):
                    stim_in, y_hat, td_in = stim.make_batch(test_task, test = True)
                    accuracy[test_task] = sess.run(model.accuracy, feed_dict={x:stim_in, td:td_in, y:y_hat, mask:mk,droput_keep_pct:1.0})
            else:
                accuracy = [-1]

            print('Task ',task, ' Mean ', np.mean(accuracy), ' First ', accuracy[0], ' Last ', accuracy[-1])

            if par['save_analysis'] and (par['task']=='mnist' or (par['task']=='cifar' and task > 0)):
                save_results = {'task': task, 'accuracy': accuracy, 'big_omegas': big_omegas, 'par': par}
                pickle.dump(save_results, open(par['save_dir'] + 'analysis.pkl', 'wb'))

    print('\nModel execution complete.')


def determine_top_down_weights():

    # file to store/load top-down weights
    td_weight_fn = par['save_dir'] + 'top_down_weights_' + par['clamp'] + '_' + par['task'] + '_dfl_.pkl'
    top_down.TrainTopDown(td_weight_fn)



try:
    main()
except KeyboardInterrupt:
    quit('\nQuit by KeyboardInterrupt.')
