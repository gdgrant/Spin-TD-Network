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

    def __init__(self, input_data, td_data, target_data, droput_keep_pct, learning_rate, gate_conv = 1):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data         = input_data
        self.td_data            = td_data
        self.target_data        = target_data
        self.droput_keep_pct    = droput_keep_pct
        self.learning_rate      = learning_rate
        self.gate_conv          = gate_conv # Only for CIFAR; used to gate changes to convolution weights after first task

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        if par['task'] == 'cifar':
            print('INPUT', self.input_data)

            conv_weights = pickle.load(open('./encoder_testing/conv_weights.pkl','rb'))
            print(conv_weights['conv2d/bias'])
            #conv1 = tf.nn.relu(tf.nn.conv2d(input=self.input_data, filter=conv_weights['conv2d/kernel'],strides=[1,1,1,1],padding='SAME'))

            conv1 = tf.layers.conv2d(inputs=self.input_data,filters=32, kernel_size=[3, 3], kernel_initializer = \
                tf.constant_initializer(conv_weights['conv2d/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d/bias']), \
                strides=1, activation=tf.nn.relu, padding = 'SAME')


            conv2 = tf.layers.conv2d(inputs=conv1,filters=32, kernel_size=[3, 3], kernel_initializer = \
                tf.constant_initializer(conv_weights['conv2d_1/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_1/bias']), \
                strides=1, activation=tf.nn.relu, padding = 'SAME')


            #conv2 = tf.nn.relu(tf.nn.conv2d(input=conv1, filter=conv_weights['conv2d_1/kernel'],strides=[1,1,1,1],padding='SAME'))
            conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')
            print('conv2',conv2)
            #conv3 = tf.nn.relu(tf.nn.conv2d(input=conv2, filter=conv_weights['conv2d_2/kernel'],strides=[1,1,1,1],padding='SAME'))
            #conv4 = tf.nn.relu(tf.nn.conv2d(input=conv3, filter=conv_weights['conv2d_3/kernel'],strides=[1,1,1,1],padding='SAME'))
            conv3 = tf.layers.conv2d(inputs=conv2,filters=64, kernel_size=[3, 3], kernel_initializer = \
                tf.constant_initializer(conv_weights['conv2d_2/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_2/bias']), \
                strides=1, activation=tf.nn.relu, padding = 'SAME')


            conv4 = tf.layers.conv2d(inputs=conv3,filters=64, kernel_size=[3, 3], kernel_initializer = \
                tf.constant_initializer(conv_weights['conv2d_3/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_3/bias']), \
                strides=1, activation=tf.nn.relu, padding = 'SAME')


            conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='SAME')
            print('conv4',conv4)
            #conv5 = tf.nn.relu(tf.nn.conv2d(input=conv4, filter=conv_weights['conv2d_4/kernel'],strides=[1,1,1,1],padding='SAME'))
            #conv6 = tf.nn.relu(tf.nn.conv2d(input=conv5, filter=conv_weights['conv2d_5/kernel'],strides=[1,1,1,1],padding='SAME'))
            #conv6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
            #print('conv6 ', conv6)
            """
            #conv1 = tf.layers.conv2d(inputs=self.input_data,filters=32,kernel_size=[3, 3], strides=1,activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(inputs=conv1,filters=32,kernel_size=[3, 3], strides=1,activation=tf.nn.relu)
            $pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            pool2 = tf.nn.dropout(pool2, tf.constant(0.5,dtype=np.float32)+self.droput_keep_pct/2)
            conv3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[3, 3],strides=1,activation=tf.nn.relu)
            conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[3, 3],strides=1,activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
            pool4 = tf.nn.dropout(pool4, tf.constant(0.5,dtype=np.float32)+self.droput_keep_pct/2)
            print('POOL4', pool4)
            """
            self.x = tf.reshape(conv4,[par['batch_size'], -1])


        elif par['task'] == 'mnist':
            self.x = self.input_data

        self.spike_loss = 0
        self.td_gating = []
        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
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
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['layer_dims'][n+1]]), trainable = True)


                if n < par['n_layers']-2:
                    dend_activity = tf.nn.relu(tf.tensordot(self.x, W, ([1],[0]))  + b)
                    print('dend_activity', dend_activity)
                    self.x = tf.nn.dropout(tf.reduce_sum(dend_activity*self.td_gating[n], axis=1), self.droput_keep_pct)

                    self.spike_loss += tf.reduce_mean(self.x)

                else:
                    if par['dendrites_final_layer']:
                        dend_activity = tf.tensordot(self.x, W, ([1],[0])) + b
                        self.y = tf.nn.softmax(tf.reduce_sum(dend_activity*self.td_gating[n], axis=1), dim = 1)
                        print('Y',self.y)
                    else:
                        self.y = tf.nn.softmax(tf.matmul(self.x,W) + b, dim = 1)
                        print('Y',self.y)



    def optimize(self):

        epsilon = 1e-4
        #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)

        # Use all trainable variables, except those in the convolutional layers
        #variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        print('variables ', variables)

        adam_optimizer = AdamOpt.AdamOpt(variables, learning_rate = self.learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)

        small_omega_var = {}
        gates = {}
        previous_weights_mu_minus_1 = {}
        self.big_omega_var = {}
        aux_loss = 0.0

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        #for var, task_num in zip(variables, range(n_tasks)):
        if par['omega_c'] > 0:
            for var in variables:

                gates[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )

                previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
                self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

                aux_loss += tf.reduce_sum(tf.multiply( self.big_omega_var[var.op.name], tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))

                reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name], var ) )


                update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name],  tf.div(small_omega_var[var.op.name], \
                	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))

            # After each task is complete, call update_big_omega and reset_small_omega
            self.update_big_omega = tf.group(*update_big_omega_ops)
            #new_big_omega_var = big_omega_var

            # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
            self.reset_small_omega = tf.group(*reset_small_omega_ops)

        self.task_loss = -tf.reduce_mean( self.target_data*tf.log(self.y+epsilon) + (1.-self.target_data)*tf.log(1.-self.y+epsilon) )

        self.total_loss = self.task_loss + 0.04*self.spike_loss

        # Gradient of the loss function for the current task
        gradients = optimizer.compute_gradients(self.task_loss, var_list = variables)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        for var in variables:
            if var.op.name.find('conv') == 0:
                # convolutional layer
                tf.assign(gates[var.op.name], self.gate_conv)
            else:
                # fully connected layer
                layer_num = int([s for s in var.op.name if s.isdigit()][0])
                var_dim = var.get_shape()[0].value
                if par['clamp'] == 'dendrites' and (layer_num < par['n_layers']-2 or par['dendrites_final_layer']):
                    td_gating = tf.tile(tf.reduce_mean(self.td_gating[layer_num], axis=0, keep_dims = True), [var_dim, 1, 1])
                    tf.assign(gates[var.op.name], td_gating)
                else:
                    tf.assign(gates[var.op.name], 1)

        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        if par['omega_c'] > 0:
        	#m1,v,dg = adam_optimizer.compute_gradients(self.total_loss + par['omega_c']*aux_loss, gates)
            self.train_op = adam_optimizer.compute_gradients(self.total_loss + par['omega_c']*aux_loss, gates)
        else:
            m1,v,dg = adam_optimizer.apply_gradients(self.total_loss + par['omega_c']*aux_loss, gates)
        	#self.train_op = adam_optimizer.apply_gradients(self.total_loss, gates)
        #self.train_op = tf.group(*m1)
        self.delta_grads = adam_optimizer.return_means()
        self.delta_grads1 = adam_optimizer.return_delta_grads()


        # This is called every batch
        #print(small_omega_var.keys())
        if par['omega_c'] > 0:
            for i, (grad,var) in enumerate(gradients):
                print(i,var.op.name)
                update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name], self.delta_grads1[var.op.name]*gradients[i][0] ) )

            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

        #self.train_op = adam_optimizer.apply_gradients()

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.target_data,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def main():

    determine_top_down_weights()

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
    y   = tf.placeholder(tf.float32, [ par['batch_size'], par['layer_dims'][-1]], 'out')
    droput_keep_pct = tf.placeholder(tf.float32, [] , 'dropout')
    learning_rate = tf.placeholder(tf.float32, [] , 'learning_rate')
    gate_conv = tf.placeholder(tf.float32, [] , 'gate_conv')

    stim = stimulus.Stimulus()


    with tf.Session() as sess:
        model   = Model(x, td, y, droput_keep_pct, learning_rate, gate_conv)
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        for task in range(0,par['n_tasks']):

            #gate = 1 if (par['task'] == 'mnist' or task == 0) else 0
            gate = 0
            keep_pct = par['keep_pct'] if (par['task'] == 'mnist' or task > 0) else 1.0
            #gate = 1

            for i in range(par['n_train_batches']):

                stim_in, y_hat, td_in = stim.make_batch(task, test = False)

                # learning rate is set to zero for first 100 epochs of cifar task
                # this is to prevent interference from AdamOptimizer
                lr = par['learning_rate'] if (par['task']=='mnist' or (par['task']=='cifar' and i > -1)) else 0

                #stim_in += np.random.normal(0,0.02,size=stim_in.shape)
                #print(stim_in.shape, y_hat.shape, td_in.shape, np.mean(td_in,axis=0))

                if par['omega_c'] > 0:
                    _,loss,spike_loss,_,delta_grads = sess.run([model.train_op,model.task_loss,model.spike_loss,model.update_small_omega, model.delta_grads], \
                        feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct:keep_pct, learning_rate: lr, gate_conv: gate})
                else:
                    sess.run(model.train_op, feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct: par['keep_pct'], \
                        learning_rate: lr, gate_conv: gate})

                if i//100 == i/100:
                    dgs = [np.mean(dg) for dg in delta_grads.values()]
                    #dgs = [np.mean(g) for g,v in delta_grads]
                    print(i, loss, spike_loss, dgs)

            # if training on the cifar task, don't update omegas on the 0th task
            if par['omega_c'] > 0 and (par['task']=='mnist' or (par['task']=='cifar' and task > -1)):
                sess.run(model.update_big_omega,feed_dict={td:td_in})
                sess.run(model.reset_small_omega)
                big_omegas = sess.run(model.big_omega_var)

            if par['task']=='mnist':
                accuracy = np.zeros((task+1))
                for test_task in range(task+1):
                    stim_in, y_hat, td_in = stim.make_batch(test_task, test = True)
                    accuracy[test_task] = sess.run(model.accuracy, feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct:1.0, gate_conv: gate})
            elif par['task']=='cifar' and task > -1:
                accuracy = np.zeros((task+1))
                for test_task in range(0,task+1):
                    stim_in, y_hat, td_in = stim.make_batch(test_task, test = True)
                    accuracy[test_task] = sess.run(model.accuracy, feed_dict={x:stim_in, td:td_in, y:y_hat, droput_keep_pct:1.0, gate_conv: gate})
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
    #td_weight_fn = par['save_dir'] + 'top_down_weights' + '.pkl'

    if par['train_top_down']:

        print('Training top-down weights')
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = top_down.TrainTopDown()
            sess.run(tf.global_variables_initializer())

            for i in range(par['n_batches_top_down']):
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

        top_down_results = {'W_td': W_td, 'td_cases': par['td_cases']}
        par['W_td0'] = W_td
        print('Finished training top-down weights... saving data')

        pickle.dump(top_down_results, open(td_weight_fn,'wb'))

    else:
        print('Loading top-down weights')
        top_down_results = pickle.load(open(td_weight_fn,'rb'))
        par['W_td0'] = top_down_results['W_td']
        par['td_cases']  = top_down_results['td_cases']

try:
    main()
except KeyboardInterrupt:
    quit('\nQuit by KeyboardInterrupt.')
