import numpy as np
import tensorflow as tf
from itertools import product

class AdamOpt:

    """
    Example of use:

    optimizer = AdamOpt.AdamOpt(variables, learning_rate=self.lr)
    self.train = optimizer.compute_gradients(self.loss, gate=0)
    gvs = optimizer.return_gradients()
    self.g = gvs[0][0]
    self.v = gvs[0][1]
    """

    def __init__(self, variables, learning_rate = 0.001):

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.t = 0
        self.variables = variables
        self.learning_rate = learning_rate

        self.m = {}
        self.v = {}
        self.delta_grads = {}
        for var in self.variables:
            self.m[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.v[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.delta_grads[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

        self.grad_descent = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)


    def reset_params(self):

        self.t = 0
        reset_op = []
        for var in self.variables:
            reset_op.append(tf.assign(self.m[var.op.name], tf.zeros(var.get_shape())))
            reset_op.append(tf.assign(self.v[var.op.name], tf.zeros(var.get_shape())))
            reset_op.append(tf.assign(self.delta_grads[var.op.name], tf.zeros(var.get_shape())))

        return tf.group(*reset_op)


    def compute_gradients(self, loss, gate, apply = True):

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)

        self.t += 1
        lr = self.learning_rate*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        #lr = self.learning_rate
        self.update_var_op = []

        for (grads, _), var in zip(self.gradients, self.variables):
            new_m = self.beta1*self.m[var.op.name] + (1-self.beta1)*grads
            new_v = self.beta2*self.v[var.op.name] + (1-self.beta2)*grads*grads
            #delta_grad =  - lr*gate[var.op.name]*new_m/(tf.sqrt(new_v) + self.epsilon)

            delta_grad = - lr*new_m/(tf.sqrt(new_v) + self.epsilon)
            delta_grad = tf.clip_by_norm(delta_grad, 1)

            #delta_grad = self.dendritic_competition(delta_grad, var.op.name)

            self.update_var_op.append(tf.assign(self.m[var.op.name], new_m))
            self.update_var_op.append(tf.assign(self.v[var.op.name], new_v))
            #self.update_var_op.append(tf.assign(self.delta_grads[var.op.name], delta_grad))
            self.update_var_op.append(tf.assign(self.delta_grads[var.op.name], delta_grad))
            if apply:
                self.update_var_op.append(tf.assign_add(var, delta_grad))

        return tf.group(*self.update_var_op)

    def dendritic_competition(self, delta_grad, var_name):

        corrected_delta_grads = []

        if var_name.find('W') == -1 or var_name.find('2') > 0:
            return delta_grad
        else:
            print(var_name, var_name.find('2'))
            delta_grad_branches = tf.unstack(delta_grad, axis=2)
            corrected_delta_grads = []

            # cycle through dendrites, post-synaptic neurons
            for delta_branch in delta_grad_branches:
                #corrected_delta_grads.append(delta_branch*tf.nn.softmax(tf.abs(3*delta_branch), dim = 0))
                s = tf.exp(tf.abs(delta_branch))
                corrected_delta_grads.append(delta_branch*s/(1e-6+tf.reduce_mean(s)))

            corrected_delta_grads = tf.stack(corrected_delta_grads, axis = 2)
            print(var_name, ' corrected_delta_grads', corrected_delta_grads)
            return corrected_delta_grads

    def return_delta_grads(self):
        return self.delta_grads

    def return_means(self):
        return self.m

    def return_grads_and_vars(self):
        return self.gradients
