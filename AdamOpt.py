import numpy as np
import tensorflow as tf

class AdamOpt:

    def __init__(self, variables, learning_rate = 0.001):

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.variables = variables
        self.learning_rate = learning_rate
        self.reset_params()

        self.grad_descent = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

    def reset_params(self):

        self.m = {}
        self.v = {}
        self.delta_grads = {}
        self.t = 0
        for var in self.variables:
            self.m[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.v[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.delta_grads[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

    def compute_gradients(self, loss, gate):

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)

        self.t += 1
        lr = self.learning_rate*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)

        self.update_var_op = []

        for grads, var in self.gradients:
            new_m = self.beta1*self.m[var.op.name] + (1-self.beta1)*grads
            new_v = self.beta2*self.v[var.op.name] + (1-self.beta2)*grads*grads
            #delta_grad = - lr*gate[var.op.name]*new_m/(new_v + self.epsilon)
            delta_grad = - lr*new_m/(new_v + self.epsilon)

            # update var by delta gradient
            self.update_var_op.append(tf.assign_add(var, delta_grad))

            self.update_var_op.append(tf.assign(self.m[var.op.name], new_m))
            self.update_var_op.append(tf.assign(self.v[var.op.name], new_v))
            self.update_var_op.append(tf.assign(self.delta_grads[var.op.name], delta_grad))

        return tf.group(*self.update_var_op)



    def return_delta_grads(self):

        return self.delta_grads

    def return_means(self):

        return self.delta_grads

    def return_gradients(self):

        return self.gradients

    def apply_gradients(self, loss, gate):

        self.compute_gradients(loss, gate)
        self.apply_grads = []

        for var in self.variables:
            print('Applying grad ', var.op.name)
            print(var)
            self.apply_grads.append(tf.assign_add(var, self.delta_grads[var.op.name]))

        return tf.group(*self.apply_grads)
