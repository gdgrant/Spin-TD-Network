import tensorflow as tf
import numpy as np
import os

import AdamOpt

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
SIZE = [40]*2

class Model:

    def __init__(self, x, y, learning_rate):
        print('Loading model.')
        self.x = x
        self.y_hat = y
        self.lr = learning_rate

        self.run_model()
        self.optimize()


    def run_model(self):
        w = tf.get_variable('w', initializer=tf.zeros(SIZE), trainable=True)
        self.y = tf.matmul(self.x, w)


    def optimize(self):

        self.loss = tf.reduce_mean(tf.square(self.y-self.y_hat))
        variables = [var for var in tf.trainable_variables()]

        if True:
            # Adam optimizer scenario
            optimizer = AdamOpt.AdamOpt(variables, learning_rate=self.lr)
            self.train = optimizer.compute_gradients(self.loss, gate=0)
            gvs = optimizer.return_gradients()

            self.g = gvs[0][0]
            self.v = gvs[0][1]

        else:
            # GD optimizer scenario
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            gvs = optimizer.compute_gradients(self.loss)
            self.train = optimizer.apply_gradients(gvs)

            self.g = tf.reduce_mean(gvs[0][0])
            self.v = tf.reduce_mean(gvs[0][1])


def main():
    print('Running with size', SIZE)

    # Reset TensorFlow graph
    tf.reset_default_graph()

    x    = tf.placeholder(tf.float32, SIZE, 'x')
    y    = tf.placeholder(tf.float32, SIZE, 'y')
    lr   = tf.placeholder(tf.float32, [])
    data = np.ones(SIZE, dtype=np.float32)

    with tf.Session() as sess:
        model   = Model(x, y, lr)
        sess.run(tf.global_variables_initializer())

        print('Model loaded.\n')

        x_hat = np.ones(SIZE, dtype=np.float32)*10
        y_hat = np.ones(SIZE, dtype=np.float32)

        for m in range(500):
            _, loss, grad, var = sess.run([model.train, model.loss, model.g, model.v], \
                                feed_dict={x:x_hat, y:y_hat, lr:0.001})

            l = 'Loss: ' + str(loss)
            g = 'Grad: ' + str(np.mean(grad))
            v = 'Var:  ' + str(np.mean(var))
            if m%10==0:
                print(str(m).ljust(5), '|', l.ljust(20), '|', g.ljust(20), '|', v.ljust(20))


main()
