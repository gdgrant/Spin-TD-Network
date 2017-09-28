"""
2017/06/16 Gregory Grant
"""

import numpy as np
from parameters import *
import pickle


class Stimulus:

    def __init__(self):

        self.train_data, self.test_data = self.generate_mnist_tuning()
        self.td_tuning = par['td_cases']


    def generate_mnist_tuning(self):
        from mnist import MNIST
        mndata = MNIST('./mnist/data/original')
        train_images, train_labels = mndata.load_training()
        test_images, test_labels   = mndata.load_testing()

        self.num_train_examples = 60000
        self.num_test_examples  = 10000
        self.num_outputs        = 10

        train_tuning = np.array(train_images)/255
        test_tuning  = np.array(test_images)/255
        train_in     = [train_tuning]
        test_in      = [test_tuning]
        for t in range(par['n_tasks']-1):
            permutation = np.random.permutation(784)
            train_in.append(train_tuning[:,permutation])
            test_in.append(test_tuning[:,permutation])
        train_in = np.array(train_in)
        test_in  = np.array(test_in)

        train_out = np.zeros([self.num_train_examples,self.num_outputs])
        test_out  = np.zeros([self.num_test_examples,self.num_outputs])
        for n, l in enumerate(train_labels):
            train_out[n,l] = 1
        for n, l in enumerate(test_labels):
            test_out[n,l] = 1

        return [train_in, train_out], [test_in, test_out]


    def generate_cifar_tuning(self):
        c10_train_images = []
        c10_train_labels = []
        for i in range(5):
            with open('./cifar/cifar-10-batches-py/data_batch_'+str(i+1), 'rb') as c:
                batch = pickle.load(c, encoding='bytes')
                c10_train_images.append(batch[b'data'])
                c10_train_labels.append(batch[b'labels'])
        c10_train_images = np.concatenate(c10_train_images)
        c10_train_labels = np.concatenate(c10_train_labels)

        with open('./cifar/cifar-10-batches-py/test_batch', 'rb') as c:
            batch = pickle.load(c, encoding='bytes')
            c10_test_images = batch[b'data']
            c10_test_labels = batch[b'labels']

        with open('./cifar/cifar-100-python/train', 'rb') as c:
            batch = pickle.load(c, encoding='bytes')
            c100_train_images = batch[b'data']
            c100_train_labels = batch[b'labels']

        with open('./cifar/cifar-100-python/test', 'rb') as c:
            batch = pickle.load(c, encoding='bytes')
            c100_test_images = batch[b'data']
            c100_test_labels = batch[b'labels']

        ######### IN PROGRESS #########

    def make_batch(self, task_id, test=False):
        if test:
            stim_tuning     = self.test_data[0]
            output_tuning   = self.test_data[1]
            num_examples    = self.num_test_examples
        else:
            stim_tuning     = self.train_data[0]
            output_tuning   = self.train_data[1]
            num_examples    = self.num_train_examples

        stim = []
        td   = []
        out  = []
        for i in range(par['batch_size']):
            sample_id = np.random.randint(0,num_examples)
            stim.append(stim_tuning[task_id,sample_id])
            td.append(self.td_tuning[task_id])
            out.append(output_tuning[sample_id])

        return np.array(stim), np.array(td), np.array(out)

s = Stimulus()
st, t, o = s.make_batch(0, test=True)
print(np.shape(st))
print(np.shape(t))
print(np.shape(o))
