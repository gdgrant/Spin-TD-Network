"""
2017/06/16 Gregory Grant, Nicolas Masse
"""

import numpy as np
from parameters import *
import pickle


class Stimulus:

    def __init__(self):

        if par['task'] == 'mnist':
            self.generate_mnist_tuning()
        elif par['task'] == 'cifar':
            self.cifar10_dir = './cifar/cifar-10-python/'
            self.cifar100_dir = './cifar/cifar-100-python/'
            self.num_cifar_labels = 110
            self.cifar_labels_per_task = 10
            self.generate_cifar_tuning()
            self.find_cifar_indices()



    def generate_mnist_tuning(self):

        from mnist import MNIST
        mndata = MNIST('./mnist/data/original')
        self.mnist_train_images, self.mnist_train_labels = mndata.load_training()
        self.mnist_test_images, self.mnist_test_labels   = mndata.load_testing()

        self.num_train_examples = len(self.mnist_train_images)
        self.num_test_examples  = len(self.mnist_test_images)
        self.num_outputs        = 10

        self.mnist_train_images = np.array(self.mnist_train_images)
        self.mnist_test_images = np.array(self.mnist_test_images)

        self.mnist_permutation = []
        for t in range(par['n_tasks']):
            self.mnist_permutation.append(np.random.permutation(784))

    def generate_cifar_tuning(self):

        self.cifar_train_images = np.array([])
        self.cifar_train_labels = np.array([])

        """
        Load CIFAR-10 data
        """
        for i in range(5):
            x =  pickle.load(open(self.cifar10_dir + 'data_batch_' + str(i+1),'rb'), encoding='latin1')
            self.cifar_train_images = np.vstack((self.cifar_train_images, x['data'])) if self.cifar_train_images.size else  x['data']
            labels = np.reshape(np.array(x['labels']),(-1,1))
            self.cifar_train_labels = np.vstack((self.cifar_train_labels, labels))  if self.cifar_train_labels.size else labels

        x =  pickle.load(open(self.cifar10_dir + 'test_batch','rb'), encoding='latin1')
        self.cifar_test_images = np.array(x['data'])
        self.cifar_test_labels = np.reshape(np.array(x['labels']),(-1,1))

        """
        Load CIFAR-100 data
        """
        x =  pickle.load(open(self.cifar100_dir + 'train','rb'), encoding='latin1')
        self.cifar_train_images = np.vstack((self.cifar_train_images, x['data']))
        labels = np.reshape(np.array(x['fine_labels'])+10,(-1,1))
        self.cifar_train_labels = np.vstack((self.cifar_train_labels, labels))

        x =  pickle.load(open(self.cifar100_dir + 'test','rb'), encoding='latin1')
        self.cifar_test_images = np.vstack((self.cifar_test_images, x['data']))
        labels = np.reshape(np.array(x['fine_labels'])+10,(-1,1))
        self.cifar_test_labels = np.vstack((self.cifar_test_labels, labels))


    def find_cifar_indices(self):

        self.cifar_train_ind = []
        self.cifar_test_ind = []

        for i in range(0, self.num_cifar_labels, self.cifar_labels_per_task):
            self.cifar_train_ind.append(np.where((self.cifar_train_labels>=i)*(self.cifar_train_labels<i+self.cifar_labels_per_task))[0])
            self.cifar_test_ind.append(np.where((self.cifar_test_labels>=i)*(self.cifar_test_labels<i+self.cifar_labels_per_task))[0])


    def generate_cifar_batch(self, task_num, test = False):

        if test:
            ind = self.cifar_test_ind[task_num]
        else:
            ind = self.cifar_train_ind[task_num]
        #q = np.random.permutation(len(ind))[:par['batch_size']]
        q = np.random.randint(0,len(ind),par['batch_size'])
        batch_data = np.zeros((par['batch_size'], 32,32,3), dtype = np.float32)
        batch_labels = np.zeros((par['batch_size'], self.cifar_labels_per_task), dtype = np.float32)
        for i in range(par['batch_size']):
            if test:
                k = int(self.cifar_test_labels[ind[q[i]]] - task_num*self.cifar_labels_per_task)
                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.cifar_test_images[ind[q[i]], :],(1,32,32,3)))/255
            else:
                k = int(self.cifar_train_labels[ind[q[i]]] - task_num*self.cifar_labels_per_task)
                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.cifar_train_images[ind[q[i]], :],(1,32,32,3)))/255

        return batch_data, batch_labels


    def generate_mnist_batch(self, task_num, test = False):

        if test:
            #q = np.random.permutation(self.num_test_examples)[:par['batch_size']]
            q = np.random.randint(0,self.num_test_examples,par['batch_size'])
        else:
            #q = np.random.permutation(self.num_train_examples)[:par['batch_size']]
            q = np.random.randint(0,self.num_train_examples,par['batch_size'])

        batch_data = np.zeros((par['batch_size'], 28**2), dtype = np.float32)
        batch_labels = np.zeros((par['batch_size'], self.num_outputs), dtype = np.float32)
        for i in range(par['batch_size']):
            if test:
                k = self.mnist_test_labels[q[i]]
                batch_labels[i, k] = 1
                batch_data[i, :] = self.mnist_test_images[q[i]][self.mnist_permutation[task_num]]
            else:
                k = self.mnist_train_labels[q[i]]
                batch_labels[i, k] = 1
                batch_data[i, :] = self.mnist_train_images[q[i]][self.mnist_permutation[task_num]]

        return batch_data, batch_labels

    def make_batch(self, task_num, test=False):

        if par['task'] == 'mnist':
            batch_data, batch_labels = self.generate_mnist_batch(task_num, test)
        elif par['task'] == 'cifar':
            batch_data, batch_labels = self.generate_cifar_batch(task_num, test)
        else:
            print('Unrecognized task')

        top_down = np.tile(np.reshape(par['td_cases'][task_num, :],(1,-1)),(par['batch_size'],1))

        return batch_data/255, batch_labels, top_down
