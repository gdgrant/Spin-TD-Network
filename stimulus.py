"""
2017/06/16 Gregory Grant
"""

import numpy as np
from parameters import *


class Stimulus:

    def __init__(self):

        self.stim_tuning, self.outputs = self.generate_mnist_tuning()
        self.td_tuning = par['td_cases']


    def generate_mnist_tuning(self):
        from mnist import MNIST
        mndata = MNIST('./mnist/data/original')
        images, labels = mndata.load_training()

        self.num_examples = 60000
        self.num_outputs  = 10

        stim_tuning = np.array(images)/255
        tasks = [stim_tuning]
        for t in range(par['n_tasks']-1):
            permutation = np.random.permutation(784)
            tasks.append(stim_tuning[:,permutation])
        tasks = np.array(tasks)

        label_vector = np.zeros([self.num_examples,self.num_outputs])
        for n, l in enumerate(labels):
            label_vector[n,l] = 1

        return tasks, label_vector


    def make_batch(self, task_id):
        stim = []
        td   = []
        out  = []
        for i in range(par['batch_size']):
            sample_id = np.random.randint(0,self.num_examples)
            stim.append(self.stim_tuning[task_id,sample_id])
            td.append(self.td_tuning[task_id])
            out.append(self.outputs[sample_id])

        return np.array(stim), np.array(td), np.array(out)
