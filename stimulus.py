"""
2017/10/10 Gregory Grant, Nicolas Masse


Associated Retrieval Task:
10 tasks, (ten letters, ten numbers for each sequence)
input_space = [370] (260 for ten letters, 100 for ten digits, 10 for ??)

Example:
t = 0 : all letters (a,b,c,d,e)
t = 1 : all numbers (1,2,3,4,5)

t = 2 : all letters (a,b,c,d,e)
t = 3 : all numbers (1,2,3,4,5)

t = 4 : ??
t = 5 : letter cue  (b,b,b,b,b)
t = 6 : response    (2)
"""

import numpy as np
from parameters import *
import matplotlib.pyplot as plt
import time


class Stimulus:

    def __init__(self):
        self.input_shape  = par['input_shape']
        self.output_shape = par['output_shape']

        self.blank_character()
        self.timeline()

        self.let_feed = {
            'a'       : par['cue_space_size'],
            'size'    : [par['samples_per_trial']*2],
            'replace' : False
            }
        self.num_feed = {
            'a'       : par['sample_space_size'],
            'size'    : [par['samples_per_trial']*2],
            'replace' : True
            }


    def blank_character(self):
        self.blank = {'on':np.ones(10), 'off':np.zeros(10)}


    def timeline(self):
        self.t = []
        for i in range(par['n_input_phases']+1):
            self.t.append(par['dead_time']+par['steps_per_input']*i)


    def id_to_vector(self, n, size):
        v = np.zeros(size)
        v[n] = 1
        return v


    def split_1d(self, a):
        if a.shape[0]//2 != a.shape[0]/2:
            raise Exception('Bad split_1d size.  Use even number of elements')
        if len(a.shape) != 1:
            raise Exception('Bad split_1d size.  Use 1d array.')
        return a[:a.shape[0]//2], a[a.shape[0]//2:]


    def get_input_sets(self):
        l = []
        n = []
        c = []
        for i in range(par['batch_size']):
            l.append(np.random.choice(**self.let_feed))
            n.append(np.random.choice(**self.num_feed))
            c.append(np.random.choice(l[-1]))

        letters = np.stack(l, axis=1)
        numbers = np.stack(n, axis=1)
        cues    = np.array(c)

        return letters, numbers, cues


    def input_across_time(self, letters, numbers, cues, task_id):
        batch = np.zeros(self.input_shape)
        end_of_cues = par['samples_per_trial']*par['cue_space_size']
        end_of_samples = end_of_cues + par['samples_per_trial']*par['sample_space_size']

        # Blank character
        batch[self.t[4]:self.t[5],end_of_samples:end_of_samples+par['blank_character_size'],:] = 1.

        # TD input
        batch[:, -par['n_td']:,:] = par['td_cases'][task_id][np.newaxis,:,np.newaxis]

        # Create each batch
        for b in range(par['batch_size']):
            let = self.split_1d(letters[:,b])
            num = self.split_1d(numbers[:,b])

            for (lid, l), (nid, n) in zip(enumerate(let[0]), enumerate(num[0])):
                la = lid     * par['cue_space_size']
                lb = (lid+1) * par['cue_space_size']
                na = nid     * par['sample_space_size'] + end_of_cues
                nb = (nid+1) * par['sample_space_size'] + end_of_cues

                l1 = self.id_to_vector(l, par['cue_space_size'])
                batch[self.t[0]:self.t[1],la:lb,b] = l1

                n1 = self.id_to_vector(n, par['sample_space_size'])
                batch[self.t[1]:self.t[2],na:nb,b] = n1

            for (lid, l), (nid, n) in zip(enumerate(let[1]), enumerate(num[1])):
                la = lid     * par['cue_space_size']
                lb = (lid+1) * par['cue_space_size']
                na = nid     * par['sample_space_size'] + end_of_cues
                nb = (nid+1) * par['sample_space_size'] + end_of_cues

                l2 = self.id_to_vector(l, par['cue_space_size'])
                batch[self.t[2]:self.t[3],la:lb,b] = l2

                n2 = self.id_to_vector(n, par['sample_space_size'])
                batch[self.t[3]:self.t[4],na:nb,b] = n2

            c = np.concatenate([self.id_to_vector(cues[b], par['cue_space_size'])]*par['samples_per_trial'], axis=0)
            batch[self.t[5]:self.t[6],:end_of_cues,b] = c

        return batch


    def output_across_time(self, letters, numbers, cues, task_id):
        batch = np.zeros(self.output_shape)
        for b in range(par['batch_size']):
            i = np.where(letters[:,b]==cues[b])[0][0]
            v = self.id_to_vector(numbers[:,b][i], par['sample_space_size'])
            batch[-par['steps_per_input']:,:,b] = v

        return batch


    def make_batch(self, task_id):
        letters, numbers, cues = self.get_input_sets()

        input_batch  = self.input_across_time(letters, numbers, cues, task_id)
        output_batch = self.output_across_time(letters, numbers, cues, task_id)

        return input_batch, output_batch

s = Stimulus()
i,o = s.make_batch(np.random.choice(par['n_tasks']))
