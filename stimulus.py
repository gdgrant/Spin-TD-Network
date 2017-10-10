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


class Stimulus:

    def __init__(self):
        self.input_shape  = [par['n_steps'], par['n_inputs']+par['n_td'], par['batch_size']]
        self.output_shape = [par['n_steps'], par['n_output'], par['batch_size']]

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

    def split(self, a):
        print('Splitting')
        print(a)
        quit()


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
        batch[self.t[4]:self.t[5],end_of_samples:end_of_samples+10,:] = 1.

        # TD input
        batch[:,-par['n_td']:,:] = self.id_to_vector(task_id, par['n_td'])[np.newaxis,:,np.newaxis]
        print('Batching')
        for b in range(par['batch_size']):
            for (lid, l), (nid, n) in zip(enumerate(letters[:,2*b:2*b+2]), enumerate(numbers[:,2*b:2*b+2])):
                la = lid     * par['cue_space_size']
                lb = (lid+1) * par['cue_space_size']
                na = nid     * par['sample_space_size'] + end_of_cues
                nb = (nid+1) * par['sample_space_size'] + end_of_cues

                l1 = self.id_to_vector(l[0], par['cue_space_size'])
                batch[self.t[0]:self.t[1],la:lb,b] = l1


                n1 = self.id_to_vector(n[0], par['sample_space_size'])
                print(n1.shape)
                print(batch[self.t[1]:self.t[2],na:nb,b].shape)
                batch[self.t[1]:self.t[2],na:nb,b] = n1

                l2 = self.id_to_vector(l[1], par['cue_space_size'])
                batch[self.t[2]:self.t[3],la:lb,b] = l2

                n2 = self.id_to_vector(n[1], par['sample_space_size'])
                batch[self.t[3]:self.t[4],na:nb,b] = n2

            c = np.concatenate([self.id_to_vector(cues[b], par['cue_space_size'])]*par['samples_per_trial'], axis=0)
            batch[self.t[5]:self.t[6],:end_of_cues,b] = c

        return batch


    def make_batch(self):
        task_id = np.random.choice(par['n_tasks'])
        letters, numbers, cues = self.get_input_sets()
        print(letters[:,0])
        print(numbers[:,0])
        print(cues[0])

        print(letters.shape, numbers.shape, cues.shape)

        batch = self.input_across_time(letters, numbers, cues, task_id)

        plt.imshow(batch[:,:,0])
        plt.show()

s = Stimulus()
s.make_batch()
