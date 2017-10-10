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


class Stimulus:

    def __init__(self):
        self.input_shape  = [par['n_steps'], par['n_inputs']+par['n_td'], par['batch_size']]
        self.output_shape = [par['n_steps'], par['n_output'], par['batch_size']]
        self.set_feed = {
            'a'       : 26,
            'size'    : [par['samples_per_trial']],
            'replace' : False
            }

    def get_input_sets(self):
        l = []
        n = []
        c = []
        for i in range(par['batch_size']):
            l.append(np.random.choice(**self.set_feed))
            n.append(np.random.choice(**self.set_feed))
            c.append(np.random.choice(l[-1]))

        letters = np.stack(l, axis=1)
        numbers = np.stack(n, axis=1)
        cues    = np.array(c)

        return letters, numbers, cues

    def make_batch(self):
        letters, numbers, cues = self.get_input_sets()
        print(letters[:,0])
        print(numbers[:,0])
        print(cues[0])


s = Stimulus()
s.make_batch()
