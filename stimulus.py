"""
2017/06/16 Gregory Grant, Nicolas Masse


Associated Retrieval Task:
10 tasks, (ten letters, ten numbers for each sequence)
input_space = [370] (260 for ten letters, 100 for ten digits, 10 for ??)

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
        pass
