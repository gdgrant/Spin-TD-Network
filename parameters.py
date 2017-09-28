### Parameters for RNN research
### Authors: Nicolas Masse, Gregory Grant, Catherine Lee, Varun Iyer
### Date:    3 August, 2017

import numpy as np
import tensorflow as tf
from itertools import product

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # General parameters
    'save_dir'          : './savedir/',
    'loss_function'     : 'cross_entropy',    # cross_entropy or MSE
    'td_loss_type'      : 'pairwise_random',
    'learning_rate'     : 0.005,
    'connection_prob'   : 1.,

    # Task specs
    'n_tasks'           : 10,

    # Network shape
    'n_td'              : 20,
    'n_dendrites'       : 3,
    'layer_dims'        : [28**2, 40, 40, 10],

    # Training specs
    'batch_size'        : 8,
    'num_train_batches' : 300,
    'num_test_batches'  : 20,
    'num_iterations'    : 2,
    'switch_iteration'  : 1,
}

############################
### Dependent parameters ###
############################

def generate_init_weight(dims):
    n = np.float32(np.random.gamma(shape=0.25, scale=1.0, size=dims))
    n *= (np.random.rand(*dims) < par['connection_prob'])
    return np.float32(n)


def gen_td_cases():

    # will create par['n_tasks'] number of tunings, each with exactly n non-zero elements equal to one
    # the distance between all tuned will be d
    n = 3
    d = 4
    tuning = np.zeros([par['n_tasks'], par['n_td']])
    for i in range(par['n_tasks']):
        q = np.random.permutation(par['n_td'])[:n]
        if i == 0:
            tuning[i, q] = 1
        else:
            found_tuning = False
            while not found_tuning:
                potential_tuning = np.zeros((par['n_td']))
                potential_tuning[q] = 1
                found_tuning = True
                for j in range(i-1):
                    pair_dist = np.sum(np.abs(potential_tuning - tuning[j,:]))
                    if not (pair_dist >= d-2 and pair_dist <= d):
                        found_tuning = False
                        q = np.random.permutation(par['n_td'])[:n]
                        #print(i, pair_dist, q)
                        break
            tuning[i, :] = potential_tuning

    return tuning

def gen_td_targets():

    td_targets = []
    for n, td in product(range(par['n_layers']-1), par['td_cases']):
        target = np.zeros((par['n_tasks'],par['n_dendrites'], par['layer_dims'][n+1]))
        for i, j in product(range(par['n_tasks']), range(par['layer_dims'][n+1])):
            q = np.random.randint(par['n_dendrites'])
            target[i,q,j] = 1
        td_targets.append(target)

    return td_targets

def update_dependencies():
    """
    Updates all parameter dependencies
    """

    par['n_layers'] = len(par['layer_dims'])
    par['td_cases'] = gen_td_cases()
    par['td_targets'] = gen_td_targets()


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
    update_dependencies()


update_dependencies()
print("--> Parameters successfully loaded.\n")
