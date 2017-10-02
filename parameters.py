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
    'save_dir'              : './savedir/',
    'loss_function'         : 'cross_entropy',    # cross_entropy or MSE
    'learning_rate'         : 0.001,
    'train_top_down'        : False,
    'task'                  : 'cifar',
    'save_analysis'         : True,

    # Task specs
    'n_tasks'               : 11,

    # Network shape
    'n_td'                  : 40,
    'n_dendrites'           : 3,
    #'layer_dims'            : [28**2, 2000, 2000, 10], # mnist
    'layer_dims'            : [1600, 600, 600, 10], #cifar
    'dendrites_final_layer' : False,

    # Dropout
    'keep_pct'              : 1.0,

    # Training specs
    'batch_size'            : 1024,
    'n_train_batches'       : 2000,
    'n_batches_top_down'    : 20000,

    # Omega parameters
    'omega_c'               : 0.1,
    'omega_xi'              : 0.1,

    # Projection of top-down activity
    # Only one can be True
    'clamp'                 : 'dendrites', # can be either 'dendrites' or 'neurons' or None
    'n_clamp'                : 5

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

    par['td_cases'] = np.zeros((par['n_tasks'], par['n_td']), dtype = np.float32)

    if par['clamp'] == 'neurons':
        for n in range(par['n_tasks']):
            par['td_cases'][n, n%par['n_td']] = 1


    elif par['clamp'] == 'dendrites':
        n = 3
        min_dist = 4
        for i in range(par['n_tasks']):
            q = np.random.permutation(par['n_td'])[:n]
            if i == 0:
                par['td_cases'][i, q] = 1
            else:
                found_tuning = False
                while not found_tuning:
                    potential_tuning = np.zeros((par['n_td']))
                    potential_tuning[q] = 1
                    found_tuning = True
                    for j in range(i-1):
                        pair_dist = np.sum(np.abs(potential_tuning - par['td_cases'][j,:]))
                        if pair_dist < min_dist:
                            found_tuning = False
                            q = np.random.permutation(par['n_td'])[:n]
                            #print(i, pair_dist, q)
                            break
                par['td_cases'][i, :] = potential_tuning

def gen_td_targets():

    par['td_targets'] = []
    for n in range(par['n_layers']-1):
        td = np.zeros((par['n_tasks'],par['n_dendrites'], par['layer_dims'][n+1]), dtype = np.float32)
        for i in range(par['layer_dims'][n+1]):

            if par['clamp'] == 'dendrites':
                for t in range(0, par['n_tasks'], par['n_dendrites']):
                    q = np.random.permutation(par['n_dendrites'])
                    for j, d in enumerate(q):
                        if t+j<par['n_tasks']:
                            td[t+j,d,i] = 1

            elif par['clamp'] == 'neurons':
                for t in range(par['n_tasks']):
                    if t%par['n_td'] == i%par['n_td']:
                        td[t,:,i] = 1

        par['td_targets'].append(td)


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    par['n_layers'] = len(par['layer_dims'])
    gen_td_cases()
    gen_td_targets()


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
