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
    'train_top_down'        : True,
    'task'                  : 'mnist',

    # Task specs
    'n_tasks'               : 30,

    # Network shape
    'n_td'                  : 40,
    'n_dendrites'           : 5,
    'layer_dims'            : [28**2, 400, 400, 10],
    'dendrites_final_layer' : False,

    # Dropout
    'keep_pct'              : 1.0,

    # Training specs
    'batch_size'            : 1024,
    'n_train_batches'       : 4000,
    'n_batches_top_down'    : 40000,

    # Omega parameters
    'omega_c'               : 0.1,
    'omega_xi'              : 0.1,

    # Projection of top-down activity
    # Only one can be True
    'clamp'                 : 'dendrites' # can be either 'dendrites' or 'neurons' or None

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
    min_dist = 4
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
                    if pair_dist < min_dist:
                        found_tuning = False
                        q = np.random.permutation(par['n_td'])[:n]
                        #print(i, pair_dist, q)
                        break
            tuning[i, :] = potential_tuning

    return tuning

def gen_td_targets():

    td_targets = []

    if par['clamp'] == 'dendrites':

        for n in range(par['n_layers']-1):
            td = np.zeros((par['n_tasks'],par['n_dendrites'], par['layer_dims'][n+1]), dtype = np.float32)
            for i in range(par['layer_dims'][n+1]):
                for t in range(0, par['n_tasks'], par['n_dendrites']):
                    q = np.random.permutation(par['n_dendrites'])
                    for j, d in enumerate(q):
                        td[t+j,d,i] = 1
                    #print(td[:,:,i])
                    #quit()
            td_targets.append(td)

    elif par['clamp'] == 'neurons':

        divide_network = 5 # number of groups networks is to be divided into

        for n in range(par['n_layers']-1):
            td = np.zeros((par['n_tasks'],par['n_dendrites'], par['layer_dims'][n+1]), dtype = np.float32)
            for i in range(par['layer_dims'][n+1]):
                for t in range(par['n_tasks']):
                    if t%divide_network == i%divide_network:
                        td[t,:,i] = 1

    else:
        td = []

    return td_targets

    """
    td_targets = []
    for task in range(par['n_tasks']):
        target_layer = []
        for n in range(par['n_layers']-1):
            target = np.zeros((par['n_dendrites'], par['layer_dims'][n+1]))
            for i in range(par['layer_dims'][n+1]):
                q = np.random.randint(par['n_dendrites'])
                target[q,i] = 1
            target_layer.append(target)
        td_targets.append(target_layer)


    return td_targets
    """

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
