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

    # will create par['n_tasks'] number of templates, each with exactly n non-zero elements equal to one
    # the distance between all templated will be d
    n = 3
    d = 4
    template = np.zeros([par['n_tasks'], par['n_td']])
    for i in range(par['n_tasks']):
        q = np.random.permutation(par['n_td'])[:n]
        if i == 0:
            template[i, q] = 1
        else:
            found_template = False
            while not found_template:

                potential_template = np.zeros((par['n_td']))
                potential_template[q] = 1
                found_template = True
                for j in range(i-1):
                    pair_dist = np.sum(np.abs(potential_template - template[j,:]))
                    if not (pair_dist >= d-2 and pair_dist <= d):
                        found_template = False
                        q = np.random.permutation(par['n_td'])[:n]
                        #print(i, pair_dist, q)
                        break
            template[i, :] = potential_template

    return template

def gen_td_targets():

    td_targets = []
    for n in range(par['n_layers']-1):
        for td in par['td_cases']:
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


    """

    # Weight matrix sizes
    par['stim_to_hidden_dims']  = [par['n_hidden'], par['n_dend'], par['n_stim']]
    par['td_to_hidden_dims']    = [par['n_hidden'], par['n_dend'], par['n_td']]
    par['hidden_to_out_dims']   = [par['n_out'], par['n_hidden']]

    # Initial weight matrix states
    par['w_stim0']  = generate_init_weight(par['stim_to_hidden_dims'])
    par['w_td0']    = generate_init_weight(par['td_to_hidden_dims'])
    par['w_out0']   = generate_init_weight(par['hidden_to_out_dims'])

    par['b_hid0']   = np.zeros([par['n_hidden'],1], dtype=np.float32)
    par['b_out0']   = np.zeros([par['n_out'],1], dtype=np.float32)

    par['td_cases'] = gen_td_cases()

    # TD loss helper functions
    if par['td_loss_type'] == 'z_dot':
        par['TD_Z'] = np.ones([par['n_tasks'], par['n_hidden'], par['n_tasks'], par['n_hidden']])
        par['TD_Z'] /= par['n_dend']
        for i, j in product(range(par['n_tasks']), range(par['n_hidden'])):
            par['TD_Z'][i,j,i,j] = 0 #1/par['n_dend']

    elif par['td_loss_type'] == 'pairwise_random':
        par['TD_Z'] = np.zeros([par['n_tasks'], par['n_hidden'], par['n_dend']])
        for i, j in product(range(par['n_tasks']), range(par['n_hidden'])):
            q = np.random.randint(par['n_dend'])
            par['TD_Z'][i,j,q] = 1

    """

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
