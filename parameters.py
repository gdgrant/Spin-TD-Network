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
    'learning_rate'     : 0.001,
    'connection_prob'   : 1.,

    # Task specs
    'n_tasks'           : 10,

    # Network shape
    'n_stim'            : 784,
    'n_td'              : 10,
    'n_hidden'          : 12,
    'n_dend'            : 3,
    'n_out'             : 10,

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
    if par['n_tasks'] > par['n_td']:
        raise Exception('Use more TD neurons than tasks.')
    if par['n_td']%par['n_tasks'] != 0:
        raise Exception('Use an integer multiple of n_tasks for n_td.')

    m = par['n_td']//par['n_tasks']
    template = np.zeros([par['n_tasks'], par['n_td']])
    for n in range(par['n_tasks']):
        if n == par['n_tasks']-1:
            template[n, n*m:]=1
        else:
            template[n, n*m:n*m+m]=1

    return template


def update_dependencies():
    """
    Updates all parameter dependencies
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
