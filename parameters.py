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
    'n_tasks'               : 10,   # Number of pairwise variations
    'samples_per_trial'     : 10,   # Half the total number of presented pairs
    'cue_space_size'        : 26,   # Number of letters
    'sample_space_size'     : 20,   # Number of numbers
    'blank_character_size'  : 10,   # Number of inputs assigned to the blank character
    'dead_time'             : 10,   # Inactive time steps before any stimulus
    'steps_per_input'       : 20,   # Number of time steps for each input phase
    'n_input_phases'        : 7,    # Precisely what it says -- this is fixed

    # Network shape
    'n_td'                  : 10,
    'n_dendrites'           : 2,
    'n_hidden'              : 50,
    'dendrites_final_layer' : False,

    # Training specs
    'batch_size'            : 8,
    'n_train_batches'       : 1000,
    'n_batches_top_down'    : 15000,

    # Omega parameters
    'omega_c'               : 1,
    'omega_xi'              : 0.001,

    # Projection of top-down activity
    # Only one can be True
    'clamp'                 : 'dendrites', # can be either 'dendrites' or 'neurons' or None

}

############################
### Dependent parameters ###
############################

def gen_td_cases():

    # will create par['n_tasks'] number of tunings, each with exactly n non-zero elements equal to one
    # the distance between all tuned will be d

    par['td_cases'] = np.zeros((par['n_tasks'], par['n_td']), dtype = np.float32)

    if par['clamp'] == 'neurons':
        for n in range(par['n_tasks']):
            par['td_cases'][n, n%par['n_td']] = 1


    elif par['clamp'] == 'dendrites':

        # these params below work but no real rationale why I chose them
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

    # Currently only applies to hidden layer!

    td = np.zeros((par['n_tasks'], par['n_dendrites'], par['n_hidden']), dtype=np.float32)
    for i in range(par['n_hidden']):
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

    par['td_targets'] = td


def update_dependencies():
    """
    Updates all parameter dependencies
    """
    par['n_inputs'] = par['samples_per_trial'] \
                      * (par['cue_space_size'] + par['sample_space_size']) \
                      + par['blank_character_size']
    par['n_output'] = par['sample_space_size']
    par['n_steps']  = par['dead_time'] + par['steps_per_input']*par['n_input_phases']

    par['input_shape']  = [par['n_steps'], par['n_inputs']+par['n_td'], par['batch_size']]
    par['output_shape'] = [par['n_steps'], par['n_output'], par['batch_size']]

    gen_td_cases()
    gen_td_targets()

    par['output_time_mask'] = np.ones(par['output_shape'])
    par['output_time_mask'][:par['dead_time'],:,:] = 0.


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
