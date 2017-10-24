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
    'stabilization'         : 'pathinit', # 'EWC' (Kirkpatrick method) or 'pathint' (Zenke method)
    'learning_rate'         : 0.001,
    'train_top_down'        : False,
    'task'                  : 'mnist',
    'save_analysis'         : True,
    'train_convolutional_layers' : False,

    # Task specs
    'n_tasks'               : 100,


    # Network shape
    'n_td'                  : 100,
    'n_dendrites'           : 1,
    'layer_dims'            : [28**2, 400, 400, 10], # mnist
    #'layer_dims'            : [4096, 500, 500, 100], #cifar
    'dendrites_final_layer' : False,
    'pct_active_neurons'    : 1.0,

    # Dropout
    'keep_pct'              : 0.5,

    # Training specs
    'batch_size'            : 128,
    'n_train_batches'       : 2000,
    'n_batches_top_down'    : 15000,

    # Omega parameters
    'omega_c'               : 0.05*256,
    'omega_xi'              : 0.1,
    'last_layer_mult'       : 2,
    'scale_factor'          : 1,

    # Projection of top-down activity
    # Only one can be True
    'clamp'                 : None, # can be either 'dendrites' or 'neurons' or None

}

############################
### Dependent parameters ###
############################

def gen_td_cases():

    # will create par['n_tasks'] number of tunings, each with exactly n non-zero elements equal to one
    # the distance between all tuned will be d

    par['td_cases'] = np.zeros((par['n_tasks'], par['n_td']), dtype = np.float32)

    #if par['clamp'] == 'neurons':
    for n in range(par['n_tasks']):
        par['td_cases'][n, n%par['n_td']] = 1


def gen_td_targets():

    m = round(1/par['pct_active_neurons'])
    print('Clamping: selecting every ', m, ' neuron')

    par['td_targets'] = []
    par['W_td0'] = []
    for n in range(par['n_layers']-1):
        td = np.zeros((par['n_tasks'],par['n_dendrites'], par['layer_dims'][n+1]), dtype = np.float32)
        Wtd = np.zeros((par['n_td'],par['n_dendrites'], par['layer_dims'][n+1]), dtype = np.float32)
        for i in range(par['layer_dims'][n+1]):

            if par['clamp'] == 'dendrites':
                for t in range(0, par['n_tasks'], par['n_dendrites']):
                    q = np.random.permutation(par['n_dendrites'])
                    for j, d in enumerate(q):
                        if t+j<par['n_tasks']:
                            td[t+j,d,i] = 1
                            Wtd[t+j,d,i] = 1
                """
                for t in range(0, par['n_tasks']):
                    td[t,t%par['n_dendrites'],i] = 1
                    Wtd[t,t%par['n_dendrites'],i] = 1
                """


            elif par['clamp'] == 'neurons':
                for t in range(0, par['n_tasks'], m):
                    q = np.random.permutation(m)[0]
                    if t+q<par['n_tasks']:
                        td[t+q,:,i] = 1
                        Wtd[t+q,:,i] = 1
                    """
                    for j, d in enumerate(q):
                        if t+j<par['n_tasks']:
                            td[t+j,:,i] = 1
                            Wtd[t+j,:,i] = 1

                    if t%par['n_td'] == i%par['n_td']:
                        td[t,:,i] = 1
                        Wtd[t,:,i] = 1
                    """

            elif par['clamp'] is None:
                td[:,:,:] = 1
                Wtd[:,:,:] = 1

        par['td_targets'].append(td)
        par['W_td0'].append(Wtd)


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
        print('Updating : ', key, ' -> ', val)
    update_dependencies()


#update_dependencies()
print("--> Parameters successfully loaded.\n")
