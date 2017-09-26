### Parameters for RNN research
### Authors: Nicolas Masse, Gregory Grant, Catherine Lee, Varun Iyer
### Date:    3 August, 2017

import numpy as np
import tensorflow as tf
import itertools

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # Setup parameters
    'stimulus_type'         : 'att',    # multitask, att, mnist
    'save_dir'              : './savedir/',
    'debug_model'           : False,
    'load_previous_model'   : False,
    'processor_affinity'    : [0, 1],   # Default is [], for no preference
    'notify_on_completion'  : False,
    'test_with_optimizer'   : False,

    # hidden layer shape
    'n_hidden'          : 100,
    'den_per_unit'      : 3,

    # Tuning function data
    'tuning_height'     : 1,        # magnitutde scaling factor for von Mises
    'kappa'             : 1,        # concentration scaling factor for von Mises
    'catch_rate'        : 0.2,      # catch rate when using variable delay
    'match_rate'        : 0.5,      # number of matching tests in certain tasks

    # Cost parameters/function
    'spike_cost'        : 1e-2,
    'dend_cost'         : 1e-3,
    'wiring_cost'       : 5e-7,
    'motif_cost'        : 0e-2,
    'omega_cost'        : 0.0,
    'loss_function'     : 'cross_entropy',    # cross_entropy or MSE


    # Training specs
    'batch_train_size'      : 100,
    'num_train_batches'     : 300,
    'num_test_batches'      : 20,
    'num_iterations'        : 2,
    'switch_rule_iteration' : 1,
}

############################
### Dependent parameters ###
############################

def generate_weight(dims, connection_prob):
    n = np.float32(np.random.gamma(shape=0.25, scale=1.0, size=dims))
    n *= (np.random.rand(*dims) < connection_prob)
    return n


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val

    update_dependencies()


def spectral_radius(A):
    """
    Compute the spectral radius of each dendritic dimension of a weight array,
    and normalize using square room of the sum of squares of those radii.
    """
    if A.ndim == 2:
        return np.max(abs(np.linalg.eigvals(A)))
    elif A.ndim == 3:
        # Assumes the second axis is the target (for dendritic setup)
        r = 0
        for n in range(np.shape(A)[1]):
            r = r + np.max(abs(np.linalg.eigvals(np.squeeze(A[:,n,:]))))

        return r / np.shape(A)[1]


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    par['input_to_hidden_dend_dims'] = [par['n_hidden'], par['den_per_unit'], par['num_stim_tuned']]
    par['input_to_hidden_soma_dims'] = [par['n_hidden'], par['num_stim_tuned']]

    par['td_to_hidden_dend_dims']     = [par['n_hidden'], par['den_per_unit'], par['n_input'] - par['num_stim_tuned']]
    par['td_to_hidden_soma_dims']     = [par['n_hidden'], par['n_input'] - par['num_stim_tuned']]

    par['hidden_to_hidden_dend_dims'] = [par['n_hidden'], par['den_per_unit'], par['n_hidden']]
    par['hidden_to_hidden_soma_dims'] = [par['n_hidden'], par['n_hidden']]

    # Generate random masks
    generate_masks()

    if par['mask_connectivity'] < 1:
        reduce_connectivity()

    # Generate input weights
    par['w_stim_dend0'] = generate_weight(par['input_to_hidden_dend_dims'], par['connection_prob_in'])
    par['w_stim_soma0'] = generate_weight(par['input_to_hidden_soma_dims'], par['connection_prob_in'])

    par['w_td_dend0'] = generate_weight(par['td_to_hidden_dend_dims'], par['connection_prob_in'])
    par['w_td_soma0'] = generate_weight(par['td_to_hidden_soma_dims'], par['connection_prob_in'])

    par['w_stim_dend0'] *= par['w_stim_dend_mask']
    par['w_stim_soma0'] *= par['w_stim_soma_mask']

    par['w_td_dend0'] *= par['w_td_dend_mask']
    par['w_td_soma0'] *= par['w_td_soma_mask']

update_dependencies()
print("--> Parameters successfully loaded.\n")
