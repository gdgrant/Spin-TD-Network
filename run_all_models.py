import numpy as np
from parameters import *
from itertools import product
import model
import sys


def try_model(save_fn,gpu_id):
    # GPU designated by first argument (must be integer 0-3)
    try:
        print('Selecting GPU ',  sys.argv[1])
        assert(int(sys.argv[1]) in [0,1,2,3])
    except AssertionError:
        quit('Error: Select a valid GPU number.')

    try:
        # Run model
        model.main(save_fn, sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

###############################################################################
###############################################################################
###############################################################################

N = par['batch_size']
omega_c_vals = [0.01*N, 0.00625*N, 0.0125*N, 0.025*N, 0.05*N, 0.1*N, 0.2*N, \
                0.4*N, 0.8*N, 1.6*N, 3.2*N, 4*N, 6*N, 8*N]
c_override = None

# Fixed parameters:
drop_keep_pct = 0.5
omega_xi = 0.1

def base():
    updates = {
    'layer_dims'            : [4096, 2000, 2000, 100],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : None,
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : 'pathint',
    'omega_c'               : 0,
    'pct_active_neurons'    : 0.25
    }

    update_parameters(updates)
    save_fn = 'cifar_n2000_no_stabilization_np_inp_drop.pkl'
    try_model(save_fn, sys.argv[1])


def pathint():
    updates = {
    'layer_dims'            : [4096, 2000, 2000, 100],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : None,
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : 'pathint',
    'pct_active_neurons'    : 0.25
    }

    for i in [1,2,3,4,5,6,7]:
        updates['omega_c'] = omega_c_vals[i]

        update_parameters(updates)
        save_fn = 'cifar_n2000_pathint_oc'+str(i)+'_no_inp_drop.pkl'
        try_model(save_fn, sys.argv[1])


def EWC():
    updates = {
    'layer_dims'            : [4096, 2000, 2000, 100],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : None,
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : 'EWC',
    'pct_active_neurons'    : 0.25
    }

    for i in [4,5,6,7,8,9,10]:
        updates['omega_c'] = omega_c_vals[i]

        update_parameters(updates)
        save_fn = 'cifar_n2000_EWC_oc'+str(i)+'_no_inp_drop.pkl'
        try_model(save_fn, sys.argv[1])


def split_models(stab, c_set):
    updates = {
    'layer_dims'            : [4096, 2000, 2000, 100],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 0.8,
    'clamp'                 : 'split',
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : stab,
    'pct_active_neurons'    : 0.25
    }

    for i in c_set:
        updates['omega_c'] = omega_c_vals[i]

        update_parameters(updates)
        save_fn = 'cifar_n2000_split_'+stab+'_oc'+str(i)+'_inp_drop20.pkl'
        try_model(save_fn, sys.argv[1])


def partial_models(stab, c_set):
    updates = {
    'layer_dims'            : [4096, 2000, 2000, 100],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : 'partial',
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : stab,
    'pct_active_neurons'    : 0.25
    }

    for i in c_set:
        updates['omega_c'] = omega_c_vals[i]

        update_parameters(updates)
        save_fn = 'cifar_n2000_partial_'+stab+'_oc'+str(i)+'_inp_drop20.pkl'
        try_model(save_fn, sys.argv[1])


def full_models(stab, c_set):
    updates = {
    'layer_dims'            : [4096, 2000, 2000, 100],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 0.8,
    'clamp'                 : 'neurons',
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : stab,
    'pct_active_neurons'    : 0.25
    }

    for i in c_set:
        updates['omega_c'] = omega_c_vals[i]

        update_parameters(updates)
        save_fn = 'cifar_n2000_full_'+stab+'_1of5_oc'+str(i)+'_inp_drop20.pkl'
        try_model(save_fn, sys.argv[1])


def full_models_mnist(stab):
    updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 0.8,
    'clamp'                 : 'neurons',
    'omega_xi'              : omega_xi,
    'task'                  : 'mnist',
    'n_tasks'               : 100,
    'n_td'                  : 100,
    'stabilization'         : stab,
    'pct_active_neurons'    : 0.2
    }

    omega_c_vals = [0.01*N, 0.00625*N, 0.00625*N/2, 0.0125*N]

    for i in [1,2,3]:
        updates['omega_c'] = omega_c_vals[i]

        update_parameters(updates)
        save_fn = 'cifar_n2000_'+stab+'_rnd_1of5_oc'+str(i)+'.pkl'
        try_model(save_fn, sys.argv[1])


full_models_mnist('pathint')
quit()

pathint_cset = [1,2,3,4,5,6,7]
EWC_cset     = [4,5,6,7,8,9,10]


print('Running Phase 1 of CIFAR Models', '\n'+'-'*79)

print('Base Network', '\n'+'-'*79)
base()
print('Raw SI Network', '\n'+'-'*79)
pathint()
print('Raw EWC Network', '\n'+'-'*79)
EWC()


print('Running Phase 2 of CIFAR Models', '\n'+'-'*79)

print('Partial SI', '\n'+'-'*79)
partial_models('pathint', pathint_cset)
print('Partial EWC', '\n'+'-'*79)
partial_models('EWC', EWC_cset)


print('Running Phase 3 of CIFAR Models', '\n'+'-'*79)

print('Split SI', '\n'+'-'*79)
split_models('pathint', pathint_cset)
print('Split EWC', '\n'+'-'*79)
split_models('EWC', EWC_cset)


print('Running Phase 4 of CIFAR Models', '\n'+'-'*79)

print('Full SI', '\n'+'-'*79)
full_models('pathint', pathint_cset)
print('Full EWC', '\n'+'-'*79)
full_models('EWC', EWC_cset)



# Command for observing python processes:
# ps -A | grep python
