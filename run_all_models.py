import numpy as np
from parameters import *
from itertools import product
import model
import sys, os
import pickle


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

"""
TODO:
Run SI with omega 0.1 and 0.2


"""

N = par['batch_size']
omega_c_vals = N*np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
omega_c_vals_extra = N*np.array([0.004, 0.006, 0.008, 0.012, 0.016, 0.024, 0.032, 0.04, 0.06, 0.08])
xi_vals = [0.001, 0.01, 0.1]

# Fixed parameters:
drop_keep_pct = 0.5
num_versions = 5

mnist_updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'omega_xi'              : 0.1,
    'n_tasks'               : 100,
    'n_td'                  : 100,
    'task'                  : 'mnist',
    'save_dir'              : './savedir/mnist/',
    'n_train_batches'       : 3906,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False,
    'pct_active_neurons'    : 0.2,
    }

cifar_updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'n_tasks'               : 20,
    'n_td'                  : 20,
    'task'                  : 'cifar',
    'save_dir'              : './savedir/cifar/',
    'n_train_batches'       : 977,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False,
    'pct_active_neurons'    : 0.25,
    }

multi_updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10]}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5]}


def recurse_best(data_dir, prefix):

    if 'SI' in prefix and 'cifar' in prefix:
        # Select 'xi' to look for
        for xi_str in ['_xi0', '_xi1']:

            # Get filenames
            name_and_data = []
            for full_fn in os.listdir(data_dir):
                if full_fn.startswith(prefix):
                    x = pickle.load(open(data_dir + full_fn, 'rb'))
                    name_and_data.append((full_fn, x['accuracy_full'][-1], x['par']['omega_c']))

            # Find number of c's and v's
            cids = []
            vids = []
            for (f, _, _) in name_and_data:
                if f[-12] not in cids:
                    cids.append(f[-12])
                if f[-9] not in vids:
                    vids.append(f[-9])

            # Scan across c's and v's for accuracies
            accuracies = np.zeros((len(cids), len(vids)))
            for (c_id, v_id) in product(sorted(cids), sorted(vids)):
                text_c = 'omega'+str(c_id)
                text_v = '_v'+str(v_id)
                for full_fn in os.listdir(data_dir):
                    if full_fn.startswith(prefix) and text_c in full_fn and text_v in full_fn and xi_str in full_fn:
                        accuracies[int(c_id),int(v_id)] = pickle.load(open(data_dir + full_fn, 'rb'))['accuracy_full'][-1]

            # Aggregate and sort averages
            averages = []
            for c, a in zip(cids, np.mean(accuracies, axis=1)):
                averages.append((c, a))
            averages = sorted(averages, key=lambda x: -x[1])

            # Find neighbor c
            neighbor = 0
            neighbor_acc = -1
            for c, a in averages:
                if int(c) == int(averages[0][0])+1 or int(c) == int(averages[0][0])-1:
                    if a > neighbor_acc:
                        neighbor = c

            # Calculate new c
            c0 = omega_c_vals[int(averages[0][0])]
            c1 = omega_c_vals[int(neighbor)]
            cR = (c0+c1)/2

            # Get optimal parameters
            for full_fn in os.listdir(data_dir):
                if full_fn.startswith(prefix) and 'omega'+averages[0][0] in full_fn and xi_str in full_fn:
                    opt_pars = pickle.load(open(data_dir + full_fn, 'rb'))['par']

            # Update parameters and run versions
            update_parameters(opt_pars)
            update_parameters({'omega_c' : cR})
            for i in range(num_versions):
                save_fn = prefix + '_omegaR' + xi_str + '_v' + str(i) + '.pkl'
                try_model(save_fn, sys.argv[1])
                print(save_fn, cR)

    else:

        # Get filenames
        name_and_data = []
        for full_fn in os.listdir(data_dir):
            if full_fn.startswith(prefix):
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                name_and_data.append((full_fn, x['accuracy_full'][-1], x['par']['omega_c']))

        # Find number of c's and v's
        cids = []
        vids = []
        for (f, _, _) in name_and_data:
            if f[-8] not in cids:
                cids.append(f[-8])
            if f[-5] not in vids:
                vids.append(f[-5])

        print(name_and_data)
        print(cids)
        print(vids)

        # Scan across c's and v's for accuracies
        accuracies = np.zeros((len(cids)))
        count = np.zeros((len(cids)))
        omegas = np.zeros((len(cids)))
        cids = sorted(cids)
        vids = sorted(vids)

        for (c_id, v_id) in product(cids, vids):
            text_c = 'omega'+str(c_id)
            text_v = '_v'+str(v_id)
            for full_fn in os.listdir(data_dir):
                if full_fn.startswith(prefix) and text_c in full_fn and text_v in full_fn:
                    print('c_id', c_id)
                    x = pickle.load(open(data_dir + full_fn, 'rb'))
                    accuracies[int(c_id)] += x['accuracy_full'][-1]
                    count[int(c_id)] += 1
                    omegas[int(c_id)] = x['par']['omega_c']

        accuracies /= count
        print('accuracies ', accuracies)

        ind_sorted = np.argsort(accuracies)
        print('Sorted ind ', ind_sorted)
        if ind_sorted[-1] > ind_sorted[-2] or ind_sorted[-1] == len(ind_sorted)-1: # to the right
            cR = (omegas[ind_sorted[-1]] + omegas[ind_sorted[-1]-1])/2
        else:
            cR = (omegas[ind_sorted[-1]] + omegas[ind_sorted[-1]+1])/2

        print('omegas ', omegas)
        print('cR = ', cR)

        """
        # Aggregate and sort averages
        averages = []
        for c, a in zip(cids, np.mean(accuracies, axis=1)):
            averages.append((c, a))
        averages = sorted(averages, key=lambda x: -x[1])

        # Find neighbor c
        neighbor = 0
        neighbor_acc = -1
        for c, a in averages:
            if int(c) == int(averages[0][0])+1 or int(c) == int(averages[0][0])-1:
                if a > neighbor_acc:
                    neighbor = c

        # Calculate new c
        #c0 = omega_c_vals[int(averages[0][0])]
        #c1 = omega_c_vals[int(neighbor)]
        #cR = (c0+c1)/2
        """

        # Get optimal parameters
        for full_fn in os.listdir(data_dir):
            if full_fn.startswith(prefix) and 'omega'+cids[ind_sorted[0]] in full_fn:
                opt_pars = pickle.load(open(data_dir + full_fn, 'rb'))['par']

        # Update parameters and run versions

        update_parameters(opt_pars)
        update_parameters({'omega_c' : cR})
        for i in range(num_versions):
            save_fn = prefix + '_omegaR_v' + str(i) + '.pkl'
            try_model(save_fn, sys.argv[1])
            print(save_fn, cR)


#recurse_best('./savedir/cifar/', 'partial_cifar_EWC')
#quit()

def base():

    for i in range(num_versions):
        if par['task'] == 'cifar' and not par['multihead']:
            save_fn = 'cifar_no_stabilization_v' + str(i) + '.pkl'
        elif par['task'] == 'cifar' and par['multihead']:
            save_fn = 'cifar_MH_no_stabilization_v' + str(i) + '.pkl'
        elif par['task'] == 'mnist':
            save_fn = 'mnist_no_stabilization_v' + str(i) + '.pkl'
        else:
            print('ERROR!!!')
            quit()

        try_model(save_fn, sys.argv[1])



def SI_500perms():

    omegas = omega_c_vals[:5]
    update_parameters({'stabilization': 'pathint', 'save_dir':'./savedir/mnist/500perms/', 'n_tasks':500, 'n_td':500})

    if par['clamp'] is not None and par['input_drop_keep_pct'] < 1:
        prefix = par['clamp'] + '_InpDO_'
    elif par['clamp'] is not None:
        prefix = par['clamp'] + '_'
    else:
        prefix = ''


    for i in range(num_versions):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            if par['task'] == 'mnist':
                save_fn = 'mnist_SI_500perms_1of12_omega' + str(j) + '_v' + str(i) + '.pkl'
                try_model(prefix + save_fn, sys.argv[1])
            else:
                print('ERROR!!!')
                quit()

def SI():

    omegas = omega_c_vals[:8]
    xis = xi_vals[:2]
    update_parameters({'stabilization': 'pathint'})

    if par['clamp'] is not None and par['input_drop_keep_pct'] < 1:
        prefix = par['clamp'] + '_InpDO_'
    elif par['clamp'] is not None:
        prefix = par['clamp'] + '_'
    else:
        prefix = ''

    for i in range(num_versions):
        for j in [3,2,4,1,0,5]:

            update_parameters({'omega_c': omegas[j]})
            if par['task'] == 'cifar':
                for k in range(len(xis)):
                    update_parameters({'omega_xi': xis[k]})
                    if par['multihead']:
                        save_fn = 'cifar_MH_SI_omega' + str(j) + '_xi' + str(k) + '_v' + str(i) + '.pkl'
                    else:
                        save_fn = 'cifar_SI_omega' + str(j) + '_xi' + str(k) + '_v' + str(i) + '.pkl'
                    try_model(prefix + save_fn, sys.argv[1])
            elif par['task'] == 'mnist':
                save_fn = 'mnist_SI_omega' + str(j) + '_v' + str(i) + '.pkl'
                for n in [4,6]:
                    update_parameters({'pct_active_neurons': 1/n})
                    save_fn = 'mnist_SI_1of' + str(n) +'_omega' + str(j) + '_v' + str(i) + '.pkl'
                    try_model(prefix + save_fn, sys.argv[1])
            else:
                print('ERROR!!!')
                quit()

def EWC():

    omegas = omega_c_vals[3:]
    update_parameters({'stabilization': 'EWC'})

    if par['clamp'] is not None and par['input_drop_keep_pct'] < 1:
        prefix = par['clamp'] + '_InpDO_'
    elif par['clamp'] is not None:
        prefix = par['clamp'] + '_'
    else:
        prefix = ''

    for i in range(1,num_versions):
        for j in [8,7,6,5]:

            update_parameters({'omega_c': omegas[j]})
            if par['task'] == 'cifar' and par['multihead']:
                save_fn = 'cifar_MH_EWC_omega' + str(j) + '_v' + str(i) + '.pkl'
            elif par['task'] == 'cifar' and not par['multihead']:
                save_fn = 'cifar_EWC_omega' + str(j) + '_v' + str(i) + '.pkl'
            elif par['task'] == 'mnist':
                save_fn = 'mnist_EWC_omega' + str(j) + '_v' + str(i) + '.pkl'
            else:
                print('ERROR!!!')
                quit()

            try_model(prefix + save_fn, sys.argv[1])


#recurse_best('./savedir/mnist/', 'neurons_InpDO_mnist_SI')
#quit()


print('SI Multiple gates, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'neurons'})
SI()
quit()

print('SI + 500, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'neurons' ,'pct_active_neurons': 1/12})
SI_500perms()



"""
print('SI + Partial, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'partial'})
SI()

print('EWC + Partial, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'partial'})
EWC()
quit()
"""

"""
print('-'*79)
print('EWC + Full, CIFAR')
update_parameters(cifar_updates)
update_parameters({'clamp':'neurons'})
EWC()

print('-'*79)
print('SI + Full, CIFAR')
update_parameters(cifar_updates)
update_parameters({'clamp':'neurons'})
SI()
quit()
"""


"""
print('SI + Partial + Split, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'split'})
SI()

print('EWC + Partial + Split, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'split'})
EWC()

quit()
"""

"""
print('SI + Full, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'neurons'})
SI()

print('EWC + Full, MNIST')
update_parameters(mnist_updates)
update_parameters({'input_drop_keep_pct':0.8})
update_parameters({'clamp':'neurons'})
EWC()

quit()
"""

"""
print('-'*79)
print('SI, CIFAR')
update_parameters(cifar_updates)
update_parameters({'clamp':None})
SI()
quit()
"""

"""
print('-'*79)
print('EWC + Partial, CIFAR')
update_parameters(cifar_updates)
update_parameters({'clamp':'partial'})
EWC()


print('-'*79)
print('EWC + Partial + Split, CIFAR')
update_parameters(cifar_updates)
update_parameters(cifar_split_updates)
update_parameters({'clamp':'split'})
EWC()


quit()

"""

"""
print('-'*79)
print('SI + Partial, CIFAR')
update_parameters(cifar_updates)
update_parameters({'clamp':'partial'})
SI()

"""
print('-'*79)
print('SI + Partial + Split, CIFAR')
update_parameters(cifar_updates)
update_parameters(cifar_split_updates)
update_parameters({'clamp':'split'})
SI()


"""

print('SI + Full, CIFAR')
update_parameters(cifar_updates)
update_parameters({'clamp':None})
update_parameters({'pct_active_neurons':1.0})
SI()
quit()
"""






if int(sys.argv[1])==0:
    ### QUEUE - GPU 0

    print('-'*79)
    print('Base EWC, CIFAR')
    update_parameters(cifar_updates)
    update_parameters({'clamp':None})
    EWC()

    print('-'*79)
    print('Base EWC, CIFAR Multiheaded')
    update_parameters(cifar_updates)
    update_parameters(multi_updates)
    update_parameters({'clamp':None})
    EWC()

    print('-'*79)
    print('SI + Partial, MNIST')
    update_parameters(mnist_updates)
    update_parameters({'clamp':'partial'})
    SI()

    print('-'*79)
    print('SI + Partial, CIFAR')
    update_parameters(cifar_updates)
    update_parameters({'clamp':'partial'})
    SI()

    print('-'*79)
    print('SI + Partial, CIFAR Multiheaded')
    update_parameters(cifar_updates)
    update_parameters(multi_updates)
    update_parameters({'clamp':'partial'})
    SI()

elif int(sys.argv[1])==1:
    ### QUEUE - GPU 1

    print('-'*79)
    print('EWC + Partial, MNIST')
    update_parameters(mnist_updates)
    update_parameters({'clamp':'partial'})
    EWC()

    print('-'*79)
    print('EWC + Partial, CIFAR')
    update_parameters(cifar_updates)
    update_parameters({'clamp':'partial'})
    EWC()


    print('-'*79)
    print('SI + Partial + Split, MNIST')
    update_parameters(mnist_updates)
    update_parameters(mnist_split_updates)
    update_parameters({'clamp':'split'})
    SI()

    print('-'*79)
    print('SI + Partial + Split, CIFAR')
    update_parameters(cifar_updates)
    update_parameters(cifar_split_updates)
    update_parameters({'clamp':'split'})
    SI()

elif int(sys.argv[1])==2:
    ### QUEUE - GPU 2


    print('-'*79)
    print('EWC + Partial + Split, MNIST')
    update_parameters(mnist_updates)
    update_parameters(mnist_split_updates)
    update_parameters({'clamp':'split'})
    EWC()

    print('-'*79)
    print('EWC + Partial + Split, CIFAR')
    update_parameters(cifar_updates)
    update_parameters(cifar_split_updates)
    update_parameters({'clamp':'split'})
    EWC()


    print('-'*79)
    print('SI + Full, MNIST')
    update_parameters(mnist_updates)
    update_parameters({'clamp':'neurons'})
    SI()

elif int(sys.argv[1])==3:
    ### QUEUE - GPU 3

    print('-'*79)
    print('SI + Full, CIFAR')
    update_paramters(cifar_updates)
    update_parameters({'clamp':'neurons'})
    SI()


    print('-'*79)
    print('EWC + Full, MNIST')
    update_parameters(mnist_updates)
    update_parameters({'clamp':'neurons'})
    EWC()

    print('-'*79)
    print('SI + Full, CIFAR')
    update_paramters(cifar_updates)
    update_parameters({'clamp':'neurons'})
    SI()

    print('-'*79)
    print('EWC + Full, CIFAR Multiheaded')
    update_paramters(cifar_updates)
    update_parameters(multi_updates)
    update_parameters({'clamp':'neurons'})
    EWC()



"""
print('updating baseline network parameters...')
update_parameters({'clamp': None, 'omega_c': 0})

print('Running base MNIST networks...')
update_parameters(mnist_updates)
base()

print('Running base CIFAR networks, no multi-head...')
update_parameters(cifar_updates)
base()

print('Running base CIFAR networks, with multi-head...')
updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}
update_parameters(updates)
base()

print('Running SI MNIST networks...')
update_parameters(mnist_updates)
SI()

print('Running base SI networks, no multi-head...')
update_parameters(cifar_updates)
SI()


print('Running base SI networks, with multi-head...')
update_parameters(cifar_updates)
updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}
update_parameters(updates)
SI()

print('Running EWC MNIST networks...')
update_parameters(mnist_updates)
EWC()
"""




"""
def base():
    updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : None,
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : 'pathint',
    'omega_c'               : 0,
    'multihead'             : False,
    'pct_active_neurons'    : 1,
    'save_dir'              : save_dir
    }

    update_parameters(updates)
    save_fn = 'cifar_n1000_no_stabilization_np_inp_drop.pkl'
    try_model(save_fn, sys.argv[1])


def pathint(c_set):
    updates = {
    'layer_dims'            : [4096, 1000, 1000, 100],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : None,
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'n_train_batches'       : 2500,
    'stabilization'         : 'pathint',
    'multihead'             : True,
    'pct_active_neurons'    : 1,
    'save_dir'              : save_dir
    }

    #for i in c_set:
        #for j in range(len(xi_vals)):
    updates['omega_c'] = 0.2*N #omega_c_vals[5]
    updates['omega_xi'] = xi_vals[0]

    update_parameters(updates)
    save_fn = 'cifar_testing' #'cifar_n1000_pathint_oc'+str(i)+'xi' + str(j) + '.pkl'
    try_model(save_fn, sys.argv[1])


def EWC(c_set):
    updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : None,
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : 'EWC',
    'pct_active_neurons'    : 1,
    'save_dir'              : save_dir
    }

    for i in c_set:
        for j in range(len(xi_vals)):
            updates['omega_c'] = omega_c_vals[i]
            updates['omega_xi'] = xi_vals[j]

            update_parameters(updates)
            save_fn = 'cifar_n1000_EWC_oc'+str(i)+'xi' + str(j)+'.pkl'
            try_model(save_fn, sys.argv[1])


def split_models(stab, c_set):

    # The split network will have 1164 units in each hidden layer
    # (4096*1000+1000*1000+1000*10)/(4096*291+291*291+291*10) ~ 4
    # 4*291 = 1164

    updates = {
    'layer_dims'            : [4096, 1164, 1164, 5],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : 'split',
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : stab,
    'pct_active_neurons'    : 0.25,
    'save_dir'              : save_dir
    }

    for i in c_set:
        for j in range(len(xi_vals)):
            updates['omega_c'] = omega_c_vals[i]
            updates['omega_xi'] = xi_vals[j]

            update_parameters(updates)
            save_fn = 'cifar_n1164_split_'+stab+'_oc'+str(i)+'xi' + str(j)+'.pkl'
            try_model(save_fn, sys.argv[1])


def partial_models(stab, c_set):
    updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : 'partial',
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : stab,
    'pct_active_neurons'    : 1,
    'save_dir'              : save_dir
    }

    for i in c_set:
        for j in range(len(xi_vals)):
            updates['omega_c'] = omega_c_vals[i]
            updates['omega_xi'] = xi_vals[j]

            update_parameters(updates)
            save_fn = 'cifar_n1000_partial_'+stab+'_oc'+str(i)+'xi' + str(j)+'.pkl'
            try_model(save_fn, sys.argv[1])


def full_models(stab, c_set):
    updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : 'neurons',
    'omega_xi'              : omega_xi,
    'task'                  : 'cifar',
    'n_tasks'               : 20,
    'stabilization'         : stab,
    'multihead'             : False,
    'pct_active_neurons'    : 0.25,
    'save_dir'              : save_dir
    }

    for i in c_set:
        for j in range(len(xi_vals)):
            updates['omega_c'] = omega_c_vals[i]
            updates['omega_xi'] = xi_vals[j]

            update_parameters(updates)
            save_fn = 'cifar_n1000_full_'+stab+'_1of4_oc'+str(i)+'xi' + str(j)+'.pkl'
            try_model(save_fn, sys.argv[1])


def pathint_mnist(cset):
    updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 1.0,
    'clamp'                 : None,
    'omega_xi'              : 0.1,
    'task'                  : 'mnist',
    'n_tasks'               : 500,
    'n_td'                  : 500,
    'stabilization'         : 'pathint',
    'save_dir'              : './savedir/perm_mnist_500perms/'
    }


    for i in cset:
        updates['omega_c'] = omega_c_vals[i]

        update_parameters(updates)
        save_fn = 'cifar_n2000_pathint_oc'+str(i)+'.pkl'
        try_model(save_fn, sys.argv[1])


def full_models_mnist(stab):
    updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'drop_keep_pct'         : drop_keep_pct,
    'input_drop_keep_pct'   : 0.8,
    'clamp'                 : 'neurons',
    'omega_xi'              : 0.1,
    'task'                  : 'mnist',
    'n_tasks'               : 500,
    'n_td'                  : 500,
    'stabilization'         : stab,
    'save_dir'              : './savedir/perm_mnist_500perms/'
    }

    # 1/n_neurons pct_active_neurons
    n_neurons = [12]
    omega_c_vals = [1.5*N*0.005*(2**i) for i in range(10)]

    for i in range(1,10):
        for j in n_neurons:
            updates['omega_c'] = omega_c_vals[i]
            updates['pct_active_neurons'] = 1/j

            update_parameters(updates)
            save_fn = 'cifar_n2000_'+stab+'_1of' + str(j) + '_oc'+str(i)+'B.pkl'
            try_model(save_fn, sys.argv[1])



pathint_cset = [0,1,2,3,4,5,6]
EWC_cset     = [2,3,4,5,6,7,8,9,10]

#full_models_mnist('pathint')
#pathint_mnist([3,5,7])
pathint([0])
quit()


print('Running Phase 1 of CIFAR Models', '\n'+'-'*79)

print('Base Network', '\n'+'-'*79)
base()

print('Raw SI Network', '\n'+'-'*79)
pathint(pathint_cset)

print('Raw EWC Network', '\n'+'-'*79)
EWC(pathint_cset)


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
"""
