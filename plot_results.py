import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
from itertools import product

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"

data_dir = 'C:/Users/nicol/Projects/GitHub/Spin-TD-Network/savedir/perm_mnist_no_topdown'

def plot_fig2():

    savedir = './savedir/mnist/'
    all_same_scale = True

    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,6))

    # Figure 2A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    base    = 'mnist_no_stabilization'
    EWC     = 'mnist_EWC'
    SI      = 'mnist_SI'
    #SI= 'neurons_InpDO_mnist_SI'

    b3 = plot_best_result(ax1, savedir, SI, col=[0,0,1], label='SI')
    b1 = plot_best_result(ax1, savedir, base, col=[1,0,0], label='Base')
    b2 = plot_best_result(ax1, savedir, EWC, col=[0,1,0], label='EWC')

    print('A: ', b1, b2, b3)


    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [0.0,1],[0,100],[])

    # Figure 2B
    # Plotting: SI+TD Partial, SI, EWC+TD Partial, EWC
    # No dropout on input layer
    SI_TDP  = 'partial_InpDO_mnist_SI'
    EWC_TDP = 'partial_InpDO_mnist_EWC'


    b2 = plot_best_result(ax2, savedir, SI, col=[0,0,1], label='SI')
    b4 = plot_best_result(ax2, savedir, EWC, col=[0,1,0], label='EWC')
    b1 = plot_best_result(ax2, savedir, SI_TDP, col=[0,0.7,0.7], label='SI+TD Par.')
    b3 = plot_best_result(ax2, savedir, EWC_TDP, col=[0.7,0,0.7], label='EWC+TD Par.')

    print('B: ', b4)

    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(True)
    ax2.set_xlim(0,100)
    add_subplot_details(ax2, [0.0,1],[0,100],[])


    # Figure 2C
    # Plotting: SI+TD Partial Split, SI+TD Partial, EWC+TD Partial, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TDPS  = 'split_InpDO_mnist_SI'
    EWC_TDPS = 'split_InpDO_mnist_EWC'

    b1 = plot_best_result(ax3, savedir, SI_TDP, col=[0,0.7,0.7], label='SI+TD Par.')
    b3 = plot_best_result(ax3, savedir, EWC_TDP, col=[0.7,0,0.7], label='EWC+TD Par.')
    b2 = plot_best_result(ax3, savedir, SI_TDPS, col=[0.7,0.7,0], label='Split SI+TD Par.')
    b4 = plot_best_result(ax3, savedir, EWC_TDPS, col=[0.7,0.7,0.7],label='Split EWC+TD Par.')

    print('C: ', b2, b4)

    ax3.legend(ncol=2, fontsize=9)
    ax3.grid(True)
    ax3.set_xlim(0,100)
    add_subplot_details(ax3, [0.0,1],[0,100],[])


    # Figure 2D
    # Plotting: SI+TD Partial Split, SI+TD Full, EWC+TD Full, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TD    = 'neurons_InpDO_mnist_SI'
    EWC_TD   = 'neurons_InpDO_mnist_EWC'

    b2 = plot_best_result(ax4, savedir, SI_TDPS, col=[0.7,0.7,0], label='Split SI+TD Par.')
    b4 = plot_best_result(ax4, savedir, EWC_TDPS, col=[0.7,0.7,0.7],label='Split EWC+TD Par.')
    b1 = plot_best_result(ax4, savedir, SI_TD, col=[0,0,1], label='SI+TD Full')
    b3 = plot_best_result(ax4, savedir, EWC_TD, col=[0,1,0], label='EWC+TD Full')

    print('D: ', b1, b3)

    ax4.grid(True)
    ax4.set_xlim(0,100)
    add_subplot_details(ax4, [0.0,1],[0,100],[])
    ax4.legend(ncol=2, fontsize=9)


    plt.tight_layout()
    plt.show()


def mnist_table():
    savedir = './savedir/perm_mnist/archive'

    base     = 'mnist_n2000_no_stabilization'     # archive1
    SI       = 'perm_mnist_n2000_d1_no_topdown'   # archive0
    EWC      = 'mnist_n2000_EWC'                  # archive1

    SI_TDP   = 'perm_mnist_n2000_d1_bias'         # archive0
    EWC_TDP  = 'perm_mnist_n2000_d1_EWC_bias'     # archive1

    SI_TDPS  = 'mnist_n2000_pathint_split_oc'     # archive2
    EWC_TDPS = 'mnist_n2000_EWC_split_oc'         # archive2

    SI_TDF   = 'perm_mnist_n2000_d1_1of5'         # archive0
    EWC_TDF  = 'perm_mnist_n2000_d1_EWC_1of5'     # archive1

    archs = [1,0,1,0,1,2,2,0,1]
    names = ['Base', 'SI', 'EWC', 'SI + Partial', 'EWC + Partial', \
             'SI + Partial + Split', 'EWC + Partial + Split',\
             'SI + Full', 'EWC + Full']
    locs  = [base, SI, EWC, SI_TDP, EWC_TDP, SI_TDPS, EWC_TDPS, SI_TDF, EWC_TDF]

    with open('mnist_table_data.tsv', 'w') as f:
        f.write('Name\tC\tT1\tT10\tT20\tT50\tT100\n')
        for a, s, n in zip(archs, locs, names):
            c_opt, acc = retrieve_best_result(savedir+str(a)+'/', s)
            f.write(n + '\t' + str(c_opt) + '\t' + str(acc[0])
                                          + '\t' + str(acc[9])
                                          + '\t' + str(acc[19])
                                          + '\t' + str(acc[49])
                                          + '\t' + str(acc[99]) + '\n')


def cifar_table():
    savedir = './savedir/cifar_no_multihead/'

    base     = 'cifar_n1000_no_stabilization'
    SI       = 'cifar_n1000_pathint'
    EWC      = 'cifar_n1000_EWC'

    SI_TDP   = 'cifar_n1000_partial_pathint'
    EWC_TDP  = 'cifar_n1000_partial_EWC'

    SI_TDPS  = 'cifar_n1164_split_pathint'
    EWC_TDPS = 'cifar_n1164_split_EWC'

    SI_TDF   = 'cifar_n1000_full_pathint'
    EWC_TDF  = 'cifar_n1000_full_EWC'

    names = ['Base', 'SI', 'EWC', 'SI + Partial', 'EWC + Partial', \
             'SI + Partial + Split', 'EWC + Partial + Split',\
             'SI + Full', 'EWC + Full']
    locs  = [base, SI, EWC, SI_TDP, EWC_TDP, SI_TDPS, EWC_TDPS, SI_TDF, EWC_TDF]

    with open('cifar_table_data.tsv', 'w') as f:
        f.write('Name\tC\tT1\tT10\tT20\n')
        for s, n in zip(locs, names):
            c_opt, acc = retrieve_best_result(savedir, s)
            f.write(n + '\t' + str(c_opt) + '\t' + str(acc[0])
                                          + '\t' + str(acc[9])
                                          + '\t' + str(acc[19]) + '\n')



def fig2_inset():
    f, ax = plt.subplots(1,1)
    # Figure 2D
    # Plotting: SI+TD Partial Split, SI+TD Full, EWC+TD Full, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TD    = 'perm_mnist_n2000_d1_1of5'
    SI_TDPS  = 'mnist_n2000_pathint_split_oc'
    EWC_TD   = 'perm_mnist_n2000_d1_EWC_1of5'
    EWC_TDPS = 'mnist_n2000_EWC_split_oc'

    b2 = plot_best_result(ax, './savedir/perm_mnist/archive2/', SI_TDPS, col=[0.7,0.7,0], label='Split SI+TD Par.')
    b4 = plot_best_result(ax, './savedir/perm_mnist/archive2/', EWC_TDPS, col=[0.7,0.7,0.7],label='Split EWC+TD Par.')
    b1 = plot_best_result(ax, './savedir/perm_mnist/archive0/', SI_TD, col=[0,0,1], label='SI+TD Full')
    b3 = plot_best_result(ax, './savedir/perm_mnist/archive1/', EWC_TD, col=[0,1,0], label='EWC+TD Full')
    ax.grid(True)
    ax.set_yticks([0.85,0.90,0.95,1.0])
    ax.set_xlim(0,100)
    add_subplot_details(ax, [0.85,1],[])

    plt.tight_layout()
    plt.show()


def plot_fig3():

    savedir = './savedir/cifar/'
    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,6))

    # Figure 3A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    base    = 'cifar_no_stabilization'
    EWC     = 'cifar_EWC'
    SI      = 'cifar_SI'

    base_MH = 'cifar_MH_no_stabilization'
    EWC_MH  = 'cifar_MH_EWC'
    SI_MH   = 'cifar_MH_SI'

    b1 = plot_best_result(ax1, savedir, base, col=[1,0,0], label='base')
    b2 = plot_best_result(ax1, savedir, EWC, col=[0,1,0], label='EWC')
    b3 = plot_best_result(ax1, savedir, SI, col=[0,0,1], label='SI')

    b4 = plot_best_result(ax1, savedir, base_MH, col=[1,1,0], label='base')
    b5 = plot_best_result(ax1, savedir, EWC_MH, col=[0.5,1,0.5], label='EWC')
    b6 = plot_best_result(ax1, savedir, SI_MH, col=[0,1,1], label='SI')

    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    add_subplot_details(ax1, ylim = [0,1], xlim = [0,20])

    # Figure 3B
    # Plotting: SI+TD Partial, SI, EWC+TD Partial, EWC
    # No dropout on input layer
    SI_TDP  = 'partial_cifar_SI'
    EWC_TDP = 'partial_cifar_EWC'

    b2 = plot_best_result(ax2, savedir, SI, col=[0,0,1], label='SI')
    b4 = plot_best_result(ax2, savedir, EWC, col=[0,1,0], label='EWC')
    b1 = plot_best_result(ax2, savedir, SI_TDP, col=[0,0.7,0.7], label='SI+TD Par.')
    b3 = plot_best_result(ax2, savedir, EWC_TDP, col=[0.7,0,0.7], label='EWC+TD Par.')

    ax2.set_xlim(0,100)
    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(True)
    add_subplot_details(ax2, ylim = [0,1], xlim = [0,20])

    # Figure 3C
    # Plotting: SI+TD Partial Split, SI+TD Partial, EWC+TD Partial, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TDPS  = 'split_cifar_SI'
    EWC_TDPS = 'split_cifar_EWC'

    b1 = plot_best_result(ax3, savedir, SI_TDP, col=[0,0.7,0.7], label='SI+TD Par.')
    b3 = plot_best_result(ax3, savedir, EWC_TDP, col=[0.7,0,0.7], label='EWC+TD Par.')
    b2 = plot_best_result(ax3, savedir, SI_TDPS, col=[0.7,0.7,0], label='Split SI+TD Par.')
    b4 = plot_best_result(ax3, savedir, EWC_TDPS, col=[0.7,0.7,0.7],label='Split EWC+TD Par.')

    ax3.set_xlim(0,100)
    ax3.legend(ncol=2, fontsize=9)
    ax3.grid(True)
    add_subplot_details(ax3, ylim = [0,1], xlim = [0,20])

    # Figure 3D
    # Plotting: SI+TD Partial Split, SI+TD Full, EWC+TD Full, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TD    = 'neurons_cifar_SI'
    EWC_TD   = 'neurons_cifar_EWC'

    b2 = plot_best_result(ax4, savedir, SI_TDPS, col=[0.7,0.7,0], label='Split SI+TD Par.')
    b4 = plot_best_result(ax4, savedir, EWC_TDPS, col=[0.7,0.7,0.7],label='Split EWC+TD Par.')
    b1 = plot_best_result(ax4, savedir, SI_TD, col=[0,0,1], label='SI+TD Full')
    b3 = plot_best_result(ax4, savedir, EWC_TD, col=[0,1,0], label='EWC+TD Full')

    ax4.set_xlim(0,100)
    ax4.legend(ncol=2, fontsize=9)
    ax4.grid(True)
    add_subplot_details(ax4, ylim = [0,1], xlim = [0,20])

    plt.tight_layout()
    plt.show()

def plot_mnist_figure():

    f = plt.figure(figsize=(6,2.5))

    # SI only, no top-down
    SI_fn = 'perm_mnist_n2000_d1_no_topdown_omega'
    # SI only + top-down
    SI_td_fn = 'perm_mnist_n2000_d1_bias_omega'
    # SI only + split in 5 + top-down
    SI_split_fn = 'perm_mnist_n735_d1_bias_20tasks'
    # SI + INH TD, selecting one out of 4
    SI_inh_td4_fn = 'perm_mnist_n2000_d1_1of4_omega'
    # SI + INH TD, selecting one out of 5
    SI_inh_td5_fn = 'perm_mnist_n2000_d1_1of5_omega'
    # SI + INH TD, selecting one out of 6
    SI_inh_td6_fn = 'perm_mnist_n2000_d1_1of6_omega'
    # SI + INH TD, selecting one out of 3
    SI_inh_td3_fn = 'perm_mnist_n2000_d1_1of3_omega'
    # SI + INH TD, selecting one out of 2
    SI_inh_td2_fn = 'perm_mnist_n2000_d1_1of2_omega'

    ax = f.add_subplot(1, 2, 1)
    plot_best_result(ax, data_dir, SI_fn, col = [0,0,1], description = 'SI only')
    plot_best_result(ax, data_dir, SI_td_fn, col = [1,0,0])
    plot_best_result(ax, data_dir, SI_split_fn, col = [0,1,0], split = 5)
    plot_best_result(ax, data_dir, SI_inh_td5_fn, col = [1,0,1], description = 'SI + TD date 80%')
    add_subplot_details(ax, [0.8, 1],[0.85, 0.9,0.95])

    ax = f.add_subplot(1, 2, 2)
    plot_best_result(ax, data_dir, SI_inh_td5_fn, col = [1,0,1])
    plot_best_result(ax, data_dir, SI_inh_td4_fn, col = [0,1,0])
    plot_best_result(ax, data_dir, SI_inh_td6_fn, col = [0,0,1], description = 'SI + TD date 86.67%')
    plot_best_result(ax, data_dir, SI_inh_td3_fn, col = [0,1,1])
    plot_best_result(ax, data_dir, SI_inh_td2_fn, col = [0,0,0])
    add_subplot_details(ax, [0.9, 1], [0.95])

    plt.tight_layout()
    plt.savefig('Fig1.pdf', format='pdf')
    plt.show()

def add_subplot_details(ax, ylim = [0,1], xlim = [0,100],yminor = []):

    d = ylim[1] - ylim[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([ylim[0], ylim[1]])
    for i in yminor:
        ax.plot([0,100],[i,i],'k--')
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylabel('Mean task accuracy')
    ax.set_xlabel('Task number')

def plot_best_result(ax, data_dir, prefix, col = [0,0,1], split = 1, description = [], label=None):


    # Get filenames
    name_and_data = []
    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix):
            x = pickle.load(open(data_dir + full_fn, 'rb'))
            name_and_data.append((full_fn, x['accuracy_full'][-1], x['par']['omega_c']))

    # Find number of c's and v's
    cids = []
    vids = []
    xids = []
    for (f, _, _) in name_and_data:
        if 'xi' in f:
            if f[-12] not in cids:
                cids.append(f[-12])
            if f[-9] not in vids:
                vids.append(f[-9])
        else:
            if f[-8] not in cids:
                cids.append(f[-8])
            if f[-5] not in vids:
                vids.append(f[-5])

    accuracies = np.zeros((len(cids)))
    count = np.zeros((len(cids)))
    cids = sorted(cids)
    vids = sorted(vids)

    #print(cids)
    #quit()

    for i, c_id, in enumerate(cids):
        for v_id in vids:
            #print(i, c_id, v_id)
            text_c = 'omega'+str(c_id)
            text_v = '_v'+str(v_id)
            for full_fn in os.listdir(data_dir):
                if full_fn.startswith(prefix) and text_c in full_fn and text_v in full_fn:
                    #print('c_id', c_id)
                    x = pickle.load(open(data_dir + full_fn, 'rb'))
                    accuracies[i] += x['accuracy_full'][-1]
                    count[i] += 1

    accuracies /= count
    #print('accuracies ', accuracies)
    ind_best = np.argsort(accuracies)[-1]
    task_accuracy = []

    for v_id in vids:
        text_c = 'omega'+str(cids[ind_best])
        text_v = '_v'+str(v_id)
        for full_fn in os.listdir(data_dir):
            if full_fn.startswith(prefix) and text_c in full_fn and text_v in full_fn:
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                task_accuracy.append(x['accuracy_full'])

    task_accuracy = np.mean(np.stack(task_accuracy),axis=0)


    if split > 1:
        task_accuracy = np.array(task_accuracy)
        task_accuracy = np.tile(np.reshape(task_accuracy,(-1,1)),(1,split))
        task_accuracy = np.reshape(task_accuracy,(1,-1))[0,:]

    if not description == []:
        print(description , ' ACC after 10 trials = ', task_accuracy[9],  ' after 30 trials = ', task_accuracy[29],  \
            ' after 100 trials = ', task_accuracy[99])

    ax.plot(np.arange(1, np.shape(task_accuracy)[0]+1), task_accuracy, color = col, label=label)

    return task_accuracy[[9,99]]

def retrieve_best_result(data_dir, fn):
    best_accuracy = -1
    val_c = 0.
    for f in os.listdir(data_dir):
        if f.startswith(fn):
            x = pickle.load(open(data_dir+f, 'rb'))
            if x['accuracy_full'][-1] > best_accuracy:
                best_accuracy = x['accuracy_full'][-1]
                task_accuracy = x['accuracy_full']
                val_c         = x['par']['omega_c']

    return val_c, task_accuracy


mnist_table()
cifar_table()
#plot_fig3()
#plot_fig2()
#fig2_inset()
