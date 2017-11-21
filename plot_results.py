import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"

data_dir = 'C:/Users/nicol/Projects/GitHub/Spin-TD-Network/savedir/perm_mnist_no_topdown'

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

def add_subplot_details(ax, ylim = [0,1], yminor = []):

    d = ylim[1] - ylim[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([ylim[0], ylim[1]])
    for i in yminor:
        ax.plot([0,100],[i,i],'k--')
    ax.set_ylabel('Mean task accuracy')
    ax.set_xlabel('Task number')

def plot_best_result(ax, data_dir, fn, col = [0,0,1], split = 1, description = []):

    best_accuracy = -1
    for file in os.listdir(data_dir):
        if file.startswith(fn):
            x = pickle.load(open(data_dir + '/' + file, 'rb'))
            if x['accuracy_full'][-1] > best_accuracy:
                best_accuracy = x['accuracy_full'][-1]
                task_accuracy = x['accuracy_full']

    if split > 1:
        task_accuracy = np.array(task_accuracy)
        task_accuracy = np.tile(np.reshape(task_accuracy,(-1,1)),(1,split))
        task_accuracy = np.reshape(task_accuracy,(1,-1))[0,:]

    if not description == []:
        print(description , ' ACC after 10 trials = ', task_accuracy[9],  ' after 30 trials = ', task_accuracy[29],  \
            ' after 100 trials = ', task_accuracy[99])

    ax.plot(task_accuracy, color = col)
