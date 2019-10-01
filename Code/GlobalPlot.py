#
#   Plot the comparative boxplot for the three models :
#               - Simple CNN
#               - WeightSharing CNN
#               - AuxiliaryLoss CNN
#
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatch
import numpy as np
import pickle

def plotResults():
    # %% load pickles
    data = {}
    data_loss = {}

    with open('./pickles/simpleCNN_acc.pickle', 'rb') as handle:
        data['simple CNN'] = pickle.load(handle)

    with open('pickles/weightSharing_acc.pickle', 'rb') as handle:
        data['Weight sharing CNN'] = pickle.load(handle)

    with open('pickles/AuxiliaryLoss_acc.pickle', 'rb') as handle:
        data['Weight sharing \n+ \nAuxiliary loss CNN'] = pickle.load(handle)

    with open('pickles/simpleCNN_losses.pickle', 'rb') as handle:
        data_loss['simple CNN'] = pickle.load(handle)

    with open('pickles/weightSharing_losses.pickle', 'rb') as handle:
        data_loss['Weight Sharing CNN'] = pickle.load(handle)

    with open('pickles/AuxiliaryLoss_losses.pickle', 'rb') as handle:
        data_loss['Auxiliary Loss CNN'] = pickle.load(handle)

    # %% plot the boxplot
    train_color = 'forestgreen'
    test_color = 'steelblue'
    boxdict1 = dict(linestyle='-', linewidth=2.5, color=train_color)
    boxdict2 = dict(linestyle='-', linewidth=2.5, color=test_color)
    whiskerdict1 = dict(linestyle='-', linewidth=2.5, color=train_color)
    whiskerdict2 = dict(linestyle='-', linewidth=2.5, color=test_color)
    mediandict = dict(linestyle='-', linewidth=2, color='tomato')

    plt.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=(16,12))
    gs = gridspec.GridSpec(2, 3, hspace=0.5)
    ax1 = fig.add_subplot(gs[0, :])
    axs = []
    for i in range(3):
        axs.append(fig.add_subplot(gs[1,i]))

    # plot boxplot
    i = 1
    names = []
    for name, values in data.items():
        ax1.boxplot(values[0], positions = [i], widths = 0.6, showfliers=False, showcaps=False, boxprops=boxdict1, whiskerprops=whiskerdict1, medianprops=mediandict)
        ax1.boxplot(values[1], positions = [i+1], widths = 0.6, showfliers=False, showcaps=False, boxprops=boxdict2, whiskerprops=whiskerdict2, medianprops=mediandict)
        ax1.scatter(np.random.normal(i, 0.03, values[0].shape[0]), values[0], c='darkgray', alpha=0.8)
        ax1.scatter(np.random.normal(i+1, 0.03, values[1].shape[0]), values[1], c='darkgray', alpha=0.8)
        names.append(name)
        i += 2

    ax1.set_xticks([k+0.5 for k in range(1,i,2)])
    ax1.set_xticklabels(names)
    ax1.set_xlim(0.5, i-0.5)
    ax1.set_ylabel('Accuracy [%]')
    ax1.set_title('Train & Test accuracies for all the models')

    # plot losses
    for idx, (name, losses) in enumerate(data_loss.items()):
        for l in losses:
            axs[idx].plot(range(1,len(l)+1,1), l)
        axs[idx].set_title('Losses for ' + name)
        axs[idx].set_xlabel('epochs')
        axs[idx].set_ylabel('Cross Entropy Loss')
        axs[idx].set_xlim(left=1)

    handles = [mpatch.Patch(facecolor=train_color), mpatch.Patch(facecolor=test_color) ]
    labels = ['Train', 'Test']
    lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, 0.48), bbox_transform=fig.transFigure, ncol=4)

    plt.show()
