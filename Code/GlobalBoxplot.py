#
#   Plot the comparative boxplot for the three models :
#               - Simple CNN
#               - WeightSharing CNN
#               - AuxiliaryLoss CNN
#
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
import pickle

# %% load pickles
data = {}

with open('accuracy_pickles/simpleCNN_final.pickle', 'rb') as handle:
    data['simple CNN'] = pickle.load(handle)

with open('accuracy_pickles/WeightSharing_final.pickle', 'rb') as handle:
    data['Weight sharing CNN'] = pickle.load(handle)

with open('accuracy_pickles/AuxiliaryLoss_final.pickle', 'rb') as handle:
    data['Weight sharing \n+ \nAuxiliary loss CNN'] = pickle.load(handle)

print(data['Weight sharing \n+ \nAuxiliary loss CNN'][0].mean())
print(data['Weight sharing \n+ \nAuxiliary loss CNN'][0].std())
print(data['Weight sharing \n+ \nAuxiliary loss CNN'][1].mean())
print(data['Weight sharing \n+ \nAuxiliary loss CNN'][1].std())

#data['Weight sharing \n+ \nAuxiliary loss CNN']=[arr/100 for arr in data['Weight sharing \n+ \nAuxiliary loss CNN']]
#data['Weight sharing CNN']=[arr/100 for arr in data['Weight sharing CNN']]

# %% plot the boxplot
train_color = 'forestgreen'
test_color = 'steelblue'
boxdict1 = dict(linestyle='-', linewidth=2.5, color=train_color)
boxdict2 = dict(linestyle='-', linewidth=2.5, color=test_color)
whiskerdict1 = dict(linestyle='-', linewidth=2.5, color=train_color)
whiskerdict2 = dict(linestyle='-', linewidth=2.5, color=test_color)
mediandict = dict(linestyle='-', linewidth=2, color='tomato')

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(1,1,figsize=(16,7))
i = 1
names = []
for name, values in data.items():
    #ax.boxplot(values, positions = [i, i+1], widths = 0.6, showfliers=False, showcaps=False, boxprops=boxdict, whiskerprops=whiskerdict, medianprops=mediandict)
    ax.boxplot(values[0], positions = [i], widths = 0.6, showfliers=False, showcaps=False, boxprops=boxdict1, whiskerprops=whiskerdict1, medianprops=mediandict)
    ax.boxplot(values[1], positions = [i+1], widths = 0.6, showfliers=False, showcaps=False, boxprops=boxdict2, whiskerprops=whiskerdict2, medianprops=mediandict)
    ax.scatter(np.random.normal(i, 0.03, values[0].shape[0]), values[0], c='darkgray', alpha=0.8)
    ax.scatter(np.random.normal(i+1, 0.03, values[1].shape[0]), values[1], c='darkgray', alpha=0.8)
    names.append(name)
    i += 2

ax.set_xticks([k+0.5 for k in range(1,i,2)])
ax.set_xticklabels(names)
ax.set_xlim(0.5, i-0.5)
ax.set_ylabel('Accuracy [%]')
ax.set_title('Train & Test accuracies for all the models')

handles = [mpatch.Patch(facecolor=train_color), mpatch.Patch(facecolor=test_color) ]
labels = ['Train', 'Test']
lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, -0.03), bbox_transform=fig.transFigure, ncol=4)

fig.tight_layout()
fig.savefig('./Figures/'+'performance.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()
