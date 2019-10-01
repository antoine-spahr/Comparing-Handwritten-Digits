import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.nn import functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt # for plotting
import pickle # for accuracies saving
import math
import dlc_practical_prologue as prologue
from data_augmentaion import *

# %% creat network models here
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # kernel to preprocess image input into 64x1 vector
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # linear function for the 2 layer MLP
        self.fc1 = nn.Linear(64, 230) # weight for layer 1
        self.fc2 = nn.Linear(230, 100) # weight for layer 2
        self.fc3 = nn.Linear(100, 2) # weight for layer 2

        self.dropout = nn.Dropout(0.15)

    # Define the network structure by defining the forward method
    def forward(self, x):

        x = F.selu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.selu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=3))

        x = self.dropout(F.selu(self.fc1(x.view(-1, 64))))
        x = self.dropout(F.selu(self.fc2(x)))
        x = self.fc3(x)
        return x

def evaluate_architecture(model_class, N_train=15, N_sample=2000, N_epochs=25, minibatch_size=100, eta=0.1, l2=1e-3, criterion=nn.CrossEntropyLoss(), extend=False, use_GPU_power = False):
    # <*> DOCSTRING
    '''
        Load, Train and get accuracy over N training of the given model (given network architecture)

        INPUT: model -> [pytorch model class] architecture to be trained and test as a daugther class of nn.Module (the class name
                                        should be pass in order to call 'model=class_name()'')
                                        The output layer of the model must be of 2 neurons : 2 classes (img1 <= img2 or img1 > img2)
               N_train -> [integer] number of training to be performed (default is 15)
               N_sample -> [integer] number of sample to load from MNIST data (default is 1000)
               N_epochs -> [integer] number of epochs to use for training (default is 25)
               minibatch_size -> [integer] size of minibatch to use, it must be smaller than N_sample (default is 100)
               eta -> [double] learning rate used for the gradient descent (default is 0.1)
               criterion -> [pytorch nn.Module.loss] loss fucntion to optimize (default is nn.CrossEntropyLoss)
               extend -> [bolean] Perform data augmentation, to a given number of samples, depending on the type of data augmentation (default is false)
               use_GPU_power -> [bolean] send all stuff to GPU, if available (default is False)

        OUTPUT: accuracies_train -> [pytorch.Tensor] contains the N_train train accuracies
                accuracies_test -> [pytorch.Tensor] contains the N_train test accuracies
                losses_val -> [] sum loss over minibatch of each training

    '''
    # <*!>

    # storing tensors
    accuracies_train = torch.empty(N_train)
    accuracies_test = torch.empty(N_train)
    losses_vals = []

    for n in range(N_train):
        print('>>> Training {0}'.format(n))
        model = model_class()
        if use_GPU_power and cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = model.to(device)
        ########################
        # LOAD PAIR MNIST DATA #
        ########################
        train_input, train_target, train_class, test_input, test_target, test_class = prologue.generate_pair_sets(N_sample)
        # normalize samples
        train_input.div_(255)
        test_input.div_(255)

        if extend:
            train_input, train_target, train_class = ExtendPairData(train_input, train_target, train_class, mode='random_permutation')
            train_input, train_target, train_class = FlipImages(train_input, train_target, train_class)

        # set require_autograd=True
        train_input, train_target, train_class = Variable(train_input), Variable(train_target), Variable(train_class)
        train_input = train_input.to(device)
        train_target = train_target.to(device)
        train_class = train_class.to(device)

        test_input, test_target, test_class = Variable(test_input), Variable(test_target), Variable(test_class)
        test_input = test_input.to(device)
        test_target = test_target.to(device)
        test_class = test_class.to(device)
        ###############
        # Train Model #
        ###############
        model.train(True)
        losses_vals.append(train_model(model, train_input, train_target, N_epochs, minibatch_size, eta, l2, criterion.to(device)))

        ####################
        # Compute accuracy #
        ####################
        model.train(False)
        accuracies_train[n] = compute_accuracy(model, train_input, train_target, minibatch_size)
        accuracies_test[n] = compute_accuracy(model, test_input, test_target, minibatch_size)

    return accuracies_train, accuracies_test, losses_vals

def train_model(model, input, target, N_epochs=25, minibatch_size=100, eta=0.1, l2=1e-3, criterion=nn.CrossEntropyLoss()):
    # <*> DOCSTRING
    '''
        Train the model

        INPUT: model -> [pytorch model] architecture to be trained and test.
               input -> [pytorch tensor n_samplex2x14x14] training samples
               target -> [pytorch tensor n_samplex20] the targets
               N_epochs -> [integer] number of epochs to use for training (default is 25)
               minibatch_size -> [integer] size of minibatch to use, it must be smaller than N_sample (default is 100)
               eta -> [double] learning rate used for the gradient descent (default is 0.1)
               l2 -> [double] weight decay for the optimizer (default is 1e-3)
               criterion -> [pytorch nn.Module.loss] loss fucntion to optimize (default is nn.CrossEntropyLoss)

        OUTPUT: None (inplace modification of model)
    '''
    #<*!>

    losses = []
    # use the SGD as optimizer (no moment and no momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=l2)

    # iterate over epochs
    for e in range(N_epochs):
        print('-> epoch {0}'.format(e))
        sum_loss = 0.0
        # distribute samples randomly in batches at each epochs
        batches_idx = list(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(input), minibatch_size, drop_last=False))
        #batches_idx = list(torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(input), minibatch_size, drop_last=False))

        for b in batches_idx:
            # forward pass
            output = model(input[b])
            # compute the loss
            loss = criterion(output, target[b])
            sum_loss += loss.item()
            if torch.isnan(loss): raise ValueError()
            # reset gardient
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # perform the gradient descent step
            optimizer.step()

        losses.append(sum_loss)

    return losses

def compute_accuracy(model, data_input, data_target, minibatch_size):
    # <*> DOCSTRING
    '''
        Compute the accuracy by minibatches for the given model. The prediction
        is correct if the model output (one_hot_label for both images) is such that
        nbr1 <= nbr2.

        INPUT: model -> [pytorch model] architecture to be trained and test.
               data_input -> [pytorch tensor n_samplex2x14x14] samples
               data_target -> [pytorch tensor n_samplex20] target
               minibatch_size -> [integer] size of minibatch to use

        OUTPUT: accuracy -> [double] the accuracy of prediction of the model on the inputs
    '''
    # <*!>

    accuracy = 0.0

    # compute accuracy by minibatches
    for b in range(0, data_input.size(0), minibatch_size):
        # get the one_hot_label output by the model
        pred = model(data_input[b:b+minibatch_size]).argmax(dim=1)

        # get the number of correctly predicted
        accuracy += (pred == data_target[b:b+minibatch_size]).sum().item()

    # return the average number of coreectly predicted
    return accuracy/data_input.size(0)

# %% define parameters
N_train=20
N_sample=1000
N_epochs=40
minibatch_size=100
eta=3e-3
l2 = 2.5e-5
criterion=nn.CrossEntropyLoss()

# %% train & test
train_accuracies, test_accuracies, losses = evaluate_architecture(SimpleNet, N_train, N_sample, N_epochs, minibatch_size, eta, l2, criterion, extend = True, use_GPU_power = False)

print('>> Train accuracy : {0:.2%} +/- {1:.2%}'.format(train_accuracies.mean().item(), train_accuracies.std().item()))
print('>> Test accuracy : {0:.2%} +/- {1:.2%}'.format(test_accuracies.mean().item() ,test_accuracies.std().item()))

# %% save accuracies as pickles for common boxplot
with open('accuracy_pickles/simpleCNN_final.pickle', 'wb') as handle:
    pickle.dump([train_accuracies.numpy(), test_accuracies.numpy()], handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% plot loss for all trial
fig, ax = plt.subplots(1,1,figsize=(8,8))
for l in losses:
    ax.plot(range(1,len(l)+1,1), l)
ax.set_title('Test Convolutional Network')
ax.set_xlabel('epochs')
ax.set_ylabel('Cross Entropy Loss')
ax.set_xlim([1,N_epochs])
plt.show()

'''
# %% boxplot
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.boxplot([train_accuracies.numpy(), test_accuracies.numpy()], widths=0.6)
ax.set_xticklabels(['Train Accuracies', 'Test Accuracies'])
ax.set_ylabel('Accuracy')
plt.show()
'''
