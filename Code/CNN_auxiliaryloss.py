import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.nn import functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import math
import dlc_practical_prologue as prologue
from data_augmentaion import *

# %% create network models here
class ConvModule(nn.Module):
    ''' For a given image 1x14x14'''
    def __init__(self):
        '''  '''
        nn.Module.__init__(self)
        # kernel to preprocess image input into 64x1 vector
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) # 1) --> output : 10x10 (14-5+1) 32 Channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)# 3) --> output : 3x3 (5-3+1) 64 Channels

    def forward(self, x):
        '''  '''
        x = F.selu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)) # 2) kernel 2x2; strides 2 --> output: 5x5 and 32 channels
        x = F.selu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=3)) # 4) kernel 3x3; stride 3 --> output 1x1 and 64 channels
        return x # return 1x1x64

class MLP_share(nn.Module):
    ''' For a given image. '''
    def __init__(self, dropout_rate, nUnit_share):
        ''' '''
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(64, nUnit_share) # weight for layer 1
        self.fc2 = nn.Linear(nUnit_share, 10) # weight for layer 2. Output layer of 10 units
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        '''  '''
        x = self.dropout(F.selu(self.fc1(x.view(-1, 64))))
        x = self.dropout(F.selu(self.fc2(x)))
        return x # 1x10

class MLP(nn.Module):
    ''' Common to both images. 2 times 1x10 that are stacked '''
    def __init__(self, dropout_rate, nUnit):
        ''' '''
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(20, nUnit) # weight for layer 1
        self.fc2 = nn.Linear(nUnit,100)
        self.fc3 = nn.Linear(100, 2) # weight for layer 2
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        '''  '''
        x = self.dropout(F.selu(self.fc1(x)))
        x = self.dropout(F.selu(self.fc2(x)))
        x = self.fc3(x)
        return x # 1x2

class Net(nn.Module):
    def __init__(self, dropout_rate, nUnit_share, nUnit):
        super(Net, self).__init__()
        # Convolution module
        self.conv_module = ConvModule()
        # MLP_share module
        self.MLP_share_module = MLP_share(dropout_rate, nUnit_share)
        # Final MLP Module
        self.MLP_module = MLP(dropout_rate, nUnit)


    # Define the network structure by defining the forward method
    def forward(self, x):
        # input 2x14x14 --> extract 2x 1x14x14
        x1, x2 = x[:,0:1,:,:], x[:,1:2,:,:]

        # shared convolutionnal layers and shared MLP layer
        x1 = self.conv_module.forward(x1)
        x2 = self.conv_module.forward(x2)
        x1 = self.MLP_share_module.forward(x1)
        x2 = self.MLP_share_module.forward(x2)

        # merge the Convolutional output into a 20x1
        x_final = torch.cat((x1, x2), dim=1)

        # Final MLP
        x_final = self.MLP_module.forward(x_final)

        return x1, x2, x_final # 10x1 (for aux. loss1), 10x1 (for aux loss2), 2x1 (for main loss)

def evaluate_architecture(model_class, N_train=15, N_sample=1000, N_epochs=25, minibatch_size=100, eta=0.1, l2=0.0, criterion=nn.CrossEntropyLoss(), use_GPU_power = False, extend_data = False, type_of_extend = 'random_permutation', dropout = 0.2, alpha = 1, beta = 0.2, nUnit_share = 100, nUnit = 70):
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
               l2 -> [double] l2 regularization strength (Weight decay parameter of Adam optimizer)
               criterion -> [pytorch nn.Module.loss] loss fucntion to optimize (default is nn.CrossEntropyLoss)
               use_GPU_power -> [bolean] send all stuff to GPU, if available (default is False)
               extend_data -> [bolean] Perform data augmentation, to a given number of samples, depending on the type of data augmentation (default is false)
               type_of_extend -> [string] type of data augmentation. Cf. ExtendPairData function. (default is 'random_permutation')
               dropout -> Define a dropout level for the model
               alpha -> Define the weight of the auxiliary losses in final loss
               beta -> Define the weight of main loss in final loss
               nUnit_share -> Define the number of unit in the first hidden layer of MLP_share Module
               nUnit -> Define the number of units in the first hidden layer of MLP Module

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

        # Create the model with the defined dropout rate, and number of hidden unit
        model = model_class(dropout, nUnit_share, nUnit)
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
        # convert to vector one hot label
        train_class_hot = torch.cat((prologue.convert_to_one_hot_labels(train_input, train_class[:,0]), prologue.convert_to_one_hot_labels(train_input, train_class[:,1])), dim=1).long()
        test_class_hot = torch.cat((prologue.convert_to_one_hot_labels(test_input, test_class[:,0]), prologue.convert_to_one_hot_labels(test_input, test_class[:,1])), dim=1).long()

        if extend_data:
            train_input, train_target, train_class = ExtendPairData(train_input, train_target, train_class, type_of_extend)
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
        losses_vals.append(train_model(model, train_input, train_target, train_class, N_epochs, minibatch_size, eta, l2, criterion.to(device), alpha, beta))

        ####################
        # Compute accuracy #
        ####################
        model.train(False) #Do not dropout when testing
        accuracies_train[n] = compute_accuracy(model, train_input, train_target, minibatch_size)
        accuracies_test[n] = compute_accuracy(model, test_input, test_target, minibatch_size)

    return accuracies_train, accuracies_test, losses_vals

def train_model(model, input, target, classes, N_epochs=25, minibatch_size=100, eta=0.1, l2=0.0, criterion=nn.CrossEntropyLoss(), alpha = 1, beta = 0.2):
    # <*> DOCSTRING
    '''
        Train the model

        INPUT: model -> [pytorch model] architecture to be trained and test.
                                        The output layer of the model must be of 20 neurons : the first 10 neurons is the predicted digit of image 1 as one_hot_label
                                                                                              the last 10 neurons is the predicted digit of image 2 as one_hot_label
               input -> [pytorch tensor n_samplex2x14x14] training samples
               target -> [pytorch tensor n_samplex20] if on_hotlabel is True [pytorch tensor n_samplex1] if False
               N_epochs -> [integer] number of epochs to use for training (default is 25)
               minibatch_size -> [integer] size of minibatch to use, it must be smaller than N_sample (default is 100)
               eta -> [double] learning rate used for the gradient descent (default is 0.1)
               l2 -> [double] l2 regularization strength (Weight decay parameter of Adam optimizer)
               criterion -> [pytorch nn.Module.loss] loss fucntion to optimize (default is nn.CrossEntropyLoss)
               alpha -> Define the weight of the auxiliary losses in final loss
               beta -> Define the weight of main loss in final loss

        OUTPUT: None (inplace modification of model)
    '''
    #<*!>

    losses = []
    # use the SGD as optimizer (no moment and no momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=l2)


    # iterate over epochs
    for e in range(N_epochs):
        sum_loss = 0.0
        # distribute samples randomly in batches at each epochs
        batches_idx = list(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(input), minibatch_size, drop_last=False))

        for b in batches_idx:
            # forward pass
            output_img1, output_img2, output_final = model(input[b]) #ignore intermediary losses

            # compute the auxiliary loss of MLP_share output on image 1
            loss_img1 = criterion(output_img1, classes[b,0].long())
            sum_loss += loss_img1.item()
            if torch.isnan(loss_img1): raise ValueError()

            # compute the auxiliary loss of MLP_share output on image 2
            loss_img2 = criterion(output_img2, classes[b,1].long())
            sum_loss += loss_img2.item()
            if torch.isnan(loss_img2): raise ValueError()

            # compute the final loss
            loss_final = criterion(output_final, target[b].long())
            sum_loss += loss_final.item()
            if torch.isnan(loss_final): raise ValueError()

            # reset gardient
            #optimizer.zero_grad()
            model.zero_grad()
            # backward pass

            #Weighted sum of auxiliary loss and final loss
            loss_all = alpha*(loss_img1 + loss_img2) + beta*loss_final
            loss_all.backward()
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

        INPUT:

        OUTPUT:
    '''
    # <*!>

    accuracy = 0.0

    # compute accuracy by minibatches
    for b in range(0, data_input.size(0), minibatch_size):
        # get the one_hot_label output by the models
        pred_img1,pred_img2,pred_final = model(data_input[b:b+minibatch_size])
        pred_final = pred_final.argmax(dim=1)

        # get the number of correctly predicted
        accuracy += (pred_final == data_target[b:b+minibatch_size]).sum().item()

    # return the average number of coreectly predicted
    return accuracy/data_input.size(0)

# %% define parameters
#### Final Model with that architecture ####

N_train=20
N_sample=1000
N_epochs=40
minibatch_size=100
eta = 3e-3#eta=0.005179
l2 = 2.5e-05 #l2=2.636e-05
criterion=nn.CrossEntropyLoss()
dropout_rate = 0.15 #0.1147
alpha = 1
beta = 0.5
nUnitShare = 120
nUnit = 230

# %% train & test
train_accuracies_test, test_accuracies_test, losses_test = evaluate_architecture(Net, N_train, N_sample, N_epochs, minibatch_size, eta, l2, criterion, use_GPU_power = True, extend_data = True, type_of_extend = 'random_permutation', dropout = dropout_rate, alpha = alpha, beta = beta, nUnit_share = nUnitShare, nUnit = nUnit)

print('>> Train accuracy : {0:.2f} +/- {1:.2f}'.format(train_accuracies_test.mean().item(), train_accuracies_test.std().item()))
print('>> Test accuracy : {0:.2f} +/- {1:.2f}'.format(test_accuracies_test.mean().item() ,test_accuracies_test.std().item()))

# %% Save Result with Pickle for final Boxplot
import pickle
with open('accuracy_pickles/AuxiliaryLoss_final.pickle', 'wb') as handle:
    pickle.dump([train_accuracies_test.numpy(), test_accuracies_test.numpy()], handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% plot loss for all trial
fig, ax = plt.subplots(1,1,figsize=(8,8))
for l in losses_test:
    ax.plot(range(1,len(l)+1,1), l)
ax.set_title('Test Convolutional Network')
ax.set_xlabel('epochs')
ax.set_ylabel('Cross Entropy Loss')
ax.set_xlim([1,N_epochs])
plt.show()


# %% boxplot
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.boxplot([train_accuracies_test.numpy(), test_accuracies_test.numpy()], widths=0.6)
ax.set_xticklabels(['Train Accuracies', 'Test Accuracies'])
ax.set_ylabel('Accuracy')
plt.show()
