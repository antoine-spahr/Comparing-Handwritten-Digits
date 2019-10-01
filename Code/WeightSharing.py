import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.nn import functional as F
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import math
import dlc_practical_prologue as prologue
from data_augmentaion import *

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
    def __init__(self):
        ''' '''
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(64, 120) # weight for layer 1
        self.fc2 = nn.Linear(120, 10) # weight for layer 2
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        '''  '''
        x = self.dropout(F.selu(self.fc1(x.view(-1, 64))))
        x = self.dropout(F.selu(self.fc2(x)))
        return x # 1x10

class MLP(nn.Module):
    ''' Common to both images. 2 times 1x10 that are stacked '''
    def __init__(self):
        ''' '''
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(20, 230) # weight for layer 1
        self.fc2 = nn.Linear(230, 100) # weight for layer 2
        self.fc3 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        '''  '''
        x = self.dropout(F.selu(self.fc1(x)))
        x = self.dropout(F.selu(self.fc2(x)))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutin module
        self.conv_module = ConvModule()
        # MLP_share module
        self.MLP_share_module = MLP_share()
        # Final MLP Module
        self.MLP_module = MLP()


    # Define the network structure by defining the forward method
    def forward(self, x):
        # input 2x14x14 --> extract 2x 1x14x14
        x1, x2 = x[:,0:1,:,:], x[:,1:2,:,:]

        # shared convolutionnal layers and shared MLP layer
        x1 = self.conv_module.forward(x1)
        x2 = self.conv_module.forward(x2)
        x1 = self.MLP_share_module.forward(x1)
        x2 = self.MLP_share_module.forward(x2)

        #x1 = x1.view(-1,10) # We need to view because when we use the MLP and not MLP_share view is not present in MLP as is it the case for MLP_share
        #x2 = x2.view(-1,10)

        # merge the Convolutional output into a 20x1
        x_final = torch.cat((x1, x2), dim=1)
        # MLP
        x_final = self.MLP_module.forward(x_final)
        return x1, x2, x_final # 10x1, 10x1, 2x1

def evaluate_architecture(model_class, N_train=15, N_sample=2000, N_epochs=25, minibatch_size=100, eta=0.1, criterion=nn.CrossEntropyLoss(), extend = False, use_GPU_power = False):
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

        if extend :
            train_input, train_target, train_class = ExtendPairData(train_input, train_target, train_class,mode='random_permutation')
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
        losses_vals.append(train_model(model, train_input, train_target, N_epochs, minibatch_size, eta, criterion.to(device)))

        ####################
        # Compute accuracy #
        ####################
        model.train(False) #Do not dropout
        accuracies_train[n] = compute_accuracy(model, train_input, train_target, minibatch_size)
        accuracies_test[n] = compute_accuracy(model, test_input, test_target, minibatch_size)

    return accuracies_train, accuracies_test, losses_vals

def train_model(model, input, target, N_epochs=25, minibatch_size=100, eta=0.1, criterion=nn.CrossEntropyLoss()):
    # <*> DOCSTRING
    '''
        Train the model

        INPUT: model -> [pytorch model] architecture to be trained and test.
               input -> [pytorch tensor n_samplex2x14x14] training samples
               target -> [pytorch tensor n_samplex20] the target
               N_epochs -> [integer] number of epochs to use for training (default is 25)
               minibatch_size -> [integer] size of minibatch to use, it must be smaller than N_sample (default is 100)
               eta -> [double] learning rate used for the gradient descent (default is 0.1)
               criterion -> [pytorch nn.Module.loss] loss fucntion to optimize (default is nn.CrossEntropyLoss)

        OUTPUT: None (inplace modification of model)
    '''
    #<*!>

    losses = []
    weight_decay = 2.5e-5

    # use the SGD as optimizer (no moment and no momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)

    # iterate over epochs
    for e in range(N_epochs):
        sum_loss = 0.0

        # distribute samples randomly in batches at each epochs
        batches_idx = list(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(input), minibatch_size, drop_last=False))

        for b in batches_idx:
            # forward pass
            _ ,_ ,output_final = model(input[b]) #ignore intermediary losses
            # compute the loss
            loss = criterion(output_final, target[b])
            sum_loss += loss.item()
            if torch.isnan(loss): raise ValueError()
            # reset gardient
            model.zero_grad()
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
        # get the one_hot_label output by the models
        _,_,pred_final = model(data_input[b:b+minibatch_size])
        pred_final = pred_final.argmax(dim=1)

        # get the number of correctly predicted
        accuracy += (pred_final == data_target[b:b+minibatch_size]).sum().item()

    # return the average number of coreectly predicted
    return accuracy/data_input.size(0)

# %% define parameters
N_train=20
N_sample=1000
N_epochs=40
minibatch_size=100
eta=3e-3
criterion=nn.CrossEntropyLoss()

# %% train & test
train_accuracies_test, test_accuracies_test, losses_test = evaluate_architecture(Net, N_train, N_sample, N_epochs, minibatch_size, eta, criterion, extend = True, use_GPU_power = True)
print('>> Train accuracy : {0:.2f} +/- {1:.2f}'.format(train_accuracies_test.mean().item(), train_accuracies_test.std().item()))
print('>> Test accuracy : {0:.2f} +/- {1:.2f}'.format(test_accuracies_test.mean().item() ,test_accuracies_test.std().item()))

# save output
with open('accuracy_pickles/WeightSharing_final.pickle','wb') as handle:
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
