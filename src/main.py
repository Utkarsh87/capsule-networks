import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from capsnet import CapsuleNetwork
from loss import CapsuleLoss

import argparse
import torch
from torchvision import datasets, transforms

def get_train_dataloader():
    train_dataset = datasets.MNIST(root='data', train=True, download=True, 
                                   transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    return train_loader

def get_test_dataloader():
    test_dataset = datasets.MNIST(root='data', train=False, download=True,
                                  transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return test_loader


batch_size = 32
train_loader = get_train_dataloader()

model = CapsuleNetwork()
print(30*"="+" Architecture "+30*"=")
print(model)
print(74*"=")

print("\nSizes of parameters: ")
for name, param in model.named_parameters():
    print(f"{name}: {list(param.size())}")
n_params = sum([p.nelement() for p in model.parameters()])
# The coupling coefficients b_ij are not included in the parameter list,
# we need to add them mannually, which is 1152 * 10 = 11520.
print('\nTotal number of parameters: %d \n' % (n_params+11520))

GPU_AVAILABLE = torch.cuda.is_available()
if(GPU_AVAILABLE):
	print("Training on GPU")
	model = model.cuda()
else:
	print("Only CPU available, training on CPU")

criterion = CapsuleLoss()
optimizer = optim.Adam(model.parameters())

def train(model, criterion, optimizer, n_epochs, print_every=300):
    '''
    Trains a capsule network and prints out training batch loss statistics.
	Saves model parameters if *validation* loss has decreased.
	
	param model: trained capsule network
	param criterion: capsule loss function
	param optimizer: optimizer for updating network weights
	param n_epochs: number of epochs to train for
	param print_every: batches to print and save training loss, default = 100
	return: list of recorded training losses
	'''

    # track training loss over time
    losses = []

    for epoch in range(1, n_epochs+1):

        # initialize training loss
        train_loss = 0.0
        
        model.train() # set to train mode
    
        # get batches of training image data and targets
        for batch_i, (images, target) in enumerate(train_loader):

            # reshape and get target class
            target = torch.eye(10).index_select(dim=0, index=target)

            if GPU_AVAILABLE:
                images, target = images.cuda(), target.cuda()

            # zero out gradients
            optimizer.zero_grad()
            # get model outputs
            caps_output, reconstructions, y = model(images)
            # calculate loss
            loss = criterion(caps_output, target, images, reconstructions)
            # perform backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() # accumulated training loss
            
            # print and record training stats
            if batch_i != 0 and batch_i % print_every == 0:
                avg_train_loss = train_loss/print_every
                losses.append(avg_train_loss)
                print('Epoch: {} \tTraining Loss: {:.8f}'.format(epoch, avg_train_loss))
                train_loss = 0 # reset accumulated training loss
        
    return losses

n_epochs = 10
losses = train(model, criterion, optimizer, n_epochs=n_epochs)

import matplotlib.pyplot as plt

plt.plot(losses)
plt.title("Training Loss")
plt.savefig("lossplot.jpg")
plt.show()
