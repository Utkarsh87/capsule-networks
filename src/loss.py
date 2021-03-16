import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLoss(nn.Module):
    
    def __init__(self):
        '''
        Constructs a CapsuleLoss module
        '''
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum') # cumulative loss, equiv to size_average=False

    def forward(self, x, labels, images, reconstructions):
        '''
        param x: digit capsule outputs
        param labels: 
        param images: the original MNIST image input data
        param reconstructions: reconstructed MNIST image data
        return: weighted margin and reconstruction loss, averaged over a batch
        '''
        batch_size = x.size(0)

        ## ---------------------------- ##
        ## 1. Margin loss ##
        ## ---------------------------- ##
        
        # get magnitude of digit capsule vectors, v_c
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        # calculate "correct" and incorrect loss
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        
        # sum the losses, with a lambda = 0.5
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        ## ---------------------------- ##
        ## 2. Reconstruction loss ##
        ## ---------------------------- ##

        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        # return a weighted, summed loss, averaged over a batch size
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
