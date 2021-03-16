import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCaps(nn.Module):
    
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        '''
        Take output of the conv layer as the input and output a set of 
        output vectors(8 output vectors, one for each "capsule")

        (batch_size, 20, 20, 256) -> 8*(batch_size, 6, 6, 32)
		8*(batch_size, 6, 6, 32) -> 8*(batch_size, 1152, 1)
		
		param num_capsules: number of capsules to create
		param in_channels: input depth of features, default value = 256
		param out_channels: output depth of the convolutional layers, default value = 32
        '''
        super(PrimaryCaps, self).__init__()

        # creating a list of convolutional layers for each capsule I want to create
        # all capsules have a conv layer with the same parameters
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)])
    
    def forward(self, x):
        '''
		param x: the input; features from a convolutional layer
        return: a set of normalized, capsule output vectors
        '''
        batch_size = x.size(0)

        # reshape convolution outputs to be (batch_size, 1152, 1)
        u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        # stack up output vectors, u, one for each capsule
        u = torch.cat(u, dim=-1) # (batch_size, 1152, 8)
        u_squash = self.squash(u)
        return u_squash
    
    def squash(self, input_tensor):
        '''
        Squashes an input Tensor so it has a magnitude between 0-1.
        
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        '''
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor
