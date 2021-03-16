import torch
import torch.nn as nn
import torch.nn.functional as F

GPU_AVAILABLE = torch.cuda.is_available()

# if(GPU_AVAILABLE):
#     print('Training on GPU!')
# else:
#     print('Only CPU available')

def softmax(input_tensor, dim=1):
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    # un-transpose result
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)

# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    '''
    Performs dynamic routing between two capsule layers.

    param b_ij: initial log probabilities that capsule i should be coupled to capsule j
    param u_hat: input, weighted capsule vectors, W u
    param squash: given, normalizing squash function
    param routing_iterations: number of times to update coupling coefficients
    return: v_j, output capsule vectors
    '''    
    # update b_ij, c_ij for number of routing iterations
    for iteration in range(routing_iterations):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = softmax(b_ij, dim=2)

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squashing to get a normalized vector output, v_j
        v_j = squash(s_j)

        # if not on the last iteration, calculate agreement and new b_ij
        if iteration < routing_iterations - 1:
            # agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            
            # new b_ij
            b_ij = b_ij + a_ij
    
    return v_j # return latest v_j

class DigitCaps(nn.Module):
    
    def __init__(self, num_capsules=10, previous_layer_nodes=32*6*6, 
                 in_channels=8, out_channels=16):
        '''
        Constructs an initial weight matrix, W, and sets class variables.
        
        param num_capsules: number of capsules to create
        param previous_layer_nodes: dimension of input capsule vector, default value = 1152
        param in_channels: number of capsules in previous layer, default value = 8
        param out_channels: dimensions of output capsule vector, default value = 16
        '''
        super(DigitCaps, self).__init__()

        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes # vector input (dim=1152)
        self.in_channels = in_channels # previous layer's number of capsules

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, 
                                          in_channels, out_channels))

    def forward(self, u):
        '''
		Defines the feedforward behavior.
        
        param u: the input; vectors from the previous PrimaryCaps layer
        return: a set of normalized, capsule output vectors
        '''
        # adding batch_size dims and stacking all u vectors
        u = u[None, :, :, None, :]
        # 4D weight matrix
        W = self.W[:, None, :, :, :]
        
        # calculating u_hat = W*u
        u_hat = torch.matmul(u, W)

        # getting the correct size of b_ij
        # setting them all to 0, initially
        b_ij = torch.zeros(*u_hat.size())
        
        # moving b_ij to GPU, if available
        if GPU_AVAILABLE:
            b_ij = b_ij.cuda()

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j # return final vector outputs
    
    
    def squash(self, input_tensor):
        '''
        Squashes an input Tensor so it has a magnitude between 0-1.
        
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        '''
        # same squash function as in PrimaryCaps
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor
