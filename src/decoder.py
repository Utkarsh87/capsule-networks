import torch
import torch.nn as nn
import torch.nn.functional as F

GPU_AVAILABLE = torch.cuda.is_available()

# if(GPU_AVAILABLE):
#     print('Training on GPU!')
# else:
#     print('Only CPU available')

class DenseDecoder(nn.Module):
    
    def __init__(self, input_vector_length=16, input_capsules=10, hidden_dim=512):
        '''
        Constructs a decoder by stacking dense layers.

        param input_vector_length: dimension of input capsule vector, default value = 16
        param input_capsules: number of capsules in previous layer, default value = 10
        param hidden_dim: dimensions of hidden layers, default value = 512
        '''
        super(DenseDecoder, self).__init__()
        
        # calculate input_dim
        input_dim = input_vector_length * input_capsules
        
        # define dense network
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, 28*28), # can be reshaped into 28*28 image
            nn.Sigmoid() # sigmoid activation to get output pixel values in a range from 0-1
            )
        
    def forward(self, x):
        '''        
        param x: the input; vectors from the previous DigitCaps layer
        return: two things, reconstructed images and the class scores, y
        '''
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        # find the capsule with the maximum vector length
        # here, vector length indicates the probability of a class' existence
        _, max_length_indices = classes.max(dim=1)
        
        # create a sparse class matrix
        sparse_matrix = torch.eye(10) # 10 is the number of classes in MNIST
        
        if GPU_AVAILABLE:
            sparse_matrix = sparse_matrix.cuda()
        
        # get the class scores from the "correct" capsule
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        
        # create reconstructed pixels
        x = x * y[:, :, None]

        # flatten image into a vector shape (batch_size, vector_dim)
        flattened_x = x.contiguous().view(x.size(0), -1)
        
        # create reconstructed image vectors
        reconstructions = self.dense(flattened_x)
        
        # return reconstructions and the class scores, y
        return reconstructions, y

class CustomReshape(nn.Module):
    def __init__(self):
        super(CustomReshape, self).__init__()

    def forward(self, x):
        return x.reshape(-1, 16, 7, 7)

class CustomPad(nn.Module):
    def __init__(self, p2d=(1,0,1,0)):
        super(CustomPad, self).__init__()
        self.p2d = p2d

    def forward(self, x):
        return F.pad(x, self.p2d, "constant", 0)

class DeconvDecoder(nn.Module):
    
    def __init__(self, input_vector_length=16, input_capsules=10, img_size=28, img_channels=1):
        '''
        Constructs a decoder using transposed convolutional layers

        param input_vector_length: dimension of input capsule vector, default value = 16
        param input_capsules: number of capsules in previous layer, default value = 10
        param hidden_dim: dimensions of hidden layers, default value = 512
        '''
        super(DeconvDecoder, self).__init__()
        
        # calculate input_dim
        input_dim = input_vector_length * input_capsules
        
        # define deconv network
        self.deconv = nn.Sequential(
            nn.Linear(input_dim, 7*7*16),
            nn.ReLU(inplace=True),
            CustomReshape(),
            nn.BatchNorm2d(num_features=16, momentum=0.8),
            nn.ConvTranspose2d(in_channels=16, out_channels=64,
                            kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                            kernel_size=3, stride=2, padding=1),
            CustomPad(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                            kernel_size=3, stride=2, padding=1),
            CustomPad(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1,
                            kernel_size=3, stride=1, padding=1),
            nn.Flatten(start_dim=1),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        '''        
        param x: the input; vectors from the previous DigitCaps layer
        return: two things, reconstructed images and the class scores, y
        '''
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        # find the capsule with the maximum vector length
        # here, vector length indicates the probability of a class' existence
        _, max_length_indices = classes.max(dim=1)
        
        # create a sparse class matrix
        sparse_matrix = torch.eye(10) # 10 is the number of classes in MNIST
        
        if GPU_AVAILABLE:
            sparse_matrix = sparse_matrix.cuda()
        
        # get the class scores from the "correct" capsule
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        
        # create reconstructed pixels
        x = x * y[:, :, None]

        # flatten image into a vector shape (batch_size, vector_dim)
        flattened_x = x.contiguous().view(x.size(0), -1)
        
        # create reconstructed image vectors
        reconstructions = self.deconv(flattened_x)
        
        # return reconstructions and the class scores, y
        return reconstructions, y