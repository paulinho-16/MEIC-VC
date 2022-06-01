import torch
from torch import nn
import torch.nn.functional as F

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.pool_size = 2
        self.nb_filters = 32
        self.kernel_size = 3

        """
        nn.Conv2d(1, self.nb_filters, self.kernel_size),
        nn.ReLU(),
        nn.Conv2d(self.nb_filters, self.nb_filters, self.kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(self.pool_size),

        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(307328, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 4),
        """
        
        """
        nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(self.pool_size, self.pool_size),
    
        nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(self.pool_size, self.pool_size),
        
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(self.pool_size, self.pool_size),
        
        nn.Flatten(),
        nn.Linear(82944,1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512,6)
        """
        self.layers = nn.Sequential(
            nn.Conv2d(1, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.nb_filters, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),

            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(307328, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

        self.conv1 = nn.Conv2d(3, self.nb_filters, self.kernel_size)
        self.pool = nn.MaxPool2d(self.pool_size)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, self.kernel_size)
        self.fc1 = nn.Linear(self.nb_filters * 48 * 48, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

        # print('x_shape:', x.shape)
        # logits = self.layers(x)
        #return logits