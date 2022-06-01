import torch
from torch import nn
from torchvision import models

class BasicVersion():
    def __init__(self, model_name):
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, 4)

            self.loss_fn = nn.CrossEntropyLoss() # already includes the Softmax activation
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        elif model_name == 'ResNet':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 10)

            self.loss_fn = nn.CrossEntropyLoss() # already includes the Softmax activation
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def get_model(self):
        return (self.model, self.loss_fn, self.optimizer)