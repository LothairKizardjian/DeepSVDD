import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.transforms import ToTensor, Lambda


class Mnist_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(32, 32, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(32, 64, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(64, 64, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Flatten(),
            nn.Linear(64, config.output_channels, bias=False)
            # Pas moyen de récup le shape après le flatten directement ?
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv_stack(x)
        #print(x.shape)
        return x
