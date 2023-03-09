import os
import sys
import copy
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler, TensorDataset
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, BatchNorm1d, Dropout, Flatten, BCELoss

# kernel size, channel size, stride, paddings are hyper parameters and can be tuned
class CNN(Module):   
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 64, kernel_size=[5,2], stride=1, padding=1), # change the input channel based on your data
            # Conv2d(len(act_list), 64, kernel_size=[5,2], stride=1, padding=1), # change the input channel based on your data
            ReLU(inplace=True),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=[1,2]),
            Dropout(0.25),
            # Defining another 2D convolution layer
            Conv2d(64, 64, kernel_size=[5,2], stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=[1,2]),
            Dropout(0.25),

            Conv2d(64, 32, kernel_size=[5,2], stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=[1,2]),
            Dropout(0.25),
            Flatten(),
            Linear(14784,128), # change the dimension based on your own data
            ReLU(inplace=True),
            BatchNorm1d(128),
            Dropout(0.5),
            Linear(128, 16),
            ReLU(inplace=True),
            BatchNorm1d(16),
            Dropout(0.5),
            Linear(16, num_classes),
            torch.nn.Softmax(dim=1)
        )

    # Defining the forward pass    
    def forward(self, x):
        # x = torch.reshape(x, (-1, 2, 128, 1)) # change to your data dimension, the example here is 2 channel x 128 samples/channel
        x = self.cnn_layers(x)
        return x
    
def get_eval_model(num_classes, model_path):
    model = CNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model