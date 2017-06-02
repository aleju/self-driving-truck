from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
import torch.optim as optim
from torch.autograd import Variable
from config import Config
import numpy as np
from lib import actions as actionslib
from lib.util import to_cuda, to_variable
import imgaug as ia
import random

ANGLE_BIN_SIZE = 5
GPU = 0

class SteeringWheelTrackerCNNModel(nn.Module):
    def __init__(self):
        super(SteeringWheelTrackerCNNModel, self).__init__()

        self.c1 = nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=1)
        self.fc1 = nn.Linear(32*(32//4)*(64//4), 16)
        self.fc2 = nn.Linear(16, 360//ANGLE_BIN_SIZE)

    def forward(self, inputs, softmax=False):
        x = inputs
        x = F.relu(self.c1(x))
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 32*(32//4)*(64//4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if softmax:
            x = F.softmax(x)
        return x

    def forward_image(self, subimg, softmax=False, volatile=False, requires_grad=True, gpu=GPU):
        subimg = np.float32([subimg/255]).transpose((0, 3, 1, 2))
        subimg = to_cuda(to_variable(subimg, volatile=volatile, requires_grad=requires_grad), GPU)
        return self.forward(subimg, softmax=softmax)
