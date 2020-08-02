import math

import numpy as np
import torch
import torch.nn as nn


class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r_list = list(torch.split(x_r, 3, dim=1))

        for r in r_list:
            x = x + r * (torch.pow(x, 2) - x)
        return x
        
