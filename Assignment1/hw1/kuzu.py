# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################################################################################
# Part1 Question1
# based on the hw1 https://www.cse.unsw.edu.au/~cs9444/20T3/hw1/index.html
# The kind of NN layer -----> linear function -----> torch.nn.Linear(INPUT,OUTPUT)
# Page 5-6 Linear function Source: https://www.cse.unsw.edu.au/~cs9444/20T3/lect/1page/2b_Pytorch.pdf
# Output Layer -----> log softmax ----->torch.nn.LogSoftmax(DIMENSION)
# Page 12 log softmax Source: https://www.cse.unsw.edu.au/~cs9444/20T3/lect/1page/2b_Pytorch.pdf
# You can also find the reason why the parameter of logsoftmax dimension is 1.
#########################################################################################################
class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        # This part of code is for Part1 Question1
        self.pic_size = pow(28,2)
        self.output_num = 10
        self.dimension = 1
        self.layer1 = torch.nn.Linear(self.pic_size,self.output_num)
        self.logSoftMax = torch.nn.LogSoftmax(self.dimension)

    def forward(self, x):
        # INSERT CODE HERE
        # This part of code is for Part1 Question1
        auto_start = -1
        to_shape = [auto_start,self.pic_size]
        x = x.reshape(to_shape)

        layer1_output = self.layer1(x)
        final_output = self.logSoftMax(layer1_output)
        return final_output # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE

    def forward(self, x):
        # INSERT CODE HERE
        return 0 # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE

    def forward(self, x):
        # INSERT CODE HERE
        return 0 # CHANGE CODE HERE
