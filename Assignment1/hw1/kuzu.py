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
        # 
        self.layer1 = torch.nn.Linear(self.pic_size,self.output_num)
        self.logSoftMax = torch.nn.LogSoftmax(dim = self.dimension)

    def forward(self, x):
        # INSERT CODE HERE
        # This part of code is for Part1 Question1
        auto_start = -1
        to_shape = [auto_start,self.pic_size]
        x = x.reshape(to_shape)

        layer1_output = self.layer1(x)
        final_output = self.logSoftMax(layer1_output)
        return final_output # CHANGE CODE HERE


#########################################################################################################
# Part1 Question2
# In the second question, we should design a 2 layes nerual network.
# hidden node -----> tanh -----> nn.Tanh()
# Page 6 Source: https://www.cse.unsw.edu.au/~cs9444/20T3/lect/1page/2b_Pytorch.pdf
# output node -----> log softmax -----> nn.LogSoftmax
# Page 12 Source: https://www.cse.unsw.edu.au/~cs9444/20T3/lect/1page/2b_Pytorch.pdf
# Can also use the container nn.sequential to set up this nerual network
# Source: https://stackoverflow.com/questions/59916814/how-to-create-a-pytorch-nn-with-2-hidden-layer-with-nn-sequential/59916948#59916948
#########################################################################################################
class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # picture size
        self.pic_size = pow(28,2)
        # number of output
        self.output_num = 10
        self.dimension = 1
        # changing hidden_nodes_num can increase the accuracy
        # 100   -----> 8371 84%
        # 1000  -----> 8438 84% - 85% -----> the best
        # 10000 -----> 8382 83%
        hidden_nodes_num = 1000

        # Setting the Neural Network layer
        self.layer1 = torch.nn.Linear(self.pic_size,hidden_nodes_num)
        self.layer2 = torch.nn.Linear(hidden_nodes_num,self.output_num)
        self.tanh = torch.nn.Tanh()
        self.logSoftMax = torch.nn.LogSoftmax(dim=self.dimension)
        
        # using the Sequential to wrap up these layers
        self.model = torch.nn.Sequential(
            self.layer1,
            self.tanh,
            self.layer2,
            self.logSoftMax
        )

    def forward(self, x):
        # INSERT CODE HERE
        auto_start = -1
        output = self.model(x.reshape([auto_start,self.pic_size]))
        return output # CHANGE CODE HERE

#########################################################################################################
# How to change the parameter in the maxpool to improve the accuracy
# https://stackoverflow.com/questions/54788554/deep-conv-model-number-of-parameters
# How to use conv2d in pytorch?
# Source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
#########################################################################################################
class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # Setting Constant variable
        self.dimension = 1
        self.lim_size = 49 * 6 * 6

        # Define the layer of NN
        self.convolutional_layer1 = torch.nn.Conv2d(1,98,3,1,1)
        self.convolutional_layer2 = torch.nn.Conv2d(98,49,3,1,1)
        # Max pool to improve the accuracy
        # 3,2 -----> image overlap -----> more accurate
        self.maxpool = torch.nn.MaxPool2d(3,2)
        self.linear_layer = torch.nn.Linear(self.lim_size, 100)
        self.relu = torch.nn.ReLU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=self.dimension)

        # Can also combine them into one model
        # Using the sequential function to wrap them up
        self.conv_model1 = torch.nn.Sequential(
            self.convolutional_layer1,
            self.relu,
            self.maxpool
        )

        # Using the sequential function to wrap them up
        self.conv_model2 = torch.nn.Sequential(
            self.convolutional_layer2,
            self.relu,
            self.maxpool
        )


    def forward(self, x):
        # Put input value in model1
        model1_output = self.conv_model1(x)
        # Put the result from model1_output into model2
        model2_output = self.conv_model2(model1_output)
        
        # the value of auto_start is -1, means the NN will start itself
        auto_start = -1
        conv_output = model2_output.reshape([auto_start,self.lim_size])
        
        # Using the linear_layer and logsoftmax get the final result
        linear_output = self.linear_layer(conv_output)
        final_output = self.logsoftmax(linear_output)
        return final_output # CHANGE CODE HERE
