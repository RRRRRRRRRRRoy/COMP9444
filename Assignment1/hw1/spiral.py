# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE

    def forward(self, input):
        # INSERT CODE HERE
        output = 0*input[:,0] # CHANGE CODE HERE
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE

    def forward(self, input):
        # INSERT CODE HERE
        output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
