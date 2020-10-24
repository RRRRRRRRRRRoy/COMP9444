# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        # Part2 Question1
        # The CNN part is same as Class Netfull -----> kuzu.py 
        self.dimension = 1
        self.input_size = 2
        self.output_size = 1

        self.layer1 = torch.nn.Linear(self.input_size,num_hid)
        self.layer2 = torch.nn.Linear(num_hid,self.output_size)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        # using the Sequential to wrap up
        self.model = torch.nn.Sequential(
            self.layer1,
            self.tanh,
            self.layer2,
            self.sigmoid
        )

    def forward(self, input):
        # INSERT CODE HERE
        # Part2 Question1
        # These code of processing part is pseudocode from the hw1.pdf
        # From Part2 Question 1 
        # Source: https://www.cse.unsw.edu.au/~cs9444/20T3/hw1/index.html 
        # using slice to get x
        x = input[:,0]
        # using slice to get y
        y = input[:,1]
        # sqrt use torch version
        # this can help us easy to do the calculation
        
        # change to vector
        r = torch.sqrt(x*x+y*y)
        r_vector = r.reshape([-1,1])
        a = torch.atan2(y,x).reshape([-1,1])
        a_vector = a.reshape([-1,1])
        
        # connect vector -----> cat function ----->  connect 2 vectors
        # How to use cat function?
        # Source: https://pytorch.org/docs/stable/generated/torch.cat.html
        input = torch.cat([r_vector,a_vector],dim = self.dimension)
        output = self.model(input)
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        # Part2 Question2
        # Tip: the more hid_num, the faster converge
        # Variables
        self.input_size = 2
        self.output_size = 1
        # NN Layers
        self.layer1 = torch.nn.Linear(2,num_hid)
        self.layer2 = torch.nn.Linear(num_hid,num_hid)   
        self.layer3 = torch.nn.Linear(num_hid,1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        # rawNet Model
        self.model_rawNet = torch.nn.Sequential(
            self.layer1,
            self.tanh,
            self.layer2,
            self.tanh,
            self.layer3,
            self.sigmoid
        )


    def forward(self, input):
        # INSERT CODE HERE
        output = self.model_rawNet(input)
        # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
