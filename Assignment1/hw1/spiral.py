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
        self.model1 = torch.nn.Sequential(
            self.layer1,
            self.tanh
        )
        self.model2 = torch.nn.Sequential(
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
        output_model1 = self.model1(input)
        
        # Answer of Q5
        self.Q5Result = output_model1

        final_output = self.model2(output_model1)

        return final_output

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
        self.model1_rawNet = torch.nn.Sequential(
            self.layer1,
            self.tanh
        )
        self.model2_rawNet = torch.nn.Sequential(
            self.layer2,
            self.tanh
        )
        self.model3_rawNet = torch.nn.Sequential(
            self.layer3,
            self.sigmoid
        )


    def forward(self, input):
        # INSERT CODE HERE
        output_model1 = self.model1_rawNet(input)
        self.h1 = output_model1
        output_model2 = self.model2_rawNet(output_model1)
        self.h2 = output_model2
        final_output = self.model3_rawNet(output_model2)

        # CHANGE CODE HERE
        return final_output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
