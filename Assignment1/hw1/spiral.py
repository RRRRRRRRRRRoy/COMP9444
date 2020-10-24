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
        self.h1 = output_model1

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
        
        # seconde way writing this code
        # input = self.layer1(input)
        # input = self.tanh(input)
        # self.h1 = input
        # input = self.layer2(input)
        # input = self.tanh(input)
        # self.h2 = input
        # input = self.layer3(input)
        # final_output = self.sigmoid(input)

        # CHANGE CODE HERE
        return final_output

def check_number_layer(net, layer):
    if(layer == 1):
        mid_layer = net.h1
    elif(layer == 2):
        mid_layer = net.h2
    return mid_layer

def graph_hidden(net, layer, node):
    # INSERT CODE HERE
    # These code is modified from graph_output
    # Source: spiral_main.py
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again
        # get the number of the layers
        mid_layer = check_number_layer(net, layer)
        mid_layer_2_filtered = mid_layer[:,node]
        # Whether checking is because of the codomain of Tanh and sigmoid
        # we need to do the filting
        pred = (mid_layer_2_filtered >= 0).float()

        # plot function computed by model
        # same come with graph_output
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
