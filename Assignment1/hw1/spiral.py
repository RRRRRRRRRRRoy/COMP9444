# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


#########################################################################################################
# In the Part 2 can set a lager hid_num and initial weight to speed up
#########################################################################################################
# Part 2 Question 1 and QUestion 2
# The Boundry of PolarNet is the straight line
# Using the straight line to cut the dot and get the final result
# Using nn.cat function to connect 2 vectors
#########################################################################################################

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        # Part2 Question1
        # The CNN part is same as Class Netfull -----> kuzu.py 
        self.dimension = 1
        # the input size and output size
        self.input_size = 2
        self.output_size = 1

        # 2 linear layers
        # And 2 activation fuction
        self.layer1 = torch.nn.Linear(self.input_size,num_hid)
        self.layer2 = torch.nn.Linear(num_hid,self.output_size)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        # using the Sequential to wrap up
        # model1 -----> layer1 and TanH
        self.model1 = torch.nn.Sequential(
            self.layer1,
            self.tanh
        )
        # model2 ------> layer2 and sigmoid
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
        # this code is based on the pseudocode of hw1.pdf
        # Source: https://www.cse.unsw.edu.au/~cs9444/20T3/hw1/index.html 
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
        # Adding the mid layer to get the middle output
        self.h1 = output_model1
        final_output = self.model2(output_model1)
        return final_output

#########################################################################################################
# Part 2 Question 3 and QUestion 4
# The Boundry of RawNet is the curve line
# The reason of curve line is that the RawNet is try to overfit each dot in the graph
# In this way, rawNet can get the final result -----> convergence
# Therefore you can find some blue curve lines across the orange line
#########################################################################################################
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
        # Using 3 Linear Layers to initialize the rawNet
        self.layer1 = torch.nn.Linear(2,num_hid)
        self.layer2 = torch.nn.Linear(num_hid,num_hid)   
        self.layer3 = torch.nn.Linear(num_hid,1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        # rawNet Model
        # model1 -----> layer1 and tanh
        self.model1_rawNet = torch.nn.Sequential(
            self.layer1,
            self.tanh
        )
        # model2 -----> layer2 and tanh
        self.model2_rawNet = torch.nn.Sequential(
            self.layer2,
            self.tanh
        )
        # model3 -----> layer3 and sigmoid
        self.model3_rawNet = torch.nn.Sequential(
            self.layer3,
            self.sigmoid
        )


    def forward(self, input):
        # INSERT CODE HERE
        output_model1 = self.model1_rawNet(input)
        # Get the answer of Q5
        # This answer is for 2 middle layers situation
        self.h1 = output_model1
        output_model2 = self.model2_rawNet(output_model1)
        # The second middle layer
        self.h2 = output_model2
        final_output = self.model3_rawNet(output_model2)
        # CHANGE CODE HERE
        return final_output

# Using this method to check the number of middle layer the NN has
# If number is 1 -----> polar
# If number is 2 -----> rawNet
def check_number_layer(net, layer):
    if(layer == 1):
        mid_layer = net.h1
    elif(layer == 2):
        mid_layer = net.h2
    return mid_layer

def graph_hidden(net, layer, node):
    # INSERT CODE HERE
    # These code is modified from graph_output
    # graph_output function in spiral_main.py  
    # Source: https://www.cse.unsw.edu.au/~cs9444/20T3/hw1/hw1.zip
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
        # The tanh is from -1 to 1
        # The sigmoid is from 0 to 1
        # It is necessary to select the correct part by checking whether the value is larger than 0
        pred = (mid_layer_2_filtered >= 0).float()

        # plot function computed by model
        # same come with graph_output which is is spiral_main.py
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
