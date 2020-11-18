#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

"""

###############################################################################
##### Model Reviews(The Answer of the question in the assignment2 pdf): ######
###############################################################################

Firstly, doing the emotion analysis is similar to text categorization task. 
In other words, these two tasks are parallel. This is because in these two
tasks, we also use the LSTM to do the feature extraction. After getting the
features, we also use the artificial neural network to categorize.
After programming, we can find the result of the emotion analysis is nearly 90%
To increase the performance of the model, we also use the features which is provided
by the previous classification task to calculate the attention map. You can also
find this part in the code with the Source from the Internet. In our model, consider the
emotion analysis task, this model predict the probability whether it is a positive sample.
Furthermore, as for classification task, this model can provide the probability of 
the category each output belongs to.

In my program, I try to set the parameter as 64 instead of 56. This is because the result
provided by 64 is better than 56. In the optimizer, I chose to use Adam and set the learning
rate as the default which is 0.01. Also, I would like to set all these parameters into a same dict
and also put the layer together by using torch.nn.Sequential function. This can help me reduce the time 
to find and change the parameters

Also, consider the optimizer in this model, we can choose both Adam and SGD.
If you have a gpu which supports CUDA, you can choose SGD. If not, you can use
Adam instead. A most interesting point should be postes here, when using SGD, if the learning 
rate you set is not good, your loss result will keep positive and hard to converge.
In this situation, the loss will in the range from 0.490 - 0.495. Also, the weighted score is 
nearly 22(less eaual than 22). Also, an inappropriate batch size can also cause the overfitting.
These are what we should pay more attention in this assignment.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################


def tokenise(sample):
    processed = sample.split()
    return processed


def preprocessing(sample):
    return sample


def postprocessing(batch, vocab):
    return batch


# this dimension should be set as 300
stopWords = {}
wordVectors = GloVe(name='6B', dim=300)
parameters_dict = {"dimension": 1, 
                    "dropout":0.5, 
                    "input_size":300,
                    "hidden_size":75,
                    "rate_Layer_numer":2,
                    "layer_input_size":75*2,
                    "encode_output_size":64,
                    "category_Layer_numer":1,
                    "rate_layer1_output":1,
                    "category_Layer1_output":5}

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################


def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # Attention: If you are using Torch 1.6 you can use tensor.int()
    # But if you use torch 1.2 plz change it to long(avoiding type error)
    ratingoutput = (ratingOutput > 0.5).long()
    categoryOutput = categoryOutput.argmax(dim=parameters_dict["dimension"])
    return ratingoutput , categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################


class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        # These part is for rating analysis
        self.lstm_rate = torch.nn.LSTM(
            parameters_dict["input_size"], parameters_dict["hidden_size"], num_layers=parameters_dict["rate_Layer_numer"], 
            batch_first=True, bidirectional=True, dropout=parameters_dict["dropout"]
        )
        self.fullconnection_rate_attention = torch.nn.Linear(
            parameters_dict["layer_input_size"], parameters_dict["layer_input_size"]
            )
        self.fullconnection_rate_layer1 = torch.nn.Linear(
            parameters_dict["layer_input_size"], parameters_dict["rate_layer1_output"]
        )


        # This part is for the output categorize
        self.lstm_category = torch.nn.LSTM(
            parameters_dict["input_size"], parameters_dict["hidden_size"], num_layers=parameters_dict["category_Layer_numer"], 
            batch_first=True, bidirectional=True, dropout=parameters_dict["dropout"]
        )
        self.fullconnection_category_encode = torch.nn.Linear(
            parameters_dict["layer_input_size"], parameters_dict["encode_output_size"]
        )
        self.fullconnection_category_attention = torch.nn.Linear(
            parameters_dict["layer_input_size"], parameters_dict["layer_input_size"])
        self.fullconnection_category_layer1 = torch.nn.Linear(
            parameters_dict["encode_output_size"], parameters_dict["category_Layer1_output"])

        # activation function
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(parameters_dict["dropout"])
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=parameters_dict["dimension"])

        # Try to put these layers together
        # Using the attention method needs to find the middle part
        # Therefore using nn.Sequential to separate them
        # Source: https://stackoverflow.com/questions/63914843/layernorm-inside-nn-sequential-in-torch
        self.rate_attention_layer = torch.nn.Sequential(
            self.fullconnection_rate_attention,
            self.sigmoid
        )
        self.category_attention_layer = torch.nn.Sequential(
            self.fullconnection_category_attention,
            self.sigmoid
        )

        self.rate_output_layer = torch.nn.Sequential(
            self.fullconnection_rate_layer1,
            self.sigmoid
        )
        self.category_output_layer = torch.nn.Sequential(
            self.fullconnection_category_encode,
            self.relu,
            self.fullconnection_category_layer1,
            self.softmax
        )


    def forward(self, input, length):
        input_data = torch.nn.utils.rnn.pack_padded_sequence(
            input, length, batch_first=True
        )
        # The LSTM return by torch
        # Here is the sample: output, (hn, cn) = rnn(input, (h0, c0))
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        
        # For rate
        rate_o, (rate_h, rate_c) = self.lstm_rate(input_data)
        # slice method
        # Using cat can connect 2 tensors
        # Source: https://pytorch.org/docs/stable/generated/torch.cat.html

        # Here is the BiLSTM
        # Source: https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        rate_last_hidden = torch.cat(
            [rate_h[-2, :, :], rate_h[-1, :, :]], dim=parameters_dict["dimension"]
        )

        # For category
        category_o, (category_h, category_c) = self.lstm_category(input_data)
        # slice method
        # Using cat can connect 2 tensors
        # Source: https://pytorch.org/docs/stable/generated/torch.cat.html
        
        # Here is the BiLSTM
        # Source: https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        category_last_hidden = torch.cat(
            [category_h[-2, :, :], category_h[-1, :, :]], dim=parameters_dict["dimension"]
        )

        # Attention Map part which is to increase the accuracy
        # Here is the sample code and introduction about using attention to increase the accuracy
        # Source1: https://stackoverflow.com/questions/51817479/vscode-please-clean-your-repository-working-tree-before-checkout
        # Source2: https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
        rate_attention = self.rate_attention_layer(rate_last_hidden)
        category_attention = self.category_attention_layer(category_last_hidden)

        category_last_hidden = category_last_hidden * rate_attention
        rate_last_hidden = rate_last_hidden * category_attention

        rate_output = self.rate_output_layer(rate_last_hidden)
        rate_output = rate_output.squeeze()
        category_output = self.category_output_layer(category_last_hidden)

        return rate_output, category_output


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        # How to use BCELoss
        # Source: https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        self.rate_BCELoss = tnn.BCELoss()
        # How to use NLLLoss
        # Source: https://discuss.pytorch.org/t/understanding-nllloss-function/23702
        self.category_NLLLoss = tnn.NLLLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingTarget = ratingTarget.float()

        # rate_loss = self.rate_BCELoss(ratingOutput, ratingTarget)
        # category_loss = self.category_NLLLoss(categoryOutput, categoryTarget)

        return self.rate_BCELoss(ratingOutput, ratingTarget) + self.category_NLLLoss(categoryOutput, categoryTarget)



net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

# When you find the best model change this parameter as 1
# This means using all these data to train this model
trainValSplit = 0.95

# The larger batchsize the slower speed of training.
# Sometimes it can reduce the overfitting problem.
# Attention: If you set this value too large, your the memory of graphics card will overflow
batchSize = 32

# Set up epochs based on your PC
epochs = 10

# optimiser = toptim.Adam(net.parameters(), lr=0.01)
# optimiser = toptim.SGD(net.parameters(), lr=0.0000001)
optimiser = toptim.Adam(net.parameters(), lr=0.001
)