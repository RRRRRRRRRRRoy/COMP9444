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
    ratingoutput = (ratingOutput > 0.5).long()
    categoryOutput =categoryOutput.argmax(dim=parameters_dict["dimension"])
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
        #output, (hn, cn) = rnn(input, (h0, c0))

        # The LSTM return by torch
        # Here is the sample: output, (hn, cn) = rnn(input, (h0, c0))
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        category_o, (category_h, category_c) = self.lstm_category(input_data)
        category_h_part1 = category_h[-2, :, :]
        category_h_part2 = category_h[-1, :, :]
        category_last_hidden = torch.cat(
            [category_h_part1, category_h_part2], dim=parameters_dict["dimension"]
        )

        rate_o, (rate_h, rate_c) = self.lstm_rate(input_data)
        rate_h_part1 = rate_h[-2, :, :]
        rate_h_part2 = rate_h[-1, :, :]
        rate_last_hidden = torch.cat(
            [rate_h_part1, rate_h_part2], dim=parameters_dict["dimension"]
            )

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
        self.rate_BCELoss = tnn.BCELoss()
        self.category_NLLLoss = tnn.NLLLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingTarget = ratingTarget.float()

        rate_loss = self.rate_BCELoss(ratingOutput, ratingTarget)
        category_loss = self.category_NLLLoss(categoryOutput, categoryTarget)

        return rate_loss + category_loss



net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.95
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.01)
# optimiser = toptim.SGD(net.parameters(), lr=0.0000001)
