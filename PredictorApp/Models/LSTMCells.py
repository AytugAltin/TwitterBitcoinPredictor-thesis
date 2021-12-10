import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Device import DEVICE


class LSTMCELLS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, batch_size,window_size):
        super(LSTMCELLS, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 3
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.name = "LSTMCELLS"

        self.output_nodes = 1

        self.number_of_nodes_1 = window_size
        self.number_of_nodes_2 = window_size
        self.number_of_nodes_3 = window_size

        self.lstm1 = nn.LSTMCell(self.input_dim, self.number_of_nodes_1)
        self.lstm2 = nn.LSTMCell(self.number_of_nodes_1, self.number_of_nodes_2)
        self.lstm3 = nn.LSTMCell(self.number_of_nodes_2, self.number_of_nodes_3)
        self.linear = nn.Linear(self.number_of_nodes_3, self.output_nodes)

        self.new_epoch = True
        self.relu = nn.ReLU()

    def get_name(self):
        return "name"

    def get_model_dict_info(self):
        dict = {
            "HiddenDim": self.hidden_dim,
            "NumLayers": self.num_layers,
            "InputDim": self.input_dim,
            "Dropout": self.dropout,
            "OutputDim": self.output_dim
        }
        return dict

    def reset_cell(self, batch_size):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE),
                            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)

        h_t = torch.zeros(batch_size, self.number_of_nodes_1).to(DEVICE)
        c_t = torch.zeros(batch_size, self.number_of_nodes_1).to(DEVICE)
        h_t2 = torch.zeros(batch_size, self.number_of_nodes_2).to(DEVICE)
        c_t2 = torch.zeros(batch_size, self.number_of_nodes_2).to(DEVICE)
        h_t3 = torch.zeros(batch_size, self.number_of_nodes_3).to(DEVICE)
        c_t3 = torch.zeros(batch_size, self.number_of_nodes_3).to(DEVICE)

        h_t, c_t = self.lstm1(input_tensor.squeeze(1), (h_t, c_t))
        h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))

        output = self.linear(h_t3)

        # out = output.view(-1, self.hidden_dim)
        # out = self.relu(out)
        # out = self.linear(out)
        # out = self.relu(out)
        # out = self.fc_2(out)
        return output
