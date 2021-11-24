import torch
import torch.nn as nn
from torch.autograd import Variable

from Device import DEVICE


class EXPMODEL2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, batch_size):
        super(EXPMODEL2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_1 = nn.Linear(self.hidden_dim, 64)
        self.fc_2 = nn.Linear(64, self.output_dim)

        self.relu = nn.ReLU()
        self.new_epoch = True

        self.h = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))
        self.c = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))

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
        self.h = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))
        self.c = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))

    def forward(self, input_tensor):
        self.lstm.flatten_parameters()

        if self.new_epoch:
            self.reset_cell(batch_size=input_tensor.size(0))
            self.new_epoch = False

        h_0 = self.h.detach()
        c_0 = self.c.detach()

        output, (self.h, self.c) = self.lstm(input_tensor, (h_0, c_0))
        out = output.view(-1, self.hidden_dim)
        out = self.relu(out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out
