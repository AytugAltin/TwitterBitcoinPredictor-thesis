import torch
import torch.nn as nn
from torch.autograd import Variable

from Device import DEVICE


class EXPMODEL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, batch_size):
        super(EXPMODEL, self).__init__()
        self.name = "EXPMODEL"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_1 = nn.Linear(self.hidden_dim, self.output_dim)
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
        self.lstm.flatten_parameters()
        batch_size = input_tensor.size(0)
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))

        output, (hn, cn) = self.lstm(input_tensor, (h_0, c_0))
        out = output.view(-1, self.hidden_dim)
        out = self.relu(out)
        out = self.fc_1(out)
        # out = self.relu(out)
        # out = self.fc_2(out)
        return out
