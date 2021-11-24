import torch
import torch.nn as nn

from Device import DEVICE


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, batch_size):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.dropout = dropout
        self.output_dim = output_dim

        self.rnn = nn.RNN(
            self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.new_epoch = True
        self.relu = nn.ReLU()

        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(DEVICE)

    def get_model_dict_info(self):
        dict = {
            "HiddenDim": self.hidden_dim,
            "NumLayers": self.num_layers,
            "InputDim": self.input_dim,
            "Dropout": self.dropout,
            "OutputDim": self.output_dim
        }
        return dict

    def forward(self, x):
        h0 = self.h0

        out, self.h0 = self.rnn(x, h0.detach())

        out = out[:, -1, :]

        out = self.relu(out)
        out = self.fc(out)
        return out