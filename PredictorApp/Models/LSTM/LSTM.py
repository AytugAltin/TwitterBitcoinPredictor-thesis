import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Device import DEVICE


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTM,self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            dropout=dropout, num_layers=num_layers, batch_first=True)

        self.lstm.to(DEVICE)

        self.fc = nn.Linear(hidden_dim, output_dim).to(DEVICE)

        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.num_layers = num_layers
        self.dropout = dropout

        name = ''
        name += "(hidden=" + str(hidden_dim) + ")"
        name += "(layer=" + str(num_layers) + ")"
        name += "(input=" + str(input_dim) + ")"
        name += "(dropout=" + str(dropout) + ")"
        name += "(output=" + str(output_dim) + ")"

        self.name = name

    def forward(self, input_tensor):
        h_0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_dim)
        h_0.to(DEVICE)

        c_0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_dim)
        c_0.to(DEVICE)

        out, (h_n, c_n) = self.lstm(input_tensor, (h_0, c_0))

        out = out[:, -1, :].to(DEVICE)
        output = self.fc(out)

        return output
