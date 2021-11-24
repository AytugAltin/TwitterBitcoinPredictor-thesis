import torch
import torch.nn as nn

from Device import DEVICE


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, batch_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            dropout=dropout, num_layers=num_layers, batch_first=True)

        self.lstm.to(DEVICE)

        self.fc = nn.Linear(hidden_dim, output_dim).to(DEVICE)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.new_epoch = True
        self.batch_size = batch_size

        name = ''
        name += "(hidden=" + str(hidden_dim) + ")"
        name += "(layer=" + str(num_layers) + ")"
        name += "(input=" + str(input_dim) + ")"
        name += "(dropout=" + str(dropout) + ")"
        name += "(output=" + str(output_dim) + ")"
        self.name = name

    def get_name(self):
        return self.name

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
        self.reset_cell(input_tensor.size(0))
        out, (_) = self.lstm(input_tensor, self.hidden_cell)

        out = out[:, -1, :]
        out = out.to(DEVICE)

        output = self.fc(out)

        return output
