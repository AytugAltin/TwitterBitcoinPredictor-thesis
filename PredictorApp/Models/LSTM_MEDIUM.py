import torch
from torch import nn
from Device import DEVICE


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob
        ).to(DEVICE)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_dim)

        name = ''
        name += "(hidden=" + str(hidden_size) + ")"
        name += "(layer=" + str(num_layers) + ")"
        name += "(input=" + str(input_size) + ")"
        name += "(dropout=" + str(dropout_prob) + ")"
        name += "(output=" + str(output_dim) + ")"

        self.name = name

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :].to(DEVICE)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

    def get_name(self):
        return self.name
