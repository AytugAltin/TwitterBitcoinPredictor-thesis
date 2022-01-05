import torch
import torch.nn as nn

from Device import DEVICE


class DenseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, batch_size,
                 BERT_size, dense_out=10):
        super(DenseLSTM, self).__init__()

        self.BERT_input_size = BERT_size
        self.dense_out = dense_out

        self.fc_bert = nn.Linear(self.BERT_input_size, dense_out).to(DEVICE)

        lstm_input_dim = int(input_dim + dense_out)
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_dim,
                            dropout=dropout, num_layers=num_layers, batch_first=True)

        self.lstm.to(DEVICE)

        self.fc = nn.Linear(hidden_dim, output_dim).to(DEVICE)
        self.new_epoch = True

        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.input_dim = input_dim + BERT_size

        name = ''
        name += "(hidden=" + str(hidden_dim) + ")"
        name += "(layer=" + str(num_layers) + ")"
        name += "(input=" + str(self.input_dim) + ")"
        name += "(lstm_input=" + str(lstm_input_dim) + ")"
        name += "(dense_out=" + str(dense_out) + ")"
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
        batch_size = input_tensor.size(0)
        bert = input_tensor[:, :, :self.BERT_input_size]
        non_bert = input_tensor[:, :, :-self.BERT_input_size]
        bert_new = self.fc_bert(bert)

        input_tensor = torch.cat((non_bert, bert_new), dim=2)

        self.reset_cell(batch_size)

        out, (_) = self.lstm(input_tensor, self.hidden_cell)

        out = out[:, -1, :]
        out = out.to(DEVICE)

        output = self.fc(out)

        return output
