import torch
import torch.nn as nn

from Device import DEVICE


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, batch_size,
                 BERT_size,windows_size):
        super(CNNLSTM, self).__init__()

        BERT_input_size = BERT_size * (windows_size+1)

        self.conv1 = torch.nn.Conv1d(in_channels=768, out_channels=1, kernel_size=3)
        input_dim = input_dim - BERT_input_size + 1
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            dropout=dropout, num_layers=num_layers, batch_first=True)

        self.lstm.to(DEVICE)

        self.fc = nn.Linear(hidden_dim, output_dim).to(DEVICE)

        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.num_layers = num_layers
        self.dropout = dropout
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

    def reset_cell(self,batch_size):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE),
                            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE))

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        bert = input_tensor[:, :, :2304].reshape([batch_size, 768, 3])
        non_bert = input_tensor[:, :, :-2304]
        bert_new = self.conv1(bert)
        bert_new = bert_new.reshape([batch_size, 1, 1])
        input_tensor = torch.cat((non_bert, bert_new), dim=2)

        self.reset_cell(batch_size)

        out, (_) = self.lstm(input_tensor,self.hidden_cell)

        out = out[:, -1, :]
        out = out.to(DEVICE)

        output = self.fc(out)

        return output
