from Models.LSTMCells import LSTMCELLS
from Models.RNN import RNNModel
from Models.EXPMODEL import EXPMODEL
from Models.EXPMODEL2 import EXPMODEL2
from Models.Hybrid.CNNLSTM import CNNLSTM
from Models.Hybrid.DenseLSTM import DenseLSTM

# try:
#     Models["LSTM"] = LSTMModel(input_dim, hidden_dim, num_layers, output_dim,
#                                dropout, BATCH_SIZE).to(DEVICE)
#     Models["RNN"] = RNNModel(input_dim, hidden_dim, num_layers, output_dim,
#                              dropout, BATCH_SIZE).to(DEVICE)
#     Models["LSTMCELLS"] = LSTMCELLS(input_dim, hidden_dim, output_dim,
#                              dropout, BATCH_SIZE).to(DEVICE)
#
#     Models["EXPMODEL2"] = EXPMODEL2(input_dim, hidden_dim, num_layers, output_dim,
#                                     dropout, BATCH_SIZE).to(DEVICE)
#     Models["DenseLSTM"] = DenseLSTM(input_dim, hidden_dim, num_layers, output_dim,
#                                     dropout, BATCH_SIZE, BERT_SIZE).to(DEVICE)
#     Models["CNNLSTM"] = CNNLSTM(input_dim, hidden_dim, num_layers, output_dim,
#                                 dropout, BATCH_SIZE, BERT_SIZE, window_size).to(DEVICE)
# except Exception as e:
#     print(e)