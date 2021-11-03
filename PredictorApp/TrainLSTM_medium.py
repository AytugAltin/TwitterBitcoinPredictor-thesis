import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from Device import DEVICE
from FeatureExtraction import *
from Models.LSTM import LSTMModel
from Trainer import Trainer

if __name__ == '__main__':
    dataset = CombinedDataset(csv_file="Data/2018tweets/Objects/(60Min)bert.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = extract_features(dataset)

    scaler = MinMaxScaler()

    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    batch_size = 64

    train_features = torch.Tensor(X_train_arr).to(DEVICE)
    train_targets = torch.Tensor(y_train_arr).to(DEVICE)
    val_features = torch.Tensor(X_val_arr).to(DEVICE)
    val_targets = torch.Tensor(y_val_arr).to(DEVICE)
    test_features = torch.Tensor(X_test_arr).to(DEVICE)
    test_targets = torch.Tensor(y_test_arr).to(DEVICE)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    input_size = len(X_train.columns)
    output_dim = 1
    hidden_size = 128
    num_layers = 3
    dropout = 0.2
    n_epochs = 10000
    learning_rate = 1e-3
    weight_decay = 1e-6

    model = LSTMModel(input_size, hidden_size, num_layers, output_dim, dropout).to(DEVICE)

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    opt = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)

    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_size,
              test_loader_one=test_loader_one, X_test=X_test, scaler=scaler)

    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_size)


