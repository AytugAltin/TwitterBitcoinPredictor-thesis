from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from Device import DEVICE
from Evaluate import plot_predictions
from FeatureExtraction import *
from Models.LSTM import LSTMModel

if __name__ == '__main__':
    dataset = CombinedDataset(csv_file="Data/2018tweets/Objects/(60Min).csv")

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
    dropout = 0.6
    n_epochs = 10000
    learning_rate = 1e-3
    weight_decay = 1e-6

    model = LSTMModel(input_size, hidden_size, num_layers, output_dim, dropout).to(DEVICE)

    model.load_state_dict(torch.load(
        "Models/Saved/(hidden=128)(layer=3)(input=605)(dropout=0.6)(output=1)/model-9"))

    plot_predictions(model, test_loader, input_size, X_test, scaler)