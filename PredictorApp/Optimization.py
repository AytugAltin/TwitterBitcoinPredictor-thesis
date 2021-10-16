import datetime
from datetime import datetime
import numpy as np
from Dataset import *
from pathlib import Path
from Device import DEVICE
from sklearn.preprocessing import MinMaxScaler

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.model_index = 0

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def save_model(self):

        model_dir = "Models/Saved/" + self.model.get_name()
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = model_dir + "/model-" + str(self.model_index)
        self.model_index += 1
        if self.model_index % 10 == 0:
            self.model_index = 0

        torch.save(self.model.state_dict(), model_path)

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features,
              test_loader_one, X_test,scaler):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(DEVICE)
                    y_val = y_val.to(DEVICE)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 50) | (epoch % 10 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

            if epoch % 10 == 0:
                predictions, values = self.evaluate(test_loader_one, batch_size=1, n_features=n_features)
                df_result = format_predictions(predictions, values, X_test,scaler)
                print(df_result.head(3))
                print(df_result.sample(3))
                self.save_model()

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(DEVICE)
                y_test = y_test.to(DEVICE)
                self.model.eval()
                yhat = self.model(x_test).to(DEVICE)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values


def format_predictions(predictions, values, df_test,scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df
