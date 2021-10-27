from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Dataset import *
from Device import DEVICE
from Evaluate import evaluate_model, plot_predictions


class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.model_index = 0
        self.best_model_score = -20

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

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features,
              test_loader_one, X_test, scaler):

        for epoch in range(1, n_epochs + 1):
            # Forward
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            # Backward
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
                print(f"[{epoch}/{n_epochs}] "
                      f"Training loss: {training_loss:.4f}\t "
                      f"Validation loss: {validation_loss:.4f}")

            if epoch % 100 == 0:
                self.evaluate(test_loader_one, batch_size=1, n_features=n_features, X_test=X_test, scaler=scaler)

    def evaluate(self, test_loader_one, batch_size, n_features, X_test, scaler):

        df_result, result_metrics = evaluate_model(model=self.model, test_loader=test_loader_one, batch_size=batch_size
                                                   , n_features=n_features, X_test=X_test, scaler=scaler)

        plot_predictions(self.model, test_loader_one, n_features, X_test, scaler)
        if result_metrics["r2"] > self.best_model_score:
            self.best_model_score = result_metrics["r2"]
            self.save_model()

        print(result_metrics)
        print("best_model_score", self.best_model_score)

    def save_model(self):
        model_dir = "Models/Saved/" + self.model.get_name()
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = model_dir + "/model-" + str(self.model_index)
        self.model_index += 1
        if self.model_index % 10 == 0:
            self.model_index = 0

        torch.save(self.model.state_dict(), model_path)
