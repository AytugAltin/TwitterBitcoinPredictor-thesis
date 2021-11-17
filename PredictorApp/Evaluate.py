import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Dataset import *
from Device import DEVICE

from matplotlib import pyplot as plt


def evaluate_model(model, test_loader, X_test, scaler, batch_size=1, n_features=1):
    predictions, values = get_predictions(model, test_loader, batch_size=batch_size, n_features=n_features)
    df_result = format_predictions(predictions, values, X_test, scaler)
    result_metrics = calculate_metrics(df_result)

    return df_result, result_metrics


def get_predictions(model, test_loader, batch_size=1, n_features=1):
    with torch.no_grad():
        predictions = []
        values = []
        for x_test, y_test in test_loader:
            x_test = x_test.view([batch_size, -1, n_features]).to(DEVICE)
            y_test = y_test.to(DEVICE)
            model.eval()
            yhat = model(x_test).to(DEVICE)
            predictions.append(yhat.cpu().detach().numpy())
            values.append(y_test.cpu().detach().numpy())

    return predictions, values


def format_predictions(predictions, values, df_test, scaler):
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


def calculate_metrics(df):
    return {'mae': mean_absolute_error(df.value, df.prediction),
            'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
            'mse': mean_squared_error(df.value, df.prediction),
            'r2': r2_score(df.value, df.prediction)}


def plot_predictions(df_result, title=""):
    plt.close()
    df_result.plot(y=['value', 'prediction'], title=title)
    plt.show()
