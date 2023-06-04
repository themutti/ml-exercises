import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def prepare_data():
    # load the iris dataset
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    # set the dataset types
    X = X.astype("float32")
    y = y.astype("int64")
    # apply one-hot function to y
    y = torch.tensor(y)
    y = F.one_hot(y, num_classes=3).float().numpy()
    # split dataset in train data and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_dl, test_dl


def train_model(model, train_dl):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        for inputs, targets in train_dl:
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_dl):
    model.eval()
    predictions = []
    actuals = []
    for inputs, targets in test_dl:
        yhat = model(inputs)
        yhat = yhat.detach().numpy().round()
        actual = targets.numpy()
        predictions.append(yhat)
        actuals.append(actual)
    return accuracy_score(np.vstack(predictions), np.vstack(actuals))


def predict(model, row):
    model.eval()
    row = torch.Tensor([row])
    yhat = model(row)
    return yhat.detach().numpy()


def main():
    # load the dataset and prepare the data
    train_dl, test_dl = prepare_data()
    print(f"Size of train data: {len(train_dl.dataset)}")
    print(f"Size of test data: {len(test_dl.dataset)}")
    # create the model
    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
        nn.Softmax(dim=1)
    )
    # train the model
    train_model(model, train_dl)
    # evaluate the model
    acc = evaluate_model(model, test_dl)
    print(f"Accuracy: {acc:.3f}")
    # make predictions
    yhat = predict(model, [5.5, 2.6, 4.4, 1.2])
    expected = 1
    print(f"Predicted: {np.argmax(yhat)}")
    print(f"Expected: {expected}")


if __name__ == "__main__":
    main()
