# https://www.kaggle.com/competitions/titanic

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class CSVDataset(Dataset):
    def __init__(self, path):
        # define the necessary columns
        numeric_columns = ["Age", "SibSp", "Parch", "Fare"]
        categorical_columns = ["Pclass", "Sex", "Embarked"]
        columns = [
            "Pclass", "Sex", "Age", "SibSp",
            "Parch", "Fare", "Embarked"
        ]
        # load the data
        df = pd.read_csv(path)
        # remove rows with missing values
        # df = df.dropna(subset=columns)
        # perform mean imputation for numeric columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        # perform mode imputation for categorical columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        self.X = df[columns].values
        # read y
        self.y = df["Survived"].values
        # turn labels to numbers
        self.X[:, 1] = LabelEncoder().fit_transform(self.X[:, 1])
        self.X[:, 6] = LabelEncoder().fit_transform(self.X[:, 6])
        # set the correct types and shape
        self.X = self.X.astype("float32")
        self.y = self.y.astype("float32")
        self.y = self.y.reshape((len(self.y), 1))

    @property
    def lenX(self):
        return len(self.X[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index], self.y[index]]

    def get_splits(self, test_perc=0.2):
        test_size = round(test_perc * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])


def prepare_data(path):
    dataset = CSVDataset(path)
    n_inputs = dataset.lenX
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl, n_inputs


def train_model(model, train_dl):
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
    path = os.path.join(os.path.dirname(__file__), "train.csv")
    train_dl, test_dl, n_inputs = prepare_data(path)
    print(f"Size of train data: {len(train_dl.dataset)}")
    print(f"Size of test data: {len(test_dl.dataset)}")
    # create the model
    model = nn.Sequential(
        nn.Linear(n_inputs, 15),
        nn.ReLU(),
        nn.Linear(15, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    # train the model
    train_model(model, train_dl)
    # evaluate the model
    acc = evaluate_model(model, test_dl)
    print(f"Accuracy: {acc:.3f}")
    # make predictions
    yhat = predict(model, [2, 0, 14, 1, 0, 30.0708, 0])
    expected = 1
    print(f"Predicted: {yhat.round().item():.0f} ({yhat.item():.3f})")
    print(f"Expected: {expected}")


if __name__ == "__main__":
    main()
