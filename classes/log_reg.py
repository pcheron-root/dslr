import matplotlib.pyplot as plt

# from classes.loader import Loader
# from classes.preprocessing import Preprocessing
from classes.model import Model
from classes.loss import Loss
import numpy as np
import pandas as pd
import os
import logging


class LogReg:
    def __init__(
        self,
        nb_features,
        test=False,
        thetas=None,
        batch_size=None,
        nb_epoch=100,
        learning_rate=0.1,
    ):
        self.nb_features = nb_features
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        if test:
            if thetas is None or nb_features != len(list(thetas.values())[0]):
                raise ValueError("Invalid usage of LogReg class.")
            self.models = {
                "Hufflepuff": Model(thetas=thetas["Hufflepuff"]),
                "Ravenclaw": Model(thetas=thetas["Ravenclaw"]),
                "Gryffindor": Model(thetas=thetas["Gryffindor"]),
                "Slytherin": Model(thetas=thetas["Slytherin"]),
            }
        else:
            self.models = {
                "Hufflepuff": Model(nb_feature=nb_features),
                "Ravenclaw": Model(nb_feature=nb_features),
                "Gryffindor": Model(nb_feature=nb_features),
                "Slytherin": Model(nb_feature=nb_features),
            }
        self.loss = Loss()

        logging.info("LogReg was instanciate successfully")

    def shuffle_data(self, X, y):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        return X[shuffled_idx], y[shuffled_idx]

    def train(self, X_train, y_train_dict, X_val, y_val_dict):
        losses = {}
        losses_valid = {}
        N = X_train.shape[0]

        if not self.batch_size or self.batch_size > N or self.batch_size <= 0:
            self.batch_size = N

        for house, model in self.models.items():
            losses[house] = []
            losses_valid[house] = []
            y_train = y_train_dict[house]
            y_val = y_val_dict[house]

            for i in range(self.nb_epoch):
                running_loss = 0.0
                for batch_idx in range(0, N, self.batch_size):
                    X_batch = X_train[batch_idx : batch_idx + self.batch_size]
                    y_batch = y_train[batch_idx : batch_idx + self.batch_size]
                    y_pred = model.forward(X_batch)
                    running_loss += self.loss(y_pred, y_batch) * X_batch.shape[0]
                    grads = self.loss.grads(X_batch, y_pred, y_batch)
                    model.update_thetas(grads, self.learning_rate)

                losses[house].append(running_loss / N)

                y_pred = model.forward(X_val)
                losses_valid[house].append(self.loss(y_pred, y_val))
        self.losses = losses
        self.losses_valid = losses_valid

        self.plot_loss()

    def save_parameters(self, means, stds):
        output = pd.DataFrame(
            np.array(
                [
                    means,
                    stds,
                    self.models["Gryffindor"].thetas,
                    self.models["Hufflepuff"].thetas,
                    self.models["Ravenclaw"].thetas,
                    self.models["Slytherin"].thetas,
                ]
            ).transpose(),
            columns=[
                "Mean",
                "Std",
                "Gryffindor",
                "Hufflepuff",
                "Ravenclaw",
                "Slytherin",
            ],
        )

        directory = "./out"
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            filepath = os.path.join(directory, "parameters.csv")
            output.to_csv(filepath, index=False)
        except Exception as e:
            print(e)
            exit(1)

    def plot_loss(self):
        self.plot(self.losses, self.losses_valid)

    # plot l'evolution de la loss calcule
    def plot(self, losses, losses_valid):
        epochs = [e for e in range(self.nb_epoch)]
        for i, house in enumerate(losses.keys()):
            plt.subplot(2, 2, i + 1)
            plt.plot(epochs, losses[house], label="Train")
            plt.plot(epochs, losses_valid[house], label="Validation")

            plt.title(f"House {house}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
        plt.tight_layout()
        plt.show()

    def test(self, X_test):
        y_pred = self.infere(X_test)
        output_data = []
        for i, pred in enumerate(y_pred):
            output_data.append([i, pred])
        output = pd.DataFrame(output_data, columns=["Index", "Hogwarts House"])

        directory = "./out"
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            filepath = os.path.join(directory, "houses.csv")
            output.to_csv(filepath, index=False)
        except Exception as e:
            print(e)
            exit(1)
        return output

    def infere(self, X):
        y_preds = np.zeros((X.shape[0], 4))
        houses = list(self.models.keys())
        for i, model in enumerate(self.models.values()):
            y_preds[:, i] = model.forward(X)
        predictions = np.argmax(y_preds, axis=1)
        mapToHouse = np.vectorize(lambda x: houses[x])
        predictions = mapToHouse(predictions)
        return predictions
