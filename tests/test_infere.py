import pytest
from classes.loader import Loader
from classes.preprocessing import Preprocessing
import pandas as pd
import numpy as np
from classes.log_reg import LogReg


def main():
    loader = Loader("data/dataset_train.csv")
    means, stds, thetas = loader.load_parameters("out/parameters.csv")
    preproc = Preprocessing(loader.pd_data, loader.houses)
    X_train, y_train_dict, X_val, y_val_dict = preproc.pipe()

    nb_features = X_val.shape[1]
    log_reg = LogReg(nb_features=nb_features, test=True, thetas=thetas)

    output = log_reg.test(X_val)["Hogwarts House"].values

    houses = np.array([None] * len(X_val))
    for house, arr in y_val_dict.items():
        houses[arr] = house

    precision = round(np.sum(houses == output) / float(len(houses)), 4)
    print("\033[92mprecision on valid dataset : ", precision, "\033[0m")


if __name__ == "__main__":
    main()
