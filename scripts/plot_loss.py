from classes.loader import Loader
from classes.log_reg import LogReg
from classes.preprocessing import Preprocessing
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model to sort \
            Hogwarts student in their houses"
    )

    parser.add_argument(
        "dataset_filepath", type=str, help="path to the dataset csv file."
    )

    loader = Loader(parser.parse_args().dataset_filepath)

    preproc = Preprocessing(loader.pd_data, loader.houses)
    X_train, y_train_dict, X_val, y_val_dict = preproc.pipe()
    log_reg = LogReg(X_train.shape[1], learning_rate=0.06)
    log_reg.train(X_train, y_train_dict, X_val, y_val_dict)

    log_reg.plot_loss()


if __name__ == "__main__":
    main()
