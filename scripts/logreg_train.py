from classes.loader import Loader
from classes.log_reg import LogReg
from classes.preprocessing import Preprocessing
import argparse
import pandas as pd
import numpy as np
import os
import logging


def main():
    logging.basicConfig(
        level=logging.DEBUG,  # On veut enregistrer tous les niveaux de log (DEBUG et au-dessus)
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="app.log",  # Nom du fichier où les logs seront sauvegardés
        filemode="w",  # 'w' pour écraser le fichier à chaque exécution, 'a' pour ajouter à la suite
    )

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
    log_reg = LogReg(X_train.shape[1], learning_rate=0.1)
    log_reg.train(X_train, y_train_dict, X_val, y_val_dict)

    log_reg.save_parameters(preproc.means, preproc.stds)


if __name__ == "__main__":
    main()
