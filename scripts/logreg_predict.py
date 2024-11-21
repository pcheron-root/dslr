from classes.loader import Loader
from classes.preprocessing import Preprocessing
from classes.log_reg import LogReg
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Makes predictions on a test set using logistic \
            regression model to sort Hogwarts student in their houses"
    )

    parser.add_argument(
        "dataset_filepath", type=str, help="path to the test dataset csv file."
    )

    parser.add_argument(
        "parameters_filepath",
        type=str,
        help="path to the csv file with the model parameters.",
    )

    loader = Loader(parser.parse_args().dataset_filepath, drop_nan=False)
    means, stds, thetas = loader.load_parameters(
        parser.parse_args().parameters_filepath
    )

    preproc_test = Preprocessing(loader.pd_data, means=means, stds=stds)
    X_test = preproc_test.pipe_test()

    nb_features = X_test.shape[1]
    logreg = LogReg(nb_features, test=True, thetas=thetas)
    logreg.test(X_test)


if __name__ == "__main__":
    main()
