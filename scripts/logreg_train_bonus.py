from classes.loader import Loader
from classes.log_reg import LogReg
from classes.preprocessing import Preprocessing
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model to sort \
            Hogwarts student in their houses"
    )

    parser.add_argument(
        "dataset_filepath", type=str, help="path to the dataset csv file."
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "stochastic",
        help="launches a training with a stochastic gradient descent \
            with a learning rate of 0.0005 and 100 epochs"
    )

    subparsers.add_parser(
        "minibatch",
        help="launches a training with a minibatch gradient descent \
            with a batch size of 50, a learning rate of 0.01 and 100 epochs"
    )

    parser.add_argument(
        '-l',
        '--learning-rate',
        type=float,
        required=False,
        default=0.15,
        help= "learning rate float value"
    )

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        required=False,
        default=100,
        help= "number of epochs int value",
    )

    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        required=False,
        default=None,
        help= "batch size int value"
    )

    args = parser.parse_args()
    if args.command == "stochastic":
        lr = 0.0005
        epochs = 100
        batch = 1
    elif args.command == "minibatch":
        lr = 0.01
        epochs = 100
        batch = 50
    else:
        lr = args.learning_rate
        epochs = args.epochs
        batch = args.batch_size

    loader = Loader(args.dataset_filepath)

    preproc = Preprocessing(loader.pd_data, loader.houses)
    X_train, y_train_dict, X_val, y_val_dict = preproc.pipe()

    log_reg = LogReg(X_train.shape[1],
                    learning_rate=lr,
                    nb_epoch=epochs,
                    batch_size=batch)
    
    msg = f"Launches training with following hyperparameters:\n\tlearning_rate={lr}\n\tnumber of epochs={epochs}\n\tbatch_size={batch}"
    print(f"\033[96m {msg} \033[00m")
    log_reg.train(X_train, y_train_dict, X_val, y_val_dict)
    log_reg.save_parameters(preproc.means, preproc.stds)



if __name__ == "__main__":
    main()
