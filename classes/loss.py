import numpy as np


class Loss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # cette fonction calcule les derives partielles pour une binary cross entropy loss
    # la binary cross entropy est une fonction cout utilise dans les classification
    # derive pour determiner le sens dans lequel on doit faire evoluer les parametres thetas
    # positivement ou negativement
    def grads(self, X, y_pred, y):
        y_diff = y_pred - y
        X_t = X.T
        z = np.dot(X_t, y_diff)
        grads = 1 / X.shape[0] * z
        return grads


def main():
    pass


if __name__ == "__main__":
    main()
