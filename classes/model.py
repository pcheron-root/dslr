import numpy as np

# Chaque objet Model a ses poids, un poid par feature, chaque poid va designer l'importance de la feature pour determiner si l'undividu est ...
# Les poids evoluent au cours de l'entrainement et son mis a jour a chaque epoch


class Model:
    def __init__(self, nb_feature=None, features_array=None, thetas=None):
        if features_array is not None:
            self.thetas = np.zeros(features_array.shape[1])
        elif nb_feature is not None:
            self.thetas = np.zeros(nb_feature)
        elif thetas is not None:
            self.thetas = thetas
        else:
            raise ValueError("Either nb_feature or features_array must be provided.")

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.thetas))

    def update_thetas(self, grads, learning_rate=0.1):
        self.thetas -= learning_rate * grads

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
