import pytest
import numpy as np
from classes.model import Model


def test_model_nb_feature():
    model = Model(nb_feature=5)
    assert model.thetas.shape[0] == 5
    assert np.all(model.thetas == 0)


def test_model_features_array():
    features = np.array([[1, 2, 3], [4, 5, 6]])
    model = Model(features_array=features)
    assert model.thetas.shape[0] == features.shape[1]
    assert np.all(model.thetas == 0)


def test_model_no_arguments():
    with pytest.raises(
        ValueError, match="Either nb_feature or features_array must be provided."
    ):
        Model()


def test_forward():
    X = np.array([[1, 2], [2, 3]])
    model = Model(2)
    expected_output = model.sigmoid(np.dot(X, model.thetas))

    y_pred = model.forward(X)
    np.testing.assert_almost_equal(y_pred, expected_output)


def test_update_thetas():
    grads = np.array([0.1, -0.2])
    model = Model(2)
    learning_rate = 0.1
    initial_thetas = model.thetas.copy()
    model.update_thetas(grads, learning_rate)
    expected_thetas = initial_thetas - learning_rate * grads
    np.testing.assert_almost_equal(model.thetas, expected_thetas)


def test_sigmoid():
    X = np.array([0, 1, -1])
    model = Model(2)
    expected_output = 1 / (1 + np.exp(-X))
    output = model.sigmoid(X)
    np.testing.assert_almost_equal(output, expected_output)
