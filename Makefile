
all: train predict

describe : export PYTHONPATH+=:$(PWD)
describe :
	python scripts/describe.py data/dataset_train.csv

hist : export PYTHONPATH+=:$(PWD)
hist :
	python scripts/histogram.py data/dataset_train.csv

scat : export PYTHONPATH+=:$(PWD)
scat :
	python scripts/scatter_plot.py data/dataset_train.csv

pair : export PYTHONPATH+=:$(PWD)
pair :
	python scripts/pair_plot.py data/dataset_train.csv

train : export PYTHONPATH+=:$(PWD)
train :
	python scripts/logreg_train.py data/dataset_train.csv

stochastic_train : export PYTHONPATH+=:$(PWD)
stochastic_train:
	python scripts/logreg_train_bonus.py data/dataset_train.csv stochastic

minibatch_train : export PYTHONPATH+=:$(PWD)
minibatch_train:
	python scripts/logreg_train_bonus.py data/dataset_train.csv minibatch

predict : export PYTHONPATH+=:$(PWD)
predict:
	python scripts/logreg_predict.py data/dataset_test.csv out/parameters.csv


E2E : export PYTHONPATH+=:$(PWD)
E2E :
	python tests/test_infere.py

clean :
	rm out/*
