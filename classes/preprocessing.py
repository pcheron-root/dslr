import pandas as pd
import numpy as np

class Preprocessing:
    def delete_columns(self):
        self.cleaned_dataset = self.brut_dataset.drop(
            [
                "Index",
                "First Name",
                "Last Name",
                "Birthday",
                "Best Hand",
                "Arithmancy",
                "Defense Against the Dark Arts",
                "Care of Magical Creatures",
            ],
            axis=1,
        )
        # print(self.cleaned_dataset.head(15))
        self.cleaned_dataset.replace("", np.nan, inplace=True)
        self.cleaned_dataset.astype(float)

    def __init__(self, dataset, to_predict=[], means=[], stds=[]):
        self.means = means
        self.stds = stds
        self.normalized_dataset = []
        # print(type(self.normalized_dataset))
        self.brut_dataset = dataset.reset_index(drop=True)
        self.to_predict = pd.DataFrame(to_predict, columns=["House"])
        self.cleaned_dataset = []

    def normalize(self):
        for metric in self.cleaned_dataset:
            sum = 0.0
            self.normalized_dataset.append([])
            for elem in self.cleaned_dataset[metric]:
                if elem is not None:
                    sum += elem
            self.means.append(sum / len(self.cleaned_dataset[metric]))
            sum_std = 0
            for elem in self.cleaned_dataset[metric]:
                if elem is not None:
                    sum_std += (self.means[-1] - elem) * (self.means[-1] - elem)
            self.stds.append((sum_std / len(self.cleaned_dataset[metric])) ** 0.5)
            for elem in self.cleaned_dataset[metric]:
                if elem is None:
                    self.normalized_dataset[-1].append(0.0)
                else:
                    normalized_elem = (elem - self.means[-1]) / self.stds[-1]
                    self.normalized_dataset[-1].append(normalized_elem)

        self.normalized_dataset = pd.DataFrame(self.normalized_dataset).transpose()
        self.normalized_dataset.columns = [
            "Astronomy",
            "Herbology",
            "Divination",
            "Muggle",
            "Runes",
            "History",
            "Transfiguration",
            "Potion",
            "Charms",
            "Flying",
        ]

    def denormalize(self):
        dataset = []
        for i, metric in self.normalized_dataset:
            dataset.append([])
            for elem in dataset[metric]:
                dataset[-1].append(elem * self.stds[i] + self.means[i])
        return dataset

    def create_valid_set_file(self):
        output = pd.concat([self.to_predict_val, self.valid_set], axis=1)
        # print("voici les types : ")
        # print(type(self.valid_set))
        # print(type(self.to_predict_val))

        output.to_csv("data/dataset_valid.csv", index=False)

    def split_train_valid(self, validation_ratio=0.2):
        self.valid_set = self.normalized_dataset.sample(
            frac=validation_ratio, random_state=42
        )

        self.train_set = self.normalized_dataset.drop(self.valid_set.index)
        self.to_predict_val = self.to_predict.iloc[self.valid_set.index]
        self.to_predict_train = self.to_predict.drop(self.valid_set.index)
        self.create_valid_set_file()

    def split_to_predict(
        self, houses=["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    ):
        self.to_predict_trains = {}
        self.to_predict_valids = {}
        for house in houses:
            self.to_predict_trains[house] = np.array(
                [elem == house for elem in self.to_predict_train["House"]]
            )
            self.to_predict_valids[house] = np.array(
                [elem == house for elem in self.to_predict_val["House"]]
            )

    def pipe(self):
        self.delete_columns()
        self.normalize()
        self.split_train_valid(validation_ratio=0.2)
        self.split_to_predict()
        return [
            np.array(self.train_set),
            self.to_predict_trains,
            np.array(self.valid_set),
            self.to_predict_valids,
        ]

    def normalize_test(self):
        for i, metric in enumerate(self.cleaned_dataset):
            self.normalized_dataset.append([])
            for elem in self.cleaned_dataset[metric]:
                if elem is None or np.isnan(elem):
                    normalized_elem = 0.0
                else:
                    normalized_elem = (elem - self.means[i]) / self.stds[i]
                self.normalized_dataset[i].append(normalized_elem)

        self.normalized_dataset = pd.DataFrame(self.normalized_dataset).transpose()
        self.normalized_dataset.columns = [
            "Astronomy",
            "Herbology",
            "Divination",
            "Muggle",
            "Runes",
            "History",
            "Transfiguration",
            "Potion",
            "Charms",
            "Flying",
        ]

    def pipe_test(self):
        self.delete_columns()
        self.normalize_test()
        return np.array(self.normalized_dataset)
