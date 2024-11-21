import numpy as np
import pandas as pd
import csv
# import numbers


class Loader:
    def __init__(self, file_name, drop_nan=True):
        self.data = []
        self.read_csv(file_name)
        self.drop_nan = drop_nan

    def read_csv(self, file_name):
        try:
            with open(file_name, newline="") as file:
                csv_reader = csv.reader(file)
                self.data = [line for line in csv_reader]
        except Exception as e:
            print(e)
            exit(1)

        self.houses = [student[1] for student in self.data]
        del self.houses[0]
        for student in self.data:
            del student[1]
        self.columns = self.data[0]
        self.data = self.data[1:]
        for i, student in enumerate(self.data):
            for j, valeur in enumerate(student):
                try:
                    student[j] = float(valeur)
                except ValueError:
                    student[j] = None

        self.np_data = np.array(
            np.where(self.data is None, np.nan, self.data), dtype=object
        )
        self.pd_data = pd.DataFrame(
            self.np_data,
            columns=self.columns,
        )

    def load_parameters(self, parameters_filepath):
        # parameters_df = "oui"
        try:
            parameters_df = pd.read_csv(parameters_filepath)
        except Exception as e:
            print(e)
            exit(1)
        means = np.array(parameters_df["Mean"])
        stds = np.array(parameters_df["Std"])
        houses = set(["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"])
        thetas = {}
        for col in parameters_df.columns:
            if col in houses:
                thetas[col] = np.array(parameters_df[col])
        return [means, stds, thetas]


def main():
    loader = Loader("./data/dataset_train.csv")
    print(loader.pd_data)


if __name__ == "__main__":
    main()
