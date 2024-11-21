import matplotlib.pyplot as plt
import pandas as pd
from classes.loader import Loader
from classes.plotter import Plotter
import argparse
import seaborn as sns


class Scatter(Plotter):
    def __init__(self, pd_data: pd.DataFrame):
        super().__init__(pd_data)

    def wich_course_are_corelated(self):
        # Astronomy and defense
        x = self.pd_data["Astronomy"]
        y = self.pd_data["Defense Against the Dark Arts"]
        plt.scatter(x, y, color="blue", label="Points")
        plt.xlabel("Astronomy")
        plt.ylabel("Defense Against the Dark Arts")
        plt.title("What are the two features that are similar ?")
        plt.show()


def main():
    # parse input
    parser = argparse.ArgumentParser(
        description="Plots the scatter plot of the courses score distributions that are the most correlated"
    )
    parser.add_argument(
        "dataset_filepath", type=str, help="path to the dataset csv file."
    )

    # load csv
    loader = Loader(parser.parse_args().dataset_filepath)

    # plot scatter plot
    scatter = Scatter(loader.pd_data)
    scatter.wich_course_are_corelated()


if __name__ == "__main__":
    main()
