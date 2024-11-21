import matplotlib.pyplot as plt
import pandas as pd
from classes.loader import Loader
from classes.plotter import Plotter
import argparse
import seaborn as sns


class Histogram(Plotter):
    def __init__(self, pd_data: pd.DataFrame):
        super().__init__(pd_data)

    def wich_course_has_homogeneous_dist(self):
        sns.histplot(
            data=self.pd_data,
            x="Flying",
            hue="House",
            multiple="stack",
            palette=self.colormap,
        )
        plt.xlabel("Flying class notes")
        plt.ylabel("Number of students")
        plt.title(
            "Which Hogwarts course has a homogeneous score distribution between all four houses?"
        )
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plots the histogram of the course with most homogeneous score distribution"
    )
    parser.add_argument(
        "dataset_filepath", type=str, help="path to the dataset csv file."
    )
    loader = Loader(parser.parse_args().dataset_filepath)
    loader.pd_data["House"] = loader.houses

    hist = Histogram(loader.pd_data)
    hist.wich_course_has_homogeneous_dist()


if __name__ == "__main__":
    main()
