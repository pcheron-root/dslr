import matplotlib.pyplot as plt
from classes.loader import Loader
from classes.plotter import Plotter
import argparse
import seaborn as sns


class Pair_plot(Plotter):
    def draw_pair_plot(self, loader):
        df = loader.pd_data.drop(
            ["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1
        )
        df["Houses"] = loader.houses
        new_columns_names = [
            "Arit",
            "Astro",
            "Herbo",
            "Def",
            "Div",
            "Muggle",
            "Runes",
            "Hist",
            "Trans",
            "Pot",
            "Crea",
            "Charms",
            "Flying",
            "Houses",
        ]
        df.columns = new_columns_names
        pair_plot = sns.pairplot(
            df,
            hue="Houses",
            palette=self.colormap,
            height=1.5,
            aspect=0.8,
            plot_kws={"s": 10},
        )
        pair_plot._legend.remove()
        for ax in pair_plot.axes.flatten():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plots the scatter matrix of all courses score distributions"
    )
    parser.add_argument(
        "dataset_filepath", type=str, help="path to the dataset csv file."
    )

    loader = Loader(parser.parse_args().dataset_filepath)

    pairPlot = Pair_plot(loader)
    pairPlot.draw_pair_plot(loader)


if __name__ == "__main__":
    main()
