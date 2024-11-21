import argparse
import sys
from classes.loader import Loader


class Describe:
    def __init__(self):
        self.stats = [
            "Count",
            "Mean",
            "Std",
            "Min",
            "25%",
            "50%",
            "75%",
            "Max",
        ]

    def max(self, metric):
        max = metric[0]
        for value in metric:
            if value > max:
                max = value
        return max

    def min(self, metric):
        min = metric[0]
        for value in metric:
            if value < min:
                min = value
        return min

    def mean(self, metric):
        mean = 0
        for value in metric:
            mean += value
        return mean / len(metric)

    def find_percentile(self, metric, percentile):
        sorted_metric = sorted(metric)
        size = len(sorted_metric)

        mid = (size - 1) * (percentile / 100)
        if mid.is_integer():
            return sorted_metric[int(mid)]

        return (mid % 1.0) * sorted_metric[int(mid)] + (1 - mid % 1.0) * sorted_metric[
            int(mid) + 1
        ]

    def std(self, metric, mean):
        std_squared = [(x - mean) ** 2 for x in metric]
        sum = 0
        for elem in std_squared:
            sum += elem
        variance = sum / len(metric)

        if variance < 0:
            raise ValueError("std : error usage")

        return variance**0.5

    def describe(self):
        self.describe = []
        exclude = set({"Index", "Best Hand", "First Name", "Last Name", "Birthday"})
        for metric in self.data:
            if metric in exclude:
                continue
            metric_stats = []
            metric_pd_serie = self.data[metric].dropna()
            metric_stats.append(float(len(metric_pd_serie)))
            mean = self.mean(metric_pd_serie)
            metric_stats.append(mean)
            metric_stats.append(self.std(metric_pd_serie, mean))
            metric_stats.append(self.min(metric_pd_serie))
            metric_stats.append(self.find_percentile(metric_pd_serie, 25))
            metric_stats.append(self.find_percentile(metric_pd_serie, 50))
            metric_stats.append(self.find_percentile(metric_pd_serie, 75))
            metric_stats.append(self.max(metric_pd_serie))
            self.describe.append(metric_stats)
        self.describe = [
            [self.describe[j][i] for j in range(len(self.describe))]
            for i in range(len(self.describe[0]))
        ]

        print(" " * 6, end="")
        for metric in self.data:
            if metric in exclude:
                continue
            print("|" + metric[0:15].ljust(15), end="")
        print("|")
        for stat, line in zip(self.stats, self.describe):
            print(stat.ljust(5), end=" ")
            for value in line:
                print("|", end="")
                print(("{:.5f}".format(round(value, 5))).rjust(15), end="")
            print("|")


def main():
    parser = argparse.ArgumentParser(
        description="Plots the scatter plot of the courses score distributions that are the most correlated"
    )
    parser.add_argument(
        "dataset_filepath", type=str, help="path to the dataset csv file."
    )

    loader = Loader(parser.parse_args().dataset_filepath)

    describe = Describe()
    describe.data = loader.pd_data
    describe.describe()


if __name__ == "__main__":
    main()
