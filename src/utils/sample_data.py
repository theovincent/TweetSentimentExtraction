from pathlib import Path
import random as rd
import numpy as np
import pandas as pd


class SmallData:
    def __init__(self, path_csv, nb_samples=10, percentage=None, save=False, path_save=None):
        """
        percentage = [positive, negative]
        The percentage of neutral is computed as follow : 1 - positive - negative
        """
        # Get the arguments
        if percentage is None:
            percentage = [0.3, 0.3]
        self.path = path_csv
        self.percentage = percentage
        self.nb_samples = nb_samples

        # Get the data set
        self.data = pd.read_csv(path_csv)

        # Get the numbers of samples in each class
        self.nb_pos = int(round(nb_samples * self.percentage[0]))
        self.nb_neg = int(round(nb_samples * self.percentage[1]))
        self.nb_neu = int(round(nb_samples * (1 - np.sum(self.percentage))))

        # Fill the sets
        self.small_data = None
        self.fill_small_data()

        # Shuffle small_data
        self.small_data = self.small_data.sample(frac=1)

        # Save
        if save:
            if path_save is None:
                path_save = Path("../../data/samples/sample_{}.csv".format(nb_samples))
            pass
            self.small_data.to_csv(path_save, index=False)

    def fill_small_data(self):
        # Sample the big data set
        pos = self.data[self.data['sentiment'] == "positive"].sample(self.nb_pos)
        neu = self.data[self.data['sentiment'] == "neutral"].sample(self.nb_neu)
        neg = self.data[self.data['sentiment'] == "negative"].sample(self.nb_neg)

        self.small_data = pd.concat([pos, neu, neg])

        # Change the value of the label
        mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.small_data = self.small_data.replace({'sentiment': mapping})


if __name__ == "__main__":
    CSV_NAME = "train.csv"
    CSV_PATH = Path("../../data/") / CSV_NAME

    SMALL_DATA = SmallData(CSV_PATH, save=True)

    print(SMALL_DATA.small_data)

