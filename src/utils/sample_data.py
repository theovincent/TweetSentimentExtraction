from pathlib import Path
import random as rd
import numpy as np
import pandas as pd


class SampleData:
    def __init__(self, path_csv, nb_samples=100, percentage=None, save=False, path_save=None):
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
        self.nb_neu = nb_samples - self.nb_pos - self.nb_neg

        # Fill the sets
        self.sample_data = None
        self.fill_small_data()

        # Shuffle sample_data
        self.sample_data = self.sample_data.sample(frac=1)

        # Save
        if save:
            if path_save is None:
                path_save = Path("../../data/samples/sample_{}.csv".format(nb_samples))
            pass
            self.sample_data.to_csv(path_save, index=False)

    def fill_small_data(self):
        # Sample the big data set
        pos = self.data[self.data['sentiment'] == "positive"].sample(self.nb_pos)
        neu = self.data[self.data['sentiment'] == "neutral"].sample(self.nb_neu)
        neg = self.data[self.data['sentiment'] == "negative"].sample(self.nb_neg)

        self.sample_data = pd.concat([pos, neu, neg])

        # Change the value of the label
        mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.sample_data = self.sample_data.replace({'sentiment': mapping})


if __name__ == "__main__":
    CSV_NAME = "train.csv"
    CSV_PATH = Path("../../data/") / CSV_NAME
    PERCENTAGE = [0.333, 0.333]
    NB_SAMPLES = 10

    SAMPLE_DATA = SampleData(CSV_PATH, nb_samples=NB_SAMPLES, percentage=PERCENTAGE, save=True)

    print(SAMPLE_DATA.sample_data)
