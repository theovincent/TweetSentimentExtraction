from pathlib import Path
import pandas as pd


def split(path_data, only_neutral=False):
    df_train = pd.read_csv(path_data)

    train_neutral = df_train[df_train.sentiment == 0].to_numpy()
    if only_neutral:
        train_pos = df_train[(df_train.sentiment == 1) | (df_train.sentiment == -1)].to_numpy()
        train_neg = None
    else:
        train_pos = df_train[df_train.sentiment == 1].to_numpy()
        train_neg = df_train[df_train.sentiment == -1].to_numpy()

    return train_pos, train_neutral, train_neg


if __name__ == '__main__':
    PATH_SAMPLE = Path("../../data/samples/sample_10_train.csv")
    print(split(PATH_SAMPLE, True)[0])
