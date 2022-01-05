import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def create_train_test_cartoons():
    """
    Create and save a train and a test file for cartoons
    """
    df_cartoons = pd.read_csv(config.CARTOONS_FILTERED_CSV, index_col=0)
    train, test = train_test_split(
        df_cartoons,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
        stratify=df_cartoons["movie"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(config.CARTOONS_TRAIN_CSV)
    test.to_csv(config.CARTOONS_TEST_CSV)


def create_train_test_pictures():
    """
    Create and save a train and a test file for pictures
    """
    df_pictures = pd.read_csv(config.PICTURES_FILTERED_CSV, index_col=0)
    train, test = train_test_split(
        df_pictures,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(config.PICTURES_TRAIN_CSV)
    test.to_csv(config.PICTURES_TEST_CSV)


if __name__ == "__main__":
    create_train_test_cartoons()
    create_train_test_pictures()
