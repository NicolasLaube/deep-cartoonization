import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def create_train_test_frames():
    """
    Create and save a train and a test file for frames
    """
    df_frames = pd.read_csv(config.FRAMES_FILTERED_CSV, index_col=0)
    train, test = train_test_split(
        df_frames,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
        stratify=df_frames["movie"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(config.FRAMES_TRAIN_CSV)
    test.to_csv(config.FRAMES_TEST_CSV)


def create_train_test_images():
    """
    Create and save a train and a test file for images
    """
    df_images = pd.read_csv(config.IMAGES_FILTERED_CSV, index_col=0)
    train, test = train_test_split(
        df_images,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(config.IMAGES_TRAIN_CSV)
    test.to_csv(config.IMAGES_TEST_CSV)


if __name__ == "__main__":
    create_train_test_frames()
    create_train_test_images()
