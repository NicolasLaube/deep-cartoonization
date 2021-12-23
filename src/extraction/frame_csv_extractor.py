import os
import csv
from src import config


def create_frames_csv(folder_path: str, csv_path: str) -> None:
    """Creates a csv file to list all frames names a csv file."""
    frames_names = [
        frame
        for frame in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, frame))
    ]
    with open(csv_path, "w", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        for frame in frames_names:
            writer.writerow([frame])


def create_all_frames_csv():
    """Creates csv for each movie in the frames folder."""
    for movie in os.listdir(config.FRAMES_FOLDER):
        folder_path = os.path.join(config.FRAMES_FOLDER, movie)
        csv_path = os.path.join(config.CSV_FOLDER, movie + ".csv")
        create_frames_csv(folder_path, csv_path)


if __name__ == "__main__":
    create_all_frames_csv()
