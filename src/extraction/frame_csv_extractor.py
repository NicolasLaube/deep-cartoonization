import csv
import os

from src import config


def create_cartoons_csv(folder_path: str, csv_path: str) -> None:
    """Creates a csv file to list all cartoons names a csv file."""
    cartoons_names = [
        frame
        for frame in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, frame))
    ]
    with open(csv_path, "w", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        for cartoon in cartoons_names:
            writer.writerow([cartoon])


def create_all_cartoons_csv():
    """Creates csv for each movie in the cartoons folder."""
    for movie in os.listdir(config.CARTOONS_FOLDER):
        folder_path = os.path.join(config.CARTOONS_FOLDER, movie)
        csv_path = os.path.join(config.CARTOONS_CSV, movie + ".csv")
        create_cartoons_csv(folder_path, csv_path)


if __name__ == "__main__":
    create_all_cartoons_csv()
