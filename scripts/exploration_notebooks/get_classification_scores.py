import json
import os

import pandas as pd
from tqdm import tqdm

from scripts import config
from src.evaluation.cartoon_classifier import CartoonClassifier

if __name__ == "__main__":
    classifier = CartoonClassifier()
classifier.load_model()

FOR_EPOCHS = [5, 10, 20, 30, 40, 50, 60, "max"]

df_remote_runs = pd.read_csv("logs/all_params_remote.csv")

if not os.path.exists(config.CLASSIFIER_SCORES_PATH):
    all_predictions = {}
else:
    with open(config.CLASSIFIER_SCORES_PATH, "r", encoding="utf-8") as f:
        all_predictions = json.load(f)

for _, row in df_remote_runs.iterrows():
    if row["epochs_trained_nb"] == 0:
        continue
    pictures_path = os.path.join(config.LOGS_FOLDER, row["run_id"], "pictures")
    if not os.path.exists(pictures_path):
        continue
    if not row["run_id"] in all_predictions:
        all_predictions[row["run_id"]] = {}
    print(f"\nProcessing {row['run_id']}")

    for epoch_raw in FOR_EPOCHS:
        print(f"Processing {epoch_raw}")
        epoch = min(
            row["epochs_trained_nb"] if epoch_raw == "max" else epoch_raw,
            int(row["epochs_trained_nb"]),
        )
        epoch_pictures_path = os.path.join(pictures_path, f"epoch_{epoch}")
        if not os.path.exists(epoch_pictures_path):
            continue
        if not str(epoch) in all_predictions[row["run_id"]]:
            all_predictions[row["run_id"]][str(epoch)] = []
            for picture_path in tqdm(os.listdir(epoch_pictures_path)):
                if picture_path.endswith("cartoon.png"):
                    predictions = classifier.predict_from_path(
                        os.path.join(epoch_pictures_path, picture_path)
                    )
                    all_predictions[row["run_id"]][epoch].append(
                        [float(x) for x in predictions]
                    )
            with open(config.CLASSIFIER_SCORES_PATH, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f)
        if epoch_raw == "max":
            break
