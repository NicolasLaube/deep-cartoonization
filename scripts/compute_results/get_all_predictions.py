"""To generate predictions from all the models"""

import os
import subprocess

import pandas as pd

from scripts import config
from scripts.tools.type_dict.csv_to_bash import (
    architecture_dict,
    crop_mode_dict,
    ratio_filter_dict,
)

if __name__ == "__main__":
    FOR_EPOCHS = [5, 10, 20, 30, 40, 50, 60, "max"]
    BATCH_SIZE = 10

    df_remote_runs = pd.read_csv(config.REMOTE_PARAMS_PATH)
    subprocess.run(["cd", ".."])  # pylint: disable=subprocess-run-check

    for _, row in df_remote_runs.iterrows():
        if row["epochs_trained_nb"] == 0:
            continue
        pictures_path = os.path.join(config.LOGS_FOLDER, row["run_id"], "pictures")
        if not os.path.exists(pictures_path):
            os.mkdir(pictures_path)

        for epoch_raw in FOR_EPOCHS:
            epoch = min(
                row["epochs_trained_nb"] if epoch_raw == "max" else epoch_raw,
                int(row["epochs_trained_nb"]),
            )
            epoch_pictures_path = os.path.join(pictures_path, f"epoch_{epoch}")
            if not os.path.exists(epoch_pictures_path):
                print(
                    f"\nGenerating pictures for model {row['run_id']} with epoch {epoch}"  # pylint: disable=line-too-long
                )
                gen_path = os.path.join(
                    config.WEIGHTS_FOLDER,
                    row["run_id"],
                    f"trained_gen_{epoch}.pkl",
                )
                subprocess.run(  # pylint: disable=subprocess-run-check
                    [
                        "python",
                        "-m",
                        "src.run_predict",
                        "--architecture",
                        architecture_dict[row["cartoon_gan_architecture"]],
                        "--gen-path",
                        gen_path,
                        "--nb-images",
                        "50",
                        "--new-size",
                        str(row["training_input_size"]),
                        "--crop-mode",
                        crop_mode_dict[row["picture_dataset_crop_mode"]],
                        "--ratio-filter",
                        ratio_filter_dict[row["picture_dataset_ratio_filter_mode"]],
                        "--batch-size",
                        str(BATCH_SIZE),
                        "--save-path",
                        epoch_pictures_path,
                    ]
                )
