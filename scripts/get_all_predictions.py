"""To generate predictions from all the models"""

import os
import subprocess

import pandas as pd

from src import config

if __name__ == "__main__":
    df_remote_runs = pd.read_csv(config.REMOTE_PARAMS_PATH)
    subprocess.run(["cd", ".."])  # pylint: disable=subprocess-run-check

    for _, row in df_remote_runs.iterrows():
        pictures_path = os.path.join(config.LOGS_FOLDER, row["run_id"], "pictures")
        epoch_pictures_path = os.path.join(
            pictures_path, f"epoch_{row['epochs_trained_nb']}"
        )
        if not os.path.exists(pictures_path) or not os.path.exists(epoch_pictures_path):
            print(
                f"\nGenerating pictures for model {row['run_id']} with epoch {row['epochs_trained_nb']}"  # pylint: disable=line-too-long
            )
            if not os.path.exists(pictures_path):
                os.mkdir(pictures_path)
            gen_path = os.path.join(
                config.WEIGHTS_FOLDER,
                row["run_id"],
                f"trained_gen_{row['epochs_trained_nb']}.pkl",
            )
            architecture_dict = {
                "Style Gan": "Style",
                "UNet GAN": "Unet",
                "Fixed GAN": "Fixed",
                "Modular GAN": "Modular",
                "Anime GAN": "Anime",
            }
            crop_mode_dict = {
                "RESIZE": "Resize",
                "CROP_CENTER": "Center",
                "CROP_RANDOM": "Random",
            }
            ratio_filter_dict = {
                "NO_FILTER": "None",
                "FILTER_PORTRAIT": "Portrait",
                "FILTER_LANDSCAPE": "Landscape",
            }
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
                    "5",
                    "--save-path",
                    epoch_pictures_path,
                ]
            )
