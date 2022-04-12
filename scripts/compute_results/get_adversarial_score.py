import json
import os
import subprocess
from typing import Any, Dict

import numpy as np
import pandas as pd

from scripts import config
from scripts.tools.get_name import get_disc_name, get_gen_name
from scripts.tools.type_dict.csv_to_object import (
    architecture_dict,
    crop_mode_dict,
    ratio_filter_dict,
)
from src import dataset, models
from src.pipelines.cartoonizer import Cartoonizer
from src.pipelines.discriminator import Discriminator


def get_cartoonizer(model: Dict[str, Any]) -> Cartoonizer:
    """Get cartoonizer for a given model."""
    gen_path = os.path.join(
        config.WEIGHTS_FOLDER,
        model["run_id"],
        get_gen_name(model["epoch"]),
    )
    model_row = model["row"]
    return Cartoonizer(
        infering_parameters=models.InferingParams(batch_size=BATCH_SIZE),
        architecture=architecture_dict[model_row["cartoon_gan_architecture"]],
        architecture_params=models.ArchitectureParamsNULL(),
        pictures_dataset_parameters=dataset.PicturesDatasetParameters(
            new_size=(
                model_row["training_input_size"],
                model_row["training_input_size"],
            ),
            crop_mode=crop_mode_dict[model_row["picture_dataset_crop_mode"]],
            ratio_filter_mode=ratio_filter_dict[
                model_row["picture_dataset_ratio_filter_mode"]
            ],
            nb_images=NB_IMAGES,
        ),
        gen_path=gen_path,
    )


def get_discriminator(model: Dict[str, Any]) -> Discriminator:
    """Get discriminator for a given model."""
    disc_path = os.path.join(
        config.WEIGHTS_FOLDER,
        model["run_id"],
        get_disc_name(model["epoch"]),
    )
    model_row = model["row"]
    return Discriminator(
        infering_parameters=models.InferingParams(batch_size=BATCH_SIZE),
        architecture=architecture_dict[model_row["cartoon_gan_architecture"]],
        architecture_params=models.ArchitectureParamsNULL(),
        cartoons_dataset_parameters=dataset.CartoonsDatasetParameters(
            new_size=(
                model_row["training_input_size"],
                model_row["training_input_size"],
            ),
            crop_mode=crop_mode_dict[model_row["picture_dataset_crop_mode"]],
            ratio_filter_mode=ratio_filter_dict[
                model_row["picture_dataset_ratio_filter_mode"]
            ],
            nb_images=NB_IMAGES,
        ),
        disc_path=disc_path,
    )


if __name__ == "__main__":
    FOR_EPOCHS = [5, 10, 20, 30, 40, 50, 60, "max"]
    NB_IMAGES = 20
    BATCH_SIZE = 5

    df_remote_runs = pd.read_csv(config.REMOTE_PARAMS_PATH)
    subprocess.run(["cd", ".."])  # pylint: disable=subprocess-run-check

    print("Gathering models")
    all_models = []
    for _, row in df_remote_runs.iterrows():
        for epoch_raw in FOR_EPOCHS:
            epoch = min(  # type: ignore
                int(row["epochs_trained_nb"]) if epoch_raw == "max" else epoch_raw,
                int(row["epochs_trained_nb"]),
            )
            if epoch != 0:
                all_models.append(
                    {
                        "id": f"{row['run_id']}_{epoch}",
                        "run_id": row["run_id"],
                        "epoch": epoch,
                        "row": row,
                    }
                )
            if epoch == row["epochs_trained_nb"]:
                break
    print(f"{len(all_models)} models gathered")

    print("\nGenerating scores")
    try:
        with open(config.ADVERSARIAL_SCORES_PATH, "r", encoding="utf-8") as f:
            tab_results = json.load(f)
    except FileNotFoundError:
        tab_results = {}
    for model_gen in all_models:
        if model_gen["id"] in tab_results and all(
            model_disc["id"] in tab_results[model_gen["id"]]
            for model_disc in all_models
        ):
            continue
        print(f"\nCartoonizing for model {model_gen['id']}")
        cartoonizer = get_cartoonizer(model_gen)
        cartoons = np.array(
            [
                x["cartoon"]
                for x in cartoonizer.get_cartoonized_images(nb_images=NB_IMAGES)
            ]
        )
        if not model_gen["id"] in tab_results:
            tab_results[model_gen["id"]] = {}
        for model_disc in all_models:
            if model_disc["id"] in tab_results[model_gen["id"]]:
                continue
            print(f"Discriminating for model {model_disc['id']}")
            discriminator = get_discriminator(model_disc)
            score = np.mean(
                [x["result"] for x in discriminator.discriminate_images(cartoons)]  # type: ignore
            )
            tab_results[model_gen["id"]][model_disc["id"]] = score

        with open(config.ADVERSARIAL_SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(tab_results, f)
