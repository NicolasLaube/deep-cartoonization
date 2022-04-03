"""To copy the logs and weights from the server"""

import json
import os
import shutil
from typing import Dict, List

import pandas as pd

from scripts import config
from scripts.tools.ssh_connection import SSHConnection


def get_copy_logs_from_server(run_id: str) -> Dict[str, str]:
    """Get paths for all logs from server"""
    return {
        "id": run_id,
        "run_id": run_id,
        "from": f"./cartoongan/logs/{run_id}",
        "type": "logs",
    }


def get_copy_weights_from_server(
    run_id: str, epoch_for_weights: int, weight_type: str
) -> Dict[str, str]:
    """Get paths for weights from server"""
    return {
        "id": f"{run_id}_{weight_type}_{epoch_for_weights}",
        "run_id": run_id,
        "name": f"trained_{weight_type}_{epoch_for_weights}.pkl",
        "from": f"./cartoongan/weights/{run_id}/trained_{weight_type}_{epoch_for_weights}.pkl",
        "type": "weights",
    }


def copy_logs(from_path: str, run_id: str) -> None:
    """Copy logs"""
    to_path = os.path.join(config.LOGS_FOLDER, run_id)
    if os.path.exists(to_path):
        for name in os.listdir(to_path):
            shutil.move(os.path.join(to_path, name), os.path.join(from_path, name))
        os.rmdir(to_path)
    os.rename(from_path, to_path)


def copy_weights(from_path: str, run_id: str, to_file_name: str) -> None:
    """Copy server"""
    weight_path = os.path.join(config.WEIGHTS_FOLDER, run_id)
    to_path = os.path.join(weight_path, to_file_name)
    try:
        os.mkdir(weight_path)
    except:  # pylint: disable=bare-except
        pass
    os.rename(from_path, to_path)


if __name__ == "__main__":
    COPY_EPOCHS = [5, 10, 20, 30, 40, 50, 60, "max"]

    ssh_client = SSHConnection(
        ssh_host=config.SSH_CREDENTIALS.HOST,
        ssh_user=config.SSH_CREDENTIALS.USER,
        ssh_password_path=config.SSH_CREDENTIALS.PASSWORD_PATH,
    )

    ##################################
    ### Download params from server ###
    ###################################
    print("Downloading params from server")
    ssh_client.download(
        local_path=config.REMOTE_PARAMS_PATH,
        remote_path=config.DISTANT_PATHS.PARAMS_PATH,
    )
    df_remote_runs = pd.read_csv(config.REMOTE_PARAMS_PATH)

    #############################
    ### Get files to download ###
    #############################
    to_download: List[Dict[str, str]] = []
    for _, row in df_remote_runs.iterrows():
        if (
            (not os.path.exists(os.path.join(config.LOGS_FOLDER, row["run_id"])))
            or all(
                str(row["epochs_trained_nb"]) not in file_name
                for file_name in os.listdir(
                    os.path.join(config.WEIGHTS_FOLDER, row["run_id"])
                )
            )
            and row["epochs_trained_nb"] > 0
        ):
            to_download.append(get_copy_logs_from_server(row["run_id"]))
        for epoch in COPY_EPOCHS:
            epoch_to_retrieve = (int)(
                row["epochs_trained_nb"]
                if epoch == "max"
                else min(epoch, row["epochs_trained_nb"])
            )
            if (
                not os.path.exists(
                    os.path.join(
                        config.WEIGHTS_FOLDER,
                        row["run_id"],
                        f"trained_gen_{epoch_to_retrieve}.pkl",
                    )
                )
                and epoch_to_retrieve > 0
            ):
                to_download.append(
                    get_copy_weights_from_server(
                        row["run_id"], epoch_to_retrieve, weight_type="gen"
                    )
                )
            if (
                not os.path.exists(
                    os.path.join(
                        config.WEIGHTS_FOLDER,
                        row["run_id"],
                        f"trained_disc_{epoch_to_retrieve}.pkl",
                    )
                )
                and epoch_to_retrieve > 0
            ):
                to_download.append(
                    get_copy_weights_from_server(
                        row["run_id"], epoch_to_retrieve, weight_type="disc"
                    )
                )
            if epoch_to_retrieve == row["epochs_trained_nb"]:
                break

    with open(config.TO_DOWNLOAD_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(to_download, file)

    #################################################
    ### Upload files to server and running script ###
    #################################################
    print("\nUploading json file to server")
    ssh_client.upload(
        config.TO_DOWNLOAD_JSON_PATH, config.DISTANT_PATHS.TO_DOWLOAD_JSON_PATH
    )

    print("\nRunning copy files on server")
    ssh_client.run_python_script(config.DISTANT_PATHS.COPY_FILES_SCRIPT_PATH)

    print("\nDownloading files from server")
    ssh_client.download(
        local_path=config.DOWNLOADED_TEMP_FOLDER,
        remote_path=config.DISTANT_PATHS.TO_DOWLOAD_FOLDER_PATH,
    )

    ###################################
    ### Put files in rights folders ###
    ###################################
    for dl_elt in to_download:
        download_path = os.path.join(config.DOWNLOADED_TEMP_FOLDER, dl_elt["id"])
        if dl_elt["type"] == "logs":
            print(f"\nCopying logs from run {dl_elt['run_id']}")
            copy_logs(download_path, dl_elt["run_id"])
        elif dl_elt["type"] == "weights":
            print(f"\nCopying weights {dl_elt['name']} for run {dl_elt['run_id']}")
            copy_weights(download_path, dl_elt["run_id"], dl_elt["name"])

    print("\nCleaning temp files")
    shutil.rmtree(config.DOWNLOADED_TEMP_FOLDER, ignore_errors=True)
