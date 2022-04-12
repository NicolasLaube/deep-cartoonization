"""To copy the logs and weights from the server"""

import os
import shutil
import time

import pandas as pd

from scripts import config
from scripts.tools.ssh_connection import SSHConnection


def copy_logs_from_server(ssh_connection: SSHConnection, run_id: str) -> None:
    """Get all logs from server"""
    log_path = os.path.join(config.LOGS_FOLDER, run_id)
    if os.path.exists(log_path):
        temp_path = f"{log_path}_temp"
        ssh_connection.download(
            local_path=temp_path,
            remote_path=f"~/cartoongan/logs/{run_id}",
        )
        for name in os.listdir(log_path):
            shutil.move(os.path.join(log_path, name), os.path.join(temp_path, name))
        os.rmdir(log_path)
        os.rename(temp_path, log_path)
    else:
        ssh_connection.download(
            local_path=log_path,
            remote_path=f"~/cartoongan/logs/{run_id}",
        )
    if not os.path.exists(log_path):
        raise Exception("Didn't managed to download the files")


def copy_weights_from_server(
    ssh_connection: SSHConnection, run_id: str, epoch_for_weights: int
) -> None:
    """Get weights from server"""
    weight_path = os.path.join(config.WEIGHTS_FOLDER, run_id)
    try:
        os.mkdir(weight_path)
    except:  # pylint: disable=bare-except
        pass
    ssh_connection.download(
        local_path=os.path.join(weight_path, f"trained_gen_{epoch_for_weights}.pkl"),
        remote_path=f"~/cartoongan/weights/{run_id}/trained_gen_{epoch_for_weights}.pkl",
    )


if __name__ == "__main__":
    COPY_EPOCHS = [5, 10, 30, 30, 40, 50, 60, "max"]

    ssh_client = SSHConnection(
        ssh_host="chome.metz.supelec.fr",
        ssh_user="gpu_stutz",
        ssh_password_path=config.PASSWORD_PATH,  # type: ignore
    )
    ssh_client.download(
        local_path=config.REMOTE_PARAMS_PATH,
        remote_path="~/cartoongan/logs/all_params.csv",
    )
    time.sleep(20)
    df_remote_runs = pd.read_csv(config.REMOTE_PARAMS_PATH)

    for _, row in df_remote_runs.iterrows():
        if (not os.path.exists(os.path.join(config.LOGS_FOLDER, row["run_id"]))) or all(
            str(row["epochs_trained_nb"]) not in file_name
            for file_name in os.listdir(
                os.path.join(config.WEIGHTS_FOLDER, row["run_id"])
            )
        ):
            print(f"\nCopying run {row['run_id']}")
            copy_logs_from_server(ssh_client, row["run_id"])
            time.sleep(20)
        for epoch in COPY_EPOCHS:
            epoch_to_retrieve = (int)(
                row["epochs_trained_nb"]
                if epoch == "max"
                else min(epoch, row["epochs_trained_nb"])
            )
            if not os.path.exists(
                os.path.join(
                    config.WEIGHTS_FOLDER, row["run_id"], f"trained_gen_{epoch}.pkl"
                )
            ):
                print(
                    f"\nCopying weights for epoch {epoch_to_retrieve} of run {row['run_id']}"
                )
                copy_weights_from_server(ssh_client, row["run_id"], epoch_to_retrieve)
                time.sleep(20)
