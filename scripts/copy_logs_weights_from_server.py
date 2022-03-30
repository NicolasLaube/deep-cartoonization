"""To copy the logs and weights from the server"""

import os
import shutil
import subprocess
import time

import pandas as pd

from src import config


class SCPConnection:
    """To establish a scp connection and download files from a server"""

    def __init__(self, ssh_host: str, ssh_user: str, ssh_password_path: str):
        """To init the scp connections"""
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_password_path = ssh_password_path

    def download(self, local_path: str, remote_path: str) -> None:
        """To download files on computer"""
        subprocess.check_call(
            [
                config.SCP_BASH_SCRIPT,
                self.ssh_host,
                self.ssh_user,
                self.ssh_password_path,
                local_path,
                remote_path,
            ]
        )

    def upload(self, local_path: str, remote_path: str) -> None:
        """To upload files on server"""


def copy_logs_weights_from_server(
    scp_connection: SCPConnection, run_id: str, epoch_for_weights: int
) -> None:
    """Get all logs and best weights from server"""
    log_path = os.path.join(config.LOGS_FOLDER, run_id)
    if os.path.exists(log_path):
        temp_path = f"{log_path}_temp"
        scp_connection.download(
            local_path=temp_path,
            remote_path=f"~/cartoongan/logs/{run_id}",
        )
        for name in os.listdir(log_path):
            shutil.move(os.path.join(log_path, name), os.path.join(temp_path, name))
        os.rmdir(log_path)
        os.rename(temp_path, log_path)
    else:
        scp_connection.download(
            local_path=log_path,
            remote_path=f"~/cartoongan/logs/{run_id}",
        )
    if not os.path.exists(log_path):
        raise Exception("Didn't managed to download the files")
    weight_path = os.path.join(config.WEIGHTS_FOLDER, run_id)
    try:
        os.mkdir(weight_path)
    except:  # pylint: disable=bare-except
        pass
    scp_connection.download(
        local_path=os.path.join(weight_path, f"trained_gen_{epoch_for_weights}.pkl"),
        remote_path=f"~/cartoongan/weights/{run_id}/trained_gen_{epoch_for_weights}.pkl",
    )


if __name__ == "__main__":
    scp = SCPConnection(
        ssh_host="chome.metz.supelec.fr",
        ssh_user="gpu_stutz",
        ssh_password_path=config.PASSWORD_PATH,
    )
    scp.download(
        local_path=config.REMOTE_PARAMS_PATH,
        remote_path="~/cartoongan/logs/all_params.csv",
    )
    df_remote_runs = pd.read_csv(config.REMOTE_PARAMS_PATH)

    for _, row in df_remote_runs.iterrows():
        if (not os.path.exists(os.path.join(config.LOGS_FOLDER, row["run_id"]))) or all(
            str(row["epochs_trained_nb"]) not in file_name
            for file_name in os.listdir(
                os.path.join(config.WEIGHTS_FOLDER, row["run_id"])
            )
        ):
            print(
                f"\nCopying run {row['run_id']} with epoch {row['epochs_trained_nb']}"
            )
            copy_logs_weights_from_server(scp, row["run_id"], row["epochs_trained_nb"])
            time.sleep(10)
