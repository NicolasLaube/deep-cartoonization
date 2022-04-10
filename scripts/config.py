import os

from src.config import *  # pylint: disable=wildcard-import,unused-wildcard-import

SCRIPTS_FOLDER = os.path.join(ROOT_FOLDER, "scripts")

TOOLS_FOLDER = os.path.join(SCRIPTS_FOLDER, "tools")
RESULT_SCRIPTS_FOLDER = os.path.join(SCRIPTS_FOLDER, "results")
SERVER_SCRIPTS_FOLDER = os.path.join(SCRIPTS_FOLDER, "server")

BASH_SCRIPTS_FOLDER = os.path.join(SCRIPTS_FOLDER, "tools", "bash_scripts")
SSH_BASH_SCRIPT_DOWNLOAD = os.path.join(BASH_SCRIPTS_FOLDER, "download_folder.sh")
SSH_BASH_SCRIPT_UPLOAD = os.path.join(BASH_SCRIPTS_FOLDER, "upload_folder.sh")
SSH_BASH_SCRIPT_RUN = os.path.join(BASH_SCRIPTS_FOLDER, "run_command_on_server.sh")

TO_DOWNLOAD_JSON_PATH = os.path.join(SERVER_SCRIPTS_FOLDER, "to_download.json")
DOWNLOADED_TEMP_FOLDER = os.path.join(SERVER_SCRIPTS_FOLDER, "downloaded_temp")

ADVERSARIAL_SCORES_PATH = os.path.join(RESULT_SCRIPTS_FOLDER, "adversarial_scores.json")

HISTOGRAMS_PATH = os.path.join(RESULT_SCRIPTS_FOLDER, "histograms.pkl")


class DISTANT_PATHS:  # pylint: disable=invalid-name
    """A class for all the distant paths"""

    BASE_PATH = "~"
    CARTOON_GAN_PATH = os.path.join(BASE_PATH, "cartoongan")
    LOGS_PATH = os.path.join(CARTOON_GAN_PATH, "logs")
    PARAMS_PATH = os.path.join(LOGS_PATH, "all_params.csv")
    TO_DOWLOAD_JSON_PATH = os.path.join(BASE_PATH, "to_download.json")
    TO_DOWLOAD_FOLDER_PATH = os.path.join(BASE_PATH, "to_download")
    COPY_FILES_SCRIPT_PATH = os.path.join(BASE_PATH, "copy_files_to_download.py")


class SSH_CREDENTIALS:  # pylint: disable=invalid-name
    """A class for all the ssh credentials"""

    HOST = "chome.metz.supelec.fr"
    USER = "gpu_stutz"
    PASSWORD_PATH = os.path.join(TOOLS_FOLDER, "password.txt")
