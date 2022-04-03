"""To copy logs and weights in a to_download folder"""

import json
import os
import shutil

TO_DOWLOAD_FOLDER_PATH = "./to_download"
TO_DOWNLOAD_JSON_PATH = "./to_download.json"


if __name__ == "__main__":
    with open(TO_DOWNLOAD_JSON_PATH, "r", encoding="utf-8") as file:
        to_download = json.load(file)
    shutil.rmtree(TO_DOWLOAD_FOLDER_PATH, ignore_errors=True)
    os.mkdir(TO_DOWLOAD_FOLDER_PATH)
    for file_dict in to_download:
        print(f"Copying {file_dict['from']}")
        try:
            if os.path.isfile(file_dict["from"]):
                shutil.copy(
                    file_dict["from"],
                    os.path.join(TO_DOWLOAD_FOLDER_PATH, file_dict["id"]),
                )
            else:
                shutil.copytree(
                    file_dict["from"],
                    os.path.join(TO_DOWLOAD_FOLDER_PATH, file_dict["id"]),
                )
        except Exception as e:  # pylint: disable=broad-except
            print(f"Something went wrong: {e}")
