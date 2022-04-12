"""Class to run commands on server"""

import subprocess
from typing import List, Optional

from scripts import config


class SSHConnection:
    """To establish a ssh connection and download files or execute a command from a server"""

    def __init__(self, ssh_host: str, ssh_user: str, ssh_password_path: str):
        """To init the scp connections"""
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_password_path = ssh_password_path

    def download(self, local_path: str, remote_path: str) -> None:
        """To download files on computer"""
        subprocess.check_call(
            [
                config.SSH_BASH_SCRIPT_DOWNLOAD,
                self.ssh_host,
                self.ssh_user,
                self.ssh_password_path,
                local_path,
                remote_path,
            ]
        )

    def upload(self, local_path: str, remote_path: str) -> None:
        """To upload files on server"""
        subprocess.check_call(
            [
                config.SSH_BASH_SCRIPT_UPLOAD,
                self.ssh_host,
                self.ssh_user,
                self.ssh_password_path,
                local_path,
                remote_path,
            ]
        )

    def run_python_script(
        self, script_path: str, args: Optional[List[str]] = None
    ) -> None:
        """To run a python script on server"""
        subprocess.check_call(
            [
                config.SSH_BASH_SCRIPT_RUN,
                self.ssh_host,
                self.ssh_user,
                self.ssh_password_path,
                f"python {script_path} {' '.join(args) if args is not None else ''}",
            ]
        )
