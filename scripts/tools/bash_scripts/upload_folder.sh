#!/bin/bash

# Inputs: 1=ssh_host, 2=ssh_user, 3=ssh_password, 4=local_path, 5=remote_path
echo Running sshpass -p $3 scp -r $2@$1:$5 $4
sshpass -f $3 scp -r $4 $2@$1:$5
echo Done