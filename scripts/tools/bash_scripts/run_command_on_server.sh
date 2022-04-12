#!/bin/bash

# Inputs: 1=ssh_host, 2=ssh_user, 3=ssh_password, 4=command
echo Running sshpass -f $3 ssh $2@$1 nohup $4
sshpass -f $3 ssh $2@$1 nohup $4
echo Done