#!/bin/bash

# Define variables
REMOTE_USER="root"  # User on the pod
REMOTE_HOST="89.187.159.35"  # IP address of the pod
REMOTE_PATH="/workspace/runs/test_run_1/parameters/model_parameters.pth"  # Path on the pod where the file is located
LOCAL_USER="sid"  # User on your local machine
LOCAL_HOST="192.168.216.96"  # IP address of your local machine
LOCAL_PATH="/home/sid/chess_ai_final/runs/test_run_1/parameters/model_parameters.pth"  # Path on your local machine where the file will be transferred
PORT=40074  # Port number for SSH

# Transfer the file from the pod to your local machine
scp -P $PORT $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH $LOCAL_USER@$LOCAL_HOST:$LOCAL_PATH

# Shutdown the pod after transfer
ssh -p $PORT $REMOTE_USER@$REMOTE_HOST "shutdown now"
