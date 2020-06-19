#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKIng/Face_recog
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python train_main.py -model layer3_full


