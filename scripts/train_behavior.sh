#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKING/Face_recog
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python models/train_behavior.py -model spatial_focal_one_gamma1_none -b_loss ce_focal -f_gamma 1