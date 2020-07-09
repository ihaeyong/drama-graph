#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKING/Face_recog
export PYTHONIOENCODING=utf-8
# export CUDA_VISIBLE_DEVICES=$1

python -m models.train_behavior -model num_classes_2 -b_loss ce -b_lr 0.0001

