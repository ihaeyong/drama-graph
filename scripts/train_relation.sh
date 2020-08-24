#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
# export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKIng/Face_recog
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

python models/train_relation.py -model relation


