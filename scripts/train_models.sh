#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
# export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKIng/Face_recog
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

python train_main.py -model obj47_person_relation


