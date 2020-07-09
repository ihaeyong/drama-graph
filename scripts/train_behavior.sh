#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKING/Face_recog
export PYTHONIOENCODING=utf-8
# export CUDA_VISIBLE_DEVICES=$1

python -m models.train_behavior -model b_lr_0.0001-cross_entropy-wo_global-cond3d-num_classes_2-bug_fix_pid_3 -b_loss ce -b_lr 0.0001

