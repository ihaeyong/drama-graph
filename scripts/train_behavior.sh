#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python models/train_behavior.py -model voc_person_behavior -b_loss ce
