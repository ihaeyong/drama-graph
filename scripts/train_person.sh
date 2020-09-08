#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1,$2,$3,$4,$5,$6,$7

python models/train_person.py -model voc_person
