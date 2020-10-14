#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=$HOME/workspace/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1,$2

python models/train_person.py -model voc_person_group_2gpu_init_none
