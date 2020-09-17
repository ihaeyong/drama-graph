#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/mnt/hdd/kkddhh386/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1,$2,$3,$4

python models/train_person.py -model voc_person_sgd
