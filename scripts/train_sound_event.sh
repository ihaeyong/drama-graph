#!/usr/bin/env bash

# Preprocess video dataset before training sound event detection model
export PYTHONPATH=/root/workspace/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

#train sound event detection model using train.csv and test.csv
python sound_event_detection/src/train.py