#!/usr/bin/env bash

# Preprocess video dataset before training sound event detection model
export PYTHONPATH=/root/workspace/drama-graph
export PYTHONIOENCODING=utf-8
#export CUDA_VISIBLE_DEVICES=$4

#inference sound event detection model, create prediction JSON and visualize the results
python sound_event_detection/src/inference.py