#!/usr/bin/env bash

# Preprocess video dataset before training sound event detection model
export PYTHONPATH=/root/workspace/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

#convert videos to wavs and preprocess that wavs
python sound_event_detection/src/convert.py
#extract sound features from given wavs
python sound_event_detection/src/extract_features.py