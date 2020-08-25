#!/usr/bin/env bash

# Preprocess video dataset before training sound event detection model
export PYTHONPATH=/root/workspace/drama-graph
export PYTHONIOENCODING=utf-8

#evaluate sound event detection model
python sound_event_detection/src/evaluate.py