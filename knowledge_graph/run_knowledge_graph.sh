#!/usr/bin/env bash

# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/drama-graph/knowledge_graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python main.py


