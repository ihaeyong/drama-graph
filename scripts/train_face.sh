#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKInG/Face_recog
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python train_main.py -model face


