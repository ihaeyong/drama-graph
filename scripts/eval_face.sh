#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python models/eval_face.py