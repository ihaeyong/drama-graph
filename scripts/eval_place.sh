#!/bin/bash
export PYTHONPATH=/test/
export PYTHONPATH=./test/HSE/
# support only 1 gpu.
CUDA_VISIBLE_DEVICES=0 python3 ../models/eval_place.py -model='9_lstm_load2'
