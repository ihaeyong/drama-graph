export PYTHONPATH=/root/workspaces/vtt/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python models/train_emotion.py
