# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=$HOME/workspaces/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1


python models/eval_model.py -model integration -display --episode 1
