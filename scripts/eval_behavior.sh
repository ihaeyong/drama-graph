# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/mnt/hdd/darB/VTT_210325/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1


python models/eval_behavior.py -model voc_person_behavior_210402 -display
