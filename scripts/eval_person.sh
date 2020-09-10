# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python models/eval_person.py \
       -model voc_person_sgd \
       -display \
       --conf_threshold 0.35 \
       --nms_threshold 0.5
