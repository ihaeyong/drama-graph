# Train Object Detection Model (YOLO_V2)
# export PYTHONPATH=/root/workspaces/vtt/drama-graph
# export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=7

python models/eval_object.py \
       -model object \
       -display \
       --conf_threshold 0.35 \
       --nms_threshold 0.5
