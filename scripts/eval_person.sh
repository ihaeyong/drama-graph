# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/mnt/hdd/darB/VTT_210325/drama-graph
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1

python models/eval_person.py \
       -model voc_person_group_1gpu_210402 \
       -display \
       --conf_threshold 0.35 \
       --nms_threshold 0.5
