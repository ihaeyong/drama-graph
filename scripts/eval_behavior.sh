# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKING/Face_recog
export PYTHONIOENCODING=utf-8
# export CUDA_VISIBLE_DEVICES=$1


python -m models.eval_behavior -model b_lr_0.0001-cross_entropy-wo_global-cond3d-num_classes_2

