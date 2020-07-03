# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKING/Face_recog
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1


python models/eval_behavior.py -model global_diff_reweight_focal_one_gamma3_conv1d_lowdim
