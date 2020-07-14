# Train Object Detection Model (YOLO_V2)
export PYTHONPATH=/root/workspaces/vtt/VTT_TRACKING/Face_recog
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$1


python models/eval_behavior.py -model global_diff_subset_batch1_local_wfocal_output_1_noise_lr_schedule_with_grad
