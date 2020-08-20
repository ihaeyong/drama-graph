# training params
load_ckpt  = (False,'')    # load previous ckpt with experiment code
im_size    = (448,448)     # sizeof input image
num_epoc   = int(1e+2)     # numof epoch
err_step   = int(50)       # numof steps per displaying error(loss)
val_step   = int(1e+5)     # numof steps per saving checkpoint
ckp_step   = int(1e+5)     # numof steps per saving network ckpt
bat_size   = int(4)        # sizeof batch
lr_start   = 1e-4          # learning rate param
w_decay    = 1e-5          # weight decay param
val_sets   = [7,8]         # episode indices to use for validation/test

# network params
emo_net_ch = 64

# dataset path
db_path   = '/home/datasets4/VTT_AMO/db/AnotherMissOh_images'
dict_path = '/home/datasets4/VTT_AMO/db/AnotherMissOh_Visual_emo.json'
# pretrained yolo weight path
pre_w_path = 'Yolo_v2_pytorch/trained_models/only_params_trained_yolo_voc'
# path to save ckpt
ckpt_path = 'face_emotion/ckpt'

