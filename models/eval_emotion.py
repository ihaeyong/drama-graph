import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, PBeHavCLS, FaceCLS
from Yolo_v2_pytorch.src.utils import *
import shutil
import cv2
import pickle
import numpy as np
from lib.logger import Logger

from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten

from lib.emotion import emotion_model

def get_args():
    parser = argparse.ArgumentParser(
        "You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size",
                        type=int, default=448,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The number of images per batch")
    parser.add_argument("--conf_threshold",
                        type=float, default=0.35)
    parser.add_argument("--nms_threshold",
                        type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type",
                        type=str, choices=["model", "params"],
                        default="model")
    parser.add_argument("--data_path_test",
                        type=str,
                        default="./Yolo_v2_pytorch/missoh_test/",
                        help="the root folder of dataset")

    parser.add_argument("--saved_path", type=str,
                        default="./checkpoint/emotion") # saved training path

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")

    parser.add_argument("-model", dest='model', type=str, default="emotion")
    parser.add_argument("-display", dest='display', action='store_true')
    parser.add_argument("-emo_net_ch", dest='emo_net_ch',type=int, default=64)
    args = parser.parse_args()
    return args

# emotion text to index
def emo_char_idx(emo):
    # 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    if emo == 'angry' or emo == 'anger':
        return 0
    elif emo == 'disgust':
        return 1
    elif emo == 'fear':
        return 2
    elif emo == 'happy' or emo == 'happiness':
        return 3
    elif emo == 'sad' or emo == 'sadness':
        return 4
    elif emo == 'surprise':
        return 5
    elif emo == 'neutral':
        return 6
    else:
        print('error, '+emo)
        return 6

# img crop function
def crop_img(I, x, y, w, h, center=False, mfill=False):
    im_h = I.shape[0]
    im_w = I.shape[1]

    if center:
        w0 = w // 2;    w1 = w - w0    # w = w0+w1
        h0 = h // 2;    h1 = h - h0    # h = h0+h1

        x_min = x - w0;    x_max = x+w1-1;
        y_min = y - h0;    y_max = y+h1-1;
    else:
        x_min = x;    x_max = x+w-1;
        y_min = y;    y_max = y+h-1;

    pad_l = 0;    pad_r = 0;
    pad_u = 0;    pad_d = 0;

    # bounds
    if x_min < 0:          pad_l = -x_min;            x_min = 0;
    if x_max > im_w-1:     pad_r = x_max-(im_w-1);    x_max = im_w-1;
    if y_min < 0:          pad_u = -y_min;            y_min = 0;
    if y_max > im_h-1:     pad_d = y_max-(im_h-1);    y_max = im_h-1;

    # crop & append
    J = I[y_min:y_max+1, x_min:x_max+1, :]

    # 0 size errors
    if J.shape[0] == 0 or J.shape[1] == 0:
        return np.zeros([h,w,3])

    if mfill:
        rsel = np.linspace(0, J.shape[0], 8, endpoint=False, dtype=int)
        csel = np.linspace(0, J.shape[1], 8, endpoint=False, dtype=int)
        fill = np.mean(J[rsel][:,csel], axis=(0,1))
    else:
        fill = (0,0,0)
    J = cv2.copyMakeBorder(J, pad_u,pad_d,pad_l,pad_r, cv2.BORDER_CONSTANT, value=fill)
    return J

# measure accuracy (output_logits vs labels)
def measure_acc(out, lbl):
    num_l = float(out.shape[0])
    out_s = F.softmax(out, dim=1)
    out_e = out_s.max(dim=1)[1]
    num_c = (out_e == lbl).long()
    accu  = num_c.sum() / num_l 
    return accu.detach().cpu().numpy(), num_c.cpu().numpy()

# get args.
opt = get_args()
print(opt)

# splits the episodes int train, val, test
train, val, test = Splits(num_episodes=18)

# load datasets
train_set = AnotherMissOh(train, opt.img_path, opt.json_path, False)
val_set = AnotherMissOh(val, opt.img_path, opt.json_path, False)
test_set = AnotherMissOh(test, opt.img_path, opt.json_path, False)

num_persons = len(PersonCLS)
num_faces = len(FaceCLS)
EmoCLS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sandess', 'Surprise', 'Neutral']
num_emos = len(EmoCLS)

# model path
model_path = "{}/anotherMissOh_{}.pth".format(
    opt.saved_path,opt.model)

def test(opt):
    global colors
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        device = torch.cuda.current_device()
    else:
        torch.manual_seed(123)
    # set test loader params
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    # set test loader
    test_loader = DataLoader(test_set, **test_params)

    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model1 = torch.load(model_path)
            print("loaded with gpu {}".format(model_path))
        else:
            model1 = emotion_model(yolo_w_path=None, emo_net_ch=opt.emo_net_ch)
            model1.load_state_dict(torch.load(model_path))
            print("loaded with cpu {}".format(model_path))
        model1.cuda(device)

    model1.eval()
    width, height = (1024, 768)
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height

    # emotion accuracy
    emo_accu = []
    
    # load test clips
    for iter, batch in enumerate(test_loader):
        image, info = batch

        # sort label info on fullrect
        image, label, behavior_label, object_label, face_label, frame_id = SortFullRect(
            image, info, is_train=False)
        
        if np.array(face_label).size == 0:
            continue
            
        # crop faces from img [b,3,h,w] -> [b,h,w,3]
        image = torch.cat(image)
        image_c = image.permute(0,2,3,1)
        face_crops = list()

        for i,img in enumerate(image_c):
            for j in range(len(face_label[i])):
                # face corrdinates
                fl = face_label[i][j]
                face_x, face_y, face_w, face_h = int(fl[0]), int(fl[1]), int(fl[2])-int(fl[0]), int(fl[3])-int(fl[1])
                # crop face region, resize
                img_crop = torch.Tensor( cv2.resize(crop_img(img.numpy(), int(face_x), int(face_y), int(face_w), int(face_h)).copy(), (opt.image_size, opt.image_size)) )
                # store
                face_crops.append(img_crop)

        face_crops = torch.stack(face_crops).permute(0,3,1,2) # [f,h,w,3]->[f,3,h,w]

        if torch.cuda.is_available():
            face_crops = face_crops.cuda(device)

        # emo_logits [b, 7]
        emo_logits = model_emo(face_crops)
        
        # emo_logits [b, 7]
        emo_logits = model_emo(face_crops)
        # emo_gt labels
        emo_gt = []
        for i in range(len(info[0])):
            info_emo_i = info[0][i]['persons']['emotion']
            for j in range(len(info_emo_i)):
                emo_text = info_emo_i[j]
                emo_idx = emo_char_idx(emo_text.lower())
                emo_gt.append(emo_idx)
        emo_gt = torch.Tensor(emo_gt).long().cuda(device)
        
        # check accuracy for batch
        acc_batch, corr_batch = measure_acc(emo_logits, emo_gt)
        emo_accu.append(corr_batch)

        except Exception as ex:
            print(ex)
            continue
    
    print( 'accuracy:'+ str(np.mean(emo_accu)) )


if __name__ == "__main__":
    test(opt)
