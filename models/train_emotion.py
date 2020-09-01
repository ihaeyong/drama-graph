import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, PBeHavCLS, FaceCLS
from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.loss import YoloLoss
import shutil
import cv2
import pickle
import numpy as np
from lib.logger import Logger

from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten
from lib.focal_loss import FocalLossWithOneHot, FocalLossWithOutOneHot, CELossWithOutOneHot

from lib.emotion import emotion_model

def get_args():
    parser = argparse.ArgumentParser(
        "You Only Look Once:Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int,
                        default=448,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The number of images per batch")

    # Training base Setting
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=1,
                        help="Number of epoches between testing phases")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--noobject_scale", type=float, default=0.5)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--coord_scale", type=float, default=5.0)
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter:minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter:number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")

    parser.add_argument("--pre_trained_model_type",
                        type=str, choices=["model", "params"],
                        default="model")
    parser.add_argument("--trained_model_path", type=str,
                        default="./checkpoint/detector") # Pre-training path

    parser.add_argument("--saved_path", type=str,
                        default="./checkpoint/emotion") # saved training path
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="emotion")
    parser.add_argument("-lr", dest='lr', type=float, default=1e-4)
    parser.add_argument("-clip", dest='clip', type=float, default=10.0)
    parser.add_argument("-print_interval", dest='print_interval', type=int,
                        default=100)
    parser.add_argument("-b_loss", dest='b_loss', type=str, default='ce')
    parser.add_argument("-f_gamma", dest='f_gamma', type=float, default=1.0)
    parser.add_argument("-clip_grad", dest='clip_grad',action='store_true')
    parser.add_argument("-yolo_w_path", dest='yolo_w_path',type=str, default='Yolo_v2_pytorch/trained_models/only_params_trained_yolo_voc')
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

# logger path
logger_path = 'logs/{}'.format(opt.model)
if os.path.exists(logger_path):
    print('exist_{}'.format(logger_path))
else:
    os.makedirs(logger_path)
    print('mkdir_{}'.format(logger_path))
logger = Logger(logger_path)

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        device = torch.cuda.current_device()
    else:
        torch.manual_seed(123)
    #p_learning_rate_schedule = {"0": opt.lr/10.0, "5": opt.lr/50.0}
    #b_learning_rate_schedule = {"0": opt.lr, "5": opt.lr/10.0, "10": opt.lr/100.0}

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    train_loader = DataLoader(train_set, **training_params)

    # define emotion model
    model_emo = emotion_model(yolo_w_path=opt.yolo_w_path, emo_net_ch=opt.emo_net_ch)
    model_emo.cuda(device)

    # get optim
    e_criterion = nn.CrossEntropyLoss()
    e_optimizer = torch.optim.Adam(model_emo.parameters(), lr=opt.lr, weight_decay=opt.decay, amsgrad=True)
    e_scheduler = ReduceLROnPlateau(e_optimizer, 'min', patience=3,
                                    factor=0.5, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)
    model_emo.train()
    num_iter_per_epoch = len(train_loader)

    loss_step = 0


    for epoch in range(opt.num_epoches):

        e_loss_list = []
        for iter, batch in enumerate(train_loader):

            verbose=iter % (opt.print_interval*10) == 0
            image, info = batch

            # sort label info on fullrect
            image, label, behavior_label, obj_label, face_label = SortFullRect(image, info, is_train=True)

            if np.array(face_label).size == 0 :
                print("iter:{}_face bboxs are empty".format(
                    iter, label))
                continue

            # crop faces from img [b,3,h,w] -> [b,h,w,3]
            imgae = torch.cat(image)
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
            # emo_gt labels
            emo_gt = []
            for i in range(len(info[0])):
                info_emo_i = info[0][i]['persons']['emotion']
                for j in range(len(info_emo_i)):
                    emo_text = info_emo_i[j]
                    emo_idx = emo_char_idx(emo_text.lower())
                    emo_gt.append(emo_idx)
            emo_gt = torch.Tensor(emo_gt).long().cuda(device)
            # loss
            e_optimizer.zero_grad()
            loss_emo = e_criterion(emo_logits, emo_gt)
            loss_emo.backward()
            e_optimizer.step()
                

            print("Model:{}".format(opt.model))
            print("Epoch: {}/{}, Iteration: {}/{}, lr:{:.9f}".format(
                epoch + 1, opt.num_epoches, iter + 1,
                num_iter_per_epoch, e_optimizer.param_groups[0]['lr']))
            print("+Emotion_loss:{:.2f}".format(loss_emo))
            print()

            loss_step = loss_step + 1

        print("SAVE MODEL")
        if not os.path.exists(opt.saved_path):
            os.makedirs(opt.saved_path + os.sep + "{}".format(opt.model))
            print('mkdir_{}'.format(opt.saved_path))

        torch.save(model_emo.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_{}.pth".format(
                       opt.model))
        torch.save(model_emo,
                   opt.saved_path + os.sep + "anotherMissOh_{}.pth".format(
                       opt.model))

        
if __name__ == "__main__":
    train(opt)
