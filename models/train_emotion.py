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

from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten
from lib.focal_loss import FocalLossWithOneHot, FocalLossWithOutOneHot, CELossWithOutOneHot

from lib.emotion_model import emotion_model, crop_face_emotion, EmoCLS

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
    parser.add_argument("-emo_net_ch", dest='emo_net_ch',type=int, default=64)
    

    args = parser.parse_args()
    return args


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
num_emos = len(EmoCLS)

# logger path
logger_path = 'logs/{}'.format(opt.model)
if os.path.exists(logger_path):
    print('exist_{}'.format(logger_path))
else:
    os.makedirs(logger_path)
    print('mkdir_{}'.format(logger_path))

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
    model_emo = emotion_model(opt.emo_net_ch, num_persons, device)
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
            image, label, behavior_label, obj_label, face_label, emo_label = SortFullRect(image, info, is_train=True)

            if np.array(face_label).size == 0 :
                print("iter:{}_face bboxs are empty".format(
                    iter, label))
                continue
                
            # crop faces from img [b,3,h,w] -> [b,h,w,3]
            image = torch.cat(image)
            image_c = image.permute(0,2,3,1)
            
            face_crops, emo_gt = crop_face_emotion(image_c, face_label, emo_label, opt)
            
            if torch.cuda.is_available():
                face_crops = face_crops.cuda(device).contiguous()
                emo_gt = emo_gt.cuda(device)

            # emo_logits [b, 7]
            emo_logits = model_emo(face_crops)
            
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
