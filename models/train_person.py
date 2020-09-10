import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, PBeHavCLS
from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.loss import YoloLoss
import shutil
import cv2
import pickle
import numpy as np
from lib.logger import Logger

from lib.person_model import person_model
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten
from lib.focal_loss import FocalLossWithOneHot, FocalLossWithOutOneHot, CELossWithOutOneHot
from lib.hyper_yolo import anchors

'''
----------------------------------------------
--------sgd learning on 4 gpus----------------
----------------------------------------------
01 epoch : 2.85 %
03 epoch : 4.73 %
09 epoch : 6.45 %
12 epoch : 5.49 %
20 epoch : 7.76 %
----------------------------------------------
'''

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
    parser.add_argument("--decay", type=float, default=0.0005)
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
                        default="./checkpoint/person") # saved training path
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="baseline")
    parser.add_argument("-lr", dest='lr', type=float, default=1e-5)
    parser.add_argument("-clip", dest='clip', type=float, default=10.0)
    parser.add_argument("-print_interval", dest='print_interval', type=int,
                        default=100)
    parser.add_argument("-b_loss", dest='b_loss', type=str, default='ce')
    parser.add_argument("-f_gamma", dest='f_gamma', type=float, default=1.0)
    parser.add_argument("-clip_grad", dest='clip_grad',action='store_true')

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

# logger path
logger_path = 'logs/{}'.format(opt.model)
if os.path.exists(logger_path):
    print('exist_{}'.format(logger_path))
else:
    os.makedirs(logger_path)
    print('mkdir_{}'.format(logger_path))
logger = Logger(logger_path)

def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(123)

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn,
                       "num_workers": 8}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    train_loader = DataLoader(train_set, **training_params)

    # define behavior-model
    model = person_model(num_persons, device)
    if False :
        # cause in person_model, loaded the voc pre-trained params
        trained_persons = opt.trained_model_path + os.sep + "{}".format(
            'anotherMissOh_only_params_person.pth')

        ckpt = torch.load(trained_persons)
        if optimistic_restore(model.detector, ckpt):
            print("loaded pre-trained detector sucessfully.")

    # initialization
    model.to(device)

    # get optim
    num_gpus = torch.cuda.device_count()
    # yolo detector and person
    fc_params = [p for n,p in model.named_parameters()
                 if n.startswith('person') \
                 or n.startswith('detector') \
                 and p.requires_grad]

    p_params = [{'params': fc_params, 'lr': opt.lr * num_gpus}]

    if False:
        p_optimizer = torch.optim.Adam(p_params, lr = opt.lr * num_gpus,
                                       weight_decay=opt.decay, amsgrad=True)
    else:
        p_optimizer = torch.optim.SGD(p_params, lr = opt.lr * num_gpus,
                                      momentum=opt.momentum, weight_decay=opt.decay)

    p_scheduler = ReduceLROnPlateau(p_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)

    # multi-gpus
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.train()

    criterion = YoloLoss(num_persons, anchors, device, opt.reduction)

    num_iter_per_epoch = len(train_loader)

    loss_step = 0
    for epoch in range(opt.num_epoches):
        p_loss_list = []
        for iter, batch in enumerate(train_loader):

            behavior_lr = iter % (1) == 0
            verbose=iter % (opt.print_interval*10) == 0
            image, info = batch

            # sort label info on fullrect
            image, label, behavior_label, obj_label, face_label, emo_label = SortFullRect(
                image, info, is_train=True)

            if np.array(label).size == 0 :
                print("iter:{}_person bboxs are empty".format(
                    iter, label))
                continue

            # image [b, 3, 448, 448]
            if torch.cuda.is_available():
                image = torch.cat(image).to(device)
            else:
                image = torch.cat(image)

            p_optimizer.zero_grad()

            # logits [b, 125, 14, 14]
            logits,_ = model(image)

            # losses for person detection
            loss, loss_coord, loss_conf, loss_cls = criterion(
                logits, label, device)

            loss.backward()
            p_optimizer.step()

            print("Model:{}".format(opt.model))
            print("Epoch: {}/{}, Iteration: {}/{}, lr:{:.9f}".format(
                epoch + 1, opt.num_epoches,iter + 1,
                num_iter_per_epoch, p_optimizer.param_groups[0]['lr']))
            print("+loss:{:.2f}(coord:{:.2f},conf:{:.2f},cls:{:.2f})".format(
                loss, loss_coord, loss_conf, loss_cls))
            print()

            loss_dict = {
                'total' : loss.item(),
                'coord' : loss_coord.item(),
                'conf' : loss_conf.item(),
                'cls' : loss_cls.item(),
            }

            if iter % 100 == 0:
                p_loss_list.append(loss_cls.item())

            # Log scalar values
            for tag, value in loss_dict.items():
                logger.scalar_summary(tag, value, loss_step)

            loss_step = loss_step + 1

        print("SAVE MODEL")
        if not os.path.exists(opt.saved_path):
            os.makedirs(opt.saved_path + os.sep + "{}".format('person'))
            print('mkdir_{}'.format(opt.saved_path))

        # learning rate schedular
        p_loss_avg = np.stack(p_loss_list).mean()

        p_scheduler.step(p_loss_avg)

        torch.save(model.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_{}.pth".format(opt.model))
        torch.save(model,
                   opt.saved_path + os.sep + "anotherMissOh_{}.pth".format(opt.model))

if __name__ == "__main__":
    train(opt)
