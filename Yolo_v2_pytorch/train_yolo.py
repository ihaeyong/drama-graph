import os
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from .src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, PBeHavCLS
from .src.utils import *
from .src.loss import YoloLoss
from .src.yolo_net import Yolo
from .src.yolo_tunning import YoloD
import shutil
import visdom
import cv2
import pickle
import numpy as np
from lib.logger import Logger

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
    parser.add_argument("--num_epoches", type=int, default=50)
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
    parser.add_argument("--pre_trained_model_path", type=str,
                        default="Yolo_v2_pytorch/trained_models/only_params_trained_yolo_voc") # Pre-training path

    parser.add_argument("--saved_path", type=str,
                        default="./checkpoint") # saved training path
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="baseline")

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
num_behaviors = len(PBeHavCLS)

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
    else:
        torch.manual_seed(123)
    learning_rate_schedule = {"0": 1e-5, "5": 1e-4,
                              "80": 1e-5, "110": 1e-6}

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    train_loader = DataLoader(train_set, **training_params)

    # define pre-model
    pre_model = Yolo(num_persons).cuda()
    pre_model.load_state_dict(torch.load(opt.pre_trained_model_path),
                              strict=False)

    # define model
    model = YoloD(pre_model, num_persons, num_behaviors).cuda()

    nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)

    criterion = YoloLoss(num_persons, num_behaviors, model.anchors,
                         opt.reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5,
                                momentum=opt.momentum,
                                weight_decay=opt.decay)

    model.train()
    num_iter_per_epoch = len(train_loader)

    loss_step = 0

    for epoch in range(opt.num_epoches):
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]
        for iter, batch in enumerate(train_loader):

            image, info = batch

            # sort label info on fullrect
            image, label, behavior_label = SortFullRect(image, info)

            if np.array(label).size == 0 :
                print("iter:{}_person bboxs are empty".format(
                    iter, label))
                continue

            # image [b, 3, 448, 448]
            if torch.cuda.is_available():
                image = torch.cat(image).cuda()
            else:
                image = torch.cat(image)

            optimizer.zero_grad()

            # logits [b, 125, 14, 14]
            logits, behavior_logits = model(image)

            # losses for person detection
            loss, loss_coord, loss_conf, loss_cls, loss_behavior_cls = criterion(
                logits,behavior_logits,label, behavior_label)

            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, Iteration: {}/{}, lr:{}".format(
                epoch + 1, opt.num_epoches,iter + 1,
                num_iter_per_epoch, optimizer.param_groups[0]['lr']))
            #print("---- Person Detection ---- ")
            print("+loss:{:.2f}(coord:{:.2f},conf:{:.2f},cls:{:.2f})".format(
                loss, loss_coord, loss_conf, loss_cls))
            #print("---- Person Behavior ---- ")
            print("+cls_behavior:{:.2f}".format(loss_behavior_cls))
            print()

            loss_dict = {
                'total' : loss.item(),
                'coord' : loss_coord.item(),
                'conf' : loss_conf.item(),
                'cls' : loss_cls.item(),
                'cls_behavior' : loss_behavior_cls.item()
            }

            # Log scalar values
            for tag, value in loss_dict.items():
                logger.scalar_summary(tag, value, loss_step)

            loss_step = loss_step + 1

        print("SAVE MODEL")
        torch.save(model.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_{}.pth".format(opt.model))
        torch.save(model,
                   opt.saved_path + os.sep + "anotherMissOh_{}.pth".format(
                       opt.model))

if __name__ == "__main__":
    train(opt)
