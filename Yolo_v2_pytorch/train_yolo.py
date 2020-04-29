import os
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from .src.anotherMissOh_dataset import MissOhDataset, MissOhDatasetTest
from .src.utils import *
from .src.loss import YoloLoss
from .src.yolo_net import Yolo
from .src.yolo_tunning import YoloD
import shutil
import visdom
import cv2
import pickle
import numpy as np

loss_data = {'X': [], 'Y': [], 'legend_U':['total', 'coord', 'conf', 'cls']}
visdom = visdom.Visdom(port=6005)

def get_args():
    parser = argparse.ArgumentParser(
        "You Only Look Once:Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int,
                        default=448,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=10,
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
    parser.add_argument("--pre_trained_model_path", type=str,
                        default="Yolo_v2_pytorch/trained_models/only_params_trained_yolo_voc") # Pre-training path

    parser.add_argument("--saved_path", type=str,
                        default="./checkpoint") # saved training path
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args

# not use this classes
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']

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

    training_set = MissOhDataset(opt.image_size)
    training_generator = DataLoader(training_set, **training_params)

    pre_model = Yolo(20).cuda()
    pre_model.load_state_dict(torch.load(opt.pre_trained_model_path),
                              strict=False)

    model = YoloD(pre_model, 1).cuda()

    nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)

    criterion = YoloLoss(1, model.anchors, opt.reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5,
                                momentum=opt.momentum, weight_decay=opt.decay)

    model.train()
    num_iter_per_epoch = len(training_generator)

    loss_step = 0

    for epoch in range(opt.num_epoches):
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]
        for iter, batch in enumerate(training_generator):
            image, label = batch
            if torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image

            optimizer.zero_grad()
            logits = model(image)
            loss, loss_coord, loss_conf, loss_cls = criterion(logits, label)
            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(epoch + 1, opt.num_epoches,
                                                                                                                     iter + 1, num_iter_per_epoch,
                                                                                                                     optimizer.param_groups[0]['lr'], loss,
                loss_coord,loss_conf,loss_cls))

            loss_dict = {
                'total' : loss.item(),
                'coord' : loss_coord.item(),
                'conf' : loss_conf.item(),
                'cls' : loss_cls.item()
            }

            visdom_loss(visdom, loss_step, loss_dict)
            loss_step = loss_step + 1

        print("SAVE MODEL")
        torch.save(model.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params.pth")
        torch.save(model,
                   opt.saved_path + os.sep + "anotherMissOh.pth")

def visdom_loss(visdom, loss_step, loss_dict):
    loss_data['X'].append(loss_step)
    loss_data['Y'].append([loss_dict[k] for k in loss_data['legend_U']])
    visdom.line(
        X=np.stack([np.array(loss_data['X'])] * len(loss_data['legend_U']), 1),
        Y=np.array(loss_data['Y']),
        win=30,
        opts=dict(xlabel='Step',
                  ylabel='Loss',
                  title='YOLO_V2',
                  legend=loss_data['legend_U']),
        update='append'
    )

if __name__ == "__main__":
    opt = get_args()
    train(opt)
