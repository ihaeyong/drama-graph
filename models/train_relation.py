import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, ObjectCLS, P2ORelCLS
from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.loss import YoloLoss
import shutil
import cv2
import pickle
import numpy as np
from lib.logger import Logger

from lib.relation_model import relation_model
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten
from lib.focal_loss import FocalLossWithOneHot, FocalLossWithOutOneHot, CELossWithOutOneHot
import pdb

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
                        default="./checkpoint") # Pre-training path

    parser.add_argument("--saved_path", type=str,
                        default="./checkpoint") # saved training path
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="relation")
    parser.add_argument("-lr", dest='lr', type=float, default=1e-5)
    parser.add_argument("-clip", dest='clip', type=float, default=10.0)
    parser.add_argument("-print_interval", dest='print_interval', type=int,
                        default=100)
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
num_objects = len(ObjectCLS)
num_relations = len(P2ORelCLS)

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

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    train_loader = DataLoader(train_set, **training_params)

    # define behavior-model
    model = relation_model(num_persons, num_objects, num_relations, opt, device)
    trained_persons = opt.trained_model_path + os.sep + "{}".format(
        'anotherMissOh_only_params_person.pth')
    trained_objects = opt.trained_model_path + os.sep + "{}".format(
        'anotherMissOh_only_params_object.pth')

    ckpt_person = torch.load(trained_persons)
    ckpt_object = torch.load(trained_objects)

    if optimistic_restore(model.person_detector, ckpt_person):
        print("loaded pre-trained person detector sucessfully.")

    if optimistic_restore(model.object_detector, ckpt_object):
        print("loaded pre-trained object detector sucessfully.")
    model.cuda(device)

    # get optim
    # yolo detector and person
    fc_params = [p for n,p in model.named_parameters()
                 if n.startswith('detector') \
                 or n.startswith('person') \
                 or n.startswith('object')
                 and p.requires_grad]
    # behavior model
    non_fc_params = [p for n,p in model.named_parameters()
                     if not n.startswith('detector') \
                     or not n.startswith('person')\
                     or not n.startswith('object')
                     and p.requires_grad]

    p_params = [{'params': fc_params, 'lr': opt.lr}]
    b_params = [{'params': non_fc_params, 'lr': opt.lr * 10.0}]

    p_optimizer = torch.optim.SGD(p_params, lr = opt.lr,
                                  momentum=opt.momentum,
                                  weight_decay=opt.decay)
    b_optimizer = torch.optim.SGD(b_params, lr = opt.lr * 10.0,
                                  momentum=opt.momentum,
                                  weight_decay=opt.decay)

    p_scheduler = ReduceLROnPlateau(p_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)
    b_scheduler = ReduceLROnPlateau(b_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)

    model.train()
    num_iter_per_epoch = len(train_loader)

    loss_step = 0
    criterion = nn.CrossEntropyLoss()


    for epoch in range(opt.num_epoches):
        r_logit_list = []
        r_label_list = []
        r_loss_list = []
        for iter, batch in enumerate(train_loader):

            image, info = batch

            # sort label info on fullrect
            image, label, behavior_label, obj_label, face_label, emo_label = SortFullRect(
                image, info, is_train=True)

            if np.array(label).size == 0 :
                print("iter:{}_person bboxs are empty".format(
                    iter, label))
                continue

            if np.array(obj_label).size == 0 : 
                print("iter:{}_object bboxes are empty".format(
                    iter, obj_label))
                continue

            # image [b, 3, 448, 448]
            if torch.cuda.is_available():
                image = torch.cat(image).cuda(device)
            else:
                image = torch.cat(image)
                
            p_optimizer.zero_grad()
            b_optimizer.zero_grad()

            # logits [b, 125, 14, 14]
            r_logits, r_labels = model(image, label, obj_label)


            # loss for relation

            loss = 0.0
            for idx, rr_logits in enumerate(r_logits):
                rr_logit_list = []
                rr_label_list = []
                rr_labels = r_labels[idx]
                if len(rr_logits) == 0:
                    continue
                rr_logits = torch.stack(rr_logits).squeeze(1)

                rr_labels = np.array(flatten(rr_labels))
                #b_labels = np.stack(b_labels)


                rr_labels = Variable(
                    torch.LongTensor(rr_labels).cuda(device),
                    requires_grad=False)
                print('relation_label:{}'.format(rr_labels))

                rr_label_list.append(rr_labels)
                rr_logit_list.append(rr_logits)


                rr_logits = torch.cat(rr_logit_list, 0)
                rr_labels = torch.cat(rr_label_list, 0)
                loss_relation = criterion(rr_logits, rr_labels)

                loss += loss_relation


            loss.backward()

            if opt.clip_grad:
                clip_grad_norm(
                    [(n, p) for n, p in model.named_parameters()
                     if p.grad is not None and not n.startswith('detector')],
                    max_norm=opt.clip, verbose=verbose, clip=True)
            b_optimizer.step()

            print("Model:{}".format(opt.model))
            print("Epoch: {}/{}, Iteration: {}/{}, lr:{:.9f}".format(
                epoch + 1, opt.num_epoches,iter + 1,
                num_iter_per_epoch, p_optimizer.param_groups[0]['lr']))
            print("+lr:{:.9f}, cls_relation:{:.2f}".format(
                b_optimizer.param_groups[0]['lr'],
                loss))
            print()

            loss_dict = {}

            loss_dict['cls_relation'] = loss.item()

            # Log scalar values
            for tag, value in loss_dict.items():
                logger.scalar_summary(tag, value, loss_step)

            loss_step = loss_step + 1
            r_loss_list.append(loss.item())

        print("SAVE MODEL")
        if not os.path.exists(opt.saved_path):
            os.makedirs(opt.saved_path + os.sep + "{}".format(opt_model))
            print('mkdir_{}'.format(opt.saved_path))

        # learning rate schedular
        r_loss_avg = np.stack(r_loss_list).mean()

        b_scheduler.step(r_loss_avg)

        torch.save(model.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_{}.pth".format(
                       opt.model))
        torch.save(model,
                   opt.saved_path + os.sep + "anotherMissOh_{}.pth".format(
                       opt.model))

if __name__ == "__main__":
    train(opt)
