import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, PBeHavCLS, FaceCLS, ObjectCLS, P2ORelCLS
from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.loss import YoloLoss
from Yolo_v2_pytorch.src.relation_loss import Relation_YoloLoss
import shutil
import cv2
import pickle
import numpy as np
import time
from lib.logger import Logger
from lib.place_model import place_model, resnet50, label_mapping, accuracy, AverageMeter, ProgressMeter
from lib.behavior_model import behavior_model
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten
from lib.focal_loss import FocalLossWithOneHot, FocalLossWithOutOneHot, CELossWithOutOneHot
from lib.face_model import face_model
from lib.object_model import object_model
from lib.relation_model import relation_model


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
                        default="./checkpoint/behavior") # saved training path
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
print(torch.cuda.is_available())
# splits the episodes int train, val, test
train, val, test = Splits(num_episodes=18)

# load datasets
train_set = AnotherMissOh(train, opt.img_path, opt.json_path, False)
val_set = AnotherMissOh(val, opt.img_path, opt.json_path, False)
test_set = AnotherMissOh(test, opt.img_path, opt.json_path, False)


num_persons = len(PersonCLS)
num_behaviors = len(PBeHavCLS)
num_faces = len(FaceCLS)
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
    print(torch.cuda.is_available())

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    train_loader = DataLoader(train_set, **training_params)

    # --------------- define models ---------------------------------------
    # behavior-model
    model = behavior_model(num_persons, num_behaviors, opt, device)
    trained_persons = opt.trained_model_path + os.sep + "{}".format(
        'anotherMissOh_only_params_person.pth')
    model.cuda(device)

    # face_model
    model_face = face_model(num_persons, num_faces, device)
    model_face.cuda(device)

    # emotion model

    # object model
    model_object = object_model(num_objects)
    model_object.cuda(device)

    # relation model
    model_relation = relation_model(num_objects, num_relations)
    model_relation.cuda(device)


    # place model
    pl_ckpt = torch.load('./checkpoint/resnet/resnet50_places365.pth.tar')
    state_dict = {str.replace(k,'module.',''): v for k,v in pl_ckpt['state_dict'].items()}
    fe = resnet50()
    fe.load_state_dict(state_dict, False)
    model_place = torch.nn.Sequential(fe, place_model())
    model_place = torch.nn.DataParallel(model_place).cuda(device)



    # ---------------define optimizers ------------------------------------
    # person optim
    fc_params = [p for n,p in model.named_parameters()
                 if n.startswith('detector') and p.requires_grad]

    p_params = [{'params': fc_params, 'lr': opt.lr / 10.0}]
    p_optimizer = torch.optim.SGD(p_params, lr = opt.lr / 10.0,
                                  momentum=opt.momentum,
                                  weight_decay=opt.decay)

    # behavior optim
    non_fc_params = [p for n,p in model.named_parameters()
                     if not n.startswith('detector') and p.requires_grad]

    b_params = [{'params': non_fc_params, 'lr': opt.lr * 10.0}]

    b_optimizer = torch.optim.SGD(b_params, lr = opt.lr * 10.0,
                                  momentum=opt.momentum,
                                  weight_decay=opt.decay)

    # face optim
    face_params = [p for n, p in model_face.named_parameters()]

    f_params = [{'params': face_params, 'lr': opt.lr * 10.0}]

    f_optimizer = torch.optim.SGD(f_params, lr=opt.lr * 10.0,
                                  momentum=opt.momentum,
                                  weight_decay=opt.decay)

    # emotion optim




    # object optim
    object_params = [p for n, p in model_object.named_parameters()]

    o_params = [{'params': object_params, 'lr': opt.lr * 10.0}]

    o_optimizer = torch.optim.SGD(o_params, lr=opt.lr * 10.0,
                                  momentum=opt.momentum,
                                  weight_decay=opt.decay)

    # relation optim
    relation_params = [p for n, p in model_relation.named_parameters()]

    r_params = [{'params': relation_params, 'lr': opt.lr * 10.0}]

    r_optimizer = torch.optim.SGD(r_params, lr=opt.lr * 10.0,
                                  momentum=opt.momentum,
                                  weight_decay=opt.decay)

    # place optim
    pl_optimizer = torch.optim.SGD(model_place.parameters(), lr=0.1,
                                   momentum=0.9, weight_decay=5e-4)

    # ------------ define criterions --------------------------------------
    # person criterion
    criterion = YoloLoss(num_persons, model.detector.anchors, opt.reduction)

    # behavior criterion
    if opt.b_loss == 'ce_focal':
        focal_without_onehot = FocalLossWithOutOneHot(gamma=opt.f_gamma)
    elif opt.b_loss == 'ce':
        ce_without_onehot = CELossWithOutOneHot()

    # face criterion
    f_criterion = YoloLoss(num_faces, model_face.detector.anchors, opt.reduction)


    # emotion criterion

    # object criterion
    o_criterion = YoloLoss(num_objects, model_object.detector.anchors,
                           opt.reduction)

    # relation criterion
    r_criterion = Relation_YoloLoss(num_objects, num_relations,
                                    model_relation.detector.anchors, opt.reduction)

    # place criterion
    # CrossEntropy
    # ------------ define learning schedulers -----------------------------
    # person scheduler
    p_scheduler = ReduceLROnPlateau(p_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)

    # behavior scheduler
    b_scheduler = ReduceLROnPlateau(b_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)
    # face scheduler
    f_scheduler = ReduceLROnPlateau(f_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)

    # emotion scheduler

    # object scheduler
    o_scheduler = ReduceLROnPlateau(o_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)

    # relation scheduler
    r_scheduler = ReduceLROnPlateau(r_optimizer, 'min', patience=3,
                                    factor=0.1, verbose=True,
                                    threshold=0.0001, threshold_mode='abs',
                                    cooldown=1)

    # place scheduler
    pl_scheduler = torch.optim.lr_scheduler.MultiStepLR(pl_optimizer, [int(opt.num_epoches/8), int(opt.num_epoches/4), int(opt.num_epoches/2)], gamma=0.1, last_epoch=-1)
    pl_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model.train()
    model_face.train()
    model_place.train()
    model_object.train()
    model_relation.train()
    num_iter_per_epoch = len(train_loader)

    loss_step = 0

    for epoch in range(opt.num_epoches):
        b_logit_list = []
        b_label_list = []
        b_loss_list = []
        p_loss_list = []
        f_loss_list = []
        o_loss_list = []
        r_loss_list = []


        temp_images = []
        temp_info = []
        batch_stack = []

        for iter, batch in enumerate(train_loader):

            behavior_lr = iter % (1) == 0
            verbose=iter % (opt.print_interval*10) == 0
            image, info = batch

            # sort label info on fullrect
            image, label, behavior_label, obj_label, face_label = SortFullRect(
                image, info, is_train=True)

            if np.array(label).size == 0 :
                print("iter:{}_person bboxs are empty".format(
                    iter, label))
                continue

            # image [b, 3, 448, 448]
            if torch.cuda.is_available():
                image = torch.cat(image).cuda(device)
            else:
                image = torch.cat(image)

            p_optimizer.zero_grad()

            # logits [b, 125, 14, 14]
            logits, b_logits, b_labels = model(image, label, behavior_label)

            # -------- person learning---------
            loss, loss_coord, loss_conf, loss_cls = criterion(
                logits, label, device)

            loss.backward()
            clip_grad_norm(
                [(n, p) for n, p in model.named_parameters()
                 if p.grad is not None and n.startswith('detector')],
                max_norm=opt.clip, verbose=verbose, clip=True)
            p_optimizer.step()

            # ------- behavior learning -------
            if behavior_lr:
                b_optimizer.zero_grad()

            # loss for behavior
            b_logits = torch.stack(b_logits)
            #b_logits = torch.cat(b_logits,0)

            b_labels = np.array(flatten(b_labels))
            #b_labels = np.stack(b_labels)

            # skip none behavior
            keep_idx = np.where(b_labels!=26)
            if len(keep_idx[0]) > 0:
                b_logits = b_logits[keep_idx]
                b_labels = b_labels[keep_idx]
            else:
                continue

            b_labels = Variable(
                torch.LongTensor(b_labels).cuda(device),
                requires_grad=False)
            print('behavior_label:{}'.format(b_labels))

            b_label_list.append(b_labels)
            b_logit_list.append(b_logits)

            # ------- Train Behavior module --------
            if behavior_lr:
                b_logits = torch.cat(b_logit_list, 0)
                b_labels = torch.cat(b_label_list, 0)

                if opt.b_loss == 'ce_focal':
                    loss_behavior = focal_without_onehot(b_logits, b_labels)
                elif opt.b_loss == 'ce':
                    loss_behavior = ce_without_onehot(b_logits, b_labels)

                loss_behavior.backward()

                b_logit_list = []
                b_label_list = []

                if opt.clip_grad:
                    clip_grad_norm(
                        [(n, p) for n, p in model.named_parameters()
                         if p.grad is not None and not n.startswith('detector')],
                        max_norm=opt.clip, verbose=verbose, clip=True)
                b_optimizer.step()

            # --------- Train Face module ----------
            if np.array(face_label).size != 0:
                # ------- face learning -------
                f_optimizer.zero_grad()

                # face_logits [b, 125, 14, 14]
                face_logits = model_face(image)

                # losses for face detection
                loss_face, loss_coord_face, loss_conf_face, loss_cls_face = f_criterion(
                    face_logits, face_label, device)

                loss_face.backward()
                clip_grad_norm(
                    [(n, p) for n, p in model_face.named_parameters()
                     if p.grad is not None and n.startswith('detector')],
                    max_norm=opt.clip, verbose=verbose, clip=True)
                f_optimizer.step()

            # ---------- Train Emotion module-------------


            # ---------- Train Object module--------------
            if np.array(obj_label).size != 0:
                o_optimizer.zero_grad()

                object_logits, _ = model_object(image)

                loss_object, loss_coord_object, loss_conf_object, loss_cls_object = o_criterion(
                    object_logits, obj_label, device)

                loss_object.backward()
                clip_grad_norm(
                    [(n, p) for n, p in model_object.named_parameters()
                     if p.grad is not None and n.startswith('detector')],
                     max_norm=opt.clip, verbose=verbose, clip=True)
                o_optimizer.step()


            # ---------- Train Relation module-----------
            if np.array(obj_label).size != 0:
                r_optimizer.zero_grad()

                relation_logits, _ = model_relation(image)

                loss_relation, loss_coord_relation, loss_conf_relation, loss_cls_relation, loss_rel = r_criterion(
                    relation_logits, obj_label, device)

                loss_relation.backward()
                clip_grad_norm(
                    [(n, p) for n, p in model_relation.named_parameters()
                     if p.grad is not None and n.startswith('detector')],
                    max_norm=opt.clip, verbose=verbose, clip=True)
                r_optimizer.step()


            # ---------- Train Place module---------------
            images_norm = []
            info_place = []

            for idx in range(len(image)):
                image_resize = F.interpolate(pl_normalize(image[idx]).unsqueeze(0), (224, 224)).squeeze(0)
                images_norm.append(image_resize)
                info_place.append(info[0][idx]['place'])
            info_place = label_mapping(info_place)

            # 10 frame
            pl_updated=False
            while True:
                temp_len = len(temp_images)
                temp_images += images_norm[:(10-temp_len)]
                images_norm = images_norm[(10-temp_len):]
                #print(len(info))

                temp_info += info_place[:(10-temp_len)]
                info_place = info_place[(10-temp_len):]
                temp_len = len(temp_images)
                if temp_len == 10:
                    batch_images = (torch.stack(temp_images).cuda(device))
                    batch_images = batch_images.unsqueeze(0)

                    target = torch.Tensor(temp_info).to(torch.int64).cuda(device)
                    output = model_place(batch_images)
                    pl_loss = F.cross_entropy(output, target)

                    prec1 = []; prec5 = []
                    prec1_tmp, prec5_tmp = accuracy(output, target, topk=(1, 5))
                    prec1.append(prec1_tmp.view(1, -1)); prec5.append(prec5_tmp.view(1, -1))
                    prec1 = torch.stack(prec1); prec5 = torch.stack(prec5)
                    prec1 = prec1.view(-1).float().mean(0)
                    prec5 = prec5.view(-1).float().mean(0)

                    pl_optimizer.zero_grad()
                    pl_loss.backward()
                    pl_optimizer.step()

                    end = time.time()

                    temp_images = []; temp_info = []
                    pl_updated = True
                elif temp_len < 10:
                    break

            print("Model:{}".format(opt.model))
            print("Epoch: {}/{}, Iteration: {}/{}, lr:{:.9f}".format(
                epoch + 1, opt.num_epoches,iter + 1,
                num_iter_per_epoch, p_optimizer.param_groups[0]['lr']))
            #print("---- Person Detection ---- ")
            print("+loss:{:.2f}(coord:{:.2f},conf:{:.2f},cls:{:.2f})".format(
                loss, loss_coord, loss_conf, loss_cls))
            if behavior_lr:
                print("+lr:{:.9f}, cls_behavior:{:.2f}".format(
                    b_optimizer.param_groups[0]['lr'],
                    loss_behavior))
            if np.array(face_label).size != 0:
                # print("---- Face Detection ---- ")
                print("+Face_loss:{:.2f}(coord_face:{:.2f},conf_face:{:.2f},cls_face:{:.2f})".format(
                    loss_face, loss_coord_face, loss_conf_face, loss_cls_face))
            if pl_updated:
                print("+place(loss:{:.2f},acc@1:{:.2f},acc@5:{:.2f})".format(pl_loss,prec1,prec5))
            print()

            #print("---- Object Detection ---- ")
            #print("---- Relation Detection ---- ")
            if np.array(obj_label).size != 0:
                print("+object_loss:{:.2f}(coord:{:.2f},conf:{:.2f},cls:{:.2f})".format(
                    loss_object, loss_coord_object, loss_conf_object, loss_cls_object))
                print("+relation_loss:{:.2f}(coord:{:.2f},conf:{:.2f},cls:{:.2f})".format(
                    loss_relation, loss_coord_relation, loss_conf_relation, loss_cls_relation))

            loss_dict = {
                'total' : loss.item(),
                'coord' : loss_coord.item(),
                'conf' : loss_conf.item(),
                'cls' : loss_cls.item(),
            }

            if behavior_lr:
                loss_dict['cls_behavior'] = loss_behavior.item()
                b_loss_list.append(loss_behavior.item())
                p_loss_list.append(loss_cls.item())

            if pl_updated:
                loss_dict['place'] = pl_loss.item()

            if np.array(face_label).size != 0:
                loss_dict['face_loss'] = loss_face.item()
                loss_dict['coord_face'] = loss_coord_face.item()
                loss_dict['conf_face'] = loss_conf_face.item()
                loss_dict['cls_face'] = loss_cls_face.item()

                f_loss_list.append(loss_face.item())
                f_loss_list.append(loss_coord_face.item())
                f_loss_list.append(loss_conf_face.item())
                f_loss_list.append(loss_cls_face.item())

            # Log scalar values
            for tag, value in loss_dict.items():
                logger.scalar_summary(tag, value, loss_step)

            loss_step = loss_step + 1

        print("SAVE MODEL")
        if not os.path.exists(opt.saved_path):
            os.makedirs(opt.saved_path + os.sep + "{}".format(opt.model))
            print('mkdir_{}'.format(opt.saved_path))

        # learning rate schedular
        b_loss_avg = np.stack(b_loss_list).mean()
        p_loss_avg = np.stack(p_loss_list).mean()

        p_scheduler.step(p_loss_avg)
        b_scheduler.step(b_loss_avg)
        f_scheduler.step(loss_cls_face)
        pl_scheduler.step()

        # ------------ save model params -----------------------------
        # person
        torch.save(model.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_{}.pth".format(
                       opt.model))
        torch.save(model,
                   opt.saved_path + os.sep + "anotherMissOh_{}.pth".format(
                       opt.model))
        # face
        torch.save(model_face.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_face_{}.pth".format(
                       opt.model))
        torch.save(model_face,
                   opt.saved_path + os.sep + "anotherMissOh_face_{}.pth".format(
                       opt.model))

        # emotion

        # object
        torch.save(model_object.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_object_{}.pth".format(
                       opt.model))
        torch.save(model_object,
                   opt.saved_path + os.sep + "anotherMissOh_object_{}.pth".format(
                       opt.model))

        # relation
        torch.save(model_relation.state_dict(),
                   opt.saved_path + os.sep + "anotherMissOh_only_params_relation_{}.pth".format(
                       opt.model))
        torch.save(model_relation,
                   opt.saved_path + os.sep + "anotherMissOh_relation_{}.pth".format(
                       opt.model))

        # place
        torch.save({
                    #'val_loss' : val_loss,
                    'model' : model_place.state_dict(),
                    'optimizer' : pl_optimizer.state_dict(),
                    'scheduler' : pl_scheduler.state_dict()
                    }, opt.saved_path + os.sep + "anotherMissOh_place_{}.pth".format(
                       opt.model))

if __name__ == "__main__":
    train(opt)
