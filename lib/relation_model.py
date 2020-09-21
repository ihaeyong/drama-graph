"""
@author: Haeyong Kang
"""
import torch.nn as nn
import torch

import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.ops import roi_align

from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.yolo_tunning import YoloD

from Yolo_v2_pytorch.src.rois_utils import anchorboxes
from Yolo_v2_pytorch.src.anotherMissOh_dataset import PersonCLS, ObjectCLS, P2ORelCLS

from lib.person_model import person_model
from lib.object_model import object_model

import numpy as np
import pdb

class relation_model(nn.Module):
    def __init__(self, num_persons, num_objects, num_relations, opt, device):
        super(relation_model, self).__init__()

        # just for reference (anchor information)
        num_objects = 47
        num_relations = 13
        num_persons = 20

        self.person_model = person_model(num_persons, device)
        self.object_model = object_model(num_objects).cuda(device)
        self.person_detector = self.person_model.detector
        self.object_detector = self.object_model.detector

        self.num_persons = num_persons
        self.num_relations = num_relations
        self.num_objects = num_objects

        # define convs
        self.person_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.object_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())


        self.relation_conv1d_person = nn.Sequential(
            nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
            nn.ReLU(),nn.Dropout(0.1),
            nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
            nn.ReLU(),nn.Dropout(0.1),
        )

        self.relation_conv1d_object = nn.Sequential(
            nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
            nn.ReLU(),nn.Dropout(0.1),
            nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
            nn.ReLU(),nn.Dropout(0.1),
        )

        self.relation_fc = nn.Sequential(
            nn.Linear(256 * 3 * 3 * 2, 1024),
            nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(1024, num_relations))

        self.img_size = opt.image_size
        self.conf_threshold = opt.conf_threshold
        self.nms_threshold = opt.nms_threshold
        self.device=device

        self.gt_boxes = True

    def ex_global_feat_person(self, fmap):
        box_g = torch.from_numpy(
            np.array([0,0,self.fmap_size,self.fmap_size])).cuda(
                self.device).detach()
        g_box = Variable(
            torch.zeros(1, 5).cuda(self.device)).detach()
        g_box[:,1:] = box_g

        g_fmap = roi_align(fmap[None],
                           g_box.float(),
                           (self.fmap_size//4,
                            self.fmap_size//4))

        g_fmap = self.person_conv(g_fmap)

        return g_fmap

    def ex_global_feat_object(self, fmap):
        box_g = torch.from_numpy(
            np.array([0,0,self.fmap_size,self.fmap_size])).cuda(
                self.device).detach()
        g_box = Variable(
            torch.zeros(1, 5).cuda(self.device)).detach()
        g_box[:,1:] = box_g

        g_fmap = roi_align(fmap[None],
                           g_box.float(),
                           (self.fmap_size//4,
                            self.fmap_size//4))

        g_fmap = self.object_conv(g_fmap)

        return g_fmap


    def forward(self, image, label, object_label):

        # person detector
        logits, fmap = self.person_model(image)
        object_logits, obj_fmap = self.object_model(image)
        batch = logits.size(0)

        fmap = fmap.detach()
        obj_fmap = obj_fmap.detach()

        # fmap [b, 1024, 14, 14]
        self.fmap_size = fmap.size(2)
        self.obj_fmap_size = obj_fmap.size(2)


        # persons boxes
        r_logits = []
        r_labels = []

        # training
        if len(label) > 0 and self.training:
            if len(object_label) > 0:
                for idx, box in enumerate(label):
                    num_box = len(box)
                    g_fmap = self.ex_global_feat_person(fmap[idx])

                    if num_box == 0 :
                        continue

                    box_ = np.clip(
                        np.stack(box)[:,:4].astype('float32')/self.img_size,
                        0.0, self.fmap_size) * self.fmap_size
                    box_ = torch.from_numpy(box_).cuda(self.device).detach()
                    b_box = Variable(
                        torch.zeros(num_box, 5).cuda(self.device)).detach()
                    b_box[:,1:] = torch.clamp(box_ + torch.randn(box_.shape).cuda(
                        self.device), 0, self.fmap_size)
                    i_fmap = roi_align(fmap[idx][None],
                                       b_box.float(),
                                       (self.fmap_size//4,
                                        self.fmap_size//4))

                    # local feature
                    i_fmap = self.person_conv(i_fmap)

                    i_fmap += g_fmap
                    rr_labels = []
                    rr_logits = []

                    for jdx, obj_box in enumerate(object_label[idx]):
                        obj_num_box = 1

                        obj_g_fmap = self.ex_global_feat_object(obj_fmap[idx])
                        if len(obj_box) == 0 :
                            continue

                        obj_box = [obj_box]
                        obj_box_ = np.clip(
                            np.stack(obj_box)[:,:4].astype('float32')/self.img_size,
                            0.0, self.obj_fmap_size) * self.obj_fmap_size
                        obj_box_ = torch.from_numpy(obj_box_).cuda(self.device).detach()
                        obj_b_box = Variable(
                            torch.zeros(obj_num_box, 5).cuda(self.device)).detach()
                        obj_b_box[:,1:] = torch.clamp(obj_box_ + torch.randn(obj_box_.shape).cuda(
                            self.device), 0, self.obj_fmap_size)
                        obj_i_fmap = roi_align(obj_fmap[idx][None],
                                           obj_b_box.float(),
                                           (self.obj_fmap_size//4,
                                            self.obj_fmap_size//4))

                        
                        obj_i_fmap = self.object_conv(obj_i_fmap)

                        rr_labels.append(int(obj_box[0][5]))

                        p_feat = self.relation_conv1d_person(i_fmap.view(-1,2304).unsqueeze(2))[0]
                        o_feat = self.relation_conv1d_object(obj_i_fmap.view(-1,2304).unsqueeze(2))[0]

                        r_feat = torch.cat((p_feat, o_feat),0).transpose(1,0)
                        r_logit = self.relation_fc(r_feat)

                        rr_logits.append(r_logit)

                    r_logits.append(rr_logits)
                    r_labels.append(rr_labels)

            return r_logits, r_labels


        if not self.training:
            boxes = post_processing(logits, self.img_size, PersonCLS,
                                    self.person_detector.anchors,
                                    self.conf_threshold,
                                    self.nms_threshold)

            object_boxes = post_processing(object_logits, self.img_size, ObjectCLS,
                                           self.object_detector.anchors,
                                           self.conf_threshold,
                                           self.nms_threshold)

            boxes_gt = []

            for idx, box in enumerate(label):
                b_boxes = []
                for jdx, p_box in enumerate(box):
                    p_box_ = p_box[0:4].tolist()
                    p_conf_ = [1.0]
                    p_cls_ = [PersonCLS[int(p_box[4])]]
                    p_box = np.concatenate([p_box_, p_conf_, p_cls_])
                    b_boxes.append(p_box)

                boxes_gt.append(b_boxes)
            if 1:
                boxes = boxes_gt

            object_boxes_gt = []
            for idx, box in enumerate(object_label):
                object_b_boxes = []
                for jdx, p_box in enumerate(box):
                    p_box_ = p_box[0:4].tolist()
                    p_conf_ = [1.0]
                    p_cls_ = [ObjectCLS[int(p_box[4])]]
                    p_box = np.concatenate([p_box_, p_conf_, p_cls_])
                    object_b_boxes.append(p_box)

                object_boxes_gt.append(object_b_boxes)
            if 1:
                object_boxes = object_boxes_gt


            if len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    num_box = len(box)
                    g_fmap = self.ex_global_feat_person(fmap[idx])

                    if num_box == 0 :
                        continue

                    box_ = np.clip(
                        np.stack(box)[:,:4].astype('float32'),
                        0.0, self.img_size)
                    box_ = Variable(torch.from_numpy(box_)).to(
                        self.device).detach() / self.img_size * self.fmap_size
                    b_box = Variable(
                        torch.zeros(num_box, 5).to(self.device)).detach()
                    b_box[:,1:]  = box_
                    i_fmap = roi_align(fmap[idx][None],
                                       b_box.float(),
                                       (self.fmap_size//4,
                                        self.fmap_size//4))

                    i_fmap = self.person_conv(i_fmap)

                    i_fmap += g_fmap
                    rr_logits = []

                    for jdx, obj_box in enumerate(object_label[idx]):
                        obj_num_box = 1

                        obj_g_fmap = self.ex_global_feat_object(obj_fmap[idx])
                        if len(obj_box) == 0 :
                            continue

                        obj_box = [obj_box]
                        obj_box_ = np.clip(
                            np.stack(obj_box)[:,:4].astype('float32')/self.img_size,
                            0.0, self.obj_fmap_size) * self.obj_fmap_size
                        obj_box_ = torch.from_numpy(obj_box_).cuda(self.device).detach()
                        obj_b_box = Variable(
                            torch.zeros(obj_num_box, 5).cuda(self.device)).detach()
                        obj_b_box[:,1:] = torch.clamp(obj_box_ + torch.randn(obj_box_.shape).cuda(
                            self.device), 0, self.obj_fmap_size)
                        obj_i_fmap = roi_align(obj_fmap[idx][None],
                                               obj_b_box.float(),
                                               (self.obj_fmap_size//4,
                                                self.obj_fmap_size//4))

                        obj_i_fmap = self.object_conv(obj_i_fmap)

                        p_feat = self.relation_conv1d_person(i_fmap.view(-1, 2304).unsqueeze(2))[0]
                        o_feat = self.relation_conv1d_object(obj_i_fmap.view(-1, 2304).unsqueeze(2))[0]

                        r_feat = torch.cat((p_feat, o_feat), 0).transpose(1,0)
                        r_logit = self.relation_fc(r_feat)
                        rr_logits.append(r_logit)

                    r_logits.append(rr_logits)

            return boxes, object_boxes, r_logits




