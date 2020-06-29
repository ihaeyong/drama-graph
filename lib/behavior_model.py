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
from Yolo_v2_pytorch.src.anotherMissOh_dataset import PersonCLS, PBeHavCLS

import numpy as np

class behavior_model(nn.Module):
    def __init__(self, num_persons, num_behaviors, opt, device):
        super(behavior_model, self).__init__()

        pre_model = Yolo(num_persons).cuda(device)
        self.detector = YoloD(pre_model, num_persons).cuda(device)
        self.num_persons = num_persons

        # define behavior
        self.behavior_conv = nn.Sequential(
            nn.Conv2d(1024, 512,1, 1, 0, bias=False),
            nn.Conv2d(512, 256,1, 1, 0, bias=False))

        self.behavior_fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(1024, num_behaviors))

        self.behavior_conv1d = nn.Sequential(
            nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
            nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d((1)))

        self.num_behaviors = num_behaviors
        self.img_size = opt.image_size
        self.conf_threshold = opt.conf_threshold
        self.nms_threshold = opt.nms_threshold
        self.device=device

    def is_not_blank(self, s):
        return bool(s and s.strip())

    def label_array(self, batch, label, behavior_label):

        # define label array
        label_array = np.zeros((batch,
                                self.num_persons,
                                self.num_behaviors))

        for idx, box in enumerate(label):
            for jdx, p_box in enumerate(box):
                b_label = behavior_label[idx][jdx]
                if b_label :
                    label_array[idx, int(p_box[4]), int(b_label)] = 1
                elif idx > 0:
                    # label smoothing
                    label_array[idx, :, :] = label_array[idx-1, :, :]

        return label_array

    def forward(self, image, label, behavior_label):

        # person detector
        logits, fmap = self.detector(image)
        batch = logits.size(0)

        fmap = fmap.detach()

        # fmap [b, 1024, 14, 14]
        self.fmap_size = fmap.size(2)

        # define behavior_tensor
        behavior_tensor = Variable(
            torch.zeros(batch, self.num_persons,
                        256 * 3 * 3).cuda(self.device))

        # persons boxes
        b_logits = []
        g_features = []
        b_labels = []
        # testing
        if not self.training:
            boxes = post_processing(logits, self.img_size, PersonCLS,
                                    self.detector.anchors,
                                    self.conf_threshold,
                                    self.nms_threshold)
            if len(boxes) > 0 :
                for idx, box in enumerate(boxes):
                    num_box = len(box)
                    with torch.no_grad():
                        box_ = np.clip(
                            np.stack(box)[:,:4].astype('float32'),
                            0.0, self.img_size)
                        box_ = Variable(torch.from_numpy(box_)).cuda(
                            self.device).detach() / self.img_size * self.fmap_size
                        b_box = Variable(
                            torch.zeros(num_box, 5).cuda(self.device)).detach()
                        b_box[:,1:] = box_
                        i_fmap = roi_align(fmap[idx][None],
                                           b_box.float(),
                                           (self.fmap_size//4,
                                            self.fmap_size//4))


                        #
                        box_g = torch.from_numpy(
                            np.array([0,0,14,14])).cuda(
                                self.device).detach()
                        g_box = Variable(
                            torch.zeros(1, 5).cuda(self.device)).detach()
                        g_box[:,1:] = box_g

                        g_fmap = roi_align(fmap[idx][None],
                                           g_box.float(),
                                           (self.fmap_size//4,
                                            self.fmap_size//4))

                        i_fmap = self.behavior_conv(g_fmap + i_fmap)
                        for jdx, p_box in enumerate(box):
                            p_idx = PersonCLS.index(p_box[5])
                            behavior_tensor[idx, p_idx] = i_fmap[jdx].view(-1)

                for idx, box in enumerate(boxes):
                    for jdx, p_pox in enumerate(box):
                        p_idx = PersonCLS.index(p_box[5])
                        p_feat=behavior_tensor[:,p_idx][None,:,:].transpose(1,2)
                        p_feat = self.behavior_conv1d(p_feat)[0].squeeze(1)
                        cur_b = behavior_tensor[idx, int(p_box[4])]
                        i_logit = self.behavior_fc(p_feat + cur_b)
                        b_logits.append(i_logit)

            return boxes, b_logits

        # training
        #label_array = self.label_array(batch, label, behavior_label)
        if len(behavior_label) > 0 and self.training:
            for idx, box in enumerate(label):
                num_box = len(box)
                if num_box == 0 :
                    continue

                with torch.no_grad():
                    box_ = np.clip(
                        np.stack(box)[:,:4].astype('float32')/self.img_size,
                        0.0, self.fmap_size) * self.fmap_size
                    box_ = torch.from_numpy(box_).cuda(self.device).detach()
                    b_box = Variable(
                        torch.zeros(num_box, 5).cuda(self.device)).detach()
                    b_box[:,1:] = box_

                    i_fmap = roi_align(fmap[idx][None],
                                       b_box.float(),
                                       (self.fmap_size//4,
                                        self.fmap_size//4))

                    # global feature
                    box_g = torch.from_numpy(
                        np.array([0,0,self.fmap_size,self.fmap_size])).cuda(
                        self.device).detach()
                    g_box = Variable(
                        torch.zeros(1, 5).cuda(self.device)).detach()
                    g_box[:,1:] = box_g

                    g_fmap = roi_align(fmap[idx][None],
                                       g_box.float(),
                                       (self.fmap_size//4,
                                        self.fmap_size//4))

                i_fmap = self.behavior_conv(g_fmap + i_fmap)
                for jdx, p_box in enumerate(box):
                    behavior_tensor[idx, int(p_box[4])] = i_fmap[jdx].view(-1)

                if len(behavior_label[idx]) > 0:
                    b_labels.append(behavior_label[idx])

            for idx, box in enumerate(label):
                for jdx, p_box in enumerate(box):
                    p_feat=behavior_tensor[:,int(p_box[4])][None,:,:].transpose(1,2)
                    p_feat = self.behavior_conv1d(p_feat)[0].squeeze(1)
                    cur_b = behavior_tensor[idx, int(p_box[4])]
                    i_logit = self.behavior_fc(p_feat + cur_b)
                    b_logits.append(i_logit)


            return logits, b_logits, b_labels
