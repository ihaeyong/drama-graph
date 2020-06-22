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
    def __init__(self, num_persons, num_behaviors, opt):
        super(behavior_model, self).__init__()

        pre_model = Yolo(num_persons).cuda()
        self.detector = YoloD(pre_model, num_persons).cuda()
        self.num_persons = num_persons

        # define behavior
        self.behavior_conv = nn.Sequential(
            nn.Conv2d(1024, 512,1, 1, 0, bias=False),
            nn.Conv2d(512, 256,1, 1, 0, bias=False))

        self.behavior_fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(1024, num_behaviors))

        self.img_size = opt.image_size
        self.conf_threshold = opt.conf_threshold
        self.nms_threshold = opt.nms_threshold
        self.device=None

    def forward(self, image, label, behavior_label):

        # person detector
        logits, fmap = self.detector(image)

        if self.device is None:
            self.device = logits.get_device()

        # fmap [b, 1024, 14, 14]
        self.fmap_size = fmap.size(2)

        # persons boxes
        if not self.training:
            boxes = post_processing(logits, self.fmap_size, PersonCLS,
                                    self.detector.anchors,
                                    self.conf_threshold,
                                    self.nms_threshold)

            b_logits = []
            if len(boxes) > 0 :
                for i, box in enumerate(boxes):
                    box = np.stack(box)[:,:4].astype('float32')
                    box = Variable(torch.from_numpy(box)).cuda(
                        self.device)
                    box = torch.clamp(box, min=0.0, max=self.fmap_size)
                    with torch.no_grad():
                        i_fmap = roi_align(fmap.detach()[i].unsqueeze(0),
                                           box.view(-1, 4).float(),
                                           (self.fmap_size//4,
                                            self.fmap_size//4))

                    batch = box.size(0)
                    i_fmap = self.behavior_conv(i_fmap)
                    i_logit = self.behavior_fc(i_fmap.view(batch, -1))
                    if len(box) > 0:
                        b_logits.append(i_logit)

            return boxes, b_logits

        if len(behavior_label) > 0 and self.training:
            b_logits = []
            b_labels = []
            for i, box in enumerate(label):
                if len(box) == 0 :
                    continue
                box = np.stack(box)[:,:4].astype('float32')/self.img_size
                box = Variable(torch.from_numpy(box).cuda(self.device),
                               requires_grad=False)
                box = torch.clamp(box * self.fmap_size,
                                  min=0.0, max=self.fmap_size)
                with torch.no_grad():
                    i_fmap = roi_align(fmap.detach()[i].unsqueeze(0),
                                       box.view(-1, 4).float(),
                                       (self.fmap_size//4,
                                        self.fmap_size//4))

                batch = box.size(0)
                i_fmap = self.behavior_conv(i_fmap)
                i_logit = self.behavior_fc(i_fmap.view(batch, -1))
                if len(behavior_label[i]) > 0:
                    b_logits.append(i_logit)
                    b_labels.append(behavior_label[i])

        return logits, b_logits, b_labels
