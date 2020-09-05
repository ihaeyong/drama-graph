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
from Yolo_v2_pytorch.src.anotherMissOh_dataset import PersonCLS

from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten

import numpy as np

class person_model(nn.Module):
    def __init__(self, num_persons, device):
        super(person_model, self).__init__()

        pre_model = Yolo(num_persons).cuda(device)

        if True:
            pascal_voc = './checkpoint/detector/only_params_trained_yolo_voc'
            ckpt = torch.load(pascal_voc)
            if optimistic_restore(pre_model, ckpt):
                print("loaded pre-trained poscal_voc detector sucessfully.")

        self.detector = YoloD(pre_model).cuda(device)

        # define person
        self.person_conv = nn.Conv2d(
            1024, len(self.detector.anchors) * (5 + num_persons), 1, 1, 0, bias=False)

    def forward(self, image):

        # feature map of backbone
        fmap, output_1 = self.detector(image)

        output_person_logits = self.person_conv(fmap)

        return output_person_logits, output_1
