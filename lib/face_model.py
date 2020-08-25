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
from Yolo_v2_pytorch.src.anotherMissOh_dataset import FaceCLS

import numpy as np

class face_model(nn.Module):
    def __init__(self, num_persons, num_faces, device):
        super(face_model, self).__init__()

        pre_model = Yolo(num_persons).cuda(device)

        num_face_cls = num_faces

        self.detector = YoloD(pre_model).cuda(device)

        # define face
        self.face_conv = nn.Conv2d(
            1024, len(self.detector.anchors) * (5 + num_face_cls), 1, 1, 0, bias=False)

    def forward(self, image):

        # feature map of backbone
        fmap, output_1 = self.detector(image)

        output_face_logits = self.face_conv(fmap)

        return output_face_logits
