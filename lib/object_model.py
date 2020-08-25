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
from Yolo_v2_pytorch.src.anotherMissOh_dataset import ObjectCLS

import numpy as np

class object_model(nn.Module):
    def __init__(self, num_objects):
        super(object_model, self).__init__()

        pre_model = Yolo(num_persons)
        self.detector = YoloD(pre_model)

        self.num_objects = num_objects

        self.conv = nn.Conv2d(
            1024, len(self.detector.anchors) * (5 + num_objects), 1, 1, 0, bias=False)
        

    def forward(self, input):
        # run the model through the main detector
        output = self.detector(input)

        # run each individual model
        output = self.conv(output)

        return output
