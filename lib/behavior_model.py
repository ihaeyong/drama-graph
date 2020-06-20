"""
@author: Haeyong Kang
"""
import torch.nn as nn
import torch

import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.yolo_tunning import YoloD

from Yolo_v2_pytorch.src.rois_utils import anchorboxes

class behavior_model(nn.Module):
    def __init__(self, num_persons):
        super(behavior_model, self).__init__()

        pre_model = Yolo(num_persons).cuda()
        self.detector = YoloD(pre_model, num_persons).cuda()
        self.num_persons = num_persons

        #
        self.behavior = nn.Sequential(
            nn.Conv2d(1024, 1024,1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 256,1, 1, 0, bias=False),
            nn.BatchNorm2d(256))

        #self.output_size = (output_size, output_size)

        #self.fc_behavior = nn.Sequential(
        #    nn.Linear(256 * 3 * 3, num_cls))

    def forward(self, image, label, behavior_label):

        # person detector
        import pdb; pdb.set_trace()
        logits = self.detector(image)

        # get persons boxes

        # classification


        # results

        return logits
