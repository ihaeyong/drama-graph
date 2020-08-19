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
from Yolo_v2_pytorch.src.anotherMissOh_dataset import PersonCLS, PBeHavCLS, ObjectCLS, P2ORelCLS

import numpy as np

class relation_model(nn.Module):
    def __init__(self, num_persons, num_objects, num_relations):
        super(relation_model, self).__init__()

        pre_model = Yolo(num_persons)
        self.detector = YoloD(pre_model)

        self.num_persons = num_persons
        self.num_objects = num_objects
        self.num_relations = num_relations

        self.conv_person = nn.Conv2d(
            1024, len(self.anchors) * (5 + num_cls), 1, 1, 0, bias=False)
        self.conv_objects = nn.Conv2d(
            1024, len(self.anchors) * (5 + num_objects_cls + num_relations), 1, 1, 0, bias=False)
        

    def forward(self, input):
        # run the model through the main detector
        output = self.detector(input)

        # now we have to individually run each model
        output_person = self.conv_person(output)

        # run each individual model
        output_object = self.conv_object(output)

        return output_person, output_object
