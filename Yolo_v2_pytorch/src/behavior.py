"""
@author: haeyong.kang
"""
import torch.nn as nn
import torch
from torchvision.ops import roi_align
from .rois_utils import anchorboxes
import numpy as np

class Behavior(nn.Module):
    def __init__(self, anchors, num_cls, output_size):
        super(Behavior, self).__init__()

        self.anchors = anchors
        self.num_cls = num_cls

        self.conv_behavior = nn.Sequential(
            nn.Conv2d(1024, 1024,1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 256,1, 1, 0, bias=False),
            nn.BatchNorm2d(256))
            #nn.Conv2d(1024, len(self.anchors) * (self.num_cls),
            #          1, 1, 0, bias=False))

        self.output_size = (output_size, output_size)

        self.fc_behavior = nn.Sequential(
            nn.Linear(256 * 3 * 3, num_cls))


    def forward(self, logits, output_fmap):

        boxes = anchorboxes(logits, self.anchors)

        batch, num_anchors, num_feats,_ = boxes.size()
        fmap = self.conv_behavior(output_fmap)

        # [b, 1024, 3, 3]
        roi_fmap = []
        for b in range(batch):
            b_roi_fmap = roi_align(fmap[b].unsqueeze(0),
                            boxes[b].view(-1, 4),
                            output_size=self.output_size)

            b_roi_fmap = self.fc_behavior(
                b_roi_fmap.view(-1, 256*3*3)).view(
                    1,num_anchors * self.num_cls, 14, 14)
            roi_fmap.append(b_roi_fmap)

        # behavior_logits
        #[b, 135, 14, 14]
        behavior_logits = torch.cat(roi_fmap, 0)

        return behavior_logits
