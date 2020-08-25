"""
modified by haeyong.kang
"""
import math
import torch, torchvision
from torch.autograd import Variable
import torch.nn as nn

def anchorboxes(logits, anchors):

    # num_anchors : [5]
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    if isinstance(logits, Variable):
        logits = logits.data

    if logits.dim() == 3:
        logits.unsqueeze_(0)

    # logits : [b, 125, 14, 14]
    batch, channel, height, width = logits.size()
    max_size = min(height, width)

    # -------- Compute xc,yc, w,h, box_score on Tensor -------------
    # lin_x,y : [196]
    lin_x = torch.linspace(
        0, width - 1, width).repeat(height, 1).view(height * width)
    lin_y = torch.linspace(
        0, height - 1, height).repeat(width, 1).t().contiguous().view(
            height * width)

    # anchor_w,h : [1,5,1]
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    # logits : [b, 5, 25, 196]
    logits = logits.view(batch, num_anchors, -1, height * width)
    logits[:, :, 0, :].sigmoid_().add_(lin_x) #.div_(width)
    logits[:, :, 1, :].sigmoid_().add_(lin_y) #.div_(height)
    logits[:, :, 2, :].exp_().mul_(anchor_w)  #.div_(width)
    logits[:, :, 3, :].exp_().mul_(anchor_h)  #.div_(height)
    logits[:, :, 4, :].sigmoid_()

    # scores
    with torch.no_grad():
        # cls_scores : [b, 5, 20, 196]
        cls_scores = torch.nn.functional.softmax(logits[:, :, 5:, :], 2)

        # coords : [b, 5, 196, 4]
        coords = logits.transpose(2, 3)[..., 0:4]
        coords = torch.clamp(coords, min=0.0, max=max_size)

    return coords
