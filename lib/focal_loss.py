# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    print(index)
    return mask.scatter_(1, index, ones)

class FocalLossWithOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()

class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps
        weight = np.load('./lib/behavior.npy')
        self.reweight = True
        self.weight = []
        for i in range(len(weight)):
            if i in [2, 4, 6, 7, 8, 9, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24]:
                self.weight.append(weight[i])
        self.weight = np.stack(self.weight)

        self.weight=torch.from_numpy(
            self.weight/self.weight.sum()).cuda().float()

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        if self.reweight:
            loss = F.nll_loss(logit_ls, target, weight=self.weight,
                              reduction="none")
        else:
            loss = F.nll_loss(logit_ls, target, reduction="none")
        #view = target.size() + (1,)
        #index = target.view(*view)
        index = target.unsqueeze(1)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss

        return loss.mean()

class CELossWithOutOneHot(nn.Module):
    def __init__(self, device, gamma=0, eps=1e-7):
        super(CELossWithOutOneHot, self).__init__()
        weight = np.load('./lib/behavior.npy')
        self.reweight = True
        cls_num_list = []
        for i in range(len(weight)):
            if i in [2, 4, 6, 7, 8, 9, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24]:
                cls_num_list.append(weight[i])

        beta = 0.99
        cls_num_list = np.stack(cls_num_list)
        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.weight=torch.from_numpy(per_cls_weights).to(device).float()

        self.device = device

    def forward(self, input, target):
        if self.reweight:
            loss = F.cross_entropy(input, target, weight=self.weight)
        else:
            loss = F.cross_entropy(input, target)

        return loss.to(self.device)
