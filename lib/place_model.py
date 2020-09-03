import torch
import torch.nn as nn
from torchvision.datasets.vision import VisionDataset # For custom usage # YDK
# from .vision import VisionDataset

from PIL import Image

import os, sys, math
import os.path

import torch # YDK
import json # YDK
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_dict = {'' : 9, 'beach':0, 'cafe':1, 'car':2, 'convenience store':3, 'garden':4, 'home':5, 'hospital':6, 'kitchen':7,
    'livingroom':8, 'none':9, 'office':10, 'park':11, 'playground':12, 'pub':13, 'restaurant':14, 'riverside':15, 'road':16,
    'rooftop':17, 'room':18, 'studio':19, 'toilet':20, 'wedding hall':21
}

label_dict_wo_none = {'beach':0, 'cafe':1, 'car':2, 'convenience store':3, 'garden':4, 'home':5, 'hospital':6, 'kitchen':7,
    'livingroom':8, 'none':9, 'office':10, 'park':11, 'playground':12, 'pub':13, 'restaurant':14, 'riverside':15, 'road':16,
    'rooftop':17, 'room':18, 'studio':19, 'toilet':20, 'wedding hall':21
}
def label_mapping(target):
    temp = []
    for idx in range(len(target)):
        if target[idx][0][:3] == 'con':
            target[idx][0] = 'convenience store'
        temp.append(label_dict[target[idx][0]])
    return temp

def label_remapping(target):
    inv_label_dict = {v: k for k, v in label_dict_wo_none.items()}
    temp = []
    for idx in range(len(target)):
        temp.append(inv_label_dict[target[idx]])
    return temp

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            N, T, C, H, W = x.size()
            x = x.view(-1, C, H, W)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(N, T, -1)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class place_model(nn.Module):
    def __init__(self):
        super(place_model, self).__init__()

        self.lstm_sc = torch.nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc2 = torch.nn.Linear(1024, 1)
        self.fc3 = torch.nn.Linear(1024, 22)
        self.softmax = torch.nn.Softmax(dim=1)

        # self.fe = resnet.resnet50()

        # model = torch.nn.Sequential(fe, clsf())

    def forward(self, x):

        # x = self.fe(x)

        self.lstm_sc.flatten_parameters()
        N, T = x.size(0), x.size(1)
        x = self.lstm_sc(x)[0]

        change = x.reshape(N*T, -1)
        #x = self.fc1(x)
        change = self.fc2(change)
        change = change.reshape(N, T)
        #x = x.reshape(N*T, -1)

        M, _ = change.max(1)
        w = change - M.view(-1,1)
        w = w.exp()
        w = w.unsqueeze(1).expand(-1,w.size(1),-1)
        w = w.triu(1) - w.tril()
        w = w.cumsum(2)
        w = w - w.diagonal(dim1=1,dim2=2).unsqueeze(2)
        ww = w.new_empty(w.size())
        idx = M>=0
        ww[idx] = w[idx] + M[idx].neg().exp().view(-1,1,1)
        idx = ~idx
        ww[idx] = M[idx].exp().view(-1,1,1)*w[idx] + 1
        ww = (ww+1e-10).pow(-1)
        ww = ww/ww.sum(1,True)
        x = ww.transpose(1,2).bmm(x)

        x = x.reshape(N*T, -1)
        x = self.fc3(x)
        x = x.reshape(N*T, -1)

        return x

class place_model_yolo(nn.Module):
    def __init__(self, num_persons, num_faces, device):
        super(place_model_yolo, self).__init__()

        pre_model = Yolo(num_persons).cuda(device)

        num_face_cls = num_faces

        self.detector = YoloD(pre_model).cuda(device)
        self.place_conv = nn.Sequential(nn.Conv2d(1024, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.lstm_sc = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.fc2 = torch.nn.Linear(128, 1)
        self.fc3 = torch.nn.Linear(128, 22)
        self.softmax = torch.nn.Softmax(dim=1)



        # # define face
        # self.face_conv = nn.Conv2d(
        #     1024, len(self.detector.anchors) * (5 + num_face_cls), 1, 1, 0, bias=False)

    def forward(self, image):
        N, T , C, H, W = image.size(0), image.size(1), image.size(2), image.size(3), image.size(4)
        image = image.reshape(N*T, C, H, W)
        # feature map of backbone
        fmap, output_1 = self.detector(image)
        fmap = self.place_conv(fmap)
        x = self.avgpool(fmap)
        x = x.reshape(N, T, -1)
        
        self.lstm_sc.flatten_parameters()
        N, T = x.size(0), x.size(1)
        x = self.lstm_sc(x)[0]
        
        change = x.reshape(N*T, -1)
        #x = self.fc1(x)
        change = self.fc2(change)
        change = change.reshape(N, T)
        #x = x.reshape(N*T, -1)
        
        M, _ = change.max(1)
        w = change - M.view(-1,1)
        w = w.exp()
        w = w.unsqueeze(1).expand(-1,w.size(1),-1)
        w = w.triu(1) - w.tril()
        w = w.cumsum(2)
        w = w - w.diagonal(dim1=1,dim2=2).unsqueeze(2)
        ww = w.new_empty(w.size())
        idx = M>=0
        ww[idx] = w[idx] + M[idx].neg().exp().view(-1,1,1)
        idx = ~idx
        ww[idx] = M[idx].exp().view(-1,1,1)*w[idx] + 1
        ww = (ww+1e-10).pow(-1)
        ww = ww/ww.sum(1,True)
        x = ww.transpose(1,2).bmm(x)
       
        x = x.reshape(N*T, -1)
        x = self.fc3(x)
        x = x.reshape(N*T, -1)
        
        return x
