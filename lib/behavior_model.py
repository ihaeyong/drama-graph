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
    def __init__(self, num_persons, num_behaviors, opt, device):
        super(behavior_model, self).__init__()

        pre_model = Yolo(num_persons).cuda(device)
        self.detector = YoloD(pre_model, num_persons).cuda(device)
        self.num_persons = num_persons

        # define behavior
        ### self.behavior_conv = nn.Sequential(
        ###     nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
        ###     nn.BatchNorm2d(512),
        ###     nn.LeakyReLU(0.1, inplace=True),
        ###     nn.Conv2d(512, 256, 1, 1, 0, bias=False),
        ###     nn.BatchNorm2d(256),
        ###     nn.LeakyReLU(0.1, inplace=True))

        ### self.behavior_fc = nn.Sequential(
        ###     nn.Linear(256 * 3 * 3, 1024),
        ###     nn.ReLU(),
        ###     nn.Dropout(0.1),
        ###     nn.Linear(1024, num_behaviors))
        ### # self.behavior_fc = nn.Linear(256, num_behaviors) # avgpool

        ### self.behavior_conv1d = nn.Sequential(
        ###     nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
        ###     nn.Conv1d(2304, 2304, 3, stride=1, padding=1),
        ###     nn.AdaptiveAvgPool1d((1)))

        self.conv1a = nn.Conv3d(1024, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1b = nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(0, 0, 0))
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))

        self.fc2 = nn.Linear(256, num_behaviors)

        self.relu = nn.ReLU()

        self.num_behaviors = num_behaviors
        self.img_size = opt.image_size
        self.conf_threshold = opt.conf_threshold
        self.nms_threshold = opt.nms_threshold
        self.device=device

        self.global_feat = False

    def is_not_blank(self, s):
        return bool(s and s.strip())

    def label_array(self, batch, label, behavior_label):

        # define label array
        label_array = np.zeros((batch,
                                self.num_persons,
                                self.num_behaviors))

        for idx, box in enumerate(label):
            for jdx, p_box in enumerate(box):
                b_label = behavior_label[idx][jdx]
                if b_label :
                    label_array[idx, int(p_box[4]), int(b_label)] = 1
                elif idx > 0:
                    # label smoothing
                    label_array[idx, :, :] = label_array[idx-1, :, :]

        return label_array

    def forward(self, frames, labels, behavior_labels):
        '''
            frames: F C H W
            labels: F var(num_people) 5
            behavior_labels: F var(num_behaviors), where num_people == num_behaviors
        '''
        # person detector
        logits, fmaps = self.detector(frames)
        num_frames = logits.size(0)

        fmaps = fmaps.detach()

        # fmaps: F 1024 14 14
        self.fmap_size = fmaps.size(2)

        # define behavior_tensor
        behavior_tensor = Variable(
            torch.zeros(num_frames,
                        self.num_persons,
                        1024, 3, 3).cuda(self.device))

        # testing
        if not self.training:
            b_logits_list = []

            p_boxes_list = post_processing(logits,
                                         self.img_size,
                                         PersonCLS,
                                         self.detector.anchors,
                                         self.conf_threshold,
                                         self.nms_threshold)
            if len(p_boxes_list) > 0 :
                for frm_idx, boxes in enumerate(p_boxes_list):
                    num_box = len(boxes)
                    with torch.no_grad():
                        box_ = np.clip(
                            np.stack(boxes)[:,:4].astype('float32'),
                            0.0, self.img_size)
                        box_ = Variable(torch.from_numpy(box_)).cuda(
                            self.device).detach() / self.img_size * self.fmap_size
                        b_box = Variable(
                            torch.zeros(num_box, 5).cuda(self.device)).detach()
                        b_box[:,1:] = box_
                        i_fmap = roi_align(fmaps[frm_idx][None],
                                           b_box.float(),
                                           (self.fmap_size//4,
                                            self.fmap_size//4))

                        if self.global_feat:
                            box_g = torch.from_numpy(
                                np.array(
                                    [0,0,self.fmap_size,self.fmap_size])).cuda(
                                    self.device).detach()
                            g_box = Variable(
                                torch.zeros(1, 5).cuda(self.device)).detach()
                            g_box[:,1:] = box_g

                            g_fmap = roi_align(fmaps[frm_idx][None],
                                               g_box.float(),
                                               (self.fmap_size//4,
                                                self.fmap_size//4))

                            i_fmap = i_fmap + g_fmap

                        for p_idx, p_box in enumerate(boxes):
                            p_id = PersonCLS.index(p_box[5])
                            behavior_tensor[frm_idx, p_id] = i_fmap[p_idx]

                T = 5
                for frm_idx, boxes in enumerate(p_boxes_list):
                    b_logits = []
                    for p_box in boxes:
                        p_id = PersonCLS.index(p_box[5])
                        p_fmap = behavior_tensor[:, p_id]

                        start_idx = max(frm_idx - (T // 2), 0)
                        end_idx = min(start_idx + T - 1, num_frames - 1)
                        sample_idxs = np.linspace(start_idx, end_idx, T, dtype=int)
                        x = p_fmap[sample_idxs]
                        x = x.transpose(0, 1)[None, :, :, :, :]

                        x = self.relu(self.conv1a(x))
                        x = self.relu(self.conv1b(x))
                        x = self.pool1(x)

                        x = x.squeeze(4).squeeze(3).squeeze(2)
                        x = self.fc2(x)
                        b_logit = x.squeeze(0)
                        b_logits.append(b_logit)
                    b_logits_list.append(b_logits)
            return p_boxes_list, b_logits_list

        # training
        #label_array = self.label_array(batch, labels, behavior_labels)
        if len(behavior_labels) > 0 and self.training:
            b_logits = []
            b_labels = []

            for frm_idx, gt_boxes in enumerate(labels):
                num_boxes = len(gt_boxes)
                if num_boxes == 0:
                    continue

                with torch.no_grad():
                    box_ = np.clip(
                        np.stack(gt_boxes)[:,:4].astype('float32') / self.img_size,
                        0.0, self.fmap_size) * self.fmap_size
                    box_ = torch.from_numpy(box_).cuda(self.device).detach()
                    b_box = Variable(
                        torch.zeros(num_boxes, 5).cuda(self.device)).detach()
                    b_box[:,1:] = box_

                    p_l_fmap = roi_align(fmaps[frm_idx][None],
                                         b_box.float(),
                                         (self.fmap_size//4, self.fmap_size//4))

                    # global feature
                    if self.global_feat:
                        box_g = torch.from_numpy(
                            np.array([0,0,self.fmap_size,self.fmap_size])).cuda(
                                self.device).detach()
                        g_box = Variable(
                            torch.zeros(1, 5).cuda(self.device)).detach()
                        g_box[:,1:] = box_g

                        p_g_fmap = roi_align(fmaps[frm_idx][None],
                                             g_box.float(),
                                             (self.fmap_size//4, self.fmap_size//4))
                        
                        ### p_fmap = self.behavior_conv(p_l_fmap + p_g_fmap)
                        p_fmap = p_l_fmap + p_g_fmap
                    else:
                        p_fmap = p_l_fmap
                for p_idx, p_box in enumerate(gt_boxes):
                    p_id = int(p_box[4])
                    behavior_tensor[frm_idx, p_id] = p_fmap[p_idx]

            T = 5
            assert len(labels) == len(behavior_labels)
            for frm_idx, (gt_boxes, gt_behaviors) in enumerate(zip(labels, behavior_labels)):
                assert len(gt_boxes) == len(gt_behaviors)
                for p_box, b_label in zip(gt_boxes, gt_behaviors):
                    p_id = int(p_box[4])
                    p_fmap = behavior_tensor[:, p_id]

                    start_idx = max(frm_idx - (T // 2), 0)
                    end_idx = min(start_idx + T - 1, num_frames - 1)
                    sample_idxs = np.linspace(start_idx, end_idx, T, dtype=int)
                    x = p_fmap[sample_idxs]
                    x = x.transpose(0, 1)[None, :, :, :, :]

                    x = self.relu(self.conv1a(x))
                    x = self.relu(self.conv1b(x))
                    x = self.pool1(x)

                    x = x.squeeze(4).squeeze(3).squeeze(2)
                    x = self.fc2(x)
                    b_logit = x.squeeze(0)
                    b_logits.append(b_logit)
                    b_labels.append(b_label)

            return logits, b_logits, b_labels

