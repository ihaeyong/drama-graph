import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.yolo_tunning import YoloD

EmoCLS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sandess', 'Surprise', 'Neutral']

# emotion text to index
def emo_char_idx(emo):
    # 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    if emo == 'angry' or emo == 'anger':
        return 0
    elif emo == 'disgust':
        return 1
    elif emo == 'fear':
        return 2
    elif emo == 'happy' or emo == 'happiness':
        return 3
    elif emo == 'sad' or emo == 'sadness':
        return 4
    elif emo == 'surprise':
        return 5
    elif emo == 'neutral':
        return 6
    else:
        return 6

# img crop function
def crop_img(I, x, y, w, h, center=False, mfill=False):
    im_h = I.shape[0]
    im_w = I.shape[1]

    if center:
        w0 = w // 2;    w1 = w - w0    # w = w0+w1
        h0 = h // 2;    h1 = h - h0    # h = h0+h1

        x_min = x - w0;    x_max = x+w1-1;
        y_min = y - h0;    y_max = y+h1-1;
    else:
        x_min = x;    x_max = x+w-1;
        y_min = y;    y_max = y+h-1;

    pad_l = 0;    pad_r = 0;
    pad_u = 0;    pad_d = 0;

    # bounds
    if x_min < 0:          pad_l = -x_min;            x_min = 0;
    if x_max > im_w-1:     pad_r = x_max-(im_w-1);    x_max = im_w-1;
    if y_min < 0:          pad_u = -y_min;            y_min = 0;
    if y_max > im_h-1:     pad_d = y_max-(im_h-1);    y_max = im_h-1;

    # crop & append
    J = I[y_min:y_max+1, x_min:x_max+1, :]

    # 0 size errors
    if J.shape[0] == 0 or J.shape[1] == 0:
        return np.zeros([h,w,3])

    if mfill:
        rsel = np.linspace(0, J.shape[0], 8, endpoint=False, dtype=int)
        csel = np.linspace(0, J.shape[1], 8, endpoint=False, dtype=int)
        fill = np.mean(J[rsel][:,csel], axis=(0,1))
    else:
        fill = (0,0,0)
    J = cv2.copyMakeBorder(J, pad_u,pad_d,pad_l,pad_r, cv2.BORDER_CONSTANT, value=fill)
    return J


def crop_face_emotion(image, face_label, emo_label, opt):
    face_crops = list()
    emo_gt = list()

    for i,img in enumerate(image):
        for j in range(len(face_label[i])):
            # face corrdinates
            fl = face_label[i][j]
            face_x, face_y, face_w, face_h = float(fl[0]), float(fl[1]), float(fl[2])-float(fl[0]), float(fl[3])-float(fl[1])
            # crop face region, resize
            img_crop = torch.Tensor( cv2.resize(crop_img(img.numpy(), int(face_x), int(face_y), int(face_w), int(face_h)).copy(), (opt.image_size, opt.image_size)) )
            # store
            face_crops.append(img_crop)
            # emotion labels
            if emo_label is not None:
                emo_text = emo_label[i][j]
                el = emo_char_idx(emo_text.lower())
                emo_gt.append(el)

    face_crops = torch.stack(face_crops).permute(0,3,1,2) # [f,h,w,3]->[f,3,h,w]
    emo_gt = torch.Tensor(emo_gt).long() if emo_label is not None else None

    return face_crops, emo_gt

class emotion_model(nn.Module):
    def __init__(self, emo_net_ch=64, num_persons=21, device=None):
        super(type(self), self).__init__()
        # load backbone network
        pre_model = Yolo(num_persons).cuda(device)
        self.yolo_net = YoloD(pre_model).cuda(device)

        # branch for emotion classification
        f_dim = self.yolo_net.stage3_conv1[0].weight.shape[0]
        h_dim = emo_net_ch
        self.emo_branch = nn.Sequential(*[nn.Conv2d(f_dim,h_dim,3,1,1), nn.ReLU(), nn.Conv2d(h_dim,h_dim,3,1,1), nn.ReLU(),
                                         nn.Conv2d(h_dim,h_dim,3,1,1), nn.AdaptiveAvgPool2d((1,1)), nn.ReLU(),
                                         nn.Conv2d(h_dim,7,1,1,0)])


    def forward(self, img):
        # extract feats from yolo
        fmap,_ = self.yolo_net(img)

        # branch for face emotion classification
        emo = self.emo_branch(fmap)
        emo = emo.flatten(1)

        return emo
