import torch
import torch.nn as nn
import torch.nn.functional as F

from Yolo_v2_pytorch.src.yolo_net import Yolo

class Yolo_v2_face_emotion(nn.Module):
    def __init__(self, yolo_w_path, emo_net_ch=64):
        super(type(self), self).__init__()
        # load backbone network
        self.yolo_net = Yolo(20)
        # load weights
        ckpt = torch.load(yolo_w_path)
        self.yolo_net.load_state_dict(ckpt, strict=False)
        # freeeze yolo net weights
        for p in self.yolo_net.parameters():
            p.requires_grad = False
            
        # branch for emotion classification
        f_dim = self.yolo_net.stage3_conv2.weight.shape[0]
        h_dim = emo_net_ch
        self.emo_branch = nn.Sequential(*[nn.Conv2d(f_dim,h_dim,3,1,1), nn.ReLU(), nn.Conv2d(h_dim,h_dim,3,1,1), nn.ReLU(),
                                         nn.Conv2d(h_dim,h_dim,3,1,1), nn.AdaptiveAvgPool2d((1,1)), nn.ReLU(),
                                         nn.Conv2d(h_dim,7,1,1,0), nn.Flatten(1)])
        
        
    def forward(self, img):
        # extract feats from yolo
        fmap = self.yolo_net(img)
        # branch for face emotion classification
        emo = self.emo_branch(fmap)
        return emo
        
        
        
        
        
        
        

    