import sys,os,time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import Yolo_v2_pytorch.src.face_emotion_cfg as cfg
from Yolo_v2_pytorch.src.face_emotion_dataset import Face_emotion_dataset
from lib.yolo_face_emotion import Yolo_v2_face_emotion


def measure_acc(out, lbl):
    num_l = float(out.shape[0])
    out_s = F.softmax(out, dim=1)
    out_e = out_s.max(dim=1)[1]
    num_c = (out_e == lbl[:,0]).long().sum()
    accu  = num_c / num_l 
    return accu.detach().cpu().numpy()


def test_step(net):
    net.eval()
    vdb = face_emotion_dataset(cfg, val_set=True)
    dbl = DataLoader(vdb, batch_size=cfg.bat_size, num_workers=4, pin_memory=False)
    
    v_err, v_acc = [],[]    
    for i,bat in enumerate(dbl):
        sys.stdout.write("\r"+str(i+1)+'/'+str(cfg.val_size//cfg.bat_size))
        # load batch onto gpu
        b_img, b_emo = bat[0].cuda(), bat[1].cuda()
        # net forward
        b_out = net(b_img)
        # get loss
        l_sum = loss(b_out, b_emo.flatten().long())
        # get accu
        accu  = measure_acc(b_out, b_emo)
        # store results
        v_err.append(l_sum.detach().cpu().numpy())
        v_acc.append(accu)
    
    return np.mean(v_err), np.mean(v_acc)
    

# def dataset, dataloader object
edb = Face_emotion_dataset(cfg)
dbl = DataLoader(edb, batch_size=cfg.bat_size, num_workers=4, pin_memory=False)

# def net, loss, optim
net = Yolo_v2_face_emotion(cfg).cuda()
loss = nn.CrossEntropyLoss()

# load ckpt
ckpt_file = [ckpt for ckpt in sorted(os.listdir(cfg.ckpt_path)) if ckpt.find(cfg.load_ckpt[1])>0][-1]
ckpt = torch.load(os.path.join(cfg.ckpt_path, ckpt_file))
net.load_state_dict(ckpt['model_state_dict'], strict=False)
print('load chkpt: '+ ckpt_file)
    
# run evaluation
loss, accuracy = test_step(net)
print('overall accuracy: '+accuracy)


