import sys,os,time,cv2,json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

# custom functions
from face_emotion.th_utils import *
from face_emotion.utils import imread_to_rgb, crop_img

import face_emotion.cfg as cfg


class face_emotion_dataset(Dataset):
    
    def __init__(self, cfg, val_set=False):
        # load db path and parsed dict
        self.db_path = cfg.db_path
        with open(cfg.dict_path) as f:
            self.db_dict = json.load(f)
        
        # params
        self.im_size = cfg.im_size
        
        # only for valid set
        epi_list = sorted(self.db_dict.keys())
        epi_nums = len(epi_list)
        if val_set:
            self.len = int(cfg.val_size)
            for epi_i,epi in enumerate(epi_list):
                if epi_i+1 not in cfg.val_sets:
                    self.db_dict.pop(epi)
        else:
            self.len = int(cfg.num_iter*cfg.bat_size)
            for epi_i,epi in enumerate(epi_list):
                if epi_i+1 in cfg.val_sets:
                    self.db_dict.pop(epi)
        
        
    def __len__(self):
        return self.len
    

    def __getitem__(self, idx):
        # choose episode
        epi_list = list(self.db_dict.keys())
        epi = th_choice(epi_list)
        # choose annotation
        ann_list = list(self.db_dict[epi].keys())
        ann = th_choice(ann_list)
        # choose fidx
        idx_list = list(self.db_dict[epi][ann].keys())
        idx = th_choice(idx_list)
        # choose random frame
        num_f = len(self.db_dict[epi][ann][idx])
        fsl = th_randint(0, num_f)
        
        # chosen face
        db = self.db_dict[epi][ann][idx][fsl]
        
        # load img
        fname = 'IMAGE_'+db['img']+'.jpg'
        img = imread_to_rgb(os.path.join(self.db_path, epi, ann, idx, fname))
        # face box
        bb  = db['face_rect']
        bb_x, bb_y, bb_w, bb_h = bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]
        # crop and resize img
        fimg = crop_img(img, bb_x, bb_y, bb_w, bb_h)
        if (fimg.shape[0] < 1) or (fimg.shape[1] <1):
            fimg = np.zeros([self.im_size[0], self.im_size[1], 3])
        else:
            fimg = cv2.resize(fimg.copy(), (self.im_size[0], self.im_size[1]))
        
        # emo_vec
        emo = np.array([db['emotion']])

        fimg = torch.Tensor(fimg).permute(2,0,1)
        emo = torch.Tensor(emo)
        
        return fimg, emo
        
        
        
    
    


