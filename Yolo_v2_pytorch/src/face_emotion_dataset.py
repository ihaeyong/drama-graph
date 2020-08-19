import sys,os,time,cv2,json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import face_emotion.cfg as cfg


# util functions
def th_choice(a, p=None):
    """ torch implementation of np.random.choice(), x1.1~1.5 slower than original function """
    # preliminaries
    a_l = len(a)
    if p is None:
        idx = torch.randperm(a_l)
        return a[idx[0]]
        
    elif torch.sum(p) < 1.:
        print((torch.sum(p),' p.sum() not 1'))
    
    # accumulative prob
    pa = torch.cumsum(p,0)
    
    # random (0,1)
    trnd = torch.rand(1)[0]
    
    # find
    idx = (torch.argmax((pa < trnd).type(torch.FloatTensor))+1) % a_l
    return a[idx]


def imread_to_rgb(path):
    img_in = np.flip(cv2.imread(path, flags=cv2.IMREAD_COLOR), 2)/255.
    return img_in

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


    
    
class Face_emotion_dataset(Dataset):
    
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
            print('error, '+emo)

    
    def prepare_face_emotion(self, db_path):
        with open(os.path.join(db_path,'AnotherMissOh_Visual_full.json')) as f:
            anno_full = json.load(f)

        emo_dict = dict()
        for k in sorted(list(anno_full.keys())):
            epi, ann, idx = k.split('_')
            
            if epi not in emo_dict.keys():
                emo_dict[epi] = dict()
            if ann not in emo_dict[epi].keys():
                emo_dict[epi][ann] = dict()
            
            emo_dict[epi][ann][idx] = list()
            
            for a in anno_full[k]:
                for p in a['persons']:
                    pi = p['person_info']
                    p_emo = {'emotion': self.emo_char_idx(pi['emotion'].lower()),
                            'face_rect': [pi['face_rect']['min_x'], pi['face_rect']['min_y'], pi['face_rect']['max_x'], pi['face_rect']['max_y']],
                             'img': a['frame_id'].split('_')[-1]
                            }
                    emo_dict[epi][ann][idx].append(p_emo)
                
            
            if len(emo_dict[epi][ann][idx]) == 0:
                emo_dict[epi][ann].pop(idx,None)
                
            if len(emo_dict[epi][ann].keys()) == 0:
                emo_dict[epi].pop(ann,None)
                        

        with open(os.path.join(db_path,'AnotherMissOh_Visual_emo.json'), 'w') as f:
            json.dump(emo_dict, f)
        
        return
    
    

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
        
        
        
    
    


