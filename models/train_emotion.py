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


def valid_step(net):
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
edb = face_emotion_dataset(cfg)
dbl = DataLoader(edb, batch_size=cfg.bat_size, num_workers=4, pin_memory=False)

# def net, loss, optim
net = Yolo_v2_face_emotion(cfg).cuda()
loss = nn.CrossEntropyLoss()
# def optim
optim = torch.optim.Adam(net.parameters(), lr=cfg.lr_start, weight_decay=cfg.w_decay, amsgrad=True)

# load ckpt
if cfg.load_ckpt[0]:
    # continue from ckpt
    ckpt_file = [ckpt for ckpt in sorted(os.listdir(cfg.ckpt_path)) if ckpt.find(cfg.load_ckpt[1])>0][-1]
    ckpt = torch.load(os.path.join(cfg.ckpt_path, ckpt_file))
    i_resume, exp_code = ckpt['iter'], ckpt['exp_code']
    net.load_state_dict(ckpt['model_state_dict'], strict=False)
    optim.load_state_dict(ckpt['optim_state_dict'])
    print('continue from: '+ ckpt_file)
else:
    # new experiment
    lt = time.localtime()
    exp_code = str(lt.tm_year)[2:] + str(('%02d'%lt.tm_mon)) + str(('%02d'%lt.tm_mday) + str(('%02d'%lt.tm_hour)) + str(('%02d'%lt.tm_min)) + str(('%02d'%lt.tm_sec)))
    i_resume = 0
    print('new exp code: '+ exp_code)

tic = time.time()
# for numof epoch
for i_epo in range(cfg.num_epoc):
    # for numof iter per epoch
    for i_itr, bat in enumerate(dbl):
        # current iter
        itr = i_resume + i_epo*cfg.num_iter + i_itr
        # load batch onto gpu
        b_img, b_emo = bat[0].cuda(), bat[1].cuda()
        toc0 = time.time() - tic
        
        # net forward
        tic = time.time()
        net.train()
        b_out = net(b_img)
        # get loss
        l_sum = loss(b_out, b_emo.flatten().long())
        # optim
        optim.zero_grad()
        l_sum.backward()
        optim.step()
        toc1 = time.time() - tic
        
        # error display step
        if (itr%cfg.err_step) == 0:
            out_str  = 'iter: ' + str(itr)
            out_str += ', elap: ' + str(toc0)[:6] +', '+ str(toc1)[:6]
            out_str += ', loss: ' + str(l_sum.detach().cpu().numpy())[:6]
            out_str += ', accu: ' + str(measure_acc(b_out, b_emo))[:6]
            print(out_str)
        
        # validation step
        if (itr>0) and ((itr%cfg.val_step) == 0):
            print('valid step: ')
            tic = time.time()
            err, acc = valid_step(net)
            toc = time.time() - tic
            val_str = 'elap: '+ str(toc)[:6] +', loss: ' + str(err)[:6] + ', accu: ' + str(acc)[:6]
            print(val_str)
            
        # ckpt save step
        if itr%cfg.ckp_step == 0:
            # remove old ckpts
            ckpt_files = [ckpt for ckpt in sorted(os.listdir(cfg.ckpt_path)) if ckpt.find(exp_code)>0]
            ckpt_nums = len(ckpt_files)
            if ckpt_nums >= 50: # max numof ckpt per experiment
                os.remove(os.path.join(cfg.ckpt_path, ckpt_files[0]))
            # save new ckpt
            torch.save({'iter':itr, 'exp_code':exp_code,
                        'model_state_dict':net.state_dict(),'optim_state_dict':optim.state_dict()},
                       os.path.join(cfg.ckpt_path, 'ckpt_'+exp_code+'_'+str(itr)+'.tar')
                      )
            print('saved ckpt')
        
        tic = time.time()
            
            
            

