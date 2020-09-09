import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, PBeHavCLS, FaceCLS
from Yolo_v2_pytorch.src.utils import *
import shutil
import cv2
import pickle
import numpy as np
#from lib.logger import Logger

from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten

from lib.emotion_model import emotion_model, crop_face_emotion, EmoCLS

def get_args():
    parser = argparse.ArgumentParser(
        "You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size",
                        type=int, default=448,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The number of images per batch")
    parser.add_argument("--conf_threshold",
                        type=float, default=0.35)
    parser.add_argument("--nms_threshold",
                        type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type",
                        type=str, choices=["model", "params"],
                        default="model")
    parser.add_argument("--data_path_test",
                        type=str,
                        default="./Yolo_v2_pytorch/missoh_test/",
                        help="the root folder of dataset")

    parser.add_argument("--saved_path", type=str,
                        default="./checkpoint/emotion") # saved training path

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="emotion")
    parser.add_argument("-display", dest='display', action='store_true')
    parser.add_argument("-emo_net_ch", dest='emo_net_ch',type=int, default=64)
    args = parser.parse_args()
    return args


# measure accuracy (output_logits vs labels)
def measure_acc(out, lbl):
    num_l = float(out.shape[0])
    out_s = F.softmax(out, dim=1)
    out_e = out_s.max(dim=1)[1]
    num_c = (out_e == lbl).long()
    accu  = num_c.sum() / num_l 
    return accu.detach().cpu().numpy(), num_c.cpu().numpy()

# get args.
opt = get_args()
print(opt)

# splits the episodes int train, val, test
train, val, test = Splits(num_episodes=18)

# load datasets
train_set = AnotherMissOh(train, opt.img_path, opt.json_path, False)
val_set = AnotherMissOh(val, opt.img_path, opt.json_path, False)
test_set = AnotherMissOh(test, opt.img_path, opt.json_path, False)

num_persons = len(PersonCLS)
num_faces = len(FaceCLS)
num_emos = len(EmoCLS)

# model path
model_path = "{}/anotherMissOh_{}.pth".format(
    opt.saved_path,opt.model)

def test(opt):
    global colors
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        device = torch.cuda.current_device()
    else:
        torch.manual_seed(123)
    # set test loader params
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    # set test loader
    test_loader = DataLoader(test_set, **test_params)

    # emotion model
    if True:
        model_emo = emotion_model(opt.emo_net_ch, num_persons, device)
        trained_emotion = './checkpoint/refined_models' + os.sep + "{}".format(
        'anotherMissOh_only_params_emotion_integration.pth')
        model_emo.load_state_dict(torch.load(trained_emotion))
        print("loaded with {}".format(trained_emotion))
    model_emo.cuda(device)
    model_emo.eval()

    width, height = (1024, 768)
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height

    # emotion accuracy
    emo_accu = []
    # save dir
    save_emo_img_dir = './results/emotion/'
    
    # load test clips
    for iter, batch in enumerate(test_loader):
        image, info = batch

        # sort label info on fullrect
        image, label, behavior_label, object_label, face_label, emo_label, frame_id = SortFullRect(
            image, info, is_train=False)
        
        if np.array(face_label).size == 0 :
            print("iter:{}_face bboxs are empty".format(
                iter, label))
            continue
            
        # crop faces from img [b,3,h,w] -> [b,h,w,3]
        image = torch.cat(image)
        image_c = image.permute(0,2,3,1)

        face_crops, emo_gt = crop_face_emotion(image_c, face_label, emo_label, opt)

        if torch.cuda.is_available():
            face_crops = face_crops.cuda(device).contiguous()
            emo_gt = emo_gt.cuda(device)

        # emo_logits [b, 7]
        emo_logits = model_emo(face_crops)
        
        # check accuracy for batch
        acc_batch, corr_batch = measure_acc(emo_logits, emo_gt)
        emo_accu += corr_batch.tolist()
        
        # draw
        if not os.path.exists(save_emo_img_dir):
            os.makedirs(save_emo_img_dir)
        
        emo_count = 0
        for i in range(len(image_c)):
            img_i = image_c[i].numpy()
            img_i = cv2.resize(img_i.copy(), (width, height))
            fid = frame_id[i][0].split('/')
            imd = fid[-1].split('_')[1].split('.')[0]
            fid = '%s_%s_%s_%s'%(fid[-4], fid[-3], fid[-2], imd)
            
            for j in range(len(face_label[i])):
                # face coor
                fl = face_label[i][j]
                face_x0, face_y0 = int(fl[0]/width_ratio), int(fl[1]/height_ratio)
                face_x1, face_y1 = int(fl[2]/width_ratio), int(fl[3]/height_ratio)
                # get emotion idx
                emo_ij = F.softmax(emo_logits[emo_count,:], dim=0).argmax()
                emo_ij = emo_ij.detach().cpu().numpy()
                # emotion text
                emo_txt = EmoCLS[emo_ij]                
                # img_i
                img_i_bb = cv2.rectangle(img_i, (face_x0,face_y0), (face_x1,face_y1), (1,1,0), 2)
                # img_text
                img_i_tt = cv2.putText(img_i_bb, emo_txt, (face_x0, face_y0-5), cv2.FONT_HERSHEY_SIMPLEX,
                                           1, (1,1,0), 2, cv2.LINE_AA)
                emo_count += 1

            img_out = img_i_tt*255
            # save
            fname = fid + ('_%03d.jpg'%(emo_count))
            cv2.imwrite( os.path.join(save_emo_img_dir, fname), cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR) )
                
    print( 'accuracy:'+ str(np.mean(emo_accu)) )


if __name__ == "__main__":
    test(opt)
