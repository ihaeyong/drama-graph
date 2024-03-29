import os
import glob
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import cv2
import numpy as np
from Yolo_v2_pytorch.src.utils import *
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, ObjectCLS, P2ORelCLS
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import time

from lib.relation_model import relation_model
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten
from lib.hyper_yolo import anchors
import pdb


num_persons = len(PersonCLS)
num_objects = len(ObjectCLS)
num_relations = len(P2ORelCLS)

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
                        default="./checkpoint") # saved training path

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="relation")
    parser.add_argument("-display", dest='display', action='store_true')
    #parser.add_argument("-use_gt", type=bool, default=True, action='store_true', 
    #                    help='using gt boxes for object and person')
    parser.add_argument("-use_gt", default=True, action='store_true', 
                        help='using gt boxes for object and person')
    args = parser.parse_args()
    return args

# get args.
opt = get_args()
print(opt)

tform = [
    Resize((448, 448)),  # should match to Yolo_V2
    ToTensor(),
    # Normalize(# should match to Yolo_V2
    #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
transf = Compose(tform)

# splits the episodes int train, val, test
train, val, test = Splits(num_episodes=18)

# load datasets
train_set = AnotherMissOh(train, opt.img_path, opt.json_path, False)
val_set = AnotherMissOh(val, opt.img_path, opt.json_path, False)
test_set = AnotherMissOh(test, opt.img_path, opt.json_path, False)

# model path
model_path = "{}/anotherMissOh_only_params_{}.pth".format(
    opt.saved_path,opt.model)

def test(opt):
    global colors
    colors = pickle.load(open("./Yolo_v2_pytorch/src/pallete", "rb"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(123)
    
    # set test loader params
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    # set test loader
    test_loader = DataLoader(train_set, **test_params)
    
    model1 = relation_model(num_persons, num_objects, num_relations, opt, device)
    
    ckpt = torch.load(model_path)
    
    model1.object_model.load_state_dict(torch.load("{}/anotherMissOh_only_params_{}.pth".format(opt.saved_path,'object')))

    ckpt_state_dict = ckpt

    print("--- loading {} model ---".format(model_path))
    if optimistic_restore(model1, ckpt_state_dict):
        print("loaded trained model sucessfully.")

    model1.to(device)
    model1.eval()


    width, height = (1024, 768)
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height

    # load test clips
    correct = 0
    for iter, batch in enumerate(test_loader):
        print('{}/{}'.format(iter,len(test_loader)))
        image, info = batch

        # sort label info on fullrect
        image, label, behavior_label, obj_label, face_label, emo_label, frame_id = SortFullRect(
            image, info, is_train=False)

        try:
            image = torch.cat(image,0).cuda()
        except:
            continue
        
        predictions, object_predictions, relation_predictions = model1(image, label, obj_label)
        #if len(relation_predictions) != 0 :#len(object_predictions) != 0 and len(object_predictions[0]) != 0:
        #    print(obj_label,relation_predictions)
        if obj_label == [[]] or obj_label == [] :
            if 0 in relation_predictions[0][0].topk(1).indices :
                correct = correct+1
        elif obj_label[0][0][5] in relation_predictions[0][0].topk(1).indices :
            correct = correct+1
        
            #import pdb;pdb.set_trace()
        for idx, frame in enumerate(frame_id):
            if idx == len(predictions):
                continue
            try:
                if len(predictions[idx]) == 0:
                    continue
                if len(object_predictions[idx]) == 0:
                    continue
            
            except:
                continue
            
            f_info = frame[0].split('/')
            save_dir = './results/relations/{}/{}/{}/'.format(
                f_info[4], f_info[5], f_info[6])
            
            save_mAP_gt_dir = './results/input_person/ground-truth/'
            save_mAP_det_dir = './results/input_person/detection/'
            
            save_mAP_img_dir = './results/input_person/image/'
            
            # visualize predictions
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # ground-truth
            if not os.path.exists(save_mAP_gt_dir):
                os.makedirs(save_mAP_gt_dir)
            # detection
            if not os.path.exists(save_mAP_det_dir):
                os.makedirs(save_mAP_det_dir)

            # image
            if not os.path.exists(save_mAP_img_dir):
                os.makedirs(save_mAP_img_dir)

            f_file = f_info[7]
            mAP_file = "{}_{}_{}_{}".format(f_info[4],
                                            f_info[5],
                                            f_info[6],
                                            f_info[7].replace("jpg", "txt"))
            if opt.display:
                print("mAP_file:{}".format(mAP_file))

            # ground truth
            gt_relation_cnt = 0
            if len(obj_label) > idx :
                f = open(save_mAP_gt_dir + mAP_file, mode='w+')
                for det in obj_label[idx]:
                    cls = P2ORelCLS[int(det[5])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    if opt.display:
                        print("relation_gt:{}".format(cat_det))
                    f.write(cat_det)
                    gt_relation_cnt += 1
                f.close()

                # open detection file

                f = open(save_mAP_det_dir + mAP_file, mode='w+')
            # out of try : pdb.set_trace = lambda : None
            try:
                # for some empty video clips
                img = image[idx]
                # ToTensor function normalizes image pixel values into [0,1]
                np_img = img.cpu().numpy().transpose((1,2,0)) * 255

                if len(predictions[idx]) != 0:
                    prediction = predictions[idx]
                    object_prediction = object_predictions[idx]
                    relation_prediction = relation_predictions[idx]
                    output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                    output_image = cv2.resize(output_image, (width, height))

                    # save images
                    cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                        '.txt', '.jpg'), output_image)

                    num_preds = len(prediction)
                    for jdx, pred in enumerate(prediction):
                        xmin = int(max(float(pred[0]) / width_ratio, 0))
                        ymin = int(max(float(pred[1]) / height_ratio, 0))
                        xmax = int(min((float(pred[2])) / width_ratio, width))
                        ymax = int(min((float(pred[3])) / height_ratio, height))
                        color = colors[PersonCLS.index(pred[5])]

                        cv2.rectangle(output_image, (xmin, ymin),
                                      (xmax, ymax), color, 2)

                        text_size = cv2.getTextSize(
                            pred[5] + ' : %.2f' % float(pred[4]),
                            cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                        cv2.rectangle(
                            output_image,
                            (xmin, ymin),
                            (xmin + text_size[0] + 100,
                             ymin + text_size[1] + 20), color, -1)
                        cv2.putText(
                            output_image, pred[5] + ' : %.2f' % float(pred[4]),
                            (xmin, ymin + text_size[1] + 4),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 1)

                        for kdx, obj_pred in enumerate(object_prediction):
                            xmin = int(max(float(obj_pred[0]) / width_ratio, 0))
                            ymin = int(max(float(obj_pred[1]) / height_ratio, 0))
                            xmax = int(min((float(obj_pred[2])) / width_ratio, width))
                            ymax = int(min((float(obj_pred[3])) / height_ratio, height))

                            color = colors[ObjectCLS.index(obj_pred[5])]

                            cv2.rectangle(output_image, (xmin, ymin),
                                          (xmax, ymax), color, 2)

                            text_size = cv2.getTextSize(
                                obj_pred[5] + ' : %.2f' % float(obj_pred[4]),
                                cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                            cv2.rectangle(
                                output_image,
                                (xmin, ymin),
                                (xmin + text_size[0] + 100,
                                 ymin + text_size[1] + 20), color, -1)
                            cv2.putText(
                                output_image, obj_pred[5] + ' : %.2f' % float(obj_pred[4]),
                                (xmin, ymin + text_size[1] + 4),
                                cv2.FONT_HERSHEY_PLAIN, 1,
                                (255, 255, 255), 1)

                            value, ind = relation_prediction[kdx].max(1)
                            ind = int(ind.cpu().numpy())
                            rel_ind = P2ORelCLS[ind]
                            cv2.putText(
                                output_image, '+ relation : ' + rel_ind,
                                (xmin, ymin + text_size[1] + 4 + 12),
                                cv2.FONT_HERSHEY_PLAIN, 1,
                                (255, 255, 255), 1)

                            pred_cls = rel_ind
                            cat_pred = '%s %s %s %s %s\n' % (
                                pred_cls, str(xmin), str(ymin), str(xmax), str(ymax))
                            f.write(cat_pred)
                            print("relation_pred:{}".format(cat_pred))

                        cv2.imwrite(save_dir + "{}".format(f_file),
                                    output_image)

                        if opt.display:
                            print("detected {}".format(
                                save_dir + "{}".format(f_file)))
                    else:
                        if opt.display:
                            print("non-detected {}".format(
                            save_dir + "{}".format(f_file)))
                        f.close()
            except:
                f.close()
                continue
            if gt_relation_cnt == 0:
                if os.path.exists(save_mAP_gt_dir + mAP_file):
                    os.remove(save_mAP_gt_dir + mAP_file)
                if os.path.exists(save_mAP_det_dir + mAP_file):
                    os.remove(save_mAP_det_dir + mAP_file)
    print("top k recall : {}".format(correct/iter))

if __name__ == "__main__":
    test(opt)

