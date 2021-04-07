import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from Yolo_v2_pytorch.src.utils import *
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS,PBeHavCLS
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import time

num_persons = len(PersonCLS)
num_behaviors = len(PBeHavCLS)

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
                        default="./checkpoint/behavior") # saved training path


    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh_201029/AnotherMissOh_images/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh_201029/AnotherMissOh_Visual_v5.0/")

#    parser.add_argument("--img_path", type=str,
#                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
#    parser.add_argument("--json_path", type=str,
#                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="baseline")
    parser.add_argument("-display", dest='display', action='store_true')
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
model_path = "{}/anotherMissOh_{}.pth".format(
    opt.saved_path,opt.model)

def test(opt):
    global colors

    # set test loader params
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    # set test loader
    test_loader = DataLoader(train_set, **test_params)

    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model1 = torch.load(model_path)
            print("loaded with gpu {}".format(model_path))
        else:
            model1 = Yolo(num_persons)
            model1.load_state_dict(torch.load(model_path))
            print("loaded with cpu {}".format(model_path))

    # load the color map for detection results
    colors = pickle.load(open("./Yolo_v2_pytorch/src/pallete", "rb"))

    model1.eval()
    width, height = (1024, 768)
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height

    # load test clips
    for iter, batch in enumerate(test_loader):
        image, info = batch

        # sort label info on fullrect
        image, label, behavior_label, obj_label, face_label, emo_label, frame_id = SortFullRect(
            image, info, is_train=False)

        try:
            image = torch.cat(image,0).cuda()
        except:
            continue

        #with torch.no_grad():
        # logits : [1, 125, 14, 14]
        # behavior_logits : [1, 135, 14, 14]
        predictions, b_logits = model1(image, label, behavior_label)

        for idx, frame in enumerate(frame_id):
            f_info = frame[0].split('/')
            save_dir = './results/person/{}/{}/{}/'.format(
                f_info[4], f_info[5], f_info[6])

            save_mAP_gt_dir = './results/input_person/ground-truth/'
            save_mAP_det_dir = './results/input_person/detection/'

            save_mAP_gt_beh_dir = './results/input_person/ground-truth-behave/'
            save_mAP_det_beh_dir = './results/input_person/detection-behave/'

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

            # ground-truth behavior
            if not os.path.exists(save_mAP_gt_beh_dir):
                os.makedirs(save_mAP_gt_beh_dir)

            # behavior
            if not os.path.exists(save_mAP_det_beh_dir):
                os.makedirs(save_mAP_det_beh_dir)
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
            #b_person_label = label[i]
            # save person ground truth
            gt_person_cnt = 0
            if len(label) > idx :
                f = open(save_mAP_gt_dir + mAP_file, mode='w+')
                for det in label[idx]:
                    cls = PersonCLS[int(det[4])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    if opt.display:
                        print("person_gt:{}".format(cat_det))
                    f.write(cat_det)
                    gt_person_cnt += 1
                f.close()

                #b_behavior_label = behavior_label[i]
                f = open(save_mAP_gt_beh_dir + mAP_file, mode='w+')
                for j, det in enumerate(label[idx]):
                    cls = PBeHavCLS[int(behavior_label[idx][j])].replace(' ', '_')
                    if cls == 'none':
                        continue

                    cls = cls.replace('/', '_')
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    if opt.display:
                        print("behavior_gt:{}".format(cat_det))
                    f.write(cat_det)
                f.close()

                # open detection file

                f_beh = open(save_mAP_det_beh_dir + mAP_file, mode='w+')
                f = open(save_mAP_det_dir + mAP_file, mode='w+')
            # out of try : pdb.set_trace = lambda : None
            try:
                # for some empty video clips
                img = image[idx]
                # ToTensor function normalizes image pixel values into [0,1]
                np_img = img.cpu().numpy().transpose((1,2,0)) * 255

                if len(predictions[idx]) != 0:
                    prediction = predictions[idx]
                    b_logit = b_logits[idx]
                    output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                    output_image = cv2.resize(output_image, (width, height))

                    # save images
                    cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                        '.txt', '.jpg'), output_image)

                    num_preds = len(prediction)
                    for jdx, pred in enumerate(prediction):
                        xmin = int(max(pred[0] / width_ratio, 0))
                        ymin = int(max(pred[1] / height_ratio, 0))
                        xmax = int(min((pred[2]) / width_ratio, width))
                        ymax = int(min((pred[3]) / height_ratio, height))
                        color = colors[PersonCLS.index(pred[5])]

                        cv2.rectangle(output_image, (xmin, ymin),
                                      (xmax, ymax), color, 2)
                        value, index = b_logit[jdx].max(0)

                        b_idx = index.cpu().numpy()
                        b_pred = PBeHavCLS[b_idx]
                        text_size = cv2.getTextSize(
                            pred[5] + ' : %.2f' % pred[4],
                            cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                        cv2.rectangle(
                            output_image,
                            (xmin, ymin),
                            (xmin + text_size[0] + 100,
                             ymin + text_size[1] + 20), color, -1)
                        cv2.putText(
                            output_image, pred[5] + ' : %.2f' % pred[4],
                            (xmin, ymin + text_size[1] + 4),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 1)
                        cv2.putText(
                            output_image, '+ behavior : ' + b_pred,
                            (xmin, ymin + text_size[1] + 4 + 12),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 1)

                        cv2.imwrite(save_dir + "{}".format(f_file),
                                    output_image)

                        # save detection results
                        pred_cls = pred[5]
                        pred_beh_cls = b_pred.replace(' ', '_')
                        pred_beh_cls = pred_beh_cls.replace('/', '_')
                        cat_pred = '%s %s %s %s %s %s\n' % (
                            pred_cls,
                            str(pred[4]),
                            str(xmin), str(ymin), str(xmax), str(ymax))

                        cat_pred_beh = '%s %s %s %s %s %s\n' % (
                            pred_beh_cls,
                            str(pred[4]),
                            str(xmin), str(ymin), str(xmax), str(ymax))

                        print("behavior_pred:{}".format(cat_pred_beh))
                        print("person_pred:{}".format(cat_pred))

                        f.write(cat_pred)
                        f_beh.write(cat_pred_beh)

                        if opt.display:
                            print("detected {}".format(
                                save_dir + "{}".format(f_file)))
                    else:
                        if opt.display:
                            print("non-detected {}".format(
                            save_dir + "{}".format(f_file)))
                        f.close()
                        f_beh.close()
            except:
                f.close()
                f_beh.close()
                continue
            if gt_person_cnt == 0:
                if os.path.exists(save_mAP_gt_dir + mAP_file):
                    os.remove(save_mAP_gt_dir + mAP_file)
                if os.path.exists(save_mAP_det_dir + mAP_file):
                    os.remove(save_mAP_det_dir + mAP_file)
                if os.path.exists(save_mAP_gt_beh_dir + mAP_file):
                    os.remove(save_mAP_gt_beh_dir + mAP_file)
                if os.path.exists(save_mAP_det_beh_dir + mAP_file):
                    os.remove(save_mAP_det_beh_dir + mAP_file)

if __name__ == "__main__":
    test(opt)

