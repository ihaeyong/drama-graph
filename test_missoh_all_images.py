import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from Yolo_v2_pytorch.src.utils import *
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import time

PersonCLS = ['Dokyung', 'Haeyoung1', 'Haeyoung2', 'Sukyung', 'Jinsang',
            'Taejin', 'Hun', 'Jiya', 'Kyungsu', 'Deogi',
            'Heeran', 'Jeongsuk', 'Anna', 'Hoijang', 'Soontack',
            'Sungjin', 'Gitae', 'Sangseok', 'Yijoon', 'Seohee']
PBeHavCLS = ["unknown","stand up","sit down","walk","hold","hug",
             "look at/back on",
             "drink","eat",
             "point out","dance", "look for","watch",
             "push away", "wave hands",
             "cook", "sing", "play instruments",
             "call", "destroy",
             "put arms around each other's shoulder",
             "open", "shake hands", "wave hands",
             "kiss", "high-five", "write"]
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
    parser.add_argument("--pre_trained_model_path",
                        type=str,
                        default="./checkpoint/anotherMissOh.pth")
    parser.add_argument("--data_path_test",
                        type=str,
                        default="./Yolo_v2_pytorch/missoh_test/",
                        help="the root folder of dataset")

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual/")
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
train, val, test = Splits(num_episodes=9)

# load datasets
train_set = AnotherMissOh(train, opt.img_path, opt.json_path, False)
val_set = AnotherMissOh(val, opt.img_path, opt.json_path, False)
test_set = AnotherMissOh(test, opt.img_path, opt.json_path, False)

def test(opt):
    global colors

    # set test loader params
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    # set test loader
    test_loader = DataLoader(test_set, **test_params)

    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model1 = torch.load(opt.pre_trained_model_path)
            print("loaded with gpu {}".format(opt.pre_trained_model_path))
        else:
            model1 = Yolo(num_persons)
            model1.load_state_dict(torch.load(opt.pre_trained_model_path))
            print("loaded with cpu {}".format(opt.pre_trained_model_path))

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
        image, label, behavior_label, frame_id = SortFullRect(
            image, info, is_train=False)

        for i, frame in enumerate(frame_id):
            f_info = frame[0].split('/')
            save_dir = './results/person/{}/{}/{}/'.format(
                f_info[4], f_info[5], f_info[6])

            save_mAP_gt_dir = './results/mAP/ground-truth/'
            save_mAP_det_dir = './results/mAP/detection/'
            save_mAP_img_dir = './results/mAP/image/'

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

            print("mAP_file:{}".format(mAP_file))

            # ground truth
            #b_person_label = label[i]
            #b_behavior_label = behavior_label[i]

            # save person ground truth
            with open(save_mAP_gt_dir + mAP_file, mode='w') as f:
                for dets in label:
                    for det in dets:
                        cls = PersonCLS[int(det[4])]
                        xmin = str(max(det[0] / width_ratio, 0))
                        ymin = str(max(det[1] / height_ratio, 0))
                        xmax = str(min((det[2]) / width_ratio, width))
                        ymax = str(min((det[3]) / height_ratio, height))
                        cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                        f.write(cat_det)
            f.close()

            try:
                # pdb.set_trace = lambda : None out of try
                # for some empty video clips
                img = image[i]

                # ToTensor function normalizes image pixel values into [0,1]
                np_img = img.cpu().numpy()[0].transpose((1,2,0)) * 255

                if torch.cuda.is_available():
                    img = img.cuda()

                with torch.no_grad():
                    # logits : [1, 125, 14, 14]
                    # behavior_logits : [1, 135, 14, 14]
                    logits, behavior_logits = model1(img)

                    predictions = post_processing(logits,behavior_logits,
                                                  opt.image_size,
                                                  PersonCLS,PBeHavCLS,
                                                  model1.anchors,
                                                  opt.conf_threshold,
                                                  opt.nms_threshold)
                if len(predictions) != 0:
                    predictions = predictions[0]
                    output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                    output_image = cv2.resize(output_image, (width, height))

                    # save images
                    cv2.imwrite(save_mAP_img_dir + mAP_file, output_image)

                    for pred in predictions:
                        xmin = int(max(pred[0] / width_ratio, 0))
                        ymin = int(max(pred[1] / height_ratio, 0))
                        xmax = int(min((pred[2]) / width_ratio, width))
                        ymax = int(min((pred[3]) / height_ratio, height))
                        color = colors[PersonCLS.index(pred[5])]

                        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
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
                            output_image, '+ behavior : ' + pred[6],
                            (xmin, ymin + text_size[1] + 4 + 12),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 1)

                        cv2.imwrite(save_dir + "{}".format(f_file), output_image)
                    print("detected {}".format(save_dir + "{}".format(f_file)))
                else:
                    print("non-detected {}".format(save_dir + "{}".format(f_file)))
            except:
                continue

if __name__ == "__main__":
    test(opt)

