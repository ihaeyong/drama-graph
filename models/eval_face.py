import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from Yolo_v2_pytorch.src.utils import *
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, FaceCLS

from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import time
import shutil

num_persons = len(PersonCLS)
num_faces = len(FaceCLS)

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
                        default="./checkpoint/face") # saved training path

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")

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
    test_loader = DataLoader(test_set, **test_params)

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
        image, label, behavior_label, object_label, face_label, frame_id = SortFullRect(
            image, info, is_train=False)

        # if torch.cuda.is_available():
        #     image = torch.cat(image,0).cuda()

        for idx, frame in enumerate(frame_id):
            f_info = frame[0].split('/')
            save_dir = './results/person/{}/{}/{}/'.format(
                f_info[4], f_info[5], f_info[6])

            save_mAP_gt_face_dir = './results/input_person/ground-truth-face/'
            save_mAP_det_face_dir = './results/input_person/detection-face/'

            save_mAP_img_dir = './results/input_person/image/'

            # visualize predictions
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # ground-truth face
            if not os.path.exists(save_mAP_gt_face_dir):
                os.makedirs(save_mAP_gt_face_dir)
            # face
            if not os.path.exists(save_mAP_det_face_dir):
                os.makedirs(save_mAP_det_face_dir)

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
            # save person ground truth
            gt_face_cnt = 0
            if len(face_label) > idx:
                f_face = open(save_mAP_gt_face_dir + mAP_file, mode='w')
                for det in face_label[idx]:
                    cls = PersonCLS[int(det[4])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    print("face_gt:{}".format(cat_det))
                    f_face.write(cat_det)
                    gt_face_cnt += 1
                f_face.close()

                f_face = open(save_mAP_det_face_dir + mAP_file, mode='w')
            # out of try : pdb.set_trace = lambda : None
            try:
                # for some empty video clips
                img = image[idx]
                # ToTensor function normalizes image pixel values into [0,1]
                np_img = img.squeeze(0).cpu().numpy().transpose((1,2,0)) * 255

                if torch.cuda.is_available():
                    img = img.cuda()

                # with torch.no_grad():
                # logits : [1, 125, 14, 14]
                face_logits = model1(img)

                predictions_face = post_processing(face_logits,
                                                   opt.image_size,
                                                   FaceCLS,
                                                   model1.detector.anchors,
                                                   opt.conf_threshold,
                                                   opt.nms_threshold)

                if len(predictions_face) != 0:
                    predictions_face = predictions_face[0]
                    output_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    output_image = cv2.resize(output_image, (width, height))

                    # save images
                    cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                        '.txt', '.jpg'), output_image)

                    for pred in predictions_face:
                        xmin = int(max(pred[0] / width_ratio, 0))
                        ymin = int(max(pred[1] / height_ratio, 0))
                        xmax = int(min((pred[2]) / width_ratio, width))
                        ymax = int(min((pred[3]) / height_ratio, height))
                        color = colors[FaceCLS.index(pred[5])]

                        cv2.rectangle(output_image, (xmin, ymin),
                                      (xmax, ymax), color, 2)
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

                        cv2.imwrite(save_dir + "{}".format(f_file),
                                    output_image)

                        # save detection results
                        pred_cls = pred[5]

                        cat_pred = '%s %s %s %s %s %s\n' % (
                            pred_cls,
                            str(pred[4]),
                            str(xmin), str(ymin), str(xmax), str(ymax))

                        print("face_pred:{}".format(cat_pred))

                        f_face.write(cat_pred)

                        print("detected {}".format(
                            save_dir + "{}".format(f_file)))
                    else:
                        print("non-detected {}".format(
                            save_dir + "{}".format(f_file)))
                        f_face.close()

            except Exception as ex:
                print(ex)

                f_face.close()
                continue

            if gt_face_cnt == 0:
                if os.path.exists(save_mAP_gt_face_dir + mAP_file):
                    # os.remove(save_mAP_gt_face_dir + mAP_file)
                    shutil.rmtree(save_mAP_gt_face_dir + mAP_file, ignore_errors=True)

                if os.path.exists(save_mAP_det_face_dir + mAP_file):
                    # os.remove(save_mAP_det_face_dir + mAP_file)
                    shutil.rmtree(save_mAP_det_face_dir + mAP_file, ignore_errors=True)

if __name__ == "__main__":
    test(opt)

