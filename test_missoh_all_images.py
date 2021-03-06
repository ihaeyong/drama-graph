import os
import glob
import argparse
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
import pdb

num_persons = len(PersonCLS)
num_objects= len(ObjectCLS)
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

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual/")


    parser.add_argument("-model", dest='model', type=str, default="baseline")
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
model_path = "./checkpoint/anotherMissOh_{}.pth".format(opt.model)

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
        image, label, behavior_label, object_label, frame_id = SortFullRect(
            image, info, is_train=False)

        for i, frame in enumerate(frame_id):
            f_info = frame[0].split('/')
            # save_dir = './results/person/{}/{}/{}/'.format(f_info[4], f_info[5], f_info[6])
            save_dir = './results/person/{}/{}/{}/'.format(f_info[1], f_info[2], f_info[3])

            save_mAP_gt_dir = './results/input_person/ground-truth/'
            save_mAP_det_dir = './results/input_person/detection/'

            save_mAP_gt_obj_dir = './results/input_person/ground-truth-object/'
            save_mAP_det_obj_dir = './results/input_person/detection-object/'

            save_mAP_gt_rel_dir = './results/input_person/ground-truth-relation/'
            save_mAP_det_rel_dir = './results/input_person/detection-relation/'

            save_mAP_img_dir = './results/input_person/image/'
            save_obj_dir = './results/object/{}/{}/{}/'.format(f_info[1], f_info[2], f_info[3])

            # visualize predictions
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # ground-truth
            if not os.path.exists(save_mAP_gt_dir):
                os.makedirs(save_mAP_gt_dir)
            # detection
            if not os.path.exists(save_mAP_det_dir):
                os.makedirs(save_mAP_det_dir)

            # ground-truth object
            if not os.path.exists(save_mAP_gt_obj_dir):
                os.makedirs(save_mAP_gt_obj_dir)
            # object
            if not os.path.exists(save_mAP_det_obj_dir):
                os.makedirs(save_mAP_det_obj_dir)

        	# ground-truth object
            if not os.path.exists(save_mAP_gt_rel_dir):
                os.makedirs(save_mAP_gt_rel_dir)
            # object
            if not os.path.exists(save_mAP_det_rel_dir):
                os.makedirs(save_mAP_det_rel_dir)

            # image
            if not os.path.exists(save_mAP_img_dir):
                os.makedirs(save_mAP_img_dir)

            if not os.path.exists(save_obj_dir):
                os.makedirs(save_obj_dir)

            f_file = f_info[7]
            mAP_file = "{}_{}_{}_{}".format(f_info[4],
                                            f_info[5],
                                            f_info[6],
                                            f_info[7].replace("jpg", "txt"))

            # mAP_file = "{}_{}_{}_{}".format(f_info[1],
            #                                 f_info[2],
            #                                 f_info[3],
            #                                 f_info[4].replace("jpg", "txt"))

            print("mAP_file:{}".format(mAP_file))

            # ground truth
            #b_person_label = label[i]
            # save person ground truth
            gt_person_cnt = 0
            if len(label) > i:
                f = open(save_mAP_gt_dir + mAP_file, mode='w')
                for det in label[i]:
                    cls = PersonCLS[int(det[4])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    print("person_gt:{}".format(cat_det))
                    f.write(cat_det)
                    gt_person_cnt += 1
                f.close()
                # open detection file
                f = open(save_mAP_det_dir + mAP_file, mode='w')
            else:
                f = open(save_mAP_gt_dir + mAP_file, mode='w')
                f.write('')
                f.close()
                f = open(save_mAP_det_dir + mAP_file, mode='w')

            # save face ground truth
            gt_obj_cnt = 0
            if len(object_label) > i:
                f_obj = open(save_mAP_gt_obj_dir + mAP_file, mode='w')
                for det in object_label[i]:
                    cls = ObjectCLS[int(det[4])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    print("obj_gt:{}".format(cat_det))
                    f_obj.write(cat_det)
                    gt_obj_cnt += 1
                f_obj.close()
                f_obj = open(save_mAP_det_obj_dir + mAP_file, mode='w')
            else:
                f_obj = open(save_mAP_gt_obj_dir + mAP_file, mode='w')
                f_obj.write('')
                f_obj.close()
                f_obj = open(save_mAP_det_obj_dir + mAP_file, mode='w')

            gt_rel_cnt = 0
            rel_gt = []
            if len(object_label) > i:
                f_rel = open(save_mAP_gt_rel_dir + mAP_file, mode='w')
                for det in object_label[i]:
                    cls = P2ORelCLS[int(det[5])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    rel_gt.append([xmin, ymin, xmax, ymax])
                    print("rel_gt:{}".format(cat_det))
                    f_rel.write(cat_det)
                    gt_rel_cnt += 1
                f_rel.close()
                f_rel = open(save_mAP_det_rel_dir + mAP_file, mode='w')
            else:
                f_rel = open(save_mAP_gt_rel_dir + mAP_file, mode='w')
                f_rel.write('')
                f_rel.close()
                f_rel = open(save_mAP_det_rel_dir + mAP_file, mode='w')

            # out of try : pdb.set_trace = lambda : None
            try:
                # for some empty video clips
                img = image[i]

                # ToTensor function normalizes image pixel values into [0,1]
                np_img = img.cpu().numpy()[0].transpose((1,2,0)) * 255

                if torch.cuda.is_available():
                    img = img.cuda()

                #with torch.no_grad():
                # logits : [1, 125, 14, 14]
                # behavior_logits : [1, 135, 14, 14]
                # face_logits : [1, 125, 14, 14]
                logits, object_logits = model1(img)

                predictions = post_processing(logits,
                                              opt.image_size,
                                              PersonCLS,
                                              model1.anchors,
                                              opt.conf_threshold,
                                              opt.nms_threshold)

                predictions_object = post_processing(object_logits,
                                              opt.image_size,
                                              ObjectCLS,
                                              model1.anchors,
                                              opt.conf_threshold,
                                              opt.nms_threshold)

                if len(predictions_object) != 0:
	                predictions_relation = post_processing_relations(object_logits,
	                                              opt.image_size,
	                                              ObjectCLS,
	                                              P2ORelCLS,
	                                              model1.anchors,
	                                              opt.conf_threshold,
	                                              opt.nms_threshold)

                else:
                	predictions_relation=[]

                	
                if len(predictions) != 0:
                    predictions = predictions[0]
                    output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                    output_image = cv2.resize(output_image, (width, height))

                    # save images
                    cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                        '.txt', '.jpg'), output_image)

                    for pred in predictions:
                        xmin = int(max(pred[0] / width_ratio, 0))
                        ymin = int(max(pred[1] / height_ratio, 0))
                        xmax = int(min((pred[2]) / width_ratio, width))
                        ymax = int(min((pred[3]) / height_ratio, height))
                        color = colors[PersonCLS.index(pred[5])]

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
                        print("person_pred:{}".format(cat_pred))

                        f.write(cat_pred)

                        print("detected {}".format(
                                save_dir + "{}".format(f_file)))
                    else:
                        print("non-detected {}".format(
                            save_dir + "{}".format(f_file)))
                        f.close()

                #
                if len(predictions_object) != 0:
                    predictions_object = predictions_object[0]
                    output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                    output_image = cv2.resize(output_image, (width, height))

                    # save images
                    cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                        '.txt', '.jpg'), output_image)

                    for pred in predictions_object:
                        xmin = int(max(pred[0] / width_ratio, 0))
                        ymin = int(max(pred[1] / height_ratio, 0))
                        xmax = int(min((pred[2]) / width_ratio, width))
                        ymax = int(min((pred[3]) / height_ratio, height))
                        color = colors[ObjectCLS.index(pred[5])]

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

                        cv2.imwrite(save_obj_dir + "{}".format(f_file),
                                    output_image)

                        # save detection results
                        pred_cls = pred[5]

                        cat_pred = '%s %s %s %s %s %s\n' % (
                            pred_cls,
                            str(pred[4]),
                            str(xmin), str(ymin), str(xmax), str(ymax))

                        print("object_pred:{}".format(cat_pred))

                        f_obj.write(cat_pred)

                        print("detected {}".format(
                                save_obj_dir + "{}".format(f_file)))
                    else:
                        print("non-detected {}".format(
                            save_obj_dir + "{}".format(f_file)))
                        f_obj.close()

                if len(predictions_relation) != 0:
                    predictions_relation = predictions_relation[0]
                    # output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                    # output_image = cv2.resize(output_image, (width, height))

                    # save images
                    # cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                        # '.txt', '.jpg'), output_image)
                    obj_cnt = len(predictions_object)
                    rel_cnt = 0
                    for pred in predictions_relation:
                        if rel_cnt > obj_cnt:
                            continue
                        xmin, ymin, xmax, ymax = rel_gt[rel_cnt]
                        # xmin = int(max(pred[0] / width_ratio, 0))
                        # ymin = int(max(pred[1] / height_ratio, 0))
                        # xmax = int(min((pred[2]) / width_ratio, width))
                        # ymax = int(min((pred[3]) / height_ratio, height))

                        # save detection results
                        pred_cls = pred[5]

                        cat_pred = '%s %s %s %s %s %s\n' % (
                            pred_cls,
                            str(pred[4]),
                            str(xmin), str(ymin), str(xmax), str(ymax))

                        print("relation_pred:{}".format(cat_pred))

                        f_rel.write(cat_pred)

                        print("detected {}".format(
                                save_obj_dir + "{}".format(f_file)))
                        rel_cnt += 1
                    else:
                        print("non-detected {}".format(
                            save_obj_dir + "{}".format(f_file)))
                        f_rel.close()
                #

            except Exception as ex:
                print(ex)

                f.close()
                f_obj.close()
                f_rel.close()
                continue

            # # for windows
            # f.close()
            # f_beh.close()
            # f_face.close()

            if gt_person_cnt == 0:
                if os.path.exists(save_mAP_gt_dir + mAP_file):
                    os.remove(save_mAP_gt_dir + mAP_file)
                if os.path.exists(save_mAP_det_dir + mAP_file):
                    os.remove(save_mAP_det_dir + mAP_file)

            
            if gt_obj_cnt == 0:
                if os.path.exists(save_mAP_gt_obj_dir + mAP_file):
                    os.remove(save_mAP_gt_obj_dir + mAP_file)
                if os.path.exists(save_mAP_det_obj_dir + mAP_file):
                    os.remove(save_mAP_det_obj_dir + mAP_file)
                if os.path.exists(save_mAP_gt_rel_dir + mAP_file):
                    os.remove(save_mAP_gt_rel_dir + mAP_file)
                if os.path.exists(save_mAP_det_rel_dir + mAP_file):
                    os.remove(save_mAP_det_rel_dir + mAP_file)


if __name__ == "__main__":
    test(opt)
