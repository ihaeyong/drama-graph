import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from Yolo_v2_pytorch.src.utils import *
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS, PBeHavCLS
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import time
import shutil

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
            model = torch.load(model_path)
            print("loaded with gpu {}".format(model_path))
        else:
            model = Yolo(num_persons)
            model.load_state_dict(torch.load(model_path))
            print("loaded with cpu {}".format(model_path))

    # load the color map for detection results
    colors = pickle.load(open("./Yolo_v2_pytorch/src/pallete", "rb"))

    model.eval()
    width, height = (1024, 768)
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height

    # load test clips
    for iter, batch in enumerate(test_loader):
        frames, info = batch

        # sort label info on fullrect
        frames, labels, behavior_labels, frame_ids = SortFullRect(
            frames, info, is_train=False)

        if torch.cuda.is_available():
            frames = torch.cat(frames, 0).cuda()

        #with torch.no_grad():
        # logits : [1, 125, 14, 14]
        # behavior_logits : [1, 135, 14, 14]
        pred_p_bboxes_list, pred_b_logits_list = model(frames, labels, behavior_labels)
        assert len(pred_p_bboxes_list) == len(pred_b_logits_list)
        if len(pred_p_bboxes_list) == 0:
            continue

        for frame_idx, frame_id in enumerate(frame_ids):
            f_info = frame_id[0].split('/')
            save_dir = './results/person/{}/{}/{}/'.format(
                f_info[4], f_info[5], f_info[6])

            save_mAP_gt_dir = './results/input_person/ground-truth/'
            save_mAP_det_dir = './results/input_person/detection/'

            save_mAP_gt_beh_dir = './results/input_person/ground-truth-behave/'
            save_mAP_det_beh_dir = './results/input_person/detection-behave/'

            save_mAP_img_dir = './results/input_person/image/'

            save_behavior_img_dpath = "./results/behavior"

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
            if not os.path.exists(save_behavior_img_dpath):
                os.makedirs(save_behavior_img_dpath)
            # image
            if not os.path.exists(save_mAP_img_dir):
                os.makedirs(save_mAP_img_dir)

            f_file = f_info[7]
            mAP_file = "{}_{}_{}_{}".format(f_info[4],
                                            f_info[5],
                                            f_info[6],
                                            f_info[7].replace("jpg", "txt"))
            behavior_img_fname = '_'.join([ f_info[4], f_info[5], f_info[6], f_info[7] ])
            ### img_file = '_'.join([ f_info[4], f_info[5], f_info[6], f_info[7] ])
            ### img_fpath = os.path.join(opt.img_path, f_info[4], f_info[5], f_info[6], f_info[7])
            if opt.display:
                print("mAP_file:{}".format(mAP_file))

            # ground truth
            #b_person_label = labels[i]
            # save person ground truth
            gt_person_cnt = 0
            ### if len(labels) > frame_idx:
            with open(save_mAP_gt_dir + mAP_file, mode='w+') as fin:
                for det in labels[frame_idx]:
                    cls = PersonCLS[int(det[4])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    if opt.display:
                        print("person_gt:{}".format(cat_det))
                    fin.write(cat_det)
                    gt_person_cnt += 1

            #b_behavior_labels = behavior_labels[i]
            with open(save_mAP_gt_beh_dir + mAP_file, mode='w+') as fin:
                for j, det in enumerate(labels[frame_idx]):
                    cls = PBeHavCLS[int(behavior_labels[frame_idx][j])].replace(' ', '_')
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
                    fin.write(cat_det)


            # open detection file
            f_beh = open(save_mAP_det_beh_dir + mAP_file, mode='w+')
            f = open(save_mAP_det_dir + mAP_file, mode='w+')

            # out of try : pdb.set_trace = lambda : None
            ### try:
            frame = frames[frame_idx]
            np_frame = frame.cpu().numpy().transpose((1, 2, 0)) * 255
            if len(pred_p_bboxes_list[frame_idx]) > 0:
                pred_p_bboxes = pred_p_bboxes_list[frame_idx]
                pred_b_logits = pred_b_logits_list[frame_idx]
                output_image = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                output_image = cv2.resize(output_image, (width, height))

                # save images
                cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                    '.txt', '.jpg'), output_image)

                assert len(pred_p_bboxes) == len(pred_b_logits)
                for pred_bbox, pred_b_logit in zip(pred_p_bboxes, pred_b_logits):
                    xmin = int(max(pred_bbox[0] / width_ratio, 0))
                    ymin = int(max(pred_bbox[1] / height_ratio, 0))
                    xmax = int(min((pred_bbox[2]) / width_ratio, width))
                    ymax = int(min((pred_bbox[3]) / height_ratio, height))
                    color = colors[PersonCLS.index(pred_bbox[5])]

                    cv2.rectangle(output_image, (xmin, ymin),
                                  (xmax, ymax), color, 2)

                    ### value, index = pred_b_logit.max(0)
                    pred_b_weight = torch.softmax(pred_b_logit, dim=0)
                    ( prct1, prct2 ), ( idx1, idx2 ) = pred_b_weight.topk(2, dim=0)
                    b_idx1 = idx1.cpu().numpy()
                    b_idx2 = idx2.cpu().numpy()
                    b_pred1 = PBeHavCLS[b_idx1]
                    b_pred2 = PBeHavCLS[b_idx2]
                    text_size = cv2.getTextSize(
                        pred_bbox[5] + ' : %.2f' % pred_bbox[4],
                        cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(
                        output_image,
                        (xmin, ymin),
                        (xmin + text_size[0] + 100,
                         ymin + text_size[1] + 20 + 12), color, -1)
                    cv2.putText(
                        output_image, pred_bbox[5] + ' : %.2f' % pred_bbox[4],
                        (xmin, ymin + text_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)
                    cv2.putText(
                        output_image, f'+ behavior : {b_pred1} ({prct1 * 100:.1f}%)',
                        (xmin, ymin + text_size[1] + 4 + 12),
                        cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)
                    cv2.putText(
                        output_image, f'+ behavior : {b_pred2} ({prct2 * 100:.1f}%)',
                        (xmin, ymin + text_size[1] + 4 + 12 + 12),
                        cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)


                    cv2.imwrite(save_dir + "{}".format(f_file),
                                output_image)

                    behavior_img_fpath = os.path.join(save_behavior_img_dpath,
                                                      b_pred1,
                                                      behavior_img_fname)
                    os.makedirs(os.path.dirname(behavior_img_fpath), exist_ok=True)
                    cv2.imwrite(behavior_img_fpath,
                                output_image)

                    # save detection results
                    pred_cls = pred_bbox[5]
                    pred_beh_cls = b_pred1.replace(' ', '_')
                    pred_beh_cls = pred_beh_cls.replace('/', '_')
                    cat_pred = '%s %s %s %s %s %s\n' % (
                        pred_cls,
                        str(pred_bbox[4]),
                        str(xmin), str(ymin), str(xmax), str(ymax))

                    cat_pred_beh = '%s %s %s %s %s %s\n' % (
                        pred_beh_cls,
                        str(pred_bbox[4]),
                        str(xmin), str(ymin), str(xmax), str(ymax))

                    print("behavior_pred:{}".format(cat_pred_beh))
                    print("person_pred:{}".format(cat_pred))

                    f.write(cat_pred)
                    f_beh.write(cat_pred_beh)
                    ### dest_fpath = os.path.join(save_mAP_det_beh_dir, 'images', pred_beh_cls, img_file)
                    ### dest_dpath = os.path.dirname(dest_fpath)
                    ### os.makedirs(dest_dpath, exist_ok=True)
                    ### shutil.copy(img_fpath, dest_fpath)

                    if opt.display:
                        print("detected {}".format(
                            save_dir + "{}".format(f_file)))
                else:
                    if opt.display:
                        print("non-detected {}".format(
                        save_dir + "{}".format(f_file)))
                    f.close()
                    f_beh.close()
            ### except:
            ###     f.close()
            ###     f_beh.close()
            ###     continue
            f.close()
            f_beh.close()
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

