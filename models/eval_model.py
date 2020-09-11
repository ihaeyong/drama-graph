import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from Yolo_v2_pytorch.src.utils import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.anotherMissOh_dataset import AnotherMissOh, Splits, SortFullRect, PersonCLS,PBeHavCLS, FaceCLS, ObjectCLS, P2ORelCLS
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import time

from lib.place_model import place_model, label_mapping, accuracy
from lib.behavior_model import behavior_model
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, flatten
from lib.focal_loss import FocalLossWithOneHot, FocalLossWithOutOneHot, CELossWithOutOneHot
from lib.face_model import face_model
from lib.object_model import object_model
from lib.relation_model import relation_model
from lib.emotion_model import emotion_model, crop_face_emotion, EmoCLS

num_persons = len(PersonCLS)
num_behaviors = len(PBeHavCLS)
num_faces = len(FaceCLS)
num_objects = len(ObjectCLS)
num_relations = len(P2ORelCLS)
num_emos = len(EmoCLS)

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
                        default="./checkpoint/refined_models")

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images_ver3.2/")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_ver3.2/")
    parser.add_argument("-model", dest='model', type=str, default="baseline")
    parser.add_argument("-display", dest='display', action='store_true')
    parser.add_argument("-emo_net_ch", dest='emo_net_ch',type=int, default=64)
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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        device = torch.cuda.current_device()
    else:
        torch.manual_seed(123)
    print(torch.cuda.is_available())

    # set test loader params
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    # set test loader
    test_loader = DataLoader(test_set, **test_params)

    # ---------------(1) load refined models --------------------
    # get the trained models from
    # https://drive.google.com/drive/folders/1WXzP8nfXU4l0cNOtSPX9O1qxYH2m6LIp
    # person and behavior
    if False :
        print("-----------person---behavior-------model---------------")
        model1 = behavior_model(num_persons, num_behaviors, opt, device)
        trained_persons = './checkpoint/refined_models' + os.sep + "{}".format(
        'anotherMissOh_only_params_integration.pth')
        if optimistic_restore(model1, torch.load(trained_persons)):
            #model1.load_state_dict(torch.load(trained_persons))
            print("loaded with {}".format(trained_persons))

    else:
        # pre-trained behavior model
        # step 1: person trained on voc 50 epoch
        # step 2: person feature based behavior sequence learning 100 epoch
        model1 = behavior_model(num_persons, num_behaviors, opt, device)
        trained_persons = './checkpoint/refined_models' + os.sep + "{}".format(
            'anotherMissOh_only_params_integration.pth')
        model1.load_state_dict(torch.load(trained_persons))
        print("loaded with person and behavior model {}".format(trained_persons))
    model1.cuda(device)
    model1.eval()

    # face model
    if True:
        model_face = face_model(num_persons, num_faces, device)
        trained_face = './checkpoint/refined_models' + os.sep + "{}".format(
        'anotherMissOh_only_params_face.pth')
        model_face.load_state_dict(torch.load(trained_face))
        print("loaded with {}".format(trained_face))
    model_face.cuda(device)
    model_face.eval()

    # emotion model
    if True:
        model_emo = emotion_model(opt.emo_net_ch, num_persons, device)
        trained_emotion = './checkpoint/refined_models' + os.sep + "{}".format(
        'anotherMissOh_only_params_emotion_integration.pth')
        model_emo.load_state_dict(torch.load(trained_emotion))
        print("loaded with {}".format(trained_emotion))
    model_emo.cuda(device)
    model_emo.eval()

    # object model
    if True:
        # add model
        model_object = object_model(num_objects)
        trained_object = './checkpoint/refined_models' + os.sep + "{}".format(
        'anotherMissOh_only_params_object_integration.pth')
        # model load
        print("loaded with {}".format(trained_object))
        model_object.load_state_dict(torch.load(trained_object))

    model_object.cuda(device)
    model_object.eval()


    # relation model
    if False:
        # add model
        trained_relation = './checkpoint/refined_models' + os.sep + "{}".format(
        'anotherMissOh_only_params_relation_integration.pth')
        # model load
        print("loaded with {}".format(trained_relation))

    # place model
    if True:
        model_place = place_model(num_persons, num_behaviors, device)
        # add model
        trained_place = './checkpoint/refined_models' + os.sep + "{}".format(
            'anotherMissOh_only_params_place_integration.pth')
        # model load
        print("loaded with {}".format(trained_place))
        model_place.load_state_dict(torch.load(trained_place)['model'])
    model_place.cuda(device)
    model_place.eval()

    # load the color map for detection results
    colors = pickle.load(open("./Yolo_v2_pytorch/src/pallete", "rb"))

    width, height = (1024, 768)
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height
    
    # Sequence buffers
    preds_place = []; preds_frameid = [];
    target_place = []; image_place = []
    temp_images = []; temp_frameid = []; temp_info = []
    # load test clips
    for iter, batch in enumerate(test_loader):
        image, info = batch

        # sort label info on fullrect
        image, label, behavior_label, obj_label, face_label, emo_label, frame_id = SortFullRect(
            image, info, is_train=False)

        try :
            image = torch.cat(image,0).cuda(device)
        except:
            continue

        # -----------------(2) inference -------------------------
        # person and behavior predictions
        # logits : [1, 125, 14, 14]
        # behavior_logits : [1, 135, 14, 14]
        predictions, b_logits = model1(image, label, behavior_label)

        # face
        face_logits = model_face(image)

        predictions_face = post_processing(face_logits,
                                           opt.image_size,
                                           FaceCLS,
                                           model_face.detector.anchors,
                                           opt.conf_threshold,
                                           opt.nms_threshold)

        # emotion
        if np.array(face_label).size > 0 :
            image_c = image.permute(0,2,3,1).cpu()
            face_crops, emo_gt = crop_face_emotion(image_c, face_label, emo_label, opt)
            face_crops, emo_gt = face_crops.cuda(device).contiguous(), emo_gt.cuda(device)
            emo_logits = model_emo(face_crops)
            num_img, num_face = np.array(face_label).shape[0:2]
            emo_logits = emo_logits.view(num_img, num_face, 7)

        # object

        if np.array(obj_label).size > 0 :
            object_logits, _ = model_object(image)

            predictions_object = post_processing(object_logits,
                                                 opt.image_size,
                                                 ObjectCLS,
                                                 model_object.detector.anchors,
                                                 opt.conf_threshold,
                                                 opt.nms_threshold)



        # relation

        # place
        images_norm = []; info_place = [];

        for idx in range(len(image)):
            image_resize = image[idx]
            images_norm.append(image_resize)
            info_place.append(info[0][idx]['place'])
            frame_place = frame_id.copy()
        info_place = label_mapping(info_place)

        pl_updated=False
        while True:
            temp_len = len(temp_images)
            temp_images += images_norm[:(10-temp_len)]; images_norm = images_norm[(10-temp_len):]
            temp_frameid += frame_place[:(10-temp_len)]; frame_place = frame_place[(10-temp_len):]
            temp_info += info_place[:(10-temp_len)]; info_place = info_place[(10-temp_len):]
            temp_len = len(temp_images)
            if temp_len == 10:
                batch_images = (torch.stack(temp_images).cuda(device))
                batch_images = batch_images.unsqueeze(0)
                target = torch.Tensor(temp_info).to(torch.int64).cuda(device)
                output = model_place(batch_images)
                output = torch.cat((output[:, :9], output[:, 10:]), 1) # None excluded
                preds = torch.argmax(output, -1) # (T, n_class) ->(T, ) 
                preds = preds.tolist()
                for idx in range(len(preds)):
                    if preds[idx] >= 9: preds[idx] += 1
                preds_place += preds; preds_frameid += temp_frameid;
                target_place += temp_info; image_place += temp_images
                temp_images = []; temp_info = []; temp_frameid = []
                pl_updated = True
            elif temp_len < 10:
                break
        # Save place classification files
        preds_place_txt = label_remapping(preds_place)
        target_place_txt = label_remapping(target_place)
        if len(preds_place) > 0:
            for idx, frame in enumerate(preds_frameid):
                f_info = frame[0].split('/')
                save_dir = './results/person/{}/{}/{}/'.format(
                    f_info[4], f_info[5], f_info[6])

                save_gt_place_dir = './results/place/ground-truth-place/'
                save_pred_place_dir = './results/place/prediction-place/'
                save_img_place_dir = './results/place/image/'
                if not os.path.exists(save_gt_place_dir):
                    os.makedirs(save_gt_place_dir)
                if not os.path.exists(save_pred_place_dir):
                    os.makedirs(save_pred_place_dir)
                if not os.path.exists(save_img_place_dir):
                    os.makedirs(save_img_place_dir)
                f_file = f_info[7]
                mAP_file = "{}_{}_{}_{}".format(f_info[4],
                                                f_info[5],
                                                f_info[6],
                                                f_info[7].replace("jpg", "txt"))
                f = open(save_gt_place_dir + mAP_file, mode='w+')
                f.write(target_place_txt[idx])
                print('place_gt :', target_place_txt[idx])
                f.close()

                f = open(save_pred_place_dir + mAP_file, mode='w+')
                f.write(preds_place_txt[idx])
                print('place_pred :', preds_place_txt[idx])
                
                # Save place classification visualization images
                try:
                    image_pl = image_place[idx]
                    np_img = image_pl.cpu().numpy().transpose((1,2,0)) * 255

                    output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                    output_image = cv2.resize(output_image, (width, height))

                    cv2.putText(output_image, "place : " + preds_place_txt[idx],
                        (30, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.imwrite(save_img_place_dir + mAP_file.replace(
                        '.txt', '.jpg'), output_image)
                    f.close()
                except:
                    f.close()
        preds_place = []; target_place = []; preds_frameid = []; image_place = []
        
        
        for idx, frame in enumerate(frame_id):

            # ---------------(3) mkdir for evaluations----------------------
            f_info = frame[0].split('/')
            save_dir = './results/person/{}/{}/{}/'.format(
                f_info[4], f_info[5], f_info[6])

            save_mAP_gt_dir = './results/input_person/ground-truth/'
            save_mAP_det_dir = './results/input_person/detection/'

            save_mAP_gt_beh_dir = './results/input_person/ground-truth-behave/'
            save_mAP_det_beh_dir = './results/input_person/detection-behave/'

            # face dir
            save_mAP_gt_face_dir = './results/input_person/ground-truth-face/'
            save_mAP_det_face_dir = './results/input_person/detection-face/'

            save_mAP_img_dir = './results/input_person/image/'

            save_mAP_gt_obj_dir = './results/input_person/ground-truth-object/'
            save_mAP_det_obj_dir = './results/input_person/detection-object/'

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

            # ground-truth face
            if not os.path.exists(save_mAP_gt_face_dir):
                os.makedirs(save_mAP_gt_face_dir)

            # face
            if not os.path.exists(save_mAP_det_face_dir):
                os.makedirs(save_mAP_det_face_dir)

            # object
            if not os.path.exists(save_mAP_gt_obj_dir):
                os.makedirs(save_mAP_gt_obj_dir)

            if not os.path.exists(save_mAP_det_obj_dir):
                os.makedirs(save_mAP_det_obj_dir)

            # image
            if not os.path.exists(save_mAP_img_dir):
                os.makedirs(save_mAP_img_dir)

            f_file = f_info[7]
            mAP_file = "{}_{}_{}_{}".format(f_info[4],
                                            f_info[5],
                                            f_info[6],
                                            f_info[7].replace("jpg", "txt"))
            if opt.display:
                print("frame.__len__{}, mAP_file:{}".format(len(frame_id), mAP_file))

            # --------------(4) ground truth ---------------------------------
            # save person ground truth
            gt_person_cnt = 0
            if len(label) > idx :
                # person
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

                # behavior
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

                # emotion


                # object
            gt_object_cnt = 0
            if len(obj_label) > idx :
                f_obj = open(save_mAP_gt_obj_dir + mAP_file, mode='w+')
                for det in obj_label[idx]:
                    cls = ObjectCLS[int(det[4])]
                    xmin = str(max(det[0] / width_ratio, 0))
                    ymin = str(max(det[1] / height_ratio, 0))
                    xmax = str(min((det[2]) / width_ratio, width))
                    ymax = str(min((det[3]) / height_ratio, height))
                    cat_det = '%s %s %s %s %s\n' % (cls, xmin, ymin, xmax, ymax)
                    if opt.display:
                        print("object_gt:{}".format(cat_det))
                    f_obj.write(cat_det)
                    gt_object_cnt += 1
                f_obj.close()


                # relation


                # place



                # open detection file
                f_beh = open(save_mAP_det_beh_dir + mAP_file, mode='w+')
                f = open(save_mAP_det_dir + mAP_file, mode='w+')
                f_obj = open(save_mAP_det_obj_dir + mAP_file, mode='w+')

            # face
            gt_face_cnt = 0
            if len(face_label) > idx:
                f_face = open(save_mAP_gt_face_dir + mAP_file, mode='w+')
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

                f_face = open(save_mAP_det_face_dir + mAP_file, mode='w+')

            # --------------(5) visualization of inferences ----------
            # out of try : pdb.set_trace = lambda : None
            try:
                # for some empty video clips
                img = image[idx]
                # ToTensor function normalizes image pixel values into [0,1]
                np_img = img.cpu().numpy().transpose((1,2,0)) * 255

                output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                output_image = cv2.resize(output_image, (width, height))

                if len(predictions) != 0 :
                    prediction = predictions[idx]
                    b_logit = b_logits[idx]

                    # save images
                    cv2.imwrite(save_mAP_img_dir + mAP_file.replace(
                        '.txt', '.jpg'), output_image)

                    # person and behavior
                    num_preds = len(prediction)
                    for jdx, pred in enumerate(prediction):
                        # person
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

                        # behavior
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

                        # emotion
                        fl = face_label[idx][jdx]
                        face_x0, face_y0 = int(fl[0]/width_ratio), int(fl[1]/height_ratio)
                        face_x1, face_y1 = int(fl[2]/width_ratio), int(fl[3]/height_ratio)
                        emo_ij = F.softmax(emo_logits[idx,jdx,:], dim=0).argmax().detach().cpu().numpy()
                        emo_txt = EmoCLS[emo_ij]
                        cv2.rectangle(output_image, (face_x0,face_y0),
                                      (face_x1,face_y1), (255,255,0), 1)
                        cv2.putText(output_image, emo_txt, (face_x0, face_y0-5),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1,
                                    cv2.LINE_AA)


                        if opt.display:
                            print("detected {}".format(
                                save_dir + "{}".format(f_file)))
                    else:
                        if opt.display:
                            print("non-detected {}".format(
                            save_dir + "{}".format(f_file)))
                        f.close()
                        f_beh.close()


                        # object
                if len(predictions_object) != 0:
                    prediction_object = predictions_object[0]

                    num_preds = len(prediction)
                    for jdx, pred in enumerate(prediction_object):
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

                        # save detection results
                        pred_cls = pred[5]
                        cat_pred = '%s %s %s %s %s %s\n' % (
                            pred_cls,
                            str(pred[4]),
                            str(xmin), str(ymin), str(xmax), str(ymax))

                        print("object_pred:{}".format(cat_pred))

                        f_obj.write(cat_pred)

                        if opt.display:
                            print("detected {}".format(
                                save_dir + "{}".format(f_file)))
                    else:
                        if opt.display:
                            print("non-detected {}".format(
                            save_dir + "{}".format(f_file)))
                        f_obj.close()

                        # relation

                        # place

                # face
                if len(predictions_face) != 0:

                    prediction_face = predictions_face[idx]

                    for pred in prediction_face:
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

                # save output image
                cv2.imwrite(save_dir + "{}".format(f_file), output_image)
            except:
                f.close()
                f_beh.close()
                f_face.close()
                f_obj.close()

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

            # face
            if gt_face_cnt == 0:
                if os.path.exists(save_mAP_gt_face_dir + mAP_file):
                    os.remove(save_mAP_gt_face_dir + mAP_file)
                if os.path.exists(save_mAP_det_face_dir + mAP_file):
                    os.remove(save_mAP_det_face_dir + mAP_file)

            # object
            if gt_object_cnt == 0:
                if os.path.exists(save_mAP_gt_obj_dir + mAP_file):
                    os.remove(save_mAP_gt_obj_dir + mAP_file)
                if os.path.exists(save_mAP_det_obj_dir + mAP_file):
                    os.remove(save_mAP_det_obj_dir + mAP_file)


if __name__ == "__main__":
    test(opt)

