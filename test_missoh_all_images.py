import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.yolo_net import Yolo
from PIL import Image
import time

MissOh_CLASSES = ['person']

def get_args():
    parser = argparse.ArgumentParser(
        "You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size",
                        type=int, default=448,
                        help="The common width and height for all images")
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
    args = parser.parse_args()
    return args

def test(opt):
    global colors
    #import pdb; pdb.set_trace()
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model1 = torch.load(opt.pre_trained_model_path)
        else:
            model1 = Yolo(1)
            model1.load_state_dict(torch.load(opt.pre_trained_model_path))

    colors = pickle.load(open("./Yolo_v2_pytorch/src/pallete", "rb"))

    model1.eval()

    img_list = sorted(glob.glob(os.path.join(opt.data_path_test, '*.jpg')))

    print(img_list)

    save_dir = './Yolo_v2_pytorch/anotherMissOh_Test_Result/'

    for idx,idx_img in enumerate(img_list):
        image = Image.open(idx_img)
        np_img = np.asarray(image)
        image = cv2.cvtColor(np.float32(np_img), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height

        data = Variable(torch.FloatTensor(image))

        if torch.cuda.is_available():
            data = data.cuda()

        with torch.no_grad():
            logits = model1(data)
            predictions = post_processing(logits, opt.image_size, MissOh_CLASSES, model1.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) != 0:
            predictions = predictions[0]
            output_image = cv2.cvtColor(np.float32(np_img), cv2.COLOR_RGB2BGR)
            for pred in predictions:
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
                ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
                color = colors[MissOh_CLASSES.index(pred[5])]

                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4),
                              color, -1)
                cv2.putText(
                    output_image, pred[5] + ' : %.2f' % pred[4],
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
            cv2.imwrite(save_dir + "entire_{}.png".format(idx), output_image)

if __name__ == "__main__":
    opt = get_args()
    test(opt)

