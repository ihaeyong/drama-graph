from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from lib.place_model import place_model
import lib.folder_txt as folder
import lib.resnet as resnet
import lib.transform as transform

# num_persons = len(PersonCLS)
# num_behaviors = len(PBeHavCLS)

def get_args():
    parser = argparse.ArgumentParser(
        "VTT Place Recognition")
    parser.add_argument("--image_size",
                        type=int, default=224,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The number of images per batch")
    parser.add_argument("--trained_fe_path", type=str,
                        default="../checkpoint/resnet") # Pre-training path

    parser.add_argument("--saved_path", type=str,
                        default="../checkpoint/clsf") # saved training path

    parser.add_argument("--img_path", type=str,
                        default="../../data/AnotherMissOh/AnotherMissOh_images") #/AnotherMissOh
    parser.add_argument("--json_path", type=str,
                        default="../../data/AnotherMissOh/AnotherMissOh_Visual_full.json")
    parser.add_argument("-model", dest='model', type=str, default='9_lstm_load2')


    parser.add_argument('--file_length', type=int, default=10)
    parser.add_argument('--workers', type=int, help='number of data loading workers: 0~4', default=3) 

    args = parser.parse_args()
    return args

# get args.
args = get_args()
print(args)



def validate(val_loader, model, args):
    # losses = AverageMeter('Loss', ':.4e')
    # topp = AverageMeter('Acc_sc@1', ':6.6')
    # topn = AverageMeter('Acc_nsc@1', ':6.6f')
    # top1 = AverageMeter('Acc@1', ':6.6f')
    # top5 = AverageMeter('Acc@5', ':6.6f')
    # progress = ProgressMeter(
    #     len(val_loader), [losses, top1, top5],
    #     prefix='Test: ')

    model.eval()

    # classwise = []

    place_list = list(args.class_to_idx.keys()) # in order
    del place_list[0] # delete blank

    frame_ids = list(args.id_class.keys())
    count = 0

    with torch.no_grad():
        end = time.time()
        preds_total = np.zeros((23, ))
        tar_total = np.zeros((23, ))
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            target = target.reshape(target.size(0)*target.size(1), -1).squeeze(-1).to(torch.int64)
            output2 = model(images) # (N * T) * C
            loss = F.cross_entropy(output2, target)
            # print(output2.shape) # ((N*T), n_class)
            # print(images.shape) # (N=1, T, C, H, W)

            preds = torch.argmax(output2, -1) # (T, n_class) ->(T, ) 
            preds = preds.tolist()

            # print(len(preds), images.shape) 
            for j in range(len(preds)):
                frame_id = frame_ids[count]
                # print(frame_id)
                f_info = frame_id.split('_')  # frame_id.split('_') # ['AnotherMissOh01', '015', '0684', 'IMAGE', '0000047237']
                save_dir = '../results/place/{}/{}/{}/'.format(f_info[0], f_info[1], f_info[2])#'_'.join(f_info[-2:]))

                save_gt_dir = '../results/input_place/ground-truth/' # ?
                save_pred_dir = '../results/input_place/prediction/' # ?

                # visualize predictions
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # ground-truth
                if not os.path.exists(save_gt_dir):
                    os.makedirs(save_gt_dir)
                # detection
                if not os.path.exists(save_pred_dir):
                    os.makedirs(save_pred_dir)


                img_file = '_'.join(f_info[-2:]) + '.jpg' # f_info[3]+f_info[4]

                txt_file = "{}_{}_{}_{}".format(f_info[0],
                                f_info[1],
                                f_info[2],
                                img_file.replace("jpg", "txt"))

                # print("txt_file:{}".format(txt_file))

                # visualize prediction
                img = images.squeeze(0)[j] # (1, (1*T), C, H, W) -> (C, H, W)
                np_img = img.cpu().numpy().transpose((1,2,0)) * 255
                output_image = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
                # output_image = np_img
                output_image = cv2.resize(output_image, (1024, 768))

                # print(args.id_class[frame_id])

                place_gt = place_list[args.id_class[frame_id]]
                place_pred = place_list[preds[j]]

                cv2.putText(output_image, "place : " + place_pred,
                            (20, 20),
                            cv2.FONT_HERSHEY_PLAIN, 2,
                            (255,255,255), 2)

                cv2.imwrite(save_dir + "{}".format(img_file),
                            output_image)


                print("txt_file:{} | place_gt:{:20} | place_pred:{:20}".format(
                    txt_file, place_gt, place_pred), end='\r') # HSE

                # write gt and prediction
                f_gt = open(save_gt_dir + txt_file, mode='w')
                f_pred = open(save_pred_dir + txt_file, mode='w')

                gt_str = '%s\n' % (place_gt)
                pred_str = '%s\n' % (place_pred)

                # print("place_gt:{}".format(gt_str))
                # print("place_pred:{}".format(gt_str))


                f_gt.write(gt_str)
                f_gt.close()
                f_pred.write(pred_str)
                f_pred.close()

            count += 1

            

        #     prec1 = []
        #     prec5 = []
        #         # prec1_tmp, prec5_tmp = accuracy(output2[:,:,depth].data, target[:,depth], topk=(1, 5))
            
        #     class_pred, class_tar = accuracy2(output2, target)
        #     prec1_tmp, prec5_tmp = accuracy(output2, target, topk=(1, 5))
        #     prec1.append(prec1_tmp.view(1, -1))
        #     prec5.append(prec5_tmp.view(1, -1))
        #     prec1 = torch.stack(prec1)
        #     prec5 = torch.stack(prec5)
        #     prec1 = prec1.view(-1).float().mean(0)
        #     prec5 = prec5.view(-1).float().mean(0)

        #     losses.update(loss.item(), images.size(0))
        #     top1.update(prec1.item(), images.size(0))
        #     top5.update(prec5.item(), images.size(0))
        #     if i % args.print_freq == 0:
        #         progress.display(i)
        #     preds_total += class_pred
        #     tar_total += class_tar
        # print(preds_total)
        # print(tar_total)
        # print(np.divide(preds_total, tar_total))






def main():

    tp = time.time()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose(
        [
        
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        
        ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    testset = folder.ImageFolder(root='{}'.format(args.img_path), 
        transform = transform_test, json_label_file = args.json_path, file_length=args.file_length, train=False) # transform_train?

    testset.class_to_idx[''] = 9

    print(testset.class_to_idx) # already sorted

    args.id_class = testset.id_class # {frame_id: class}'s # HSE
    args.class_to_idx = testset.class_to_idx

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)


    

    

    fe = resnet.resnet50()
    model = torch.nn.Sequential(fe, place_model())

    # load model
    model_path = os.path.join(args.saved_path, args.model+'.pt')
    checkpoint2 = torch.load(model_path)
    state_dict2 = {str.replace(k,'module.',''): v for k,v in checkpoint2['model'].items()}
    model.load_state_dict(state_dict2, False)
    print("loaded pre-trained resnet sucessfully.")
    print("loaded with gpu {}".format(model_path))

    model = torch.nn.DataParallel(model).cuda()

    validate(test_loader, model, args)
    exit()




if __name__ == '__main__':
    main()


