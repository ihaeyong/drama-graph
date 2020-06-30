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
# import jsonlines
import time
import numpy as np

from lib.place_model import place_model
import lib.folder_txt as folder
import lib.resnet as resnet
import lib.transform as transform

# epochs -> num_epoches
# dataroot -> img_path
# workers -> new
# batchSize -> batch_size
# imageSize -> image_size
# load_label (file path) -> json_path (folder)
# resnet_checkpoint -> trained_fe_path (folder)
# clsf_checkpoint (for eval mode) ->  x 
# lr -> lr
# print_freq -> print_interval
# file_length -> new
# valid -> x
# logger?

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int,
                        default=224,
                        help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=3*10,
                        help="The number of images per batch")

    # Training base Setting
    # parser.add_argument("--momentum", type=float, default=0.9)
    # parser.add_argument("--decay", type=float, default=0.0005)
    # parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=10)
    # parser.add_argument("--test_interval", type=int, default=1,
    #                     help="Number of epoches between testing phases")
    # parser.add_argument("--object_scale", type=float, default=1.0)
    # parser.add_argument("--noobject_scale", type=float, default=0.5)
    # parser.add_argument("--class_scale", type=float, default=1.0)
    # parser.add_argument("--coord_scale", type=float, default=5.0)
    # parser.add_argument("--reduction", type=int, default=32)
    # parser.add_argument("--es_min_delta", type=float, default=0.0,
                        # help="Early stopping's parameter:minimum change loss to qualify as an improvement")
    # parser.add_argument("--es_patience", type=int, default=0,
                        # help="Early stopping's parameter:number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")

    # parser.add_argument("--pre_trained_model_type",
                        # type=str, choices=["model", "params"],
                        # default="model")
    parser.add_argument("--trained_fe_path", type=str,
                        default="./checkpoint/resnet") # Pre-training path

    parser.add_argument("--saved_path", type=str,
                        default="./checkpoint/clsf") # saved training path
    # parser.add_argument("--conf_threshold", type=float, default=0.35)
    # parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--img_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_images")
    parser.add_argument("--json_path", type=str,
                        default="./data/AnotherMissOh/AnotherMissOh_Visual_full.json")
    # parser.add_argument("-model", dest='model', type=str, default="baseline")
    parser.add_argument("-lr", dest='lr', type=float, default=1e-2)
    # parser.add_argument("-clip", dest='clip', type=float, default=5.0)
    parser.add_argument("-print_interval", dest='print_interval', type=int,
                        default=10)
    # added
    parser.add_argument('--file_length', type=int, default=10)
    parser.add_argument('--workers', type=int, help='number of data loading workers: 0~4', default=3) 
    
    args = parser.parse_args()
    return args

# get args.
args = get_args()
print(args)



class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch > 1:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #lr = args.lr
    #lr = args.lr * (0.1 ** (epoch // 30))
    #for param_group in optimizer.param_groups:
    #   param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy2(output, target, topk=(1,)):
    preds = np.zeros((23, ))
    tar = np.zeros((23, ))
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        target = target.view(1, -1)
        for i in range(23):
            tar[i] = (target==i).sum()
            preds[i] = correct[target==i].sum()
        return preds, tar

def accuracy_classwise(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        correct_class = []
        for i in range(22):
            correct = correct[(target.view(1, -1) == i)]
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    topp = AverageMeter('Acc_sc@1', ':6.6f')
    topn = AverageMeter('Acc_nsc@1', ':6.6f')
    top1 = AverageMeter('Acc@1', ':6.6f')
    top5 = AverageMeter('Acc@5', ':6.6f')
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
        target = target.reshape(target.size(0)*target.size(1), -1).squeeze(-1).to(torch.int64)
        output2 = model(images) # N * T * C
        loss = F.cross_entropy(output2, target)

        prec1 = []
        prec5 = []
        prec1_tmp, prec5_tmp = accuracy(output2, target, topk=(1, 5))
        prec1.append(prec1_tmp.view(1, -1))
        prec5.append(prec5_tmp.view(1, -1))
        prec1 = torch.stack(prec1)
        prec5 = torch.stack(prec5)
        prec1 = prec1.view(-1).float().mean(0)
        prec5 = prec5.view(-1).float().mean(0)

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0:
            progress.display(i)


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

    
    trainset = folder.ImageFolder(root='{}'.format(args.img_path), 
        transform = transform_train, json_label_file = args.json_path, file_length=args.file_length, train=True)

    trainset.class_to_idx[''] = 9


    print(trainset.class_to_idx)

    print("elapsed time 1 : ", time.time() - tp)
    tp = time.time()
     # For training
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
    #                                          shuffle= False, num_workers=args.workers) # For testing
    checkpoint = torch.load(os.path.join(args.trained_fe_path, 'resnet50_places365.pth.tar'))
    # print("checkpoint load complete")
    print("loaded pre-trained resnet sucessfully.")

    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

    fe = resnet.resnet50()
    fe.load_state_dict(state_dict, False)

    model = torch.nn.Sequential(fe, place_model())

    model = torch.nn.DataParallel(model).cuda()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1, 3, 5], gamma=0.1, last_epoch=-1)

    print("elapsed time 2 : ", time.time() - tp)
    tp = time.time()


    for epoch in range(args.num_epoches): # MJJ
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)
        train(train_loader, model, optimizer, scheduler, epoch, args)
        scheduler.step()

        print("SAVE MODEL")
        
        if not os.path.exists(args.saved_path):
            os.makedirs(args.saved_path)
            print('mkdir_{}'.format(args.saved_path))

        torch.save({
                    #'val_loss' : val_loss,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, os.path.join(args.saved_path, '{}_lstm_load2.pt'.format(epoch)))






if __name__ == '__main__':
    main()



