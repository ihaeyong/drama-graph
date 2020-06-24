"""
modified by haeyong.kang
"""
import math
import torch
import torch.nn as nn
import warnings
import numpy as np
import pdb

warnings.filterwarnings("ignore", category=UserWarning)

class YoloLoss(nn.modules.loss._Loss):
    # The loss I borrow from LightNet repo.
    def __init__(self, num_classes, num_behaviors, num_objects, anchors, reduction=32,
                 coord_scale=1.0, noobject_scale=1.0,
                 object_scale=5.0, class_scale=1.0, thresh=0.6):

        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_behaviors = num_behaviors
        self.num_objects = num_objects

        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale

        self.class_scale = class_scale
        self.class_behavior_scale = class_scale

        self.thresh = thresh

        # define loss functions
        self.mse = nn.MSELoss(size_average=False)
        self.ce = nn.CrossEntropyLoss(size_average=False)

        # display labels
        self.debug = True

    def forward(self, output, output_behavior, output_object, target, target_behavior, target_object):

        # output : [b, 125, 14, 14]
        batch, channel, height, width = output.size()
        batch_obj, channel_obj, height_obj, width_obj = output_object.size()

        # --------- Get x,y,w,h,conf,cls----------------
        # output : [b, 5, 25, 196]
        output = output.view(batch, self.num_anchors, -1, height * width)
        output_object = output_object.view(batch_obj, self.num_anchors, -1, height_obj * width_obj)
        # output_behavior : [b, 5, 25, 196]
        output_behavior = output_behavior.view(
            batch, self.num_anchors, -1, height * width)

        # coord : [b, 5, 4, 196]
        coord = torch.zeros_like(output[:, :, :4, :])
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]

        coord_obj = torch.zeros_like(output_object[:,:,:4,:])
        coord_obj[:,:,:2,:] = output_object[:,:,:2,:].sigmoid()
        coord_obj[:,:,2:4,:] = output_object[:,:,2:4,:]

        # conf : [b, 5, 196]
        conf = output[:, :, 4, :].sigmoid()
        conf_obj = output_object[:,:,4,:].sigmoid()

        # cls : [b * 5, 20, 196]
        # cls : [7840, 20] = [batch * num_anchors * height * width, num_cls]
        cls = output[:, :, 5:, :].contiguous().view(
            batch * self.num_anchors, self.num_classes,
            height * width).transpose(1, 2).contiguous().view(
                -1,self.num_classes)
        cls_obj = output_object[:, :, 5:, :].contiguous().view(
            batch_obj * self.num_anchors, self.num_objects,
            height_obj * width_obj).transpose(1, 2).contiguous().view(
                -1,self.num_objects)
        # behavior_cls : [7840, 25]
        cls_behavior = output_behavior.contiguous().view(
            batch * self.num_anchors, self.num_behaviors,
            height * width).transpose(1, 2).contiguous().view(
                -1,self.num_behaviors)

        # -------- Create prediction boxes--------------
        # pred_boxes : [7840, 4]
        pred_boxes = torch.FloatTensor(
            batch * self.num_anchors * height * width, 4)
        pred_boxes_obj = torch.FloatTensor(
            batch_obj * self.num_anchors * height_obj * width_obj, 4)
        # lin_x, y : [196]
        lin_x = torch.range(0, width - 1).repeat(
            height, 1).view(height * width)
        lin_y = torch.range(0, height - 1).repeat(
            width, 1).t().contiguous().view(height * width)

        lin_x_obj = torch.range(0, width_obj - 1).repeat(
            height_obj, 1).view(height_obj * width_obj)
        lin_y_obj = torch.range(0, height_obj - 1).repeat(
            width_obj, 1).t().contiguous().view(height_obj * width_obj)

        # anchor_w, h : [5, 1]
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)
        
        anchor_w_obj = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h_obj = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        if torch.cuda.is_available():
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()
            pred_boxes_obj = pred_boxes_obj.cuda()
            lin_x_obj = lin_x_obj.cuda()
            lin_y_obj = lin_y_obj.cuda()
            anchor_w_obj = anchor_w_obj.cuda()
            anchor_h_obj = anchor_h_obj.cuda()

        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()

        pred_boxes_obj[:, 0] = (coord_obj[:, :, 0].detach() + lin_x_obj).view(-1)
        pred_boxes_obj[:, 1] = (coord_obj[:, :, 1].detach() + lin_y_obj).view(-1)
        pred_boxes_obj[:, 2] = (coord_obj[:, :, 2].detach().exp() * anchor_w_obj).view(-1)
        pred_boxes_obj[:, 3] = (coord_obj[:, :, 3].detach().exp() * anchor_h_obj).view(-1)
        pred_boxes_obj = pred_boxes_obj.cpu()

        # --------- Get target values ------------------
        if np.array(target).size != 0:
            coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls, tcls_behavior = self.build_targets(pred_boxes, target, target_behavior, height, width)

            # coord_mask : [b, 5, 4, 196]
            coord_mask = coord_mask.expand_as(tcoord)

            # tcls : [16], cls_mask : [b, 5, 196]
            tcls_person = tcls[cls_mask].view(-1).long()

            # tcls_behavior : [16]
            tcls_behavior = tcls_behavior[cls_mask].view(-1).long()

            # cls_mask : [7840, 20]
            cls_person_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)
            cls_behavior_mask = cls_mask.view(-1, 1).repeat(1, self.num_behaviors)

            if torch.cuda.is_available():
                tcoord = tcoord.cuda()
                tconf = tconf.cuda()
                coord_mask = coord_mask.cuda()
                conf_mask = conf_mask.cuda()
                tcls_person = tcls_person.cuda()
                cls_person_mask = cls_person_mask.cuda()
                tcls_behavior = tcls_behavior.cuda()

            conf_mask = conf_mask.sqrt()
            cls_person = cls[cls_person_mask].view(-1, self.num_classes)
            cls_behavior = cls_behavior[cls_behavior_mask].view(-1, self.num_behaviors)

            # --------- Compute losses --------------------
            # losses for person detection coordinates
            self.loss_coord = self.coord_scale * self.mse(
                coord * coord_mask, tcoord * coord_mask) / batch

            # losses for person detection confidence
            self.loss_conf = self.mse(conf * conf_mask, tconf * conf_mask) / batch

            # losses for person detection
            if self.debug:
                print("tcls_person:{}".format(tcls_person))
            self.loss_cls = self.class_scale * 2 * self.ce(
                cls_person, tcls_person) / batch

            # losses for behavior
            if self.debug:
                print("tcls_behavior:{}".format(tcls_behavior))
            self.loss_behavior_cls = self.class_behavior_scale * 2 * self.ce(
                cls_behavior, tcls_behavior) / batch

            # total losses
            self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
            self.loss_tot += self.loss_behavior_cls
        else:
            self.loss_tot = torch.tensor(0, dtype=torch.float).cuda()
            self.loss_coord = torch.tensor(0, dtype=torch.float).cuda()
            self.loss_conf = torch.tensor(0, dtype=torch.float).cuda()
            self.loss_cls = torch.tensor(0, dtype=torch.float).cuda()
            self.loss_behavior_cls = torch.tensor(0, dtype=torch.float).cuda()

        if np.array(target_object).size != 0:
            coord_mask_obj, conf_mask_obj, cls_mask_obj, tcoord_obj, tconf_obj, tcls_obj = self.build_object_targets(pred_boxes_obj, target_object, height_obj, width_obj)
            # coord_mask : [b, 5, 4, 196]
            coord_mask_obj = coord_mask_obj.expand_as(tcoord_obj)

            # tcls : [16], cls_mask : [b, 5, 196]
            tcls_person_obj = tcls_obj[cls_mask_obj].view(-1).long()

            # cls_mask : [7840, 20]
            cls_person_mask_obj = cls_mask_obj.view(-1, 1).repeat(1, self.num_objects)

            if torch.cuda.is_available():
                tcoord_obj = tcoord_obj.cuda()
                tconf_obj = tconf_obj.cuda()
                coord_mask_obj = coord_mask_obj.cuda()
                conf_mask_obj = conf_mask_obj.cuda()
                tcls_person_obj = tcls_person_obj.cuda()
                cls_person_mask_obj = cls_person_mask_obj.cuda()

            conf_mask_obj = conf_mask_obj.sqrt()
            cls_person_obj = cls_obj[cls_person_mask_obj].view(-1, self.num_objects)

            # --------- Compute losses --------------------
            # losses for person detection coordinates
            self.loss_coord_obj = self.coord_scale * self.mse(
                coord_obj * coord_mask_obj, tcoord_obj * coord_mask_obj) / batch_obj

            # losses for person detection confidence
            self.loss_conf_obj = self.mse(conf_obj * conf_mask_obj, tconf_obj * conf_mask_obj) / batch_obj

            # losses for person detection
            if self.debug:
                print("tcls_person:{}".format(tcls_person_obj))
            self.loss_cls_obj = self.class_scale * 2 * self.ce(
                cls_person_obj, tcls_person_obj) / batch_obj

            # total losses
            self.loss_tot_obj = self.loss_coord_obj + self.loss_conf_obj + self.loss_cls_obj
        else:
            self.loss_tot_obj = torch.tensor(0, dtype=torch.float).cuda()
            self.loss_coord_obj = torch.tensor(0, dtype=torch.float).cuda()
            self.loss_conf_obj = torch.tensor(0, dtype=torch.float).cuda()
            self.loss_cls_obj = torch.tensor(0, dtype=torch.float).cuda()

        self.loss_tot += self.loss_tot_obj

        return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_cls, self.loss_behavior_cls, self.loss_coord_obj, self.loss_conf_obj, self.loss_cls_obj

    def build_targets(self, pred_boxes, ground_truth, ground_truth_behavior,
                      height, width):

        # pred_boxes : [7840, 4]
        # ground_truth : [b, 5]
        # ground_truth_behavior : [b]
        # height : 14
        # width : 14

        # batch : [8]
        batch = len(ground_truth)

        # conf_mask : [b, 5, 196]
        conf_mask = torch.ones(
            batch, self.num_anchors, height * width,
            requires_grad=False) * self.noobject_scale

        # coord_mask : [b, 5, 1, 196]
        coord_mask = torch.zeros(
            batch, self.num_anchors, 1, height * width,
            requires_grad=False)

        # cls_mask : [b,5,196]
        cls_mask = torch.zeros(
            batch, self.num_anchors, height * width,
            requires_grad=False).byte()

        # tcoord : [b, 5, 4, 196]
        tcoord = torch.zeros(
            batch, self.num_anchors, 4, height * width,
            requires_grad=False)

        # tconf : [b, 5, 196]
        tconf = torch.zeros(
            batch, self.num_anchors, height * width,
            requires_grad=False)

        # tcls : [b, 5, 196]
        tcls = torch.zeros(
            batch, self.num_anchors, height * width,
            requires_grad=False)

        # tcls_behavior : [b, 5, 196]
        tcls_behavior = torch.zeros(
            batch, self.num_anchors, height * width,
            requires_grad=False)

        for b in range(batch):
            if len(ground_truth[b]) == 0:
                continue

            # ------- Build up tensors --------------------------------
            # cur_pred_boxes : [980, 4]
            cur_pred_boxes = pred_boxes[b * (self.num_anchors * height * width):(
                                 b + 1) * (self.num_anchors * height * width)]

            # anchors : [5, 4]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat(
                    [torch.zeros_like(self.anchors), self.anchors], 1)
            # gt : [:, 4]
            gt = torch.zeros(len(ground_truth[b]), 4)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
                gt[i, 2] = anno[2] / self.reduction
                gt[i, 3] = anno[3] / self.reduction

            # ------ Set confidence mask of matching detections to 0
            # iou_gt_pred : [:, 980]
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)

            # mask : [:, 980]
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            # conf_mask[b] : [5, 196]
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # ------ Find best anchor for each ground truth -------------
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # ------ Set masks and target values for each ground truth --
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                cls_mask[b][best_n][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = self.object_scale
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(
                    max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(
                    max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                tconf[b][best_n][gj * width + gi] = iou
                tcls[b][best_n][gj * width + gi] = int(anno[4])

                # empty labels should be considered as backgrounds
                if ground_truth_behavior[b][i] is not None:
                    anno_behavior = int(ground_truth_behavior[b][i])
                    tcls_behavior[b][best_n][gj * width + gi] = anno_behavior
                else:
                    tcls_behavior[b][best_n][gj * width + gi] = int(
                        self.num_behaviors-1)



        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls, tcls_behavior


    def build_object_targets(self, pred_boxes, ground_truth,
                      height, width):

        # pred_boxes : [7840, 4]
        # ground_truth : [b, 5]
        # ground_truth_behavior : [b]
        # height : 14
        # width : 14

        # batch : [8]
        batch = len(ground_truth)

        # conf_mask : [b, 5, 196]
        conf_mask = torch.ones(
            batch, self.num_anchors, height * width,
            requires_grad=False) * self.noobject_scale

        # coord_mask : [b, 5, 1, 196]
        coord_mask = torch.zeros(
            batch, self.num_anchors, 1, height * width,
            requires_grad=False)

        # cls_mask : [b,5,196]
        cls_mask = torch.zeros(
            batch, self.num_anchors, height * width,
            requires_grad=False).byte()

        # tcoord : [b, 5, 4, 196]
        tcoord = torch.zeros(
            batch, self.num_anchors, 4, height * width,
            requires_grad=False)

        # tconf : [b, 5, 196]
        tconf = torch.zeros(
            batch, self.num_anchors, height * width,
            requires_grad=False)

        # tcls : [b, 5, 196]
        tcls = torch.zeros(
            batch, self.num_anchors, height * width,
            requires_grad=False)

        for b in range(batch):
            if len(ground_truth[b]) == 0:
                continue

            # ------- Build up tensors --------------------------------
            # cur_pred_boxes : [980, 4]
            cur_pred_boxes = pred_boxes[b * (self.num_anchors * height * width):(
                                 b + 1) * (self.num_anchors * height * width)]

            # anchors : [5, 4]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat(
                    [torch.zeros_like(self.anchors), self.anchors], 1)
            # gt : [:, 4]
            gt = torch.zeros(len(ground_truth[b]), 4)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
                gt[i, 2] = anno[2] / self.reduction
                gt[i, 3] = anno[3] / self.reduction

            # ------ Set confidence mask of matching detections to 0
            # iou_gt_pred : [:, 980]
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)

            # mask : [:, 980]
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            # conf_mask[b] : [5, 196]
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # ------ Find best anchor for each ground truth -------------
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # ------ Set masks and target values for each ground truth --
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                cls_mask[b][best_n][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = self.object_scale
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(
                    max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(
                    max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                tconf[b][best_n][gj * width + gi] = iou
                tcls[b][best_n][gj * width + gi] = int(anno[4])


        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


def bbox_ious(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions
