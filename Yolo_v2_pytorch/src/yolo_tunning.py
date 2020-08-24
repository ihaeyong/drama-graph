"""
@author: Viet Nguyen <nhviet1009@gmail.com>
modified by Haeyong.Kang
"""

import torch.nn as nn
import torch

class YoloD(nn.Module):
    def __init__(self, pre_model,
                 anchors=[(1.3221, 1.73145),
                          (3.19275, 4.00944),
                          (5.05587, 8.09892),
                          (9.47112, 4.84053),
                          (11.2364, 10.0071)]):
        super(YoloD, self).__init__()

        self.anchors = anchors

        self.stage1_conv1 = pre_model.stage1_conv1
        self.stage1_conv2 = pre_model.stage1_conv2
        self.stage1_conv3 = pre_model.stage1_conv3
        self.stage1_conv4 = pre_model.stage1_conv4
        self.stage1_conv5 = pre_model.stage1_conv5
        self.stage1_conv6 = pre_model.stage1_conv6
        self.stage1_conv7 = pre_model.stage1_conv7
        self.stage1_conv8 = pre_model.stage1_conv8
        self.stage1_conv9 = pre_model.stage1_conv9
        self.stage1_conv10 = pre_model.stage1_conv10
        self.stage1_conv11 = pre_model.stage1_conv11
        self.stage1_conv12 = pre_model.stage1_conv12
        self.stage1_conv13 = pre_model.stage1_conv13

        self.stage2_a_maxpl = pre_model.stage2_a_maxpl
        self.stage2_a_conv1 = pre_model.stage2_a_conv1
        self.stage2_a_conv2 = pre_model.stage2_a_conv2
        self.stage2_a_conv3 = pre_model.stage2_a_conv3
        self.stage2_a_conv4 = pre_model.stage2_a_conv4
        self.stage2_a_conv5 = pre_model.stage2_a_conv5
        self.stage2_a_conv6 = pre_model.stage2_a_conv6
        self.stage2_a_conv7 = pre_model.stage2_a_conv7

        self.stage2_b_conv = pre_model.stage2_b_conv

        self.stage3_conv1 = pre_model.stage3_conv1


    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output

        output_1 = self.stage2_a_maxpl(output)
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.view(
            batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output_fmap = self.stage3_conv1(output)

        # output_1 is used for behavior learning

        return output_fmap, output_1
