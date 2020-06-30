import torch
import torch.nn as nn

class place_model(nn.Module):
    def __init__(self):
        super(place_model, self).__init__()

        self.lstm_sc = torch.nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc2 = torch.nn.Linear(1024, 1)
        self.fc3 = torch.nn.Linear(1024, 22)
        self.softmax = torch.nn.Softmax(dim=1)

        # self.fe = resnet.resnet50()

        # model = torch.nn.Sequential(fe, clsf())

    def forward(self, x):

        # x = self.fe(x)

        self.lstm_sc.flatten_parameters()
        N, T = x.size(0), x.size(1)
        x = self.lstm_sc(x)[0]
        
        change = x.reshape(N*T, -1)
        #x = self.fc1(x)
        change = self.fc2(change)
        change = change.reshape(N, T)
        #x = x.reshape(N*T, -1)
        
        M, _ = change.max(1)
        w = change - M.view(-1,1)
        w = w.exp()
        w = w.unsqueeze(1).expand(-1,w.size(1),-1)
        w = w.triu(1) - w.tril()
        w = w.cumsum(2)
        w = w - w.diagonal(dim1=1,dim2=2).unsqueeze(2)
        ww = w.new_empty(w.size())
        idx = M>=0
        ww[idx] = w[idx] + M[idx].neg().exp().view(-1,1,1)
        idx = ~idx
        ww[idx] = M[idx].exp().view(-1,1,1)*w[idx] + 1
        ww = (ww+1e-10).pow(-1)
        ww = ww/ww.sum(1,True)
        x = ww.transpose(1,2).bmm(x)
       
        x = x.reshape(N*T, -1)
        x = self.fc3(x)
        x = x.reshape(N*T, -1)

        return x

        ###########################################################

        # # person detector
        # logits, fmap = self.detector(image)

        # fmap = fmap.detach()

        # # fmap [b, 1024, 14, 14]
        # self.fmap_size = fmap.size(2)

        # # persons boxes
        # b_logits = []
        # b_labels = []
        # if not self.training:
        #     boxes = post_processing(logits, self.fmap_size, PersonCLS,
        #                             self.detector.anchors,
        #                             self.conf_threshold,
        #                             self.nms_threshold)
        #     if len(boxes) > 0 :
        #         for i, box in enumerate(boxes):
        #             num_box = len(box)
        #             with torch.no_grad():
        #                 box = np.clip(
        #                     np.stack(box)[:,:4].astype('float32'),
        #                     0.0 + 1e-3, self.fmap_size - 1e-3)
        #                 box = Variable(torch.from_numpy(box)).cuda(
        #                     self.device).detach()
        #                 b_box = Variable(
        #                     torch.zeros(num_box, 5).cuda(self.device)).detach()
        #                 b_box[:,1:] = box
        #                 i_fmap = roi_align(fmap[i][None],
        #                                    b_box.float(),
        #                                    (self.fmap_size//4,
        #                                     self.fmap_size//4))

        #             i_fmap = self.behavior_conv(i_fmap)
        #             i_logit = self.behavior_fc(i_fmap.view(num_box, -1))
        #             if num_box > 0:
        #                 b_logits.append(i_logit)

        #     return boxes, b_logits

        # if len(behavior_label) > 0 and self.training:
        #     for i, box in enumerate(label):
        #         num_box = len(box)
        #         if num_box == 0 :
        #             continue

        #         with torch.no_grad():
        #             box = np.clip(
        #                 np.stack(box)[:,:4].astype('float32')/self.img_size,
        #                 0.0 + 1e-3, self.fmap_size - 1e-3)
        #             box = torch.from_numpy(box).cuda(
        #                 self.device).detach() * self.fmap_size
        #             b_box = Variable(
        #                 torch.zeros(num_box, 5).cuda(self.device)).detach()
        #             b_box[:,1:] = box
        #             i_fmap = roi_align(fmap[i][None],
        #                                b_box.float(),
        #                                (self.fmap_size//4,
        #                                 self.fmap_size//4))

        #         batch = box.size(0)
        #         i_fmap = self.behavior_conv(i_fmap)
        #         i_logit = self.behavior_fc(i_fmap.view(batch, -1))
        #         if len(behavior_label[i]) > 0:
        #             b_logits.append(i_logit)
        #             b_labels.append(behavior_label[i])

        # return logits, b_logits, b_labels
