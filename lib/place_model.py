import torch
import torch.nn as nn
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os, sys, math
import os.path
import torch
import json 
import torch.utils.model_zoo as model_zoo
from Yolo_v2_pytorch.src.utils import *
from Yolo_v2_pytorch.src.yolo_net import Yolo
from Yolo_v2_pytorch.src.yolo_tunning import YoloD
import numpy as np
import torch.nn.functional as F
from Yolo_v2_pytorch.src.rois_utils import anchorboxes
from Yolo_v2_pytorch.src.anotherMissOh_dataset import FaceCLS
from lib.person_model import person_model


label_dict = {'' : 9, 'beach':0, 'cafe':1, 'car':2, 'convenience store':3, 'garden':4, 'home':5, 'hospital':6, 'kitchen':7,
    'livingroom':8, 'none':9, 'office':10, 'park':11, 'playground':12, 'pub':13, 'restaurant':14, 'riverside':15, 'road':16,
    'rooftop':17, 'room':18, 'studio':19, 'toilet':20, 'wedding hall':21
}

label_dict_wo_none = {'beach':0, 'cafe':1, 'car':2, 'convenience store':3, 'garden':4, 'home':5, 'hospital':6, 'kitchen':7,
    'livingroom':8, 'none':9, 'office':10, 'park':11, 'playground':12, 'pub':13, 'restaurant':14, 'riverside':15, 'road':16,
    'rooftop':17, 'room':18, 'studio':19, 'toilet':20, 'wedding hall':21
}
def label_mapping(target):
    temp = []
    for idx in range(len(target)):
        if target[idx][0][:3] == 'con':
            target[idx][0] = 'convenience store'
        temp.append(label_dict[target[idx][0]])
    return temp

def label_remapping(target):
    inv_label_dict = {v: k for k, v in label_dict_wo_none.items()}
    temp = []
    for idx in range(len(target)):
        temp.append(inv_label_dict[target[idx]])
    return temp

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
    
def place_buffer(images_norm, buffer_images):
    if len(buffer_images) == 0:
        buffer_images = images_norm
    if len(buffer_images) < 10:
        for idx in range(10-len(buffer_images)):
            buffer_images = images_norm[0] + images_norm
    assert len(buffer_images) == 10, 'Buffer failed'

    return buffer_images

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



sample_default = [105, 462, 953, 144, 108, 13, 123, 510, 1690, 19914, 1541, 126, 67, 592, 1010, 53, 2087, 0, 1547, 576, 74, 0]

def CB_loss(labels, logits, beta=0.99, gamma=0.5, samples_per_cls=sample_default, no_of_classes=22, loss_type='softmax'):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).cpu().float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot.cuda(), logits, weights.cuda(), gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot.cuda(), weight = weights.cuda())
    return cb_loss

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class place_model(nn.Module):
    def __init__(self, num_persons, num_faces, device):
        super(place_model, self).__init__()

        pre_model = Yolo(num_persons).cuda(device)

        num_face_cls = num_faces

        self.detector = YoloD(pre_model).cuda(device)
        self.place_conv = nn.Sequential(nn.Conv2d(1024, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # self.lstm_sc = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        # self.bert_fc1 = torch.nn.Linear(128, 768)
        # self.bert_fc2 = torch.nn.Linear(768, 128)
        self.bert = BERT()

        self.fc2 = torch.nn.Linear(128, 1)
        self.fc3 = torch.nn.Linear(128, 22)
        self.softmax = torch.nn.Softmax(dim=1)



        # # define face
        # self.face_conv = nn.Conv2d(
        #     1024, len(self.detector.anchors) * (5 + num_face_cls), 1, 1, 0, bias=False)

    def forward(self, image):
        N, T , C, H, W = image.size(0), image.size(1), image.size(2), image.size(3), image.size(4)
        image = image.reshape(N*T, C, H, W)
        # feature map of backbone
        fmap, output_1 = self.detector(image)
        fmap = self.place_conv(fmap)
        x = self.avgpool(fmap)
        x = x.reshape(N, T, -1)
        
        # self.lstm_sc.flatten_parameters()
        # N, T = x.size(0), x.size(1)
        # x = self.lstm_sc(x)[0]

        # x = self.bert_fc1(x)
        x = self.bert(x)
        # x = self.bert_fc2(x)
        
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

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size=0, hidden=128, n_layers=5, attn_heads=8, dropout=0.):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len])
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            # x = transformer.forward(x, mask)
            x = transformer.forward(x, None)

        return x

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super(BERTEmbedding, self).__init__()
        # self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.position = PositionalEmbedding(d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        # x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        x = sequence + self.position(sequence)
        return self.dropout(x)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        #self.activation = nn.GELU()
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
