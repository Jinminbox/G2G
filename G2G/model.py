import torch.nn as nn
import torch
import math
import numpy as np
from einops import rearrange
import copy
from sklearn.metrics import mutual_info_score
from torch import Tensor

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from typing import Union, Tuple, Optional
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

# import preprocess.SEED_pretrain as SEED_pre
from utils import *
import torchvision.models as models
import logging


# 运行位置相关的自注意力
class RelationAwareness(nn.Module):
    def __init__(self, args):
        super(RelationAwareness, self).__init__()

        self.head = args.head_num
        self.input_size = args.config["input_size"] # eeg input size on each electrode, 5
        self.location_size = args.config["location_size"] # 3
        self.expand_size = args.config["expand_size"] # expand eeg, eye, and location to the same dimen, 10

        self.location_em = nn.Linear(self.location_size, self.head*self.expand_size) # 3 --> 6*10
        self.data_em = nn.Linear(self.input_size, self.head*self.expand_size) # 5 --> 6*10
        self.eye_em = nn.Linear(10, self.head*self.expand_size) # 10 --> 6*10
        self.relu = nn.ReLU()
        self.args = args

        self.a = nn.Parameter(torch.empty(size=(2*self.expand_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, feature, location, eye):

        feature_embed = self.data_em(feature)
        location_embed = self.location_em(location)
        feature_local_embed = self.relu(feature_embed + location_embed)

        eye_embed = self.relu(self.eye_em(eye))
        eeg_eye_embed = torch.cat([feature_local_embed, eye_embed], 1)

        feature_ = rearrange(eeg_eye_embed, "b n (h d) -> b h n d", h=self.head)

        two_d_feature = self.cal_att_matrix(feature_)
        return two_d_feature

    def cal_att_matrix(self, feature):

        data = []
        batch_size, head,  N = feature.size(0), feature.size(1), feature.size(2)
        Wh1 = torch.matmul(feature, self.a[:self.expand_size, :])
        Wh2 = torch.matmul(feature, self.a[self.expand_size:, :])
        # broadcast add
        Wh2_T = rearrange(Wh2, "b n h d -> b n d h")
        e = Wh1 + Wh2_T
        return e


class ConvNet(nn.Module):
    def __init__(self, emb_size, args, cifar_flag=False):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        self.last_hidden = self.hidden * 16 if not cifar_flag else self.hidden
        # self.last_hidden = self.hidden * 1 if not cifar_flag else self.hidden
        self.emb_size = emb_size
        self.args = args

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=12,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.max = nn.MaxPool2d(kernel_size=2)

        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_second = nn.Sequential(nn.Linear(in_features=self.last_hidden * 2 ,
                                                    out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4 ,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.conv_4(out_3)
        output_data0 = self.max(out_3)
        out1 = self.layer_last(output_data.view(output_data.size(0), -1))
        out2 = self.layer_second(output_data0.view(output_data0.size(0), -1))

        out = torch.cat((out1, out2), dim=1)  # (batch_size, 256)

        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.resnet50(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.resnet18(x)
        return x



# 执行数据的预处理，包括att将数据升维和resnet提取数据特征
# ConvNet encoder
class EncoderNet(nn.Module):
    def __init__(self, args):
        super(EncoderNet, self).__init__()
        logger = logging.getLogger("model")
        self.args = args
        self.resnet_embed = 256
        self.backbone_output =  self.resnet_embed * 2

        self.relationAwareness = RelationAwareness(args = self.args)
        self.rand_order = random_1D_node(self.args.rand_ali_num, self.args.config["eeg_node_num"])
        print(self.rand_order)

        # define selected backbone
        self.backbone = None
        if self.args.backbone == "ConvNet":
            self.backbone =  ConvNet(self.resnet_embed, args=self.args)
        elif self.args.backbone == "ResNet18":
            self.backbone = ResNet18()
        elif self.args.backbone == "ResNet50":
            self.backbone = ResNet50()
        else:
            raise RuntimeError("Wrong backbone!")

        # get node location
        self.location = torch.from_numpy(return_coordinates()).to(self.args.device)

        self.l_relu = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(self.backbone_output)
        self.bn_2D = nn.BatchNorm2d(12)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

        self.mlp_0 = nn.Linear(512, self.backbone_output)
        self.mlp_1 = nn.Linear(self.backbone_output, self.args.config["num_class"])


    def forward(self, x):

        # output_original =copy.deepcopy(x.view(x.size(0), -1))
        # 随机排列运行分支
        ######################################################
        ran_list = []
        for index in range(self.args.rand_ali_num):
            x_eeg = x[:, :310]
            x_eye = x[:, 310:380]

            x_eeg = rearrange(x_eeg, 'b (h c) -> b h c', h=62) #(32,62,5)
            x_eye = rearrange(x_eye, 'b (h c) -> b h c', h=self.args.config["sup_node_num"]) #(32,6,10)

            x_random, coor_random = x_eeg[:, self.rand_order[index], :], self.location[self.rand_order[index], :]
            x_ = self.relationAwareness(x_random, coor_random, x_eye) # (batch_size, 62, 62, 3)

            ran_list.append(x_)

        x_ = torch.cat(tuple(ran_list), 1)  # (batch_size, self.args.rand_ali_num*self.head, N, N)
        x_ = self.bn_2D(x_)

        output = self.backbone(x_)

        x = self.dropout(output)
        x = self.mlp_0(x)
        x = self.l_relu(x)
        x = self.bn(x)
        x = self.mlp_1(x)

        return x


