#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:44:20 2018

@author: nickblauch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys

from familiarity.dl.vsnet import ScaleInvConv2d
from familiarity.dl.cornet_z import Identity


class L2Normalizer(nn.Module):
    """
    normalizes features such that the p-norm along dim is equal to alpha for all examples
    useful for L2-constrained softmax (Ranjan et al 2017)
    """
    def __init__(self, p, alpha, dim=1):
        super().__init__()
        self.p=p
        self.alpha=alpha
        self.dim=dim
    
    def __call__(self, inputs):
        outputs = self.alpha*F.normalize(inputs, self.p, dim=self.dim)
        return outputs
        
                    

class Vgg16(nn.Module):

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __init__(self, n_classes=2622, initialize_weights=True, conv_scales=None, l2_alpha=None):
        super().__init__()
        if conv_scales is None:
            Conv2d = nn.Conv2d
            kwargs = {}
        else:
            Conv2d = ScaleInvConv2d
            kwargs = {'scales':conv_scales}
        self.debug_feats = OrderedDict() # only used for feature verification
        self.conv1_1 = Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv2_1 = Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv3_1 = Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv4_1 = Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv5_1 = Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), **kwargs)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.fc6 = Conv2d(512, 4096, kernel_size=[7, 7], stride=(1, 1), **kwargs)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.l2_constraint = L2Normalizer(p=2, alpha=l2_alpha, dim=1) if l2_alpha is not None else Identity()
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, n_classes)
        self.prob = nn.Softmax(dim=1)

        if initialize_weights:
            self._initialize_weights()

    def forward(self, x0, get_penultimate=False):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31 = self.pool5(x30)
        x32_preflatten = self.fc6(x31)
        x32 = x32_preflatten.view(x32_preflatten.size(0), x32_preflatten.size(1))
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.l2_constraint(self.relu7(x35)) # making this one operation for backwards compatibility with layer numbers
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37) # return this for cross entropy loss

        if get_penultimate:
            return x36
        else:
            return x38

    def forward_debug(self, x0, return_layers, device='cpu'):
        acts = OrderedDict()
        acts['x0'] = x0
        acts['x1'] = self.conv1_1(acts['x0'])
        acts['x2'] = self.relu1_1(acts['x1'])
        acts['x3'] = self.conv1_2(acts['x2'])
        acts['x4'] = self.relu1_2(acts['x3'])
        acts['x5'] = self.pool1(acts['x4'])
        acts['x6'] = self.conv2_1(acts['x5'])
        acts['x7'] = self.relu2_1(acts['x6'])
        acts['x8'] = self.conv2_2(acts['x7'])
        acts['x9'] = self.relu2_2(acts['x8'])
        acts['x10'] = self.pool2(acts['x9'])
        acts['x11'] = self.conv3_1(acts['x10'])
        acts['x12'] = self.relu3_1(acts['x11'])
        acts['x13'] = self.conv3_2(acts['x12'])
        acts['x14'] = self.relu3_2(acts['x13'])
        acts['x15'] = self.conv3_3(acts['x14'])
        acts['x16'] = self.relu3_3(acts['x15'])
        acts['x17'] = self.pool3(acts['x16'])
        acts['x18'] = self.conv4_1(acts['x17'])
        acts['x19'] = self.relu4_1(acts['x18'])
        acts['x20'] = self.conv4_2(acts['x19'])
        acts['x21'] = self.relu4_2(acts['x20'])
        acts['x22'] = self.conv4_3(acts['x21'])
        acts['x23'] = self.relu4_3(acts['x22'])
        acts['x24'] = self.pool4(acts['x23'])
        acts['x25'] = self.conv5_1(acts['x24'])
        acts['x26'] = self.relu5_1(acts['x25'])
        acts['x27'] = self.conv5_2(acts['x26'])
        acts['x28'] = self.relu5_2(acts['x27'])
        acts['x29'] = self.conv5_3(acts['x28'])
        acts['x30'] = self.relu5_3(acts['x29'])
        acts['x31'] = self.pool5(acts['x30'])
        x32_preflatten = self.fc6(acts['x31'])
        acts['x32'] = x32_preflatten.view(x32_preflatten.size(0), x32_preflatten.size(1))
        acts['x33'] = self.relu6(acts['x32'])
        acts['x34'] = self.dropout6(acts['x33'])
        acts['x35'] = self.fc7(acts['x34'])
        acts['x36'] = self.l2_constraint(self.relu7(acts['x35'])) 
        acts['x37'] = self.dropout7(acts['x36'])
        acts['x38'] = self.fc8(acts['x37'])
        acts['x39'] = self.prob(acts['x38']) # we might want an explicit probability distribution

        out = {}
        for layer in return_layers:
            out['x{}'.format(layer)] = acts.pop('x{}'.format(layer)).to(device)
        del acts
        return out


def vgg_face_dag(weights_path=None, n_classes=2622, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg16(n_classes, initialize_weights=weights_path is None)
    if weights_path:
        state_dict = torch.load(weights_path)
        state_dict['fc6.weight'] = state_dict['fc6.weight'].reshape(4096, 512, 7, 7)
        model.load_state_dict(state_dict)
    model.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                 'std': [1, 1, 1],
                 'imageSize': [224, 224, 3]}
    return model

def vgg16(weights_path=None, n_classes=1000, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    if weights_path:
        state_dict = torch.load(weights_path)
        if isinstance(state_dict, tuple):
            state_dict = state_dict[0]
            state_dict_ = OrderedDict()
            for key in state_dict.keys():
                state_dict_[key.replace('module.','')] = state_dict[key]
            state_dict = state_dict_
        n_classes = len(state_dict['fc8.bias'])
    model = Vgg16(n_classes, initialize_weights=weights_path is None, **kwargs)
    if weights_path:
        model.load_state_dict(state_dict)
    model.meta = {'mean': [122.74494171142578, 114.94409942626953, 101.64177703857422],
                 'std': [1, 1, 1],
                 'imageSize': [224, 224, 3]}
    return model

class Vgg_m_face_bn_dag(nn.Module):

    def __init__(self, n_classes=2622):
        super().__init__()
        self.debug_feats = OrderedDict() # only used for feature verification
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(4096, 4096)
        self.bn55 = nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(4096, n_classes)

    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)
        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20).squeeze()
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24 = self.relu7(x23)
        x25 = self.fc8(x24)
        return x25


    def forward_debug(self, x0):
        """ This purpose of this function is to provide an easy debugging
        utility for the converted network.  Cloning is used to prevent in-place
        operations from modifying feature artefacts. You can prevent the
        generation of this function by setting `debug_mode = False` in the
        importer tool.
        """

        self.debug_feats['x0'] = x0.cpu()
        x1 = self.conv1(x0)
        self.debug_feats['x1'] = x1.cpu()
        x2 = self.bn49(x1)
        self.debug_feats['x2'] = x2.cpu()
        x3 = self.relu1(x2)
        self.debug_feats['x3'] = x3.cpu()
        x4 = self.pool1(x3)
        self.debug_feats['x4'] = x4.cpu()
        x5 = self.conv2(x4)
        self.debug_feats['x5'] = x5.cpu()
        x6 = self.bn50(x5)
        self.debug_feats['x6'] = x6.cpu()
        x7 = self.relu2(x6)
        self.debug_feats['x7'] = x7.cpu()
        x8 = self.pool2(x7)
        self.debug_feats['x8'] = x8.cpu()
        x9 = self.conv3(x8)
        self.debug_feats['x9'] = x9.cpu()
        x10 = self.bn51(x9)
        self.debug_feats['x10'] = x10.cpu()
        x11 = self.relu3(x10)
        self.debug_feats['x11'] = x11.cpu()
        x12 = self.conv4(x11)
        self.debug_feats['x12'] = x12.cpu()
        x13 = self.bn52(x12)
        self.debug_feats['x13'] = x13.cpu()
        x14 = self.relu4(x13)
        self.debug_feats['x14'] = x14.cpu()
        x15 = self.conv5(x14)
        self.debug_feats['x15'] = x15.cpu()
        x16 = self.bn53(x15)
        self.debug_feats['x16'] = x16.cpu()
        x17 = self.relu5(x16)
        self.debug_feats['x17'] = x17.cpu()
        x18 = self.pool5(x17)
        self.debug_feats['x18'] = x18.cpu()
        x19 = self.fc6(x18)
        self.debug_feats['x19'] = x19.cpu()
        x20 = self.bn54(x19)
        self.debug_feats['x20'] = x20.cpu()
        x21 = self.relu6(x20).squeeze()
        self.debug_feats['x21'] = x21.cpu()
        x22 = self.fc7(x21)
        self.debug_feats['x22'] = x22.cpu()
        x23 = self.bn55(x22)
        self.debug_feats['x23'] = x23.cpu()
        x24 = self.relu7(x23)
        self.debug_feats['x24'] = x24.cpu()
        x25 = self.fc8(x24)
        self.debug_feats['x25'] = x25.cpu()

def vgg_m_face_bn_dag(weights_path=None, n_classes=2622, map_location=None, freeze=False, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_m_face_bn_dag(n_classes)
    if weights_path:
        state_dict = torch.load(weights_path, map_location)
        model.load_state_dict(state_dict)

    if freeze:
        for child in model.children():
            if 'BatchNorm' in str(child):
                child.track_running_stats = False
    #            child.reset_parameters()


    return model
