from collections import OrderedDict, Iterable, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
import importlib

from .yolo_layer import *


class Yolov3Base(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_loss_layers(self):
        return [self.yolo_0, self.yolo_1]

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward(self, x):
        n, c, h, w = x.shape
        assert c==3 and h%32==0 and w%32==0, f"Tensor shape should be [bs, 3, x*32, y*32], was {x.shape}"
        xb = self.forward_backbone(x)
        return self.forward_yolo(xb)

    def predict(self, x, threshold=0.25):
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0) 
        outputs = self.forward(x)
        return self.boxes_from_output(outputs, threshold)
    
    def boxes_from_output(self, outputs, threshold=0.25):
        boxes = [[] for j in range(outputs[0].size(0))]
        for i, layer in enumerate(self.get_loss_layers()):
            layer_boxes = layer.get_region_boxes(outputs[i], threshold=threshold)
            for j, layer_box in enumerate(layer_boxes):
                boxes[j] += layer_box
        return boxes

    def freeze_backbone(self, requires_grad=False):
        for p in self.backbone.parameters():
            p.requires_grad=requires_grad
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
    def freeze_info(self, print_all=False):
        d = defaultdict(set)
        print("Layer: param.requires_grad")
        for name, param in self.named_parameters():
            if print_all:
                print(f"{name}: {param.requires_grad}")
            else:
                d[name.split('.')[0]].add(param.requires_grad)
        if not print_all:
            for k,v in d.items():
                print(k, ': ', v)        

    def load_backbone(self, h5_path):
        state_old = self.state_dict()
        state_new = torch.load(h5_path)

        skipped_layers = []
        for k in list(state_new.keys()):
            if state_old[k].shape != state_new[k].shape:
                skipped_layers.append(k)
                del state_new[k]

        return self.load_state_dict(state_new, strict=False), skipped_layers


###################################################################
## Common helper modules

class ConvBN(nn.Module):
 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1)//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        assert(x.data.dim() == 4)
        return nn.Upsample(scale_factor=self.stride, mode='nearest')(x)
