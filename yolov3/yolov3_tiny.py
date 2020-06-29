import torch
import torch.nn as nn
from .yolo_layer import *
from .yolov3_base import *

class Yolov3Tiny(Yolov3Base):

    def __init__(self, num_classes, use_wrong_anchors=False):
        super().__init__()

        self.num_classes = num_classes
        self.return_out_boxes = False
        self.skip_backbone = False

        self.backbone = Yolov3TinyBackbone()

        Ar = 3
        self.yolo_0_pre = nn.Sequential(OrderedDict([
            ('14_convbatch',    ConvBN(256, 512, 3, 1, 1)),
            ('15_conv',         nn.Conv2d(512, (num_classes+5)*Ar, 1, 1, 0))]))
        
        anchors0=[(81.,82.), (135.,169.), (344.,319.)]
        self.yolo_0 = YoloLayer(anchors=anchors0, stride=32, num_classes=num_classes)

        self.up_1 = nn.Sequential(OrderedDict([
            ('17_convbatch',    ConvBN(256, 128, 1, 1, 0)),
            ('18_upsample',     Upsample(2))]))

        self.yolo_1_pre = nn.Sequential(OrderedDict([
            ('19_convbatch',    ConvBN(128+256, 256, 3, 1, 1)),
            ('20_conv',         nn.Conv2d(256, (num_classes+5)*Ar, 1, 1, 0))]))

        if use_wrong_anchors:
            anchors1 = [(23.,27.),  (37.,58.),  (81.,82.)]
        else: 
            anchors1 = [(10.,14.),  (23.,27.),  (37.,58.)]

        self.yolo_1 = YoloLayer(anchors=anchors1, stride=16, num_classes=num_classes)

    def get_loss_layers(self):
        return [self.yolo_0, self.yolo_1]

    def forward_yolo(self, xb):
        x0, x1 = xb[0], xb[1]
        y0 = self.yolo_0_pre(x1)

        x_up = self.up_1(x1)
        x_up = torch.cat((x_up, x0), 1)
        y1 = self.yolo_1_pre(x_up)
        return [y0, y1]

###################################################################
## Backbone and helper modules

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x


class Yolov3TinyBackbone(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.layers =  nn.Sequential(OrderedDict([
            ('0_convbatch',     ConvBN(input_channels, 16, 3, 1, 1)),
            ('1_max',           nn.MaxPool2d(2, 2)),
            ('2_convbatch',     ConvBN(16, 32, 3, 1, 1)),
            ('3_max',           nn.MaxPool2d(2, 2)),
            ('4_convbatch',     ConvBN(32, 64, 3, 1, 1)),
            ('5_max',           nn.MaxPool2d(2, 2)),
            ('6_convbatch',     ConvBN(64, 128, 3, 1, 1)),
            ('7_max',           nn.MaxPool2d(2, 2)),
            ('8_convbatch',     ConvBN(128, 256, 3, 1, 1)),
            ('9_max',           nn.MaxPool2d(2, 2)),
            ('10_convbatch',    ConvBN(256, 512, 3, 1, 1)),
            ('11_max',          MaxPoolStride1()),
            ('12_convbatch',    ConvBN(512, 1024, 3, 1, 1)),
            ('13_convbatch',    ConvBN(1024, 256, 1, 1, 0))]))
    
    def forward(self, x):
        idx=9
        x0 = self.layers[ :idx](x)
        x1 = self.layers[idx: ](x0)
        return x0, x1