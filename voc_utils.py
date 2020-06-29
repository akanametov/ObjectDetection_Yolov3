import os
import torch
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

class TransformAnnotation(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj[0].text.lower().strip()
            bbox = obj[4]
            bndbox = [int(bb.text)-1 for bb in bbox]
            res += [bndbox + [name]]
        return res
    
class VOCDataset(Dataset):
    def __init__(self, path, input_shape=(416, 416), mode='train', dataset='voc'):
        self.path = path
        self.mode = mode
        self.S = input_shape[0]
        self.transform = Compose([Resize(input_shape), ToTensor()])
        self.target_transform = TransformAnnotation()
        
        f_path = os.path.join(path, 'ImageSets/Main/train_new.txt')
        self.files = [i.strip('\n') for i in open(f_path).readlines()]
        random.seed(42)
        random.shuffle(self.files)
        
        self.ip = os.path.join(path, 'JPEGImages/')
        self.lp = os.path.join(path, 'Annotations/')
        
        self.classes = [c.strip() for c in open('data/'+dataset+'_classes.txt').readlines()]

    def __getitem__(self, i):
        f = self.files[i]
        x = Image.open(self.ip + f + '.jpg').convert('RGB')
        W, H = x.size
        x = self.transform(x)
        if self.mode=='test':
            return x
        else:
            y = ET.parse(self.lp + f + '.xml').getroot()
            y = self.target_transform(y)
            for yi in y:
                x1, y1, x2, y2, c = yi
                ci = self.classes.index(c)
                w, h = x2-x1, y2-y1
                cx, cy = x2-w/2, y2-h/2
                yi[:] = [cx, cy, w, h, ci]
            
            scale = torch.tensor([1/W, 1/H, 1/W, 1/H, 1])
            y = torch.tensor(y)*scale
            return x, y

    def __len__(self):
        return len(self.files)
    
    def classes(self,):
        return self.classes
    
def PlotSample(x, y, dataset='voc'):
    x = ToPILImage()(x)
    y = y.numpy()
    W, H = x.size
    scale = np.array((W, H, W, H))
    thickness = (W + H)//300
    font = ImageFont.truetype(font='data/font/FiraMono-Medium.otf', size=int(0.03*H + 0.5))
    classes = [c.strip() for c in open('data/'+dataset+'_classes.txt').readlines()]
    colors = np.load('data/'+dataset+'_colors.npy')

    for yi in y:
        draw = ImageDraw.Draw(x)
        cls=classes[int(yi[-1])]
        color=colors[int(yi[-1])]
        cx, cy, w, h = yi[ :4]*scale
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        size = draw.textsize(cls, font)
        
        if y1 - size[1] >= 0:
            origin = np.array([x1, y1-size[1]])
        else:
            origin = np.array([x1, y1+1])
        
        for k in range(thickness):
            draw.rectangle([x1+k, y1+k, x2-k, y2-k], outline=tuple(color))
        # Draw Label
        draw.rectangle([tuple(origin), tuple(origin+size)], fill=tuple(color))
        draw.text(origin, cls, fill=(0, 0, 0), font=font)
        del draw
    return x