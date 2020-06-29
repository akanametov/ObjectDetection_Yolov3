import argparse
import torch
from yolov3.yolov3_tiny import *
from dutils import PictureDetection

parser = argparse.ArgumentParser(prog = 'top', description = 'Detect Objects')

parser.add_argument('-i', '--input', default='person.jpg')

args = parser.parse_args()


if __name__ == "__main__":
    
    model = Yolov3Tiny(num_classes=80)
    model.load_state_dict(torch.load('models/yolov3(tiny).h5'))
    path = "images/" + args.input
    PictureDetection(model, path, dataset='coco').show()