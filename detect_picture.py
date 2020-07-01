import argparse
import torch
from yolov3.yolov3_tiny import *
from dutils import PictureDetection

parser = argparse.ArgumentParser(prog = 'top', description = 'Object Detection')

parser.add_argument('-i', '--input', default='person.jpg')
parser.add_argument('-is', '--input_size', default='416')
parser.add_argument('-st', '--score_threshold', default='0.25')
parser.add_argument('-it', '--iou_threshold', default='0.5')

args = parser.parse_args()


if __name__ == "__main__":
    s = int(args.input_size)
    score = float(args.score_threshold)
    iou = float(args.iou_threshold)
    
    model = Yolov3Tiny(num_classes=80)
    model.load_state_dict(torch.load('models/yolov3(tiny).h5'))
    path = "images/" + args.input
    PictureDetection(model, path, size=(s, s), threshold=score, IoU=iou, dataset='coco').show()