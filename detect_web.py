import argparse
import cv2
import torch
import numpy as np
from dutils import VideoDetection
from yolov3.yolov3_tiny import *

parser = argparse.ArgumentParser(prog = 'top', description = 'Web-Object Detection')

parser.add_argument('-is', '--input_size', default='256')
parser.add_argument('-st', '--score_threshold', default='0.25')
parser.add_argument('-it', '--iou_threshold', default='0.5')

args = parser.parse_args()

if __name__ == "__main__":
    s = int(args.input_size)
    score = float(args.score_threshold)
    iou = float(args.iou_threshold)
    
    model = Yolov3Tiny(num_classes=80)
    model.load_state_dict(torch.load('models/yolov3(tiny).h5'))
    cap = cv2.VideoCapture(0)
    cap.set(3, s)
    cap.set(4, s)

    while True:
        success, img = cap.read()
        x = torch.from_numpy(img).permute(2, 0, 1)
        image = VideoDetection(model, x, size=(s, s), threshold=score, IoU=iou, dataset='coco')
        cv2.imshow('ObjectDetection', np.array(image))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break